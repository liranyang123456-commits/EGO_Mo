#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build dense trajectory comparisons from overlapping 3 s displacement edges.

Translation edges are rotated with reference orientation to isolate translation
quality. This is explicitly not a pure-IMU pose result; gyro orientation is
displayed separately in the UI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
import tools.train_seq as api
from tools.benchmark_methods import _ensemble, _predict, _ridge_fit, _ridge_predict, _strapdown
from tools.train_external_architectures import make_model, predict as predict_external

DATA = ROOT / "datasets"


def _edge_pack(t, R, p, usable, usb_t, usb, context, horizon, length):
    idx = np.flatnonzero(usable)
    X, y, edges = [], [], []
    node_for_frame = {int(frame): node for node, frame in enumerate(idx)}
    for i in idx:
        prev = idx[idx < i]
        if not len(prev):
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - horizon)))])
        dt = float(t[i] - t[j])
        if not (0.6 * horizon <= dt <= 1.5 * horizon):
            continue
        if t[j] - context < usb_t[0] or t[i] + context > usb_t[-1]:
            continue
        feat = api._grid(usb_t, usb, float(t[j]), float(t[i]), context)
        if len(feat) < length:
            feat = np.vstack((feat, np.zeros((length - len(feat), 8), np.float32)))
        X.append(feat[:length])
        y.append(R[j].T @ (p[i] - p[j]))
        edges.append((node_for_frame[j], node_for_frame[int(i)]))
    return idx, np.stack(X), np.asarray(y, np.float32), np.asarray(edges, np.int64)


def _components(n, edges):
    parent = np.arange(n)

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in edges:
        ra, rb = root(int(a)), root(int(b))
        if ra != rb:
            parent[rb] = ra
    groups = {}
    for i in range(n):
        groups.setdefault(root(i), []).append(i)
    return [np.asarray(nodes) for nodes in groups.values() if len(nodes) >= 3]


def _solve_positions(gt_p, gt_R, edges, dp):
    n = len(gt_p)
    output = np.full((n, 3), np.nan)
    for nodes in _components(n, edges):
        node_set = set(nodes.tolist())
        edge_idx = [k for k, (a, b) in enumerate(edges) if int(a) in node_set and int(b) in node_set]
        mapping = {int(node): i for i, node in enumerate(nodes)}
        rows, cols, vals = [], [], []
        rhs = []
        row = 0
        for k in edge_idx:
            a, b = (int(v) for v in edges[k])
            rows += [row, row]
            cols += [mapping[a], mapping[b]]
            vals += [-1.0, 1.0]
            rhs.append(gt_R[a] @ dp[k])
            row += 1
        # One metric anchor per connected component.
        rows.append(row)
        cols.append(0)
        vals.append(1.0)
        rhs.append(gt_p[nodes[0]])
        A = coo_matrix((vals, (rows, cols)), shape=(row + 1, len(nodes))).tocsr()
        rhs = np.asarray(rhs)
        solved = np.column_stack([lsqr(A, rhs[:, axis], atol=1e-10, btol=1e-10)[0] for axis in range(3)])
        output[nodes] = solved
    return output


def _metrics(pred, truth):
    mask = np.isfinite(pred).all(axis=1)
    err = np.linalg.norm(pred[mask] - truth[mask], axis=1) * 1000
    return {
        "nodes": int(mask.sum()),
        "ate_rmse_mm": round(float(np.sqrt(np.mean(err ** 2))), 2),
        "ate_mean_mm": round(float(err.mean()), 2),
        "ate_p90_mm": round(float(np.quantile(err, 0.9)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--extra-edges", type=Path, default=None,
                    help="npz with per-edge test predictions (e.g. session_cv_benchmark.npz) to add as methods")
    ap.add_argument("--extra-keys", nargs="*", default=["physnet", "imunet", "equal_blend"])
    args = ap.parse_args()
    api.LABELS = "pose_gt_raw"
    api.TARGET_S = 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = range(5)
    real_models = _ensemble([DATA / "traj_run_v14" / f"seq_ctx2_h3_s{s}.pt" for s in seeds], device)
    hybrid_models = _ensemble(
        [DATA / "traj_run_v17" / f"seq_ctx2_h3_r0_p35_lr0.001_s{s}.pt" for s in seeds],
        device,
    )
    zero_models = _ensemble(
        [DATA / "traj_run_v19" / f"pretrained_ctx2_h3_p35_preonly_s{s}.pt" for s in seeds],
        device,
    )
    ronin_fine = make_model("ronin_resnet").to(device)
    ronin_fine.load_state_dict(torch.load(
        DATA / "external_benchmark" / "ronin_resnet_fine_s0.pt", map_location=device
    ))
    ronin_fine.eval()
    imunet_real = make_model("imunet").to(device)
    imunet_real.load_state_dict(torch.load(
        DATA / "external_benchmark" / "imunet_real_s0.pt", map_location=device
    ))
    imunet_real.eval()
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    train_parts = [api.build(name, context, length) for name in split["train"]]
    ridge = _ridge_fit(
        np.concatenate([p[0] for p in train_parts]),
        np.concatenate([p[1] for p in train_parts]),
    )
    cases = {}

    # Real held-out session.
    real_name = split["test"][0]
    raw = np.load(DATA / "pose_gt_raw" / f"{real_name}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{real_name}.npz")["delay_usb"][0])
    usb_t, usb = load_imu(DATA / real_name / "imu_stream.csv")
    real_pack = _edge_pack(
        raw["t"] - delay, raw["R"].astype(float), raw["p"].astype(float),
        raw["usable"] == 1, usb_t, usb, context, horizon, length,
    )
    cases["real_test"] = (real_name, raw, real_pack)

    # One complete synthetic mixed-motion test session.
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    entry = next(e for e in manifest["imu"]["test"] if e["motion"] == "mixed")
    session = args.corpus / entry["path"]
    gt = np.load(session / "ground_truth.npz")
    usb_t, usb = load_imu(session / "imu_stream.csv")
    usable = np.ones(len(gt["t"]), bool)
    synth_pack = _edge_pack(
        gt["t"], gt["R"].astype(float), gt["p"].astype(float),
        usable, usb_t, usb, context, horizon, length,
    )
    cases["synthetic_test"] = (entry["name"], gt, synth_pack)

    out = DATA / "trajectory_comparison"
    out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for case, (name, gt, pack) in cases.items():
        idx, X, y, edges = pack
        R = gt["R"][idx].astype(float)
        p = gt["p"][idx].astype(float)
        pred_real = _predict(real_models, X, device)
        pred_hybrid = _predict(hybrid_models, X, device)
        edge_predictions = {
            "zero": np.zeros_like(y),
            "strapdown": _strapdown(X),
            "ridge": _ridge_predict(ridge, X),
            "zero_shot": _predict(zero_models, X, device),
            "real_only": pred_real,
            "fine_tuned": pred_hybrid,
            "blend": 0.2 * pred_real + 0.8 * pred_hybrid,
            "ronin_fine": predict_external(ronin_fine, X, device),
            "imunet_real": predict_external(imunet_real, X, device),
            "ground_truth_edges": y,
        }
        edge_predictions["architecture_blend"] = (
            0.171321 * pred_real
            + 0.709310 * pred_hybrid
            + 0.119370 * edge_predictions["imunet_real"]
        )
        if case == "real_test" and args.extra_edges is not None:
            extra = np.load(args.extra_edges, allow_pickle=True)
            assert np.allclose(extra["y"], y, atol=1e-6), "edge enumeration mismatch"
            if "equal_blend" in args.extra_keys and "equal_blend" not in extra.files:
                edge_predictions["cv_equal_blend"] = 0.5 * (extra["physnet"] + extra["imunet"])
            for key in args.extra_keys:
                if key in extra.files:
                    edge_predictions[f"cv_{key}"] = extra[key].astype(np.float32)
        trajectories = {"ground_truth": p}
        metrics = {}
        for method, pred in edge_predictions.items():
            if method == "ground_truth_edges":
                continue
            trajectory = _solve_positions(p, R, edges, pred)
            trajectories[method] = trajectory
            metrics[method] = _metrics(trajectory, p)
        np.savez(
            out / f"{case}.npz",
            t=gt["t"][idx],
            R=R.astype(np.float32),
            edges=edges,
            **{key: value.astype(np.float32) for key, value in trajectories.items()},
        )
        summary[case] = {"session": name, "edge_count": len(edges), "metrics": metrics}
        print(json.dumps({case: summary[case]}, ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
