#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-session-out benchmark over the eight non-test sessions.

Each fold trains on seven sessions and early-stops on the eighth, so every
non-test second of data is scored exactly once without touching the sealed
test session. The eight fold models are then averaged and applied once to the
sealed test session. Works for PhysNet (tools/train_physnet.py --holdout) and
for the adapted public baselines (tools/train_external_architectures.py
--only real_only --holdout).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
import tools.train_physnet as phys
from tools.train_external_architectures import make_model, predict as predict_external
from tools.build_trajectory_comparison import _solve_positions, _metrics as ate_metrics

DATA = ROOT / "datasets"


def trajectory_ate(session: str, data: dict, end_pred: np.ndarray, dense_pred: np.ndarray | None,
                   dense_weight_half_life_s: float = 1.5) -> dict:
    """ATE of the test trajectory rebuilt from predicted edges.

    `single_edge` uses one 3 s edge per window (what an end-point regressor
    can provide). `dense_graph` adds one edge from every window anchor to every
    usable chessboard frame inside the window, weighted by exp(-|dt|/half_life),
    which only a displacement-stream model can produce.
    """
    t, Rg, pg, usable = phys._load_gt(session)
    node = {int(f): i for i, f in enumerate(usable)}
    R_nodes = Rg[usable]
    p_nodes = pg[usable]
    out = {}
    edges = np.array([(node[int(a)], node[int(b)]) for a, b in data["pair"]], np.int64)
    traj = _solve_positions(p_nodes, R_nodes, edges, end_pred.astype(np.float64))
    out["single_edge"] = ate_metrics(traj, p_nodes)
    out["single_edge"]["edges"] = int(len(edges))
    traj_gt = _solve_positions(p_nodes, R_nodes, edges, data["y"].astype(np.float64))
    out["single_edge_gt_edges"] = ate_metrics(traj_gt, p_nodes)
    if dense_pred is not None:
        rows, disp, w = [], [], []
        for k, (a, _b) in enumerate(data["pair"]):
            for m in range(dense_pred.shape[1]):
                fr = int(data["dense_frame"][k, m])
                if data["dense_valid"][k, m] <= 0 or fr < 0 or fr == int(a):
                    continue
                rows.append((node[int(a)], node[fr]))
                disp.append(dense_pred[k, m])
                w.append(np.exp(-abs(t[fr] - t[int(a)]) / dense_weight_half_life_s))
        rows = np.asarray(rows, np.int64)
        disp = np.asarray(disp, np.float64)
        w = np.asarray(w)
        # weighted least squares by duplicating the edge scaling (rows scaled inside solver)
        traj_d = _solve_positions_weighted(p_nodes, R_nodes, rows, disp, w)
        out["dense_graph"] = ate_metrics(traj_d, p_nodes)
        out["dense_graph"]["edges"] = int(len(rows))
    return out


def _solve_positions_weighted(gt_p, gt_R, edges, dp, weights):
    from scipy.sparse import coo_matrix
    from scipy.sparse.linalg import lsqr
    from tools.build_trajectory_comparison import _components
    n = len(gt_p)
    output = np.full((n, 3), np.nan)
    for nodes in _components(n, edges):
        node_set = set(nodes.tolist())
        mapping = {int(node): i for i, node in enumerate(nodes)}
        sel = [k for k, (a, b) in enumerate(edges) if int(a) in node_set and int(b) in node_set]
        rows, cols, vals, rhs = [], [], [], []
        for r, k in enumerate(sel):
            a, b = (int(v) for v in edges[k])
            rows += [r, r]
            cols += [mapping[a], mapping[b]]
            vals += [-weights[k], weights[k]]
            rhs.append(weights[k] * (gt_R[a] @ dp[k]))
        r = len(sel)
        rows.append(r)
        cols.append(0)
        vals.append(1.0)
        rhs.append(gt_p[nodes[0]])
        A = coo_matrix((vals, (rows, cols)), shape=(r + 1, len(nodes))).tocsr()
        rhs = np.asarray(rhs)
        output[nodes] = np.column_stack([lsqr(A, rhs[:, ax], atol=1e-10, btol=1e-10)[0] for ax in range(3)])
    return output


def _bootstrap_ci(err_a: np.ndarray, err_b: np.ndarray, t_pair: np.ndarray, n_boot: int = 2000,
                  seed: int = 0) -> list[float]:
    """Block bootstrap of mean(err_b - err_a) over non-overlapping 3 s blocks."""
    rng = np.random.default_rng(seed)
    blocks = np.floor((t_pair[:, 0] - t_pair[:, 0].min()) / 3.0).astype(int)
    ids = np.unique(blocks)
    diff = err_b - err_a
    stats = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sel = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        stats.append(diff[sel].mean())
    return [round(float(v) * 1000, 2) for v in np.quantile(stats, (0.025, 0.975))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physnet-dir", default="physnet_cv")
    ap.add_argument("--cv-tag", default="cv", help="PhysNet fold file tag, e.g. cvx or cvg")
    ap.add_argument("--baseline-tag", default="",
                    help="fold tag of the baseline architectures when it differs from --cv-tag")
    ap.add_argument("--baselines", nargs="*", default=["imunet"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[0])
    ap.add_argument("--output", type=Path, default=DATA / "session_cv_benchmark.json")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    sessions = split["train"] + split["val"]

    report = {"protocol": {
        "folds": sessions,
        "extra_train": split.get("extra_train", []),
        "test_sessions": split["test"],
        "rule": f"train on {len(sessions) - 1} sessions (+ extra_train), early-stop on the held-out one; "
                "fold models averaged and applied once to every sealed test session",
    }, "folds": {}, "test": {}}

    # Per-fold held-out scores (already computed by the training scripts).
    for ses in sessions:
        row = {}
        for seed in args.seeds:
            m = json.loads((DATA / args.physnet_dir / f"metrics_{args.cv_tag}_{ses}_s{seed}.json").read_text(encoding="utf-8"))
            row.setdefault("physnet", []).append(m["val"])
            for arch in args.baselines:
                b = json.loads((DATA / "external_benchmark" / f"{arch}_{args.baseline_tag or args.cv_tag}_{ses}_s{seed}.json").read_text(encoding="utf-8"))
                row.setdefault(arch, []).append(b["real_only_real_val"])
        report["folds"][ses] = {
            name: {
                "err_mm": round(float(np.mean([v["err_mm"] for v in vals])), 2),
                "fast_mm": round(float(np.mean([v["fast_mm"] for v in vals])), 2),
                "r2": round(float(np.mean([v["r2"] for v in vals])), 3),
                "zero_mm": vals[0]["zero_mm"], "n": vals[0]["n"],
            } for name, vals in row.items()
        }
    total_n = sum(v["physnet"]["n"] for v in report["folds"].values())
    report["cv_mean_over_pairs"] = {
        name: round(float(sum(v[name]["err_mm"] * v[name]["n"] for v in report["folds"].values()) / total_n), 2)
        for name in report["folds"][sessions[0]]
    }
    report["cv_mean_over_sessions"] = {
        name: round(float(np.mean([v[name]["err_mm"] for v in report["folds"].values()])), 2)
        for name in report["folds"][sessions[0]]
    }
    report["cv_zero_motion_over_pairs"] = round(float(
        sum(v["physnet"]["zero_mm"] * v["physnet"]["n"] for v in report["folds"].values()) / total_n), 2)

    models = [phys.load_model(DATA / args.physnet_dir / f"physnet_{args.cv_tag}_{ses}_s{seed}.pt", device)
              for ses in sessions for seed in args.seeds]

    def baseline_models(arch):
        nets = []
        for ses in sessions:
            for seed in args.seeds:
                net = make_model(arch).to(device)
                net.load_state_dict(torch.load(
                    DATA / "external_benchmark" / f"{arch}_{args.baseline_tag or args.cv_tag}_{ses}_s{seed}.pt", map_location=device))
                net.eval()
                nets.append(net)
        return nets

    baselines = {arch: baseline_models(arch) for arch in args.baselines}

    # Post-hoc calibration fitted on the held-out CV predictions only:
    # a convex blend weight between PhysNet and the strongest baseline and a
    # global shrink factor. Both are scalars, fitted before any test session is
    # opened, so they cannot overfit the test.
    calibration = None
    if "imunet" in baselines:
        cv_phys, cv_imu, cv_y = [], [], []
        for k, ses in enumerate(sessions):
            X_ses, y_ses = api.build(ses, context, length)
            data_ses = phys.build_group([ses], context, horizon, length)
            assert np.allclose(data_ses["y"], y_ses, atol=1e-6)
            data_ses_t = phys.to_device(data_ses, device)
            fold_phys = models[k * len(args.seeds):(k + 1) * len(args.seeds)]
            fold_imu = baselines["imunet"][k * len(args.seeds):(k + 1) * len(args.seeds)]
            cv_phys.append(np.mean([m.predict(data_ses_t) for m in fold_phys], axis=0))
            cv_imu.append(np.mean([predict_external(n, X_ses, device) for n in fold_imu], axis=0))
            cv_y.append(y_ses)
        cv_phys, cv_imu, cv_y = (np.concatenate(v) for v in (cv_phys, cv_imu, cv_y))

        def cv_err(w, alpha):
            return float(np.linalg.norm(alpha * (w * cv_phys + (1 - w) * cv_imu) - cv_y, axis=1).mean() * 1000)

        grid_w = np.linspace(0, 1, 21)
        grid_a = np.linspace(0.6, 1.2, 25)
        best = min(((cv_err(w, a), w, a) for w in grid_w for a in grid_a), key=lambda r: r[0])
        _, w_best, a_best = best
        calibration = (float(w_best), float(a_best))
        report["cv_calibration"] = {
            "blend_weight_physnet": round(float(w_best), 3),
            "shrink": round(float(a_best), 3),
            "cv_err_mm_physnet": round(cv_err(1.0, 1.0), 2),
            "cv_err_mm_imunet": round(cv_err(0.0, 1.0), 2),
            "cv_err_mm_equal_blend": round(cv_err(0.5, 1.0), 2),
            "cv_err_mm_calibrated": round(best[0], 2),
        }

    # Sealed test sessions: one evaluation per fold-ensemble per session.
    report["test_sessions"] = {}
    pooled = {}
    arrays_all = {}
    for test_name in split["test"]:
        res = {}
        test_phys = phys.build_group([test_name], context, horizon, length)
        test_phys_t = phys.to_device(test_phys, device)
        phys_pred = np.mean([m.predict(test_phys_t) for m in models], axis=0)
        y = test_phys["y"]
        res["n"] = int(len(y))
        res["physnet_cv_ensemble"] = api._metrics(phys_pred, y)
        res["physnet_fold_models"] = [api._metrics(m.predict(test_phys_t), y)["err_mm"] for m in models]
        res["zero_motion"] = api._metrics(np.zeros_like(y), y)
        X_test, y_ext = api.build(test_name, context, length)
        assert np.allclose(y_ext, y, atol=1e-6)
        err_phys = np.linalg.norm(phys_pred - y, axis=1)
        arrays = {"y": y, "t_pair": test_phys["t_pair"], "pair": test_phys["pair"], "physnet": phys_pred}
        dense_pred = np.mean([m.predict_dense(test_phys_t) for m in models], axis=0)
        res["trajectory_physnet"] = trajectory_ate(test_name, test_phys, phys_pred, dense_pred)
        dv = test_phys["dense_valid"] > 0
        res["physnet_dense_stream_err_mm"] = round(float(
            np.linalg.norm(dense_pred[dv] - test_phys["dense_y"][dv], axis=1).mean() * 1000), 2)
        for arch, nets in baselines.items():
            preds = [predict_external(n, X_test, device) for n in nets]
            pred = np.mean(preds, axis=0)
            arrays[arch] = pred
            res[f"{arch}_cv_ensemble"] = api._metrics(pred, y)
            res[f"trajectory_{arch}"] = trajectory_ate(test_name, test_phys, pred, None)
            res[f"{arch}_fold_models"] = [api._metrics(p, y)["err_mm"] for p in preds]
            res[f"physnet_minus_{arch}_mm_95ci_blocks"] = _bootstrap_ci(
                err_phys, np.linalg.norm(pred - y, axis=1), test_phys["t_pair"])
        res["physnet_minus_zero_mm_95ci_blocks"] = _bootstrap_ci(
            err_phys, np.linalg.norm(y, axis=1), test_phys["t_pair"])
        if calibration is not None:
            w_best, a_best = calibration
            blend = a_best * (w_best * phys_pred + (1 - w_best) * arrays["imunet"])
            equal = 0.5 * (phys_pred + arrays["imunet"])
            arrays["cv_calibrated_blend"] = blend
            arrays["equal_blend"] = equal
            res["equal_blend_physnet_imunet"] = api._metrics(equal, y)
            res["trajectory_equal_blend"] = trajectory_ate(test_name, test_phys, equal, None)
            res["cv_calibrated_blend"] = api._metrics(blend, y)
            res["cv_calibrated_blend_minus_imunet_mm_95ci_blocks"] = _bootstrap_ci(
                np.linalg.norm(blend - y, axis=1), np.linalg.norm(arrays["imunet"] - y, axis=1),
                test_phys["t_pair"])
        report["test_sessions"][test_name] = res
        for key, value in arrays.items():
            pooled.setdefault(key, []).append(value)
            arrays_all[f"{test_name}__{key}"] = value
        print(json.dumps({test_name: {k: v for k, v in res.items() if k.endswith("ensemble") or "blend" in k}},
                         ensure_ascii=False), flush=True)

    # Backwards compatible: `test` is the fixed first test session; `test_pooled`
    # merges every sealed session when more than one exists.
    report["test"] = report["test_sessions"][split["test"][0]]
    if len(split["test"]) > 1:
        y_all = np.concatenate(pooled["y"])
        report["test_pooled"] = {"sessions": split["test"], "n": int(len(y_all)),
                                 "zero_motion": api._metrics(np.zeros_like(y_all), y_all)}
        for key in ("physnet", *args.baselines, "equal_blend", "cv_calibrated_blend"):
            if key in pooled:
                report["test_pooled"][key] = api._metrics(np.concatenate(pooled[key]), y_all)
    first = split["test"][0]
    arrays_out = {k.split("__", 1)[1]: v for k, v in arrays_all.items() if k.startswith(first + "__")}
    arrays_out.update(arrays_all)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(args.output.with_suffix(".npz"), **arrays_out)
    print(json.dumps({k: v for k, v in report.items() if k != "test_sessions"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
