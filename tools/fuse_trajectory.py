#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session-level fusion of overlapping displacement windows.

Independent windows each assume implicitly that the mean camera velocity over
their context is zero, which is exactly the component an accelerometer cannot
observe; the residual of a window model is strongly correlated with that term.
This tool replaces the per-window estimate by one sparse least-squares problem
over the whole recording whose unknowns are the camera positions at every
usable reference frame and whose constraints are

  1. displacement edges   p_m - p_a = R_a d_pred(a, m)        (all windows),
  2. velocity edges       p_j - p_i = R_ij v_pred * (t_j-t_i) (neighbours),
  3. zero-velocity edges  p_j - p_i = 0                       (still segments),

so the velocity stream and the detected still phases constrain the drift that
a single window cannot resolve. Edge weights fall off with the time span and,
when the network provides one, with the predicted variance.

Two attitude sources are reported: `ref` uses the reference orientation (the
protocol already used for trajectory comparison; attitude is supplied) and
`gyro` integrates the measured gyroscope from the first node, so only the
initial board-frame attitude comes from the reference.
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

import tools.train_seq as api
import tools.train_physnet as phys
from tools.build_trajectory_comparison import _components, _metrics as ate_metrics

DATA = ROOT / "datasets"


def _gyro_orientation(name: str, t_nodes: np.ndarray, R_first: np.ndarray) -> np.ndarray:
    """Reference attitude at the first node propagated by the measured gyroscope."""
    ses = phys.Session(name)
    q = phys._orientation_at(ses.t, ses.gyro, ses.Q, np.clip(t_nodes, ses.t[0], ses.t[-1]))
    q0 = phys._orientation_at(ses.t, ses.gyro, ses.Q, t_nodes[:1])
    rel = phys._qmul(phys._qconj(np.repeat(q0, len(q), 0)), q)          # IMU_0 <- IMU_k
    q_ci = ses.q_ci
    rel = phys._qmul(phys._qmul(np.repeat(q_ci[None], len(q), 0), rel),
                     phys._qconj(np.repeat(q_ci[None], len(q), 0)))      # C_0 <- C_k
    Rrel = phys.quat_to_mat(torch.from_numpy(rel)).numpy()
    return np.einsum("ij,njk->nik", R_first, Rrel)                       # B <- C_k


def _solve(n_nodes: int, rows: list, anchor_node: int, anchor_pos: np.ndarray):
    """rows: list of (i, j, rhs[3], weight); i == -1 means an absolute position row."""
    edges = np.array([(r[0], r[1]) for r in rows if r[0] >= 0], np.int64)
    out = np.full((n_nodes, 3), np.nan)
    for nodes in _components(n_nodes, edges):
        keep = set(nodes.tolist())
        index = {int(v): k for k, v in enumerate(nodes)}
        ri, ci, va, rhs = [], [], [], []
        row = 0
        for i, j, value, weight in rows:
            if i not in keep or j not in keep:
                continue
            ri += [row, row]
            ci += [index[i], index[j]]
            va += [-weight, weight]
            rhs.append(weight * value)
            row += 1
        base = anchor_node if anchor_node in keep else int(nodes[0])
        ri.append(row)
        ci.append(index[base])
        va.append(10.0)
        rhs.append(10.0 * (anchor_pos[base]))
        A = coo_matrix((va, (ri, ci)), shape=(row + 1, len(nodes))).tocsr()
        rhs = np.asarray(rhs)
        out[nodes] = np.column_stack(
            [lsqr(A, rhs[:, k], atol=1e-11, btol=1e-11, iter_lim=20000)[0] for k in range(3)])
    return out


def fuse_session(name: str, models: list, context: float, horizon: float, length: int,
                 device, attitude: str = "ref", half_life: float = 1.5,
                 w_vel: float = 0.0, w_zupt: float = 0.0, still_thr: float = 0.5,
                 use_var: bool = False, max_span: float = 1e9) -> dict:
    data = phys.build_group([name], context, horizon, length)
    data_t = phys.to_device(data, device)
    streams = [m.predict_streams(data_t) for m in models]
    d_pred = np.mean([s["d"] for s in streams], axis=0)
    v_pred = np.mean([s["v"] for s in streams], axis=0)
    still_p = np.mean([s["still"] for s in streams], axis=0)
    var = np.mean([s["var"] for s in streams], axis=0)
    end_pred = np.mean([s["end"] for s in streams], axis=0)

    t, Rg, pg, usable = phys._load_gt(name)
    node_of = {int(f): k for k, f in enumerate(usable)}
    t_nodes, p_nodes = t[usable], pg[usable]
    R_nodes = _gyro_orientation(name, t_nodes, Rg[usable[0]]) if attitude == "gyro" else Rg[usable]

    valid = data["dense_valid"] > 0
    frames = data["dense_frame"]
    rows = []
    # 1) displacement edges from every window anchor to every reference frame
    for k, (a, _b) in enumerate(data["pair"]):
        na = node_of[int(a)]
        for m in np.flatnonzero(valid[k]):
            fr = int(frames[k, m])
            if fr < 0 or fr == int(a):
                continue
            nj = node_of[fr]
            span = abs(t_nodes[nj] - t_nodes[na])
            if span > max_span:
                continue
            w = float(np.exp(-span / half_life))
            if use_var:
                w /= float(np.sqrt(var[k, m].mean()) + 1e-6)
            rows.append((na, nj, R_nodes[na] @ d_pred[k, m], w))

    # Node-level velocity and stillness: mean over all windows covering the node.
    vel_sum = np.zeros((len(usable), 3))
    still_sum = np.zeros(len(usable))
    count = np.zeros(len(usable))
    for k in range(len(data["pair"])):
        for m in np.flatnonzero(valid[k]):
            fr = int(frames[k, m])
            if fr < 0:
                continue
            nj = node_of[fr]
            vel_sum[nj] += v_pred[k, m]
            still_sum[nj] += still_p[k, m]
            count[nj] += 1
    seen = count > 0
    vel_node = np.zeros_like(vel_sum)
    vel_node[seen] = vel_sum[seen] / count[seen, None]
    still_node = np.zeros(len(usable))
    still_node[seen] = still_sum[seen] / count[seen]

    # 2/3) neighbour edges from the predicted velocity and the detected still phases
    if w_vel > 0 or w_zupt > 0:
        for i in range(len(usable) - 1):
            j = i + 1
            dt = float(t_nodes[j] - t_nodes[i])
            if dt <= 0 or dt > 0.6 or not (seen[i] and seen[j]):
                continue
            if w_zupt > 0 and still_node[i] > still_thr and still_node[j] > still_thr:
                rows.append((i, j, np.zeros(3), w_zupt))
            elif w_vel > 0:
                mid = 0.5 * (R_nodes[i] @ vel_node[i] + R_nodes[j] @ vel_node[j])
                rows.append((i, j, mid * dt, w_vel))

    solved = _solve(len(usable), rows, 0, p_nodes)
    result = {"session": name, "attitude": attitude, "nodes": int(len(usable)),
              "edges": int(len(rows)), "ate": ate_metrics(solved, p_nodes)}

    # Re-read the 3-s displacements from the fused trajectory.
    ok = np.isfinite(solved).all(1)
    reread, truth, base = [], [], []
    for k, (a, b) in enumerate(data["pair"]):
        na, nb = node_of[int(a)], node_of[int(b)]
        if not (ok[na] and ok[nb]):
            continue
        reread.append(R_nodes[na].T @ (solved[nb] - solved[na]))
        truth.append(data["y"][k])
        base.append(end_pred[k])
    reread = np.asarray(reread, np.float32)
    truth = np.asarray(truth, np.float32)
    base = np.asarray(base, np.float32)
    result["window_only"] = api._metrics(base, truth)
    result["fused_reread"] = api._metrics(reread, truth)
    result["still_node_fraction"] = round(float((still_node[seen] > still_thr).mean()), 3)
    if "dense_v" in data:
        speed = np.linalg.norm(data["dense_v"], axis=-1) * 1000.0
        sup = (data["dense_valid"] * data["dense_v_valid"]) > 0
        pred_still = still_p[sup] > still_thr
        true_still = speed[sup] < phys.STILL_MM_S
        true_move = speed[sup] > phys.MOVE_MM_S
        result["stillness_detector"] = {
            "recall_on_still": round(float(pred_still[true_still].mean()), 3) if true_still.any() else None,
            "false_alarm_on_moving": round(float(pred_still[true_move].mean()), 3) if true_move.any() else None,
            "predicted_still_fraction": round(float(pred_still.mean()), 3),
        }
    return result, solved, p_nodes, t_nodes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--sessions", nargs="*", default=[], help="default: the sealed test sessions")
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--half-life", type=float, default=1.5)
    ap.add_argument("--w-vel", type=float, default=0.0)
    ap.add_argument("--w-zupt", type=float, default=0.0)
    ap.add_argument("--still-thr", type=float, default=0.5)
    ap.add_argument("--use-var", action="store_true")
    ap.add_argument("--attitude", nargs="*", default=["ref", "gyro"])
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    ap.add_argument("--output", type=Path, default=DATA / "fusion_benchmark.json")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", args.horizon
    length = int(round((1.5 * args.horizon + 2 * args.context) * phys.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    sessions = args.sessions or split["test"]
    models = [phys.load_model(Path(p), device) for p in args.models]
    report = {"models": args.models,
              "config": {k: (str(v) if isinstance(v, Path) else v)
                         for k, v in vars(args).items() if k != "models"},
              "sessions": {}}
    arrays = {}
    for name in sessions:
        report["sessions"][name] = {}
        for attitude in args.attitude:
            res, solved, truth, t_nodes = fuse_session(
                name, models, args.context, args.horizon, length, device, attitude,
                args.half_life, args.w_vel, args.w_zupt, args.still_thr, args.use_var)
            report["sessions"][name][attitude] = res
            arrays[f"{name}__{attitude}"] = solved
            arrays[f"{name}__truth"] = truth
            arrays[f"{name}__t"] = t_nodes
            print(json.dumps({name: {attitude: {
                "ate_rmse_mm": res["ate"]["ate_rmse_mm"],
                "window_only_mm": res["window_only"]["err_mm"],
                "fused_reread_mm": res["fused_reread"]["err_mm"],
                "fused_r2": res["fused_reread"]["r2"],
                "edges": res["edges"],
                "stillness": res.get("stillness_detector"),
            }}}, ensure_ascii=False), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez(args.output.with_suffix(".npz"), **arrays)


if __name__ == "__main__":
    main()
