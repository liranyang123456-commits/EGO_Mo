#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Why the stillness gate improves local error but worsens trajectory error.

On the held-out recording, the gated and non-gated PhysNet fold ensembles are
compared (i) on windows where the gate is active (mean stillness probability
inside [t_a, t_b] above 0.10), along the true motion direction, and (ii) in
board-frame Z, the axis whose trajectory error differs, per 15-s interval of
the window start. The summed board-Z edges of the reference are reported next
to those of the two ensembles. (iii) `offset_accounting` follows the per-edge
board-Z difference of the two ensembles along the pose-graph paths to the
offset between the two trajectories.

    python tools/analyze_gate_trajectory.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
from tools.build_trajectory_comparison import _components, _solve_positions

DATA = ROOT / "datasets"
TEST = "traj_20260923_023422"
ACTIVE = 0.10
REST_END = 10.0     # components anchored before this time start in the initial rest


def offset_accounting(t, pg, usable, data, edges, pred, traj, p_win, active, board_z) -> dict:
    """How a per-edge difference of a few millimetres becomes the trajectory offset.

    Times are seconds from the first usable reference frame, the axis of the
    response figure (the 15-s blocks of the main report count from the first
    window start). Every node has one incoming edge at most, so each connected
    component is a tree and its solution is the running sum of the edges from
    the anchor: the board-Z offset between the gated and the non-gated
    trajectory at a node is the sum of the per-edge differences on its path.
    """
    t0 = float(t[usable][0])
    ts, te = data["t_pair"][:, 0] - t0, data["t_pair"][:, 1] - t0
    tn = t[usable] - t0
    y = data["y"]
    dz = board_z(pred["cvg"]) - board_z(pred["cvx"])
    ref_z = board_z(y)
    res_z = {tag: board_z(pred[tag]) - ref_z for tag in pred}     # signed error of every edge against the reference
    err = {tag: np.linalg.norm(pred[tag] - y, axis=1) * 1000 for tag in pred}

    def windows(lo, hi):
        m = (ts >= lo) & (ts < hi)
        return {"start_s": [lo, hi], "windows": int(m.sum()), "gate_active": int((m & active).sum()),
                "mean_p_still": round(float(p_win[m].mean()), 3), "max_p_still": round(float(p_win[m].max()), 3),
                "board_z_gated_minus_nongated_mm": round(float(dz[m].mean()), 2),
                "share_positive": round(float((dz[m] > 0).mean()), 3),
                "reference_abs_board_z_mm": round(float(np.abs(ref_z[m]).mean()), 1),
                **{f"board_z_residual_{tag}_mm": round(float(res_z[tag][m].mean()), 2) for tag in pred},
                **{f"err_{tag}_mm": round(float(err[tag][m].mean()), 2) for tag in pred}}

    comps = _components(len(usable), edges)
    comp = np.full(len(usable), -1)
    for i, nodes in enumerate(comps):
        comp[nodes] = i
    anchor = np.array([tn[nodes[0]] for nodes in comps])
    solved = comp >= 0
    first = solved & (anchor[np.maximum(comp, 0)] < REST_END)
    later = solved & ~first
    parent = {int(b): k for k, (_a, b) in enumerate(edges)}
    if len(parent) != len(edges):
        raise AssertionError("a node has two incoming edges; the path sum is not exact")

    def path(n):
        out = []
        while n in parent:
            out.append(parent[n])
            n = int(edges[parent[n], 0])
        return np.array(out, np.int64)

    P = pg[usable]
    off = (traj["cvg"][:, 2] - traj["cvx"][:, 2]) * 1000
    ez = {tag: (traj[tag][:, 2] - P[:, 2]) * 1000 for tag in traj}
    parts = ((0.0, 15.0), (15.0, 25.0), (25.0, 60.0), (60.0, 1e9))

    def nodes_between(lo, hi):
        rows, signed = [], {tag: [] for tag in res_z}
        for n in np.flatnonzero(first & (tn >= lo) & (tn < hi)):
            pe = path(int(n))
            s = ts[pe]
            rows.append([off[n], dz[pe].sum(), *[dz[pe][(s >= a) & (s < b)].sum() for a, b in parts],
                         ((s >= 25.0) & (s < 60.0)).sum(), ((s >= 15.0) & (s < 60.0)).sum()])
            for tag in res_z:      # the Z error of the node itself, split by where its edges start
                signed[tag].append([res_z[tag][pe][(s >= a) & (s < b)].sum() for a, b in parts])
        r = np.array(rows)
        m = first & (tn >= lo) & (tn < hi)
        split = {tag: [round(float(x), 1) for x in np.mean(signed[tag], axis=0)] for tag in res_z}
        return {"node_time_s": [lo, round(hi, 1)], "nodes": int(len(r)),
                "offset_mm": round(float(r[:, 0].mean()), 1), "path_sum_mm": round(float(r[:, 1].mean()), 1),
                "from_edges_starting_s": {f"{a:g}-{'end' if b > 1e8 else f'{b:g}'}": round(float(r[:, 2 + i].mean()), 1)
                                          for i, (a, b) in enumerate(parts)},
                "path_edges_starting_25_60": round(float(r[:, 6].mean()), 1),
                "path_edges_starting_15_60": round(float(r[:, 7].mean()), 1),
                **{f"z_error_{tag}_from_edges_starting_s": dict(zip(("0-15", "15-25", "25-60", "60-end"), split[tag]))
                   for tag in split},
                **{f"z_error_{tag}_mm": round(float(ez[tag][m].mean()), 1) for tag in ez}}

    by5 = []
    for b0 in range(0, 115, 5):
        m = first & (tn >= b0) & (tn < b0 + 5)
        if m.any():
            by5.append({"node_time_s": [b0, b0 + 5], "nodes": int(m.sum()), "offset_mm": round(float(off[m].mean()), 1),
                        **{f"z_error_{tag}_mm": round(float(ez[tag][m].mean()), 1) for tag in ez}})
    end = float(tn[first].max()) + 1.0
    z0 = P[first][0, 2]
    vel, vel_ok = phys.gt_velocity(t, pg, usable)
    speed = np.linalg.norm(vel[usable], axis=1) * 1000
    moving = vel_ok[usable] & (tn >= 15.0) & (tn < 60.0)
    early = (tn >= 15.0) & (tn < 40.0)
    return {
        "time_axis": "seconds from the first usable reference frame",
        "first_window_start_s": round(float(ts.min()), 2),
        "edge_duration_median_s": round(float(np.median(te - ts)), 2),
        "gate_active_windows": {"start_before_15s": int((active & (ts < 15)).sum()),
                                "start_15s_or_later": int((active & (ts >= 15)).sum()),
                                "latest_start_s": round(float(ts[active].max()), 1),
                                "latest_end_s": round(float(te[active].max()), 1)},
        "windows_by_start": [windows(0.0, 15.0), windows(15.0, 25.0), windows(25.0, 60.0),
                             windows(60.0, round(float(ts.max()) + 0.1, 1)), windows(15.0, 60.0)],
        "reference_speed_min_mm_s_15_60s": round(float(speed[moving].min()), 1),
        "graph": {"components": len(comps), "max_incoming_edges_per_node": 1,
                  "components_ending_before_22s": int(sum(tn[c].max() < 22.0 for c in comps)),
                  "two_largest": {"nodes": sorted((int(len(c)) for c in comps), reverse=True)[:2],
                                  "span_s": sorted((round(float(np.ptp(tn[c])), 1) for c in comps), reverse=True)[:2]},
                  "anchored_in_initial_rest": {"components": int((anchor < REST_END).sum()), "nodes": int(first.sum()),
                                               "anchor_s": [round(float(anchor[anchor < REST_END].min()), 1),
                                                            round(float(anchor[anchor < REST_END].max()), 1)]},
                  "anchored_after_dropout": {"components": int((anchor >= REST_END).sum()), "nodes": int(later.sum()),
                                             "anchor_s": [round(float(a), 1) for a in sorted(anchor[anchor >= REST_END])],
                                             "nodes_each": [int(len(c)) for c, a in zip(comps, anchor) if a >= REST_END]},
                  "nodes_after_75s": {"anchored_in_initial_rest": int((first & (tn >= 75)).sum()),
                                      "anchored_after_dropout": int((later & (tn >= 75)).sum())}},
        "z_rmse_mm": {"anchored_in_initial_rest": {tag: round(float(np.sqrt((ez[tag][first] ** 2).mean())), 1) for tag in ez},
                      "anchored_after_dropout": {tag: round(float(np.sqrt((ez[tag][later] ** 2).mean())), 1) for tag in ez}},
        "z_range_mm": {"reference_min": round(float((P[:, 2] - z0).min() * 1000), 1),
                       "reference_min_15_40s": round(float((P[early, 2] - z0).min() * 1000), 1),
                       **{f"{tag}_min_max": [round(float((traj[tag][first, 2] - z0).min() * 1000), 1),
                                             round(float((traj[tag][first, 2] - z0).max() * 1000), 1)] for tag in traj}},
        "offset_at_nodes": [nodes_between(15.0, 30.0), nodes_between(60.0, 75.0), nodes_between(75.0, end),
                            nodes_between(60.0, end)],
        "offset_by_5s": by5,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, Rg, pg, usable = phys._load_gt(TEST)
    data = phys.build_group([TEST], 2.0, 3.0, int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ)))
    data_t = phys.to_device(data, device)
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds = split["train"] + split["val"]
    models = {tag: [phys.load_model(DATA / "physnet_cv" / f"physnet_{tag}_{s}_s{k}.pt", device)
                    for s in folds for k in (0, 1)] for tag in ("cvx", "cvg")}
    pred = {tag: np.mean([m.predict(data_t) for m in ms], axis=0) for tag, ms in models.items()}
    still = np.mean([m.predict_streams(data_t)["still"] for m in models["cvg"]], axis=0)
    y = data["y"]
    p_win = np.zeros(len(y))
    for k in range(len(y)):
        dv = data["dense_valid"][k] > 0
        tau = data["dense_tau"][k][dv]
        sel = (tau >= int(round(2.0 * phys.HZ))) & (tau <= data["tau_b"][k])
        p_win[k] = still[k][dv][sel].mean() if sel.any() else 0.0
    active = p_win > ACTIVE
    unit = y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-9)

    def group(mask):
        return {"windows": int(mask.sum()),
                "reference_mm": round(float(np.linalg.norm(y[mask], axis=1).mean() * 1000), 2),
                **{f"along_motion_{tag}_mm": round(float((pred[tag][mask] * unit[mask]).sum(1).mean() * 1000), 2)
                   for tag in pred},
                **{f"err_{tag}_mm": round(float(np.linalg.norm(pred[tag][mask] - y[mask], axis=1).mean() * 1000), 2)
                   for tag in pred}}

    Ra = Rg[data["pair"][:, 0]]

    def board_z(v):
        return np.einsum("nij,nj->ni", Ra, v)[:, 2] * 1000

    rz = {tag: board_z(pred[tag]) - board_z(y) for tag in pred}
    ta = data["t_pair"][:, 0] - data["t_pair"][:, 0].min()
    bins = []
    for b0 in range(0, 120, 15):
        m = (ta >= b0) & (ta < b0 + 15)
        if m.any():
            bins.append({"interval_s": [b0, b0 + 15], "windows": int(m.sum()),
                         "mean_p_still": round(float(p_win[m].mean()), 3),
                         **{f"board_z_residual_{tag}_mm": round(float(rz[tag][m].mean()), 2) for tag in pred}})
    node = {int(f): i for i, f in enumerate(usable)}
    edges = np.array([(node[int(a)], node[int(b)]) for a, b in data["pair"]], np.int64)
    ate, traj = {}, {}
    for tag in pred:
        traj[tag] = _solve_positions(pg[usable], Rg[usable], edges, pred[tag].astype(np.float64))
        ok = np.isfinite(traj[tag]).all(1)
        ate[tag] = [round(float(np.sqrt(np.mean(((traj[tag][ok, a] - pg[usable][ok, a]) * 1000) ** 2))), 1)
                    for a in range(3)]
    report = {
        "active_threshold": ACTIVE,
        "gate_active": group(active), "gate_inactive": group(~active),
        "board_z_sum_mm": {"reference": round(float(board_z(y).sum()), 1),
                           **{tag: round(float(board_z(pred[tag]).sum()), 1) for tag in pred}},
        "board_z_residual_by_15s": bins,
        "per_axis_ate_mm_xyz": ate,
        "offset_accounting": offset_accounting(t, pg, usable, data, edges, pred, traj, p_win, active, board_z),
    }
    (DATA / "gate_trajectory_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
