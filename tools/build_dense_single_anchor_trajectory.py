#!/usr/bin/env python3
"""Rebuild the uninterrupted dense PhysNet and calibrated-blend trajectories.

The dense graph contains one edge from every window anchor to every usable
reference frame in that window.  Dense PhysNet edges use the same exponential
time weights as ``eval_session_cv.py``.  The calibrated-blend variant adds the
CV-calibrated end-point edge of every window at unit weight.  With a connected
graph, the weighted solver introduces one position anchor only.
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

import tools.train_physnet as phys
from tools.build_trajectory_comparison import _components
from tools.eval_session_cv import _solve_positions_weighted

DATA = ROOT / "datasets"


def metrics(trajectory: np.ndarray, truth: np.ndarray) -> dict:
    ok = np.isfinite(trajectory).all(1)
    error = np.linalg.norm(trajectory[ok] - truth[ok], axis=1) * 1000.0
    return {
        "nodes": int(ok.sum()),
        "ate_rmse_mm": round(float(np.sqrt(np.mean(error**2))), 3),
        "ate_mean_mm": round(float(error.mean()), 3),
        "ate_p90_mm": round(float(np.quantile(error, 0.9)), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", default="traj_20260923_023422")
    parser.add_argument("--half-life-s", type=float, default=1.5)
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA / "dense_single_anchor_trajectory.json",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=DATA / "trajectory_comparison" / "real_test.npz",
        help="optional existing NPZ used only for a reproducibility difference",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ))
    data = phys.build_group([args.session], 2.0, 3.0, length)
    data_t = phys.to_device(data, device)
    models = [
        phys.load_model(path, device)
        for path in sorted((DATA / "physnet_cv").glob("physnet_cvx_*_s*.pt"))
    ]
    if len(models) != 16:
        raise RuntimeError(f"expected 16 non-gated fold models, found {len(models)}")
    dense = np.mean([model.predict_dense(data_t) for model in models], axis=0)

    t, R, p, usable = phys._load_gt(args.session)
    node = {int(frame): i for i, frame in enumerate(usable)}
    rows, displacement, weights = [], [], []
    for k, (anchor, _end) in enumerate(data["pair"]):
        for m in range(dense.shape[1]):
            frame = int(data["dense_frame"][k, m])
            if data["dense_valid"][k, m] <= 0 or frame < 0 or frame == int(anchor):
                continue
            rows.append((node[int(anchor)], node[frame]))
            displacement.append(dense[k, m])
            weights.append(np.exp(-abs(t[frame] - t[int(anchor)]) / args.half_life_s))
    rows = np.asarray(rows, np.int64)
    displacement = np.asarray(displacement, np.float64)
    weights = np.asarray(weights, np.float64)

    arrays = np.load(DATA / "session_cv_matched_cvx.npz", allow_pickle=True)
    blend = arrays[f"{args.session}__cv_calibrated_blend"].astype(np.float64)
    endpoint_rows = np.asarray([
        (node[int(a)], node[int(b)]) for a, b in data["pair"]
    ], np.int64)

    truth, R_nodes = p[usable], R[usable]
    physnet_traj = _solve_positions_weighted(
        truth, R_nodes, rows, displacement, weights
    )
    blend_traj = _solve_positions_weighted(
        truth,
        R_nodes,
        np.vstack((rows, endpoint_rows)),
        np.vstack((displacement, blend)),
        np.concatenate((weights, np.ones(len(blend)))),
    )
    components = _components(len(usable), rows)
    report = {
        "session": args.session,
        "duration_s": round(float(t[usable][-1] - t[usable][0]), 1),
        "fold_models": len(models),
        "dense_edges": int(len(rows)),
        "calibrated_blend_endpoint_edges": int(len(endpoint_rows)),
        "connected_components_with_at_least_3_nodes": len(components),
        "connected_component_sizes": [int(len(c)) for c in components],
        "position_anchors": len(components),
        "dense_physnet_1anchor": metrics(physnet_traj, truth),
        "dense_blend_1anchor": metrics(blend_traj, truth),
    }

    if args.compare.is_file():
        old = np.load(args.compare)
        for key, trajectory in (
            ("dense_physnet_1anchor", physnet_traj),
            ("dense_blend_1anchor", blend_traj),
        ):
            if key in old.files:
                difference = np.linalg.norm(trajectory - old[key], axis=1) * 1000.0
                report[key]["existing_npz_max_difference_mm"] = round(
                    float(np.nanmax(difference)), 6
                )

    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        t=t[usable],
        ground_truth=truth,
        dense_physnet_1anchor=physnet_traj,
        dense_blend_1anchor=blend_traj,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
