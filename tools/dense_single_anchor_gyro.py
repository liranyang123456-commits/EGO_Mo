#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-anchor dense trajectory with gyroscope attitude, and its zero-motion baseline.

tools/build_dense_single_anchor_trajectory.py solves the uninterrupted dense
graph with the reference orientation at every node. This script solves the
same graph (same edges, weights and 16 non-gated fold models) three ways:

* reference orientation (must reproduce the stored 81.08 / 80.22 mm),
* gyroscope orientation propagated from the first node, as in
  tools/eval_pure_imu_pose.py (the "initially anchored relative pose"), and
* zero motion: every node at the single anchor position.

    python tools/dense_single_anchor_gyro.py
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
from tools.build_trajectory_comparison import _components
from tools.eval_session_cv import _solve_positions_weighted
from tools.fuse_trajectory import _gyro_orientation

DATA = ROOT / "datasets"
SESSION = "traj_20260923_023422"
HALF_LIFE_S = 1.5


def ate(trajectory: np.ndarray, truth: np.ndarray) -> float:
    ok = np.isfinite(trajectory).all(1)
    return round(float(np.sqrt(np.mean(np.sum((trajectory[ok] - truth[ok]) ** 2, axis=1))) * 1000.0), 2)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ))
    data = phys.build_group([SESSION], 2.0, 3.0, length)
    data_t = phys.to_device(data, device)
    models = [phys.load_model(path, device) for path in sorted((DATA / "physnet_cv").glob("physnet_cvx_*_s*.pt"))]
    if len(models) != 16:
        raise RuntimeError(f"expected 16 non-gated fold models, found {len(models)}")
    dense = np.mean([model.predict_dense(data_t) for model in models], axis=0)

    t, R, p, usable = phys._load_gt(SESSION)
    node = {int(frame): i for i, frame in enumerate(usable)}
    rows, displacement, weights = [], [], []
    for k, (anchor, _end) in enumerate(data["pair"]):
        for m in range(dense.shape[1]):
            frame = int(data["dense_frame"][k, m])
            if data["dense_valid"][k, m] <= 0 or frame < 0 or frame == int(anchor):
                continue
            rows.append((node[int(anchor)], node[frame]))
            displacement.append(dense[k, m])
            weights.append(np.exp(-abs(t[frame] - t[int(anchor)]) / HALF_LIFE_S))
    rows = np.asarray(rows, np.int64)
    displacement = np.asarray(displacement, np.float64)
    weights = np.asarray(weights, np.float64)
    arrays = np.load(DATA / "session_cv_matched_cvx.npz", allow_pickle=True)
    blend = arrays[f"{SESSION}__cv_calibrated_blend"].astype(np.float64)
    endpoint_rows = np.asarray([(node[int(a)], node[int(b)]) for a, b in data["pair"]], np.int64)
    truth, R_ref = p[usable], R[usable]
    R_gyro = _gyro_orientation(SESSION, t[usable], R_ref[0])
    components = _components(len(usable), rows)

    def solve(R_nodes):
        phys_traj = _solve_positions_weighted(truth, R_nodes, rows, displacement, weights)
        blend_traj = _solve_positions_weighted(
            truth, R_nodes, np.vstack((rows, endpoint_rows)), np.vstack((displacement, blend)),
            np.concatenate((weights, np.ones(len(blend)))))
        return {"dense_physnet_ate_mm": ate(phys_traj, truth), "dense_blend_ate_mm": ate(blend_traj, truth)}

    zero_traj = _solve_positions_weighted(truth, R_ref, rows, np.zeros_like(displacement), weights)
    report = {
        "session": SESSION, "nodes": int(len(usable)), "dense_edges": int(len(rows)),
        "connected_components": len(components), "position_anchors": len(components),
        "reference_orientation": solve(R_ref),
        "gyroscope_orientation_from_first_node": solve(R_gyro),
        "zero_motion_ate_mm": ate(zero_traj, truth),
    }
    (DATA / "dense_single_anchor_gyro.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
