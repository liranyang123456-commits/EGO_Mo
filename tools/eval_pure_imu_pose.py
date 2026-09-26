#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a pure-IMU relative 6-DoF pose stream.

Translation increments come from a learned IMU-only model. Orientation comes
from bias-corrected gyroscope integration. One initial reference position and
orientation set the gauge; no later image or chessboard observation is used.
This is therefore a relative pose estimate, not an absolute-pose or ground
truth estimator.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
from tools.build_trajectory_comparison import _metrics, _solve_positions
from tools.fuse_trajectory import _gyro_orientation

DATA = ROOT / "datasets"


def angle_deg(R):
    c = np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)
    return np.degrees(np.arccos(c))


def evaluate(session: str, predictions: np.lib.npyio.NpzFile, method: str):
    t, R, p, usable = phys._load_gt(session)
    t_node, R_ref, p_ref = t[usable], R[usable], p[usable]
    R_imu = _gyro_orientation(session, t_node, R_ref[0])
    frame_to_node = {int(frame): i for i, frame in enumerate(usable)}

    prefix = f"{session}__"
    pair_key = prefix + "pair" if prefix + "pair" in predictions else "pair"
    pred_key = prefix + method if prefix + method in predictions else method
    pairs_frame = predictions[pair_key]
    pred = predictions[pred_key]
    edges, dp = [], []
    for (a, b), value in zip(pairs_frame, pred):
        if int(a) in frame_to_node and int(b) in frame_to_node:
            edges.append((frame_to_node[int(a)], frame_to_node[int(b)]))
            dp.append(value)
    edges = np.asarray(edges, np.int64)
    dp = np.asarray(dp, np.float64)
    trajectory = _solve_positions(p_ref, R_imu, edges, dp)

    ori_err = angle_deg(np.einsum("nji,njk->nik", R_imu, R_ref))
    finite = np.isfinite(trajectory).all(1)
    pos_err = np.linalg.norm(trajectory[finite] - p_ref[finite], axis=1) * 1000

    rpe = {}
    for horizon in (1.0, 3.0, 10.0):
        te, re = [], []
        for i in range(len(t_node)):
            j = int(np.argmin(np.abs(t_node - (t_node[i] + horizon))))
            if j <= i or abs(t_node[j] - t_node[i] - horizon) > 0.25 * horizon:
                continue
            Rgt = R_ref[i].T @ R_ref[j]
            Rest = R_imu[i].T @ R_imu[j]
            re.append(float(angle_deg(Rest.T @ Rgt)))
            if finite[i] and finite[j]:
                dgt = R_ref[i].T @ (p_ref[j] - p_ref[i])
                dest = R_imu[i].T @ (trajectory[j] - trajectory[i])
                te.append(float(np.linalg.norm(dest - dgt) * 1000))
        rpe[f"{horizon:g}s"] = {
            "n": len(re),
            "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(re)))),
            "translation_rmse_mm": float(np.sqrt(np.mean(np.square(te))))
                if te else None,
        }
    return {
        "session": session, "method": method,
        "gauge": "reference position and attitude at first usable node only",
        "inference_after_initialization": "IMU stream only",
        "nodes": len(usable), "edges": len(edges),
        "orientation": {
            "rmse_deg": float(np.sqrt(np.mean(ori_err ** 2))),
            "mean_deg": float(np.mean(ori_err)),
            "p95_deg": float(np.quantile(ori_err, 0.95)),
            "terminal_deg": float(ori_err[-1]),
            "terminal_drift_deg_s": float(ori_err[-1] /
                                          max(t_node[-1] - t_node[0], 1e-9)),
        },
        "translation": _metrics(trajectory, p_ref),
        "translation_error_p95_mm": float(np.quantile(pos_err, 0.95)),
        "relative_pose_error": rpe,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path,
                   default=DATA / "session_cv_benchmark.npz")
    p.add_argument("--sessions", nargs="*", default=[])
    p.add_argument("--methods", nargs="*",
                   default=["physnet", "imunet", "equal_blend",
                            "cv_calibrated_blend"])
    p.add_argument("--split-file", default="trajectory_split_20260924.json")
    p.add_argument("--output", type=Path,
                   default=DATA / "pure_imu_pose_benchmark.json")
    args = p.parse_args()
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    sessions = args.sessions or split["test"]
    predictions = np.load(args.predictions, allow_pickle=True)
    report = {
        "kind": "initially_anchored_pure_imu_relative_6dof",
        "not_claimed": [
            "absolute attitude without initialization",
            "reference-grade ground truth",
            "learned orientation (orientation is gyroscope-integrated)",
        ],
        "sessions": {},
    }
    for session in sessions:
        report["sessions"][session] = {}
        for method in args.methods:
            key = f"{session}__{method}"
            if key not in predictions and method not in predictions:
                continue
            row = evaluate(session, predictions, method)
            report["sessions"][session][method] = row
            print(json.dumps({
                session: {method: {
                    "orientation_rmse_deg": row["orientation"]["rmse_deg"],
                    "ate_rmse_mm": row["translation"]["ate_rmse_mm"],
                    "rpe_3s": row["relative_pose_error"]["3s"],
                }}
            }, ensure_ascii=False))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")


if __name__ == "__main__":
    main()
