#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate one exported synthetic sequence."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.build_pose_gt import _find
from tools.build_pose_gt import _load_k
from ego_capture.sync import load_imu


def _csv_times(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as handle:
        return np.asarray([float(row["timestamp"]) for row in csv.DictReader(handle)])


def validate(session: Path) -> dict:
    meta = json.loads((session / "session_meta.json").read_text(encoding="utf-8"))
    gt = np.load(session / "ground_truth.npz")
    imu_t, imu = load_imu(session / "imu_stream.csv")
    left = sorted((session / "cam0" / "images").glob("*.jpg"))
    right_images = sorted((session / "cam1" / "images").glob("*.jpg"))
    sample = left[::max(1, len(left) // 30)]
    found = 0
    corner_rmse = []
    disparities = []
    K0, d0 = _load_k("cam0", int(gt["image_size"][0]), int(gt["image_size"][1]))
    square = float(meta["config"]["square_mm"])
    obj = np.zeros((8 * 11, 3), np.float64)
    obj[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * square
    for path in sample:
        frame = int(path.stem)
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        corners, pattern = _find(image)
        found += int(corners is not None)
        if corners is None or pattern != (11, 8):
            continue
        R_B_C = gt["R"][frame].astype(np.float64)
        p_B_C = gt["p"][frame].astype(np.float64)
        R_C_B = R_B_C.T
        t_C_B = -R_C_B @ p_B_C * 1000.0
        rvec, _ = cv2.Rodrigues(R_C_B)
        projected, _ = cv2.projectPoints(obj, rvec, t_C_B, K0, d0)
        projected = projected.reshape(8, 11, 2)
        detected = corners.reshape(8, 11, 2)
        candidates = (
            detected,
            detected[::-1, ::-1],
            detected[:, ::-1],
            detected[::-1],
        )
        corner_rmse.append(min(
            float(np.sqrt(np.mean((candidate - projected) ** 2)))
            for candidate in candidates
        ))
        right_path = session / "cam1" / "images" / path.name
        right_image = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        right_corners, right_pattern = _find(right_image)
        if right_corners is not None and right_pattern == pattern:
            disparities.append(float(corners[:, 0, 0].mean() - right_corners[:, 0, 0].mean()))
    ortho = np.max(np.abs(np.einsum("nji,njk->nik", gt["R_W_C"], gt["R_W_C"]) - np.eye(3)))
    det = np.linalg.det(gt["R_W_C"])
    physics = None
    if all(key in gt.files for key in ("imu_R_W_C", "imu_p_W_C", "imu_usb_ideal")):
        ti = gt["imu_t"].astype(np.float64)
        R_W_C = gt["imu_R_W_C"].astype(np.float64)
        p_W_C = gt["imu_p_W_C"].astype(np.float64)
        R_C_I = np.asarray(meta["R_camera_imu"], dtype=np.float64)
        lever = np.array([
            meta["config"]["imu_lever_arm_mm_x"],
            meta["config"]["imu_lever_arm_mm_y"],
            meta["config"]["imu_lever_arm_mm_z"],
        ]) / 1000.0
        R_W_I = np.einsum("nij,jk->nik", R_W_C, R_C_I)
        p_W_I = p_W_C + np.einsum("nij,j->ni", R_W_C, lever)
        velocity = np.gradient(p_W_I, ti, axis=0, edge_order=2)
        acceleration = np.gradient(velocity, ti, axis=0, edge_order=2)
        expected_acc = np.einsum(
            "nji,nj->ni",
            R_W_I,
            acceleration - np.array([0.0, 0.0, -9.80665]),
        ) / 9.80665
        rel = np.einsum("nji,njk->nik", R_W_I[:-1], R_W_I[1:])
        dt = np.diff(ti)
        expected_gyro = np.vstack((
            Rotation.from_matrix(rel[0]).as_rotvec() / dt[0],
            Rotation.from_matrix(rel).as_rotvec() / dt[:, None],
        ))
        expected_gyro = np.degrees(expected_gyro)
        ideal = gt["imu_usb_ideal"].astype(np.float64)
        physics = {
            "ideal_acc_rmse_g_xyz": np.round(
                np.sqrt(np.mean((ideal[:, :3] - expected_acc) ** 2, axis=0)), 9
            ).tolist(),
            "ideal_gyro_rmse_dps_xyz": np.round(
                np.sqrt(np.mean((ideal[1:, 3:6] - expected_gyro[1:]) ** 2, axis=0)), 9
            ).tolist(),
            "timestamp_dt_std_ms": round(float(np.std(dt) * 1000.0), 4),
        }
    report = {
        "session": session.name,
        "frames_left": len(left),
        "frames_right": len(right_images),
        "gt_frames": int(len(gt["t"])),
        "imu_samples": int(len(imu_t)),
        "imu_rate_hz": round((len(imu_t) - 1) / max(imu_t[-1] - imu_t[0], 1e-9), 3),
        "rotation_orthogonality_max": float(ortho),
        "rotation_det_range": [float(det.min()), float(det.max())],
        "checkerboard_detection_sample": f"{found}/{len(sample)}",
        "corner_truth_rmse_px_median": None if not corner_rmse else round(float(np.median(corner_rmse)), 3),
        "stereo_disparity_px_median": None if not disparities else round(float(np.median(disparities)), 2),
        "acc_norm_g_range": [
            round(float(np.linalg.norm(imu[:, 0:3], axis=1).min()), 3),
            round(float(np.linalg.norm(imu[:, 0:3], axis=1).max()), 3),
        ],
        "gyro_norm_dps_max": round(float(np.linalg.norm(imu[:, 3:6], axis=1).max()), 2),
        "nominal_stereo_warning": meta["provisional"]["stereo_extrinsic"],
        "physics_consistency": physics,
    }
    passed = (
        len(left) == len(right_images) == len(gt["t"])
        and abs(report["imu_rate_hz"] - meta["config"]["imu_hz"]) < 0.5
        and ortho < 1e-5
        and (not sample or found >= max(1, int(0.5 * len(sample))))
        and (not corner_rmse or float(np.median(corner_rmse)) < 2.0)
        and (
            physics is None
            or (
                max(physics["ideal_acc_rmse_g_xyz"]) < 5e-5
                and max(physics["ideal_gyro_rmse_dps_xyz"]) < 1e-4
            )
        )
    )
    report["passed"] = passed
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    args = ap.parse_args()
    report = validate(args.session)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
