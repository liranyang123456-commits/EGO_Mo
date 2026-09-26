#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Freeze the prospective acquisition and final-evaluation protocol.

The resulting lock is intentionally independent of the already observed test
recording. New theme-T recordings are accepted as prospectively sealed tests
only if their session_meta.json contains this protocol ID.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
DEFAULT = DATA / "study_protocol_v2.lock.json"


def canonical(payload: dict) -> bytes:
    clean = {k: v for k, v in payload.items() if k not in
             ("protocol_id", "frozen_at_utc")}
    return json.dumps(clean, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def build() -> dict:
    return {
        "schema": 1,
        "name": "EGO_Mo prospective validation v2",
        "status": "FROZEN",
        "scientific_scope": {
            "inference_input": "continuous six-axis IMU stream and timestamps only",
            "visual_data_role": "calibration, training supervision, and evaluation only",
            "output": "3-D camera displacement increments and translation trajectory",
            "not_claimed": "reference-grade ground truth or learned full absolute SE(3) pose",
        },
        "sensor_protocol": {
            "imu_hz": 200,
            "camera_resolution": [1280, 720],
            "camera_actual_fps_target": 10,
            "board": "GP050, 11x8 inner corners, 3-mm pitch",
        },
        "window_protocol": {
            "horizon_s": 3.0,
            "context_before_s": 2.0,
            "context_after_s": 2.0,
            "target_frame": "camera frame at pair start",
        },
        "prospective_acquisition": {
            "session_duration_s": 180,
            "pause_every_s": [10, 15],
            "pause_duration_s": [1, 2],
            "travel_cm": [10, 20],
            "working_distance_cm": [15, 40],
            "required_test_sessions": 4,
            "test_theme": "T",
            "independence_requirements": [
                "at least two acquisition days",
                "at least two operators or grip configurations",
                "split assigned before capture",
                "no model or threshold update after viewing a theme-T metric",
            ],
        },
        "acceptance": {
            "usb_imu_hz_min": 195,
            "ble_imu_hz_min": 150,
            "left_image_fraction_min": 0.85,
            "chessboard_usable_fraction_min": 0.70,
            "reprojection_p50_px_max_720p": 0.40,
            "frame_drops_max": 0,
            "rigidity_axis_correlation_min": 0.75,
            "rigidity_magnitude_correlation_min": 0.90,
            "rigidity_scale_range": [0.85, 1.15],
        },
        "frozen_models": {
            "physnet": {
                "input_representation": "anchor-frame 24-channel physics features",
                "width": 96,
                "weight_decay": 0.1,
                "dense_supervision": True,
                "lever_arm": True,
                "stillness_weight": 1.0,
                "stillness_gate": True,
                "still_mm_s": 3.0,
                "moving_mm_s": 8.0,
            },
            "baseline": "adapted IMUNet, same input/target/split protocol",
            "ensemble": "eight leave-one-session-out folds x two seeds",
            "blend": {
                "equal_weight_local_error": [0.5, 0.5],
                "cv_calibrated_physnet_weight": 0.65,
                "cv_calibrated_shrink": 1.05,
            },
            "uncertainty_scale": 4.559,
        },
        "final_metrics": [
            "mean 3-s Euclidean displacement error (mm)",
            "multivariate R2",
            "fastest-reference-third error (mm)",
            "translation-trajectory ATE RMSE (mm)",
            "1-sigma and 2-sigma interval coverage",
            "3-s block-bootstrap paired 95% confidence interval",
        ],
        "comparisons": [
            "zero motion",
            "ridge summary features",
            "adapted RoNIN-ResNet",
            "adapted TLIO-ResNet",
            "adapted IMUNet",
            "PhysNet",
            "predeclared PhysNet/IMUNet blends",
        ],
        "decision_rules": {
            "primary_method_claim": "pooled prospective-test mean displacement error",
            "significance": "paired 3-s block-bootstrap CI excludes zero",
            "sota_claim": "only protocol-matched evaluated methods; no cross-protocol official-score claim",
            "test_reuse": "one final evaluation bundle; a correction requires a signed amendment and new test data",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; use --force only before new data capture")
    payload = build()
    payload["protocol_id"] = hashlib.sha256(canonical(payload)).hexdigest()[:16]
    payload["frozen_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    checklist = {
        "protocol_id": payload["protocol_id"],
        "slots": [
            {"slot": f"T{i}", "theme": "T", "split": "test", "session": None,
             "day": None, "operator": None, "grip": None, "status": "PENDING"}
            for i in range(1, 5)
        ],
    }
    (args.output.parent / "prospective_test_checklist.json").write_text(
        json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "protocol_id": payload["protocol_id"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
