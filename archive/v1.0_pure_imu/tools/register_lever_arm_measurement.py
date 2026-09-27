#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Register an independent mechanical camera-to-IMU lever-arm measurement.

Coordinate convention: the left-camera frame has +x right, +y down and +z
forward. Each row is the vector from the USB-IMU sensing centre to the left
camera optical centre, expressed in that frame. The optical centre is not
usually on the lens surface; any drawing-derived correction must be entered
explicitly and its uncertainty archived.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
DEFAULT_CSV = DATA / "lever_arm_measurements.csv"
DEFAULT_OUT = DATA / "lever_arm_measured.json"


def init(path: Path, repeats: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=[
            "trial", "imu_to_camera_x_mm", "imu_to_camera_y_mm",
            "imu_to_camera_z_mm", "instrument_id", "resolution_mm",
            "optical_center_correction_mm", "operator", "notes",
        ])
        w.writeheader()
        for trial in range(1, repeats + 1):
            w.writerow({"trial": trial})
    print(f"wrote {path}; fill every coordinate before analyze")


def analyze(path: Path, output: Path):
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8-sig")))
    required = ("imu_to_camera_x_mm", "imu_to_camera_y_mm", "imu_to_camera_z_mm",
                "instrument_id", "resolution_mm")
    missing = [(r.get("trial"), k) for r in rows for k in required
               if not (r.get(k) or "").strip()]
    if missing:
        raise RuntimeError(f"incomplete mechanical measurements: {missing[:8]}")
    xyz = np.asarray([[float(r[f"imu_to_camera_{a}_mm"]) for a in "xyz"]
                      for r in rows])
    # z is measured to the lens front surface; the optical centre lies
    # `optical_center_correction_mm` behind it, i.e. towards -z.
    corr = np.asarray([float((r.get("optical_center_correction_mm") or "0").strip() or 0)
                       for r in rows])
    xyz[:, 2] -= corr
    mean = xyz.mean(0)
    std = xyz.std(0, ddof=1) if len(xyz) > 1 else np.full(3, np.nan)
    radial = np.linalg.norm(xyz - mean, axis=1)
    resolution = max(float(r["resolution_mm"]) for r in rows)
    # Rectangular resolution uncertainty plus Type-A repeatability.
    u_axis = np.sqrt(np.nan_to_num(std / np.sqrt(len(xyz))) ** 2 +
                     (resolution / np.sqrt(12)) ** 2)

    learned = []
    for report in DATA.glob("physnet_cv/metrics_cvx_*.json"):
        try:
            p = json.loads(report.read_text(encoding="utf-8"))
            if "lever_arm_mm" in p:
                learned.append(p["lever_arm_mm"])
        except (OSError, ValueError):
            pass
    learned = np.asarray(learned, float)
    comparison = None
    if len(learned):
        lm = learned.mean(0)
        comparison = {
            "n_models": int(len(learned)),
            "learned_mean_mm": lm.tolist(),
            "learned_std_mm": learned.std(0, ddof=1).tolist(),
            "difference_measured_minus_learned_mm": (mean - lm).tolist(),
            "difference_norm_mm": float(np.linalg.norm(mean - lm)),
        }

    report = {
        "kind": "independent_mechanical_lever_arm",
        "convention": {
            "vector": "USB IMU sensing centre -> left-camera optical centre",
            "expressed_in": "left-camera frame",
            "axes": {"x": "image right", "y": "image down", "z": "camera forward"},
        },
        "n_trials": len(rows),
        "mean_mm": mean.tolist(),
        "std_mm": std.tolist(),
        "standard_uncertainty_mm": u_axis.tolist(),
        "expanded_uncertainty_k2_mm": (2 * u_axis).tolist(),
        "repeatability_p95_mm": float(np.quantile(radial, 0.95)),
        "instrument_ids": sorted(set(r["instrument_id"] for r in rows)),
        "max_resolution_mm": resolution,
        "trials": rows,
        "learned_comparison_not_used_for_calibration": comparison,
        "passed": bool(len(rows) >= 3 and np.quantile(radial, 0.95) <= 2.0 and
                       np.max(2 * u_axis) <= 2.0),
        "acceptance": {
            "trials_min": 3, "repeatability_p95_mm_max": 2.0,
            "expanded_axis_uncertainty_mm_max": 2.0,
        },
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"],
                      "mean_mm": report["mean_mm"],
                      "expanded_uncertainty_k2_mm":
                          report["expanded_uncertainty_k2_mm"]},
                     ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("init"); a.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    a.add_argument("--repeats", type=int, default=5)
    b = sub.add_parser("analyze"); b.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    b.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    init(args.csv, args.repeats) if args.command == "init" else analyze(args.csv, args.output)


if __name__ == "__main__":
    main()
