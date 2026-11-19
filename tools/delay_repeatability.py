#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Repeatability of the per-recording camera-IMU delay (Section V-B).

The capture program estimates the camera-to-USB-IMU delay of every recording
by correlating image motion with the gyroscope rate (ego_capture/sync.py).
Here the same estimator is run on the two halves of each recording; if the
true delay is constant within a recording, the half-to-half difference d has
variance 2*s_half^2, and a full-recording estimate has about s_d / 2. A
robust (MAD) standard deviation is used because recordings with little
motion give occasional failed estimates.

    python tools/delay_repeatability.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import estimate_delay, image_motion, load_imu, load_times

DATA = ROOT / "datasets"
SEARCH_S = 0.15   # the +-150-ms window of the endoscope recordings reported in the paper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=DATA / "delay_repeatability.json")
    parser.add_argument("--max-date", type=int,
                        help="include only session names whose YYYYMMDD token is no later than this")
    args = parser.parse_args()
    rows = []
    for folder in (sorted(args.data_root.glob("traj_*")) + sorted(args.data_root.glob("rigid_*"))
                   + sorted(args.data_root.glob("gtraj_2*"))):
        if args.max_date is not None:
            try:
                if int(folder.name.split("_")[1]) > args.max_date:
                    continue
            except (IndexError, ValueError):
                continue
        video, times = folder / "cam0" / "video.avi", folder / "cam0" / "times.txt"
        if not (video.is_file() and times.is_file() and (folder / "imu_stream.csv").is_file()):
            continue
        tm, motion = image_motion(video, load_times(times))
        ut, uy = load_imu(folder / "imu_stream.csv")
        g = np.linalg.norm(uy[:, 3:6], axis=1)
        full = estimate_delay(tm, motion, ut, g, search_s=SEARCH_S)
        if not full["reliable"]:
            continue
        mid = np.median(tm)
        h1 = estimate_delay(tm[tm <= mid], motion[tm <= mid], ut, g, search_s=SEARCH_S)
        h2 = estimate_delay(tm[tm > mid], motion[tm > mid], ut, g, search_s=SEARCH_S)
        rows.append({"session": folder.name, "full_ms": 1000 * full["delay_s"], "half1_ms": 1000 * h1["delay_s"],
                     "half2_ms": 1000 * h2["delay_s"], "corr": round(full["corr"], 3)})
        print(rows[-1], flush=True)
    d = np.array([r["half1_ms"] - r["half2_ms"] for r in rows])
    s_d = 1.4826 * float(np.median(np.abs(d - np.median(d))))
    traj = np.array([r["full_ms"] for r in rows if r["session"].startswith("traj_")])
    report = {
        "recordings": len(rows),
        "half_difference_median_abs_ms": round(float(np.median(np.abs(d))), 1),
        "half_difference_robust_sd_ms": round(s_d, 1),
        "half_length_estimate_sd_ms": round(s_d / np.sqrt(2), 1),
        "full_length_estimate_sd_ms": round(s_d / 2, 1),
        "trajectory_delay_range_ms": [round(float(traj.min()), 1), round(float(traj.max()), 1)],
        "per_recording": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "per_recording"}, indent=2))


if __name__ == "__main__":
    main()
