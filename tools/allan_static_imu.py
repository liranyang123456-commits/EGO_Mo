#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Overlapping Allan deviation of a static USB-IMU log.

The accelerometer is reported in g and the gyroscope in deg/s, per axis, on a
logarithmic grid of averaging times. Samples are resampled to the nominal
200-Hz grid first. A gyroscope channel that never leaves zero is flagged,
because its noise lies below the 0.061-deg/s quantization step.

    python tools/allan_static_imu.py align_imu_static_20260922_115532 align_imu_static_20260922_115758
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
HZ = 200.0


def oadev(x: np.ndarray, m: int) -> float:
    """Overlapping Allan deviation for cluster size m (samples)."""
    c = np.concatenate(([0.0], np.cumsum(x)))
    n = len(x)
    if n < 2 * m + 1:
        return float("nan")
    avg = (c[m:] - c[:-m]) / m
    d = avg[m:] - avg[:-m]
    return float(np.sqrt(0.5 * np.mean(d ** 2)))


def analyze(name: str, taus: np.ndarray) -> dict:
    a = np.genfromtxt(DATA / name / "imu_stream.csv", delimiter=",", skip_header=1, usecols=range(7))
    t = a[:, 0] - a[0, 0]
    grid = np.arange(0.0, t[-1], 1.0 / HZ)
    acc = np.column_stack([np.interp(grid, t, a[:, c]) for c in (1, 2, 3)])
    gyro = np.column_stack([np.interp(grid, t, a[:, c]) for c in (4, 5, 6)])
    out = {"seconds": round(float(t[-1]), 1), "taus_s": [], "acc_adev_g": [], "gyro_adev_dps": [],
           "gyro_all_zero": bool((np.abs(a[:, 4:7]) < 1e-9).all())}
    for tau in taus:
        m = int(round(tau * HZ))
        if m < 1 or len(grid) < 2 * m + 1:
            continue
        out["taus_s"].append(float(tau))
        out["acc_adev_g"].append([oadev(acc[:, k], m) for k in range(3)])
        out["gyro_adev_dps"].append([oadev(gyro[:, k], m) for k in range(3)])
    acc_norm = np.linalg.norm(np.asarray(out["acc_adev_g"]), axis=1)
    out["acc_adev_norm_g"] = acc_norm.tolist()
    if len(acc_norm) >= 3:
        tt = np.log10(out["taus_s"][-3:])
        out["acc_log_slope_last3"] = float(np.polyfit(tt, np.log10(acc_norm[-3:]), 1)[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--output", type=Path, default=DATA / "allan_static_imu.json")
    args = ap.parse_args()
    taus = np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000])
    report = {name: analyze(name, taus) for name in args.sessions}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, r in report.items():
        pairs = ", ".join(f"{tau:g}s:{v:.2e}" for tau, v in zip(r["taus_s"], r["acc_adev_norm_g"]))
        print(f"{name} ({r['seconds']} s, gyro all zero: {r['gyro_all_zero']}): acc ADEV |g| {pairs}")
        if "acc_log_slope_last3" in r:
            print(f"  log-log slope over the last three taus: {r['acc_log_slope_last3']:+.2f}")


if __name__ == "__main__":
    sys.exit(main())
