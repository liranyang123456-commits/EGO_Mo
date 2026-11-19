#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""What the static bench logs do and do not say about the accelerometer random walk.

Reads datasets/allan_static_imu.json (tools/allan_static_imu.py) and reports

* the relative 1-sigma uncertainty of every Allan-deviation estimate,
  1 / sqrt(2 (N/n - 1)) for N samples and clusters of n samples;
* the log-log slope of the largest-axis curve up to 5 s and beyond;
* the largest rate random walk K that is compatible with the largest-axis
  deviation at the longest averaging time, sigma(tau) >= K sqrt(tau / 3), with
  and without one sigma of that estimate;
* the displacement that such a random walk adds to a 3-s window (Monte Carlo
  with the window layout of the estimator: 7-s window, demeaned, interval from
  2 to 5 s, zero velocity and displacement at the anchor);
* whether the gyroscope zeros are quantization: the smallest non-zero rate of
  a handheld recording against the 16-bit step.

    python tools/allan_static_bound.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
HZ = 200.0
G0 = 9.80665
GYRO_STEP = 4000.0 / 65536.0          # +-2000 deg/s over 16 bit
MOVING = "traj_20260923_023422"


def displacement_rms_mm(K: float, trials: int = 20000, seed: int = 0) -> float:
    """RMS displacement error per axis (mm) of a bias random walk K (g/sqrt(s))."""
    rng = np.random.default_rng(seed)
    n, a, b = int(7 * HZ), int(2 * HZ), int(5 * HZ)
    walk = np.cumsum(rng.normal(0.0, np.sqrt(1.0 / HZ), (trials, n)), axis=1) * K * G0
    lin = walk - walk.mean(1, keepdims=True)
    v = np.cumsum(lin, axis=1) / HZ
    v -= v[:, a:a + 1]
    d = np.cumsum(v, axis=1) / HZ
    d -= d[:, a:a + 1]
    return float(np.sqrt(np.mean(d[:, b] ** 2)) * 1000)


def main() -> None:
    logs = json.loads((DATA / "allan_static_imu.json").read_text(encoding="utf-8"))
    sessions, bounds = {}, []
    for name, v in logs.items():
        tau = np.array(v["taus_s"])
        adev = np.array(v["acc_adev_g"])
        n_samples = int(round(v["seconds"] * HZ))
        clusters = n_samples / (tau * HZ)
        rel = 1.0 / np.sqrt(2.0 * (clusters - 1.0))
        axis = int(np.argmax(adev[-1]))                      # largest axis at the longest tau
        largest = adev[:, axis]

        def slope(lo, hi):
            m = (tau >= lo) & (tau <= hi)
            return round(float(np.polyfit(np.log10(tau[m]), np.log10(largest[m]), 1)[0]), 2)

        k0 = largest[-1] * np.sqrt(3.0 / tau[-1])
        k1 = largest[-1] * (1.0 + rel[-1]) * np.sqrt(3.0 / tau[-1])
        bounds.append((k0, k1))
        sessions[name] = {
            "seconds": v["seconds"], "samples": n_samples, "largest_axis": "xyz"[axis],
            "taus_s": tau.tolist(), "clusters": [round(float(c), 1) for c in clusters],
            "relative_sigma": [round(float(r), 3) for r in rel],
            "largest_axis_adev_g": largest.tolist(),
            "slope_0p1_to_5s": slope(0.1, 5.0), "slope_5_to_20s": slope(5.0, 20.0),
            "random_walk_bound_g_sqrt_s": float(k0), "random_walk_bound_plus_1sigma_g_sqrt_s": float(k1),
        }
    k0 = max(b[0] for b in bounds)
    k1 = max(b[1] for b in bounds)
    moving = np.genfromtxt(DATA / MOVING / "imu_stream.csv", delimiter=",", skip_header=1, usecols=(4, 5, 6))
    nonzero = np.abs(moving[moving != 0])
    report = {
        "sessions": sessions,
        "random_walk_bound_g_sqrt_s": round(float(k0), 7),
        "random_walk_bound_plus_1sigma_g_sqrt_s": round(float(k1), 7),
        "displacement_3s_rms_mm_per_axis": {"at_bound": round(displacement_rms_mm(k0), 2),
                                            "at_bound_plus_1sigma": round(displacement_rms_mm(k1), 2)},
        "gyroscope": {"step_dps": round(GYRO_STEP, 4),
                      "static_all_zero": bool(all(v["gyro_all_zero"] for v in logs.values())),
                      "moving_recording": MOVING,
                      "moving_smallest_nonzero_dps": round(float(nonzero.min()), 4),
                      "moving_all_multiples_of_step": bool(
                          np.abs(nonzero / GYRO_STEP - np.rint(nonzero / GYRO_STEP)).max() < 0.02)},
    }
    (DATA / "allan_static_bound.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name, s in sessions.items():
        print(f"{name}: {s['seconds']} s, largest axis {s['largest_axis']}, slope 0.1-5 s {s['slope_0p1_to_5s']}, "
              f"5-20 s {s['slope_5_to_20s']}; relative sigma at tau = 5, 10, 20 s: {s['relative_sigma'][-3:]}")
    print(json.dumps({k: v for k, v in report.items() if k != "sessions"}, indent=2))


if __name__ == "__main__":
    main()
