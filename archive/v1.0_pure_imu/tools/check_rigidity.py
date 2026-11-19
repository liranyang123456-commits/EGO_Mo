#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the USB gyro turn rigidly with the camera?

At 10 camera frames/s, consecutive planar-PnP rotations are too noisy for an
axis-by-axis rigidity decision. We therefore compare relative rotations over
0.8-1.2 s, fit the constant IMU-to-camera rotation with Wahba/Kabsch on
alternating samples, and score the held-out samples. A rigid mount should
preserve the three rotation components, rotation magnitude and scale.

    python tools/check_rigidity.py traj_2026xxxx_xxxxxx [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
# A 1 s planar-PnP increment at 10 fps still includes frame-level tilt noise.
# 0.75 is used as an acceptance floor; 0.90 on every axis is reported as
# "high" quality rather than made a hard requirement.
PASS_AXIS_CORR = 0.75
PASS_MAG_CORR = 0.90
PASS_SCALE = (0.85, 1.15)
HORIZON_S = 1.0
HORIZON_TOL_S = 0.20


def _so3_log(R: np.ndarray) -> np.ndarray:
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w * th / (2.0 * np.sin(th))


def _integral(t: np.ndarray, w: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    """Trapezoidal gyro integral with interpolated interval boundaries."""
    if t1 <= t0 or t0 < t[0] or t1 > t[-1]:
        return None
    inside = (t > t0) & (t < t1)
    ts = np.concatenate(([t0], t[inside], [t1]))
    ws = np.vstack([
        [np.interp(t0, t, w[:, k]) for k in range(3)],
        w[inside],
        [np.interp(t1, t, w[:, k]) for k in range(3)],
    ])
    return np.trapezoid(ws, ts, axis=0)


def _fit_rotation(imu: np.ndarray, cam: np.ndarray) -> np.ndarray:
    u, _s, vt = np.linalg.svd(imu.T @ cam)
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    return R


def check(name: str) -> dict:
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * np.pi / 180.0
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    t = gt["t"] - float(gt["delay_usb"][0])
    ok = np.flatnonzero((gt["ok"] == 1) & (gt["reproj"] < 1.5) & (gt["ble_gyro"] < 8.0))
    wi, wc = [], []
    for pos, a in enumerate(ok[:-1]):
        later = ok[pos + 1:]
        b = int(later[np.argmin(np.abs(t[later] - (t[a] + HORIZON_S)))])
        dt = float(t[b] - t[a])
        if abs(dt - HORIZON_S) > HORIZON_TOL_S:
            continue
        g = _integral(usb_t, gyro - bg, float(t[a]), float(t[b]))
        if g is None:
            continue
        rate = np.degrees(np.linalg.norm(g)) / dt
        if rate < 3.0 or rate > 80.0:
            continue
        wi.append(g)
        wc.append(_so3_log(gt["R"][a].T @ gt["R"][b]))
    if len(wi) < 30:
        return {"name": name, "pairs": len(wi), "verdict": "too few moving pairs"}
    wi, wc = np.asarray(wi), np.asarray(wc)
    # Interleaving keeps every motion phase represented in fit and evaluation.
    fit = np.arange(len(wi)) % 2 == 0
    test = ~fit
    R_ci = _fit_rotation(wi[fit], wc[fit])
    pred = (R_ci @ wi[test].T).T
    truth = wc[test]
    corr = [float(np.corrcoef(pred[:, k], truth[:, k])[0, 1]) for k in range(3)]
    mag_corr = float(np.corrcoef(np.linalg.norm(pred, axis=1), np.linalg.norm(truth, axis=1))[0, 1])
    scale = float(np.sum(pred * truth) / max(np.sum(pred * pred), 1e-12))
    rms = np.degrees(np.sqrt(np.mean((pred - truth) ** 2, axis=0)))
    excited = np.degrees(np.std(truth, axis=0))
    enough_axes = bool(np.all(excited >= 1.0))
    passed = (
        enough_axes
        and min(corr) >= PASS_AXIS_CORR
        and mag_corr >= PASS_MAG_CORR
        and PASS_SCALE[0] <= scale <= PASS_SCALE[1]
    )
    quality = "high" if passed and min(corr) >= 0.90 else ("acceptable" if passed else "failed")
    return {
        "name": name,
        "pairs": int(len(wi)),
        "test_pairs": int(test.sum()),
        "corr_tilt_x": round(corr[0], 3),
        "corr_tilt_y": round(corr[1], 3),
        "corr_roll": round(corr[2], 3),
        "corr_magnitude": round(mag_corr, 3),
        "rotation_scale": round(scale, 3),
        "rms_error_deg_xyz": [round(float(v), 3) for v in rms],
        "excitation_std_deg_xyz": [round(float(v), 3) for v in excited],
        "R_camera_imu": np.round(R_ci, 6).tolist(),
        "criterion": {
            "axis_corr_min": PASS_AXIS_CORR,
            "magnitude_corr_min": PASS_MAG_CORR,
            "scale_range": list(PASS_SCALE),
            "excitation_std_deg_min": 1.0,
            "horizon_s": HORIZON_S,
        },
        "quality": quality,
        "verdict": "rigid" if passed else ("insufficient axis excitation" if not enough_axes else "NOT rigid"),
    }


def main() -> None:
    names = sys.argv[1:]
    if not names:
        index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
        names = [row["name"] for row in index]
    for name in names:
        print(json.dumps(check(name), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
