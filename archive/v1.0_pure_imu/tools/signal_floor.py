#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much of the camera step is visible in the USB accelerometer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
from tools.train_trajectory import TEST_NAME, VAL_NAME

GT = ROOT / "datasets" / "pose_gt"
DATA = ROOT / "datasets"


def _quat_R(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ]
    )


def _pairs(name: str):
    gt = np.load(GT / f"{name}.npz")
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    t = gt["t"][ok] - float(gt["delay_usb"][0])
    R = gt["R"][ok]
    p = gt["p"][ok]
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    acc = usb[:, 0:3] * 9.81
    gyro = usb[:, 3:6]
    quat = usb[:, 15:19]
    rows = []
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt < 0.12 or dt > 0.30:
            continue
        i0 = int(np.searchsorted(usb_t, t[i - 1]))
        i1 = int(np.searchsorted(usb_t, t[i]))
        if i1 <= i0 + 5:
            continue
        dp = R[i - 1].T @ (p[i] - p[i - 1])
        sl = slice(i0, i1)
        rows.append((dp, acc[sl], gyro[sl], quat[sl], usb_t[sl], dt))
    return rows


def _world_acc(acc, quat, use_t: bool):
    out = np.zeros_like(acc)
    for i in range(len(acc)):
        Rm = _quat_R(quat[i])
        out[i] = (Rm.T if use_t else Rm) @ acc[i]
    return out


def _best_frame(acc, quat, gyro):
    still = np.linalg.norm(gyro, axis=1) < 3.0
    if int(still.sum()) < 20:
        still = np.ones(len(acc), dtype=bool)
    scores = {}
    for use_t in (False, True):
        world = _world_acc(acc, quat, use_t)
        scores[use_t] = float(np.linalg.norm(world[still] - world[still].mean(0), axis=1).mean())
    use_t = min(scores, key=scores.get)
    return use_t, scores[use_t]


def _r2(pred, truth) -> float:
    resid = np.sum((pred - truth) ** 2)
    base = np.sum((truth - truth.mean(0)) ** 2)
    return float(1.0 - resid / max(base, 1e-12))


def _half_score(src, dst):
    mid = len(src) // 2
    if mid < 20 or len(src) - mid < 20:
        return None
    design = np.concatenate([src[:mid], np.ones((mid, 1))], 1)
    coef, *_ = np.linalg.lstsq(design, dst[:mid], rcond=None)
    pred = np.concatenate([src[mid:], np.ones((len(src) - mid, 1))], 1) @ coef
    err = np.linalg.norm(pred - dst[mid:], axis=1)
    zero = np.linalg.norm(dst[mid:], axis=1)
    return {
        "zero_mm": round(float(zero.mean() * 1000.0), 2),
        "fit_mm": round(float(err.mean() * 1000.0), 2),
        "r2": round(_r2(pred, dst[mid:]), 3),
    }


def _session_gravity(name: str, use_t: bool):
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    acc = usb[:, 0:3] * 9.81
    gyro = usb[:, 3:6]
    quat = usb[:, 15:19]
    still = np.linalg.norm(gyro, axis=1) < 3.0
    if int(still.sum()) < 50:
        still = np.ones(len(acc), dtype=bool)
    step = max(1, int(still.sum()) // 4000)
    idx = np.flatnonzero(still)[::step]
    world = _world_acc(acc[idx], quat[idx], use_t)
    return world.mean(0), float(np.linalg.norm(world - world.mean(0), axis=1).mean())


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    report = {}
    for name in names:
        rows = _pairs(name)
        acc = np.concatenate([r[1] for r in rows])
        gyro = np.concatenate([r[2] for r in rows])
        quat = np.concatenate([r[3] for r in rows])
        use_t, _spread = _best_frame(acc, quat, gyro)
        gravity, spread = _session_gravity(name, use_t)
        mean_a = []
        dp = []
        speed = []
        gt_acc = []
        for d, a, g, q, _ts, dt in rows:
            world = _world_acc(a, q, use_t)
            mean_a.append((world - gravity).mean(0))
            dp.append(d)
            speed.append(np.linalg.norm(d) / dt)
            gt_acc.append(2.0 * np.linalg.norm(d) / (dt * dt))
        mean_a = np.stack(mean_a)
        dp = np.stack(dp)
        speed = np.asarray(speed)
        fast = speed >= np.quantile(speed, 0.67)
        slow = speed <= np.quantile(speed, 0.33)
        report[name] = {
            "n": len(rows),
            "quat_transpose": use_t,
            "still_spread_mps2": round(spread, 3),
            "acc_norm_g": round(float(np.median(np.linalg.norm(acc, axis=1)) / 9.81), 3),
            "fast_lin_mps2": round(float(np.median(np.linalg.norm(mean_a[fast], axis=1))), 3),
            "slow_lin_mps2": round(float(np.median(np.linalg.norm(mean_a[slow], axis=1))), 3),
            "fast_speed_mm_s": round(float(np.median(speed[fast]) * 1000.0), 1),
            "gt_fast_acc_mps2": round(float(np.median(np.asarray(gt_acc)[fast])), 3),
            "half": _half_score(mean_a, dp),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v6" / "signal_floor.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    held = [n for n in names if n in (TEST_NAME, VAL_NAME)]
    print(json.dumps({"held": held}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
