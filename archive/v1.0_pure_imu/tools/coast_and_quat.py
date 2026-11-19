#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Two translations of the same 0.2 s label.

Vision coast uses only the previous chessboard step. Quaternion integration
removes gravity with the IMU's own attitude and keeps a leaky velocity.
"""

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
    w, x, y, z = [q[:, i] for i in range(4)]
    n = np.maximum(w * w + x * x + y * y + z * z, 1e-12)
    s = 2.0 / n
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - s * (y * y + z * z)
    R[:, 0, 1] = s * (x * y - z * w)
    R[:, 0, 2] = s * (x * z + y * w)
    R[:, 1, 0] = s * (x * y + z * w)
    R[:, 1, 1] = 1 - s * (x * x + z * z)
    R[:, 1, 2] = s * (y * z - x * w)
    R[:, 2, 0] = s * (x * z - y * w)
    R[:, 2, 1] = s * (y * z + x * w)
    R[:, 2, 2] = 1 - s * (x * x + y * y)
    return R


def _poses(name: str):
    gt = np.load(GT / f"{name}.npz")
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    delay = float(gt["delay_usb"][0])
    return gt["t"][ok] - delay, gt["R"][ok], gt["p"][ok]


def _r2(pred, truth) -> float:
    resid = np.sum((pred - truth) ** 2)
    base = np.sum((truth - truth.mean(0)) ** 2)
    return float(1.0 - resid / max(base, 1e-12))


def _pack(pred, truth) -> dict:
    err = np.linalg.norm(pred - truth, axis=1)
    speed = np.linalg.norm(truth, axis=1)
    fast = speed >= np.quantile(speed, 0.67)
    return {
        "n": int(len(err)),
        "zero_mm": round(float(speed.mean() * 1000.0), 2),
        "err_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "r2": round(_r2(pred, truth), 3),
    }


def _vision(name: str) -> dict:
    t, R, p = _poses(name)
    dt = np.diff(t)
    board = np.diff(p, axis=0)
    cam = np.stack([R[i].T @ board[i] for i in range(len(board))])
    # Predict step i from step i-1, scaled by duration. Step 0 has no history.
    pred_b = board[:-1] * (dt[1:, None] / np.maximum(dt[:-1, None], 1e-3))
    pred_c = cam[:-1] * (dt[1:, None] / np.maximum(dt[:-1, None], 1e-3))
    bands = {}
    for label, mask in (
        ("short", dt[1:] < 0.12),
        ("step", (dt[1:] >= 0.12) & (dt[1:] <= 0.30)),
        ("gap", dt[1:] > 0.30),
    ):
        if int(mask.sum()) < 5:
            continue
        bands[label] = {
            "board": _pack(pred_b[mask], board[1:][mask]),
            "cam": _pack(pred_c[mask], cam[1:][mask]),
        }
    hist = np.histogram(dt, bins=[0, 0.12, 0.30, 1.0, 10.0])
    return {"dt_counts": [int(x) for x in hist[0]], "bands": bands}


def _quat_steps(name: str, tau: float):
    t, R, p = _poses(name)
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    acc = usb[:, 0:3] * 9.81
    gyro = np.linalg.norm(usb[:, 3:6], axis=1)
    Rw = _quat_R(usb[:, 15:19])
    specific_w = np.einsum("nij,nj->ni", Rw, acc)
    still = gyro < 3.0
    gravity = specific_w[still].mean(0) if int(still.sum()) >= 50 else specific_w.mean(0)
    lin = specific_w - gravity
    v = np.zeros(3)
    pos = np.zeros((len(usb_t), 3))
    t_prev = float(usb_t[0])
    for i in range(1, len(usb_t)):
        dt = float(usb_t[i] - t_prev)
        t_prev = float(usb_t[i])
        if dt <= 0.0 or dt > 0.05:
            pos[i] = pos[i - 1]
            continue
        v = (v + lin[i] * dt) * np.exp(-dt / tau)
        pos[i] = pos[i - 1] + v * dt
    src, dst = [], []
    for i in range(1, len(t)):
        dt = float(t[i] - t[i - 1])
        if dt < 0.12 or dt > 0.30:
            continue
        i0 = int(np.searchsorted(usb_t, t[i - 1]))
        i1 = int(np.searchsorted(usb_t, t[i]))
        i0 = min(max(i0, 0), len(usb_t) - 1)
        i1 = min(max(i1, 0), len(usb_t) - 1)
        if i1 <= i0:
            continue
        src.append(Rw[i0].T @ (pos[i1] - pos[i0]))
        dst.append(R[i - 1].T @ (p[i] - p[i - 1]))
    if len(src) < 8:
        return None
    return np.stack(src), np.stack(dst)


def _half(src, dst):
    mid = len(src) // 2
    if mid < 15:
        return None
    design = np.concatenate([src[:mid], np.ones((mid, 1))], 1)
    coef, *_ = np.linalg.lstsq(design, dst[:mid], rcond=None)
    pred = np.concatenate([src[mid:], np.ones((len(src) - mid, 1))], 1) @ coef
    return _pack(pred, dst[mid:])


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    report = {"vision": {}, "quat": {}}
    for name in names:
        report["vision"][name] = _vision(name)
        print(json.dumps({"vision": {name: report["vision"][name]}}, ensure_ascii=False), flush=True)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    for tau in (2.0, 8.0):
        built = {}
        for name in names:
            got = _quat_steps(name, tau)
            if got is None:
                continue
            built[name] = got
            report["quat"].setdefault(str(tau), {})[name] = {"half": _half(*got)}
        src = np.concatenate([built[n][0] for n in train if n in built])
        dst = np.concatenate([built[n][1] for n in train if n in built])
        design = np.concatenate([src, np.ones((len(src), 1))], 1)
        coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
        for name in (VAL_NAME, TEST_NAME):
            if name not in built:
                continue
            pred = np.concatenate([built[name][0], np.ones((len(built[name][0]), 1))], 1) @ coef
            block = _pack(pred, built[name][1])
            block["half"] = report["quat"][str(tau)][name]["half"]
            block["singular"] = [round(float(v), 3) for v in np.linalg.svd(coef[:3], compute_uv=False)]
            report["quat"][str(tau)][name] = block
            print(json.dumps({f"quat{tau}": {name: block}}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v6" / "coast_and_quat.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
