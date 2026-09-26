#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict the 0.2 s camera step from rotation only.

A fixed lever arm is the skew-symmetric case. The fit here is a general
linear map from the rotation vector. The deployable rotation is the IMU
quaternion; the chessboard rotation is only an upper bound.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import exp_so3
from ego_capture.sync import load_imu
from tools.train_trajectory import TEST_NAME, VAL_NAME, _fit_rotation

GT = ROOT / "datasets" / "pose_gt"
DATA = ROOT / "datasets"


def _so3_log(R: np.ndarray) -> np.ndarray:
    cos = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(cos))
    if th < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return w * (th / (2.0 * np.sin(th)))


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
    idx = np.flatnonzero(ok)
    delay = float(gt["delay_usb"][0])
    t = gt["t"] - delay
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    quat = usb[:, 15:19]
    dR, dp, w_imu = [], [], []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))]
        dt = float(t[i] - t[j])
        if dt < 0.12 or dt > 0.30:
            continue
        rel = gt["R"][j].T @ gt["R"][i]
        dR.append(rel.astype(np.float64))
        dp.append(gt["R"][j].T @ (gt["p"][i] - gt["p"][j]))
        i0 = min(max(int(np.searchsorted(usb_t, t[j])), 0), len(usb_t) - 1)
        i1 = min(max(int(np.searchsorted(usb_t, t[i])), 0), len(usb_t) - 1)
        Rim = _quat_R(quat[i0]).T @ _quat_R(quat[i1])
        w_imu.append(_so3_log(Rim))
    return np.stack(dR), np.stack(dp), np.stack(w_imu)


def _score(pred, dp) -> dict:
    err = np.linalg.norm(pred - dp, axis=1)
    truth = np.linalg.norm(dp, axis=1)
    fast = truth >= np.quantile(truth, 0.67)
    resid = np.sum((pred - dp) ** 2)
    base = np.sum((dp - dp.mean(0)) ** 2)
    return {
        "n": int(len(err)),
        "zero_mm": round(float(truth.mean() * 1000.0), 2),
        "err_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "pred_mm": round(float(np.median(np.linalg.norm(pred, axis=1)) * 1000.0), 2),
        "r2": round(float(1.0 - resid / max(base, 1e-12)), 3),
    }


def _map_fit(src, dst):
    design = np.concatenate([src, np.ones((len(src), 1))], 1)
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return coef


def _map_apply(src, coef):
    return np.concatenate([src, np.ones((len(src), 1))], 1) @ coef


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = {}
    for name in names:
        packs[name] = _pairs(name)
        print(json.dumps({"built": name, "n": int(len(packs[name][1]))}), flush=True)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    dR_tr = np.concatenate([packs[n][0] for n in train])
    dp_tr = np.concatenate([packs[n][1] for n in train])
    w_gt = np.stack([_so3_log(rel) for rel in dR_tr])
    w_imu = np.concatenate([packs[n][2] for n in train])
    R_ci = _fit_rotation(np.stack([exp_so3(w) for w in w_imu]), dR_tr)
    w_imu_cam = np.stack([_so3_log(R_ci @ exp_so3(w) @ R_ci.T) for w in w_imu])
    coef_gt = _map_fit(w_gt, dp_tr)
    coef_imu = _map_fit(w_imu_cam, dp_tr)
    report = {"R_ci": R_ci.round(4).tolist()}
    for name in (VAL_NAME, TEST_NAME):
        dR, dp, w = packs[name]
        wgt = np.stack([_so3_log(rel) for rel in dR])
        wcam = np.stack([_so3_log(R_ci @ exp_so3(wi) @ R_ci.T) for wi in w])
        block = {
            "chessboard_rot": _score(_map_apply(wgt, coef_gt), dp),
            "imu_quat": _score(_map_apply(wcam, coef_imu), dp),
        }
        report[name] = block
        print(json.dumps({name: block}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v6" / "rot_to_step.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
