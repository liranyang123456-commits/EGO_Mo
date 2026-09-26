#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Replace each PnP pose by the gyro rotation that explains the same pixels.

The stored pose and the gyro-locked pose are fit to one virtual board.
Pairs whose locked pose stays within 1.5 px are kept. A linear map from the
gyro rotation to the new step is fit on the moving training sessions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import exp_so3
from ego_capture.sync import load_imu
from tools.train_trajectory import TEST_NAME, VAL_NAME

DATA = ROOT / "datasets"
CALIB = DATA / "calib_intrinsics_20260922_123120"


def _k(width: int, height: int):
    raw = json.loads((CALIB / "camera_calibration_cam0.json").read_text(encoding="utf-8"))
    K = np.asarray(raw["camera_matrix"], np.float64).copy()
    dist = np.asarray(raw["distortion_coefficients"], np.float64).reshape(-1, 1)
    src_w, src_h = raw["image_size"]
    K[0, 0] *= width / float(src_w)
    K[0, 2] *= width / float(src_w)
    K[1, 1] *= height / float(src_h)
    K[1, 2] *= height / float(src_h)
    return K, dist


def _obj():
    cols, rows = 11, 8
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= 3.0
    return obj


def _gyro_w(usb_t, gyro, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 0), len(usb_t) - 1)
    i1 = min(max(i1, 0), len(usb_t) - 1)
    if i1 <= i0:
        return None
    acc = np.zeros(3)
    for k in range(i0 + 1, i1 + 1):
        dt = float(usb_t[k] - usb_t[k - 1])
        if dt <= 0.0 or dt > 0.05:
            continue
        acc += gyro[k] * dt
    return np.array([-acc[1], -acc[2], acc[0]])


def _fit_t(obj, corners, R_pnp, t0, K, dist):
    rvec, _ = cv2.Rodrigues(R_pnp)
    t = t0.copy()
    for _ in range(8):
        proj, jac = cv2.projectPoints(obj, rvec, t, K, dist)
        step, *_ = np.linalg.lstsq(jac[:, 3:6], (corners - proj).reshape(-1), rcond=None)
        t = t + step
        if np.linalg.norm(step) < 1e-3:
            break
    proj, _ = cv2.projectPoints(obj, rvec, t, K, dist)
    px = float(np.sqrt(np.mean((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2)))
    return t, px


def _score(pred, dp):
    err = np.linalg.norm(pred - dp, axis=1)
    truth = np.linalg.norm(dp, axis=1)
    fast = truth >= np.quantile(truth, 0.67) if len(truth) else np.zeros(0, dtype=bool)
    resid = np.sum((pred - dp) ** 2)
    base = np.sum((dp - dp.mean(0)) ** 2)
    return {
        "n": int(len(err)),
        "zero_mm": round(float(truth.mean() * 1000.0), 2) if len(truth) else None,
        "err_mm": round(float(err.mean() * 1000.0), 2) if len(err) else None,
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2) if np.any(fast) else None,
        "r2": round(float(1.0 - resid / max(base, 1e-12)), 3),
    }


def session_pairs(name: str, obj, K, dist):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    delay = float(gt["delay_usb"][0])
    t = gt["t"] - delay
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * (np.pi / 180.0)
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    gyro = gyro - bg
    w_list, old_list, new_list, px_list = [], [], [], []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))])
        dt = float(t[i] - t[j])
        if dt < 0.12 or dt > 0.30:
            continue
        w = _gyro_w(usb_t, gyro, float(t[j]) - 0.04, float(t[i]) - 0.04)
        if w is None:
            continue
        R_j = gt["R"][j]
        R_i = gt["R"][i]
        t_stored = (-R_i.T @ gt["p"][i]) * 1000.0
        rvec, _ = cv2.Rodrigues(R_i.T)
        corners, _ = cv2.projectPoints(obj, rvec, t_stored, K, dist)
        R_lock = R_j @ exp_so3(w)
        t_new, px = _fit_t(obj, corners, R_lock.T, t_stored, K, dist)
        p_new = -R_lock @ (t_new / 1000.0)
        w_list.append(w)
        old_list.append(R_j.T @ (gt["p"][i] - gt["p"][j]))
        new_list.append(R_j.T @ (p_new - gt["p"][j]))
        px_list.append(px)
    return np.stack(w_list), np.stack(old_list), np.stack(new_list), np.asarray(px_list)


def main() -> None:
    index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    sample = np.load(DATA / "pose_gt" / f"{names[0]}.npz")
    K, dist = _k(int(sample["image_size"][0]), int(sample["image_size"][1]))
    obj = _obj()
    built = {}
    for name in names:
        # Intrinsics scale per session.
        gt = np.load(DATA / "pose_gt" / f"{name}.npz")
        K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
        built[name] = session_pairs(name, obj, K, dist)
        w, old, new, px = built[name]
        keep = px < 1.5
        print(json.dumps({
            name: {
                "n": int(len(px)),
                "fit_px": round(float(np.median(px)), 3),
                "under_1_5": int(keep.sum()),
                "old_mm": round(float(np.linalg.norm(old, axis=1).mean() * 1000), 2),
                "new_mm": round(float(np.linalg.norm(new[keep], axis=1).mean() * 1000), 2) if np.any(keep) else None,
            }
        }, ensure_ascii=False), flush=True)
    train = [n for n in ("traj_20260923_020132", "traj_20260923_020407") if n in built]
    src, dst = [], []
    for name in train:
        w, _old, new, px = built[name]
        m = px < 1.5
        src.append(w[m])
        dst.append(new[m])
    src = np.concatenate(src)
    dst = np.concatenate(dst)
    weight = np.clip(np.linalg.norm(dst, axis=1), 0, np.quantile(np.linalg.norm(dst, axis=1), 0.95))
    design = np.concatenate([src, np.ones((len(src), 1))], 1) * np.sqrt(weight)[:, None]
    coef, *_ = np.linalg.lstsq(design, dst * np.sqrt(weight)[:, None], rcond=None)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        w, old, new, px = built[name]
        m = px < 1.5
        pred = np.concatenate([w[m], np.ones((int(m.sum()), 1))], 1) @ coef
        block = {
            "old_zero": _score(np.zeros_like(old), old),
            "new_zero": _score(np.zeros_like(new[m]), new[m]),
            "gyro_map": _score(pred, new[m]),
        }
        report[name] = block
        print(json.dumps({name: block}, ensure_ascii=False), flush=True)
    out = DATA / "traj_run_v6" / "gyro_disambiguate.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
