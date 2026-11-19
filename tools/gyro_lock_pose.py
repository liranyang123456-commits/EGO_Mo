#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lock the 0.2 s rotation to the USB gyro and refit only the translation.

Train sessions show gyro X tracks camera Z. The other gyro axes are mapped
with the sign seen on those sessions. If the refit still sits on the corners,
the large chessboard tilt was not required by the image.
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
INNER = (11, 8)


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


def _obj(pattern):
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= 3.0
    return obj


def _find_all(gray):
    view = gray
    scale = 1.0
    if gray.shape[1] > 900:
        view = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        scale = 2.0
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found = []
    for pattern in (INNER, (INNER[1], INNER[0])):
        ok, corners = cv2.findChessboardCorners(view, pattern, flags)
        if not ok or corners is None or len(corners) != pattern[0] * pattern[1]:
            continue
        pts = corners.astype(np.float32) * scale
        if scale != 1.0:
            cv2.cornerSubPix(
                gray, pts, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05),
            )
        found.append((pts, pattern))
        found.append((pts[::-1].copy(), pattern))
    return found


def _gyro_w(usb_t, gyro_rad, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 0), len(usb_t) - 1)
    i1 = min(max(i1, 0), len(usb_t) - 1)
    if i1 <= i0:
        return None
    seg_t = usb_t[i0:i1 + 1]
    seg_w = gyro_rad[i0:i1 + 1]
    acc = np.zeros(3)
    for k in range(1, len(seg_t)):
        dt = float(seg_t[k] - seg_t[k - 1])
        if dt <= 0.0 or dt > 0.05:
            continue
        acc += seg_w[k] * dt
    # Train-only axis map: camera Z <- gyro X, camera X <- -gyro Y, camera Y <- -gyro Z.
    return np.array([-acc[1], -acc[2], acc[0]])


def _reproj(obj, corners, rvec, t_mm, K, dist):
    proj, _ = cv2.projectPoints(obj, rvec, t_mm, K, dist)
    return float(np.sqrt(np.mean((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2)))


def _fit_t(obj, corners, R_pnp, t0_mm, K, dist):
    rvec, _ = cv2.Rodrigues(R_pnp)
    t = t0_mm.copy()
    for _ in range(10):
        proj, jac = cv2.projectPoints(obj, rvec, t, K, dist)
        step, *_ = np.linalg.lstsq(jac[:, 3:6], (corners.reshape(-1, 2) - proj.reshape(-1, 2)).reshape(-1), rcond=None)
        t = t + step
        if np.linalg.norm(step) < 1e-3:
            break
    return t, _reproj(obj, corners, rvec, t, K, dist)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else TEST_NAME
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    delay = float(gt["delay_usb"][0])
    t = gt["t"] - delay
    times = np.asarray(
        [float(line) for line in (DATA / name / "cam0" / "times.txt").read_text(encoding="utf-8").splitlines() if line.strip()],
        dtype=np.float64,
    )
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * (np.pi / 180.0)
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    gyro = gyro - bg
    rows = []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))])
        dt = float(t[i] - t[j])
        if dt < 0.12 or dt > 0.30:
            continue
        dp = gt["R"][j].T @ (gt["p"][i] - gt["p"][j])
        rows.append((float(np.linalg.norm(dp)), int(j), int(i)))
    rows.sort(reverse=True)
    shown = 0
    for _norm, j, i in rows:
        if shown >= 6:
            break
        w = _gyro_w(usb_t, gyro, float(t[j]) - 0.04, float(t[i]) - 0.04)
        if w is None:
            continue
        fj = int(np.argmin(np.abs(times - gt["t"][j])))
        fi = int(np.argmin(np.abs(times - gt["t"][i])))
        img = cv2.imread(str(DATA / name / "cam0" / "images" / f"{fi:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        found = _find_all(img)
        if not found:
            print(json.dumps({"frame": fi, "found": False}), flush=True)
            continue
        R_j = gt["R"][j]
        R_i = gt["R"][i]
        r_stored, _ = cv2.Rodrigues(R_i.T)
        t_stored = (-R_i.T @ gt["p"][i]) * 1000.0
        best = None
        for corners, pattern in found:
            obj = _obj(pattern)
            px = _reproj(obj, corners, r_stored, t_stored, K, dist)
            if best is None or px < best[0]:
                best = (px, obj, corners)
        px_stored, obj, corners = best
        dR = exp_so3(w)
        R_lock = R_j @ dR
        t_lock, px_lock = _fit_t(obj, corners, R_lock.T, t_stored, K, dist)
        p_lock = -R_lock @ (t_lock / 1000.0)
        dp_old = R_j.T @ (gt["p"][i] - gt["p"][j])
        dp_new = R_j.T @ (p_lock - gt["p"][j])
        rel = R_j.T @ R_i
        cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
        print(json.dumps({
            "frames": [fj, fi],
            "cam_tilt_deg": round(float(np.degrees(np.arccos(cos))), 2),
            "gyro_deg": [round(float(v) * 180 / np.pi, 2) for v in w],
            "px_stored": round(px_stored, 3),
            "px_gyro_lock": round(px_lock, 3),
            "dp_old_mm": round(float(np.linalg.norm(dp_old) * 1000), 1),
            "dp_new_mm": round(float(np.linalg.norm(dp_new) * 1000), 1),
        }, ensure_ascii=False), flush=True)
        shown += 1


if __name__ == "__main__":
    main()
