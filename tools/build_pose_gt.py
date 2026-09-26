#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chessboard pose ground truth for each usable trajectory session.

Left and right cameras are solved independently. A frame is stereo-consistent
when the right pose, moved by the 46.5 mm baseline, matches the left pose.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ego_capture.rig import STEREO_BASELINE_M
from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
CALIB = DATA / "calib_intrinsics_20260922_123120"
OUT = DATA / "pose_gt"
INNER = (11, 8)
FAST = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
SLOW = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
MAX_REPROJ = 1.5


def _load_k(cam: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads((CALIB / f"camera_calibration_{cam}.json").read_text(encoding="utf-8"))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    src_w, src_h = data["image_size"]
    if (src_w, src_h) != (width, height):
        K = K.copy()
        K[0, 0] *= width / float(src_w)
        K[0, 2] *= width / float(src_w)
        K[1, 1] *= height / float(src_h)
        K[1, 2] *= height / float(src_h)
    return K, dist


def _times(path: Path) -> np.ndarray:
    if not path.is_file():
        return np.zeros(0)
    lines = path.read_text(encoding="utf-8").splitlines()
    return np.asarray([float(line) for line in lines if line.strip()], dtype=np.float64)


def _find(gray: np.ndarray):
    attempts = [(gray, 1.0)]
    if gray.shape[1] > 900:
        half = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # Fast half-resolution pass first, then full-resolution fallback. The
        # latter matters for small boards in 1080p/2K synthetic images.
        attempts = [(half, 2.0), (gray, 1.0)]
    for view, scale in attempts:
        for flags in (FAST, SLOW):
            for pattern in (INNER, (INNER[1], INNER[0])):
                ok, corners = cv2.findChessboardCorners(view, pattern, flags)
                if ok and corners is not None and len(corners) == pattern[0] * pattern[1]:
                    pts = corners.astype(np.float32) * scale
                    try:
                        cv2.cornerSubPix(
                            gray, pts, (5, 5), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05),
                        )
                    except cv2.error:
                        pass
                    return pts, pattern
    return None, None


def _pose(corners, pattern, K, dist):
    cols, rows = pattern
    obj = np.zeros((cols * rows, 3), np.float32)
    obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    obj *= 3.0
    ok, rvec, tvec = cv2.solvePnP(
        obj, corners.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean((proj.reshape(-1, 2) - corners.reshape(-1, 2)) ** 2)))
    if err > MAX_REPROJ:
        return None
    R, _ = cv2.Rodrigues(rvec)
    t = tvec.reshape(3) / 1000.0
    return R.astype(np.float64), (-R.T @ t).astype(np.float64), err


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    rel = Ra.T @ Rb
    cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _nearest_gyro(t_imu: np.ndarray, gyro: np.ndarray, t: float) -> float:
    if len(t_imu) == 0:
        return 999.0
    k = int(np.searchsorted(t_imu, t))
    k = min(max(k, 0), len(t_imu) - 1)
    if k > 0 and abs(t_imu[k - 1] - t) < abs(t_imu[k] - t):
        k -= 1
    if abs(t_imu[k] - t) > 0.05:
        return 999.0
    return float(np.linalg.norm(gyro[k]))


def process_session(name: str) -> dict:
    sess = DATA / name
    summary = json.loads((sess / "capture_summary.json").read_text(encoding="utf-8"))
    delay = float(((summary.get("sync") or {}).get("cameras") or {}).get("cam0", {}).get("delay_usb_s") or 0.0)
    t0 = _times(sess / "cam0" / "times.txt")
    t1 = _times(sess / "cam1" / "times.txt")
    imgs0 = sorted((sess / "cam0" / "images").glob("*.jpg"))
    imgs1 = {p.name: p for p in (sess / "cam1" / "images").glob("*.jpg")}
    if not imgs0:
        return {"name": name, "error": "no images"}
    sample = cv2.imread(str(imgs0[0]), cv2.IMREAD_GRAYSCALE)
    height, width = sample.shape[:2]
    K0, d0 = _load_k("cam0", width, height)
    K1, d1 = _load_k("cam1", width, height)
    bt_t, bt_y = load_imu(sess / "imu_bt.csv")
    bt_gyro = bt_y[:, 3:6] if len(bt_t) else np.zeros((0, 3))
    n = min(len(imgs0), len(t0)) if len(t0) else len(imgs0)
    R = np.tile(np.eye(3), (n, 1, 1))
    p = np.zeros((n, 3))
    ok = np.zeros(n, dtype=np.uint8)
    reproj = np.full(n, 999.0)
    stereo_ok = np.zeros(n, dtype=np.uint8)
    stereo_deg = np.full(n, 999.0)
    stereo_mm = np.full(n, 999.0)
    ble_gyro = np.full(n, 999.0)
    stamps = t0[:n] if len(t0) else np.zeros(n)
    T_c0_c1 = np.eye(4)
    T_c0_c1[:3, 3] = (STEREO_BASELINE_M, 0.0, 0.0)
    for i in range(n):
        bgr = cv2.imread(str(imgs0[i]), cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            continue
        ble_gyro[i] = _nearest_gyro(bt_t, bt_gyro, float(stamps[i]))
        corners, pattern = _find(bgr)
        if corners is None:
            continue
        left = _pose(corners, pattern, K0, d0)
        if left is None:
            continue
        Rl, pl, err = left
        R[i] = Rl.T
        p[i] = pl
        ok[i] = 1
        reproj[i] = err
        right_path = imgs1.get(f"{i:06d}.jpg")
        if right_path is None or i >= len(t1):
            continue
        gray_r = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)
        if gray_r is None:
            continue
        cr, pr = _find(gray_r)
        if cr is None:
            continue
        right = _pose(cr, pr, K1, d1)
        if right is None:
            continue
        Rr, pr_t, _err_r = right
        T_c1_o = np.eye(4)
        T_c1_o[:3, :3] = Rr
        T_c1_o[:3, 3] = -Rr @ pr_t
        T_c0_o = T_c0_c1 @ T_c1_o
        R_from_r = T_c0_o[:3, :3].T
        p_from_r = -R_from_r @ T_c0_o[:3, 3]
        # Left stored R is R_O_C = Rl.T, p is camera origin in board.
        stereo_deg[i] = _rot_angle(R[i], R_from_r)
        stereo_mm[i] = float(np.linalg.norm(p[i] - p_from_r) * 1000.0)
        if stereo_deg[i] < 5.0 and stereo_mm[i] < 20.0:
            stereo_ok[i] = 1
        if (i + 1) % 200 == 0:
            print(f"{name} {i + 1}/{n} ok {int(ok.sum())} stereo {int(stereo_ok.sum())}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT / f"{name}.npz",
        t=stamps, ok=ok, reproj=reproj, R=R, p=p,
        stereo_ok=stereo_ok, stereo_deg=stereo_deg, stereo_mm=stereo_mm,
        ble_gyro=ble_gyro, delay_usb=np.array([delay]),
        image_size=np.array([width, height]),
    )
    still = (ok == 1) & (ble_gyro < 8.0)
    return {
        "name": name,
        "frames": int(n),
        "pnp": int(ok.sum()),
        "pnp_still": int(still.sum()),
        "stereo": int(stereo_ok.sum()),
        "reproj_p50": float(np.median(reproj[ok == 1])) if ok.any() else None,
        "size": [int(width), int(height)],
        "delay_usb_ms": round(delay * 1000.0, 1),
    }


def main() -> None:
    audit = json.loads((DATA / "audit_sessions.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in audit if row.get("usable")]
    rows = []
    for name in names:
        print("start", name, flush=True)
        rows.append(process_session(name))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (OUT / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print("done", len(rows), flush=True)


if __name__ == "__main__":
    main()
