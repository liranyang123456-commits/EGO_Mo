#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Align IMU samples to each camera frame on one host clock.

Camera stamps are later than the motion: exposure plus USB delivery.
The constant lateness is the shift that best matches image-to-image change
with the gyro magnitude. IMU values at a frame are then taken at
t_frame - delay, using only samples at or before that time.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

IMU_COLS = [
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "angle_x", "angle_y", "angle_z",
    "mag_x", "mag_y", "mag_z",
    "pressure", "altitude", "temp",
    "quat_w", "quat_x", "quat_y", "quat_z",
]
QUAT = slice(15, 19)
MAX_EXTRAP_S = 0.02
SEARCH_S = 0.15
SEARCH_STEP_S = 0.001
MIN_CORR = 0.25


def load_imu(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    rows: list[list[float]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            try:
                times.append(float(rec["timestamp"]))
                rows.append([float(rec[name]) for name in IMU_COLS])
            except (KeyError, ValueError):
                continue
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(rows, dtype=np.float64)
    if len(t) == 0:
        return t, y.reshape(0, len(IMU_COLS))
    order = np.argsort(t, kind="mergesort")
    return t[order], y[order]


def load_times(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8").strip().splitlines()
    return np.asarray([float(line) for line in text if line.strip()], dtype=np.float64)


def image_motion(video: str | Path, times: np.ndarray, max_side: int = 160) -> tuple[np.ndarray, np.ndarray]:
    """Mean absolute frame difference, stamped at the midpoint of the two frames."""
    import cv2

    cap = cv2.VideoCapture(str(video))
    prev = None
    mids: list[float] = []
    mot: list[float] = []
    index = 0
    try:
        while index < len(times):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            scale = max_side / float(max(h, w))
            if scale < 1.0:
                gray = cv2.resize(gray, (max(1, int(w * scale)), max(1, int(h * scale))))
            cur = gray.astype(np.float32)
            if prev is not None:
                mids.append(0.5 * (float(times[index - 1]) + float(times[index])))
                mot.append(float(np.mean(np.abs(cur - prev))))
            prev = cur
            index += 1
    finally:
        cap.release()
    return np.asarray(mids, dtype=np.float64), np.asarray(mot, dtype=np.float64)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def estimate_delay(
    t_obs: np.ndarray,
    signal: np.ndarray,
    imu_t: np.ndarray,
    gyro_norm: np.ndarray,
    search_s: float = SEARCH_S,
    step_s: float = SEARCH_STEP_S,
) -> dict[str, float]:
    """Positive delay: camera stamp is late, so the gyro event is earlier."""
    signal = np.asarray(signal, dtype=np.float64)
    t_obs = np.asarray(t_obs, dtype=np.float64)
    if len(signal) < 8 or len(imu_t) < 8 or float(signal.std()) < 1e-6:
        return {"delay_s": 0.0, "corr": 0.0, "reliable": 0.0}
    delays = np.arange(-search_s, search_s + step_s * 0.5, step_s)
    best_d = 0.0
    best_c = -2.0
    sig = signal - signal.mean()
    for delay in delays:
        query = t_obs - delay
        inside = (query >= imu_t[0]) & (query <= imu_t[-1])
        if int(inside.sum()) < 8:
            continue
        gyro = np.interp(query[inside], imu_t, gyro_norm)
        corr = _pearson(sig[inside], gyro)
        if corr > best_c:
            best_c = corr
            best_d = float(delay)
    return {"delay_s": best_d, "corr": float(best_c), "reliable": float(best_c >= MIN_CORR)}


def causal_interp(t: np.ndarray, y: np.ndarray, t_query: np.ndarray, max_gap: float = MAX_EXTRAP_S) -> tuple[np.ndarray, np.ndarray]:
    """Linear fill at t_query from the last two samples with time <= t_query.

    Extrapolation past the last sample is at most one sample interval, and
    never more than max_gap seconds.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tq = np.asarray(t_query, dtype=np.float64)
    out = np.zeros((len(tq), y.shape[1]), dtype=np.float64)
    ok = np.zeros(len(tq), dtype=bool)
    if len(t) < 2:
        return out, ok
    idx = np.searchsorted(t, tq, side="right") - 1
    for i, k in enumerate(idx):
        if k < 1 or tq[i] < t[0]:
            continue
        t0, t1 = float(t[k - 1]), float(t[k])
        span = t1 - t0
        if span <= 1e-6 or (tq[i] - t1) > max_gap:
            continue
        alpha = min((tq[i] - t1) / span, 1.0)
        out[i] = y[k] + (y[k] - y[k - 1]) * alpha
        q = out[i, QUAT]
        n = float(np.linalg.norm(q))
        if n > 1e-8:
            out[i, QUAT] = q / n
        ok[i] = True
    return out, ok


def _gyro_norm(y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(y[:, 3:6], axis=1)


def _write_aligned(
    path: Path,
    t_frame: np.ndarray,
    delay_usb: float,
    delay_bt: float,
    usb_t: np.ndarray,
    usb_y: np.ndarray,
    bt_t: np.ndarray,
    bt_y: np.ndarray,
) -> dict[str, int]:
    t_usb = t_frame - delay_usb
    t_bt = t_frame - delay_bt
    usb, usb_ok = causal_interp(usb_t, usb_y, t_usb) if len(usb_t) else (np.zeros((len(t_frame), len(IMU_COLS))), np.zeros(len(t_frame), dtype=bool))
    bt, bt_ok = causal_interp(bt_t, bt_y, t_bt) if len(bt_t) else (np.zeros((len(t_frame), len(IMU_COLS))), np.zeros(len(t_frame), dtype=bool))
    header = ["frame_idx", "frame_timestamp", "t_query_usb", "t_query_bt", "usb_ok", "bt_ok"]
    header += [f"usb_{name}" for name in IMU_COLS]
    header += [f"bt_{name}" for name in IMU_COLS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for i in range(len(t_frame)):
            row = [
                i,
                f"{t_frame[i]:.6f}",
                f"{t_usb[i]:.6f}",
                f"{t_bt[i]:.6f}",
                int(usb_ok[i]),
                int(bt_ok[i]),
            ]
            row += [f"{v:.6f}" for v in usb[i]]
            row += [f"{v:.6f}" for v in bt[i]]
            writer.writerow(row)
    return {"frames": int(len(t_frame)), "usb_ok": int(usb_ok.sum()), "bt_ok": int(bt_ok.sum())}


def align_session(session_dir: str | Path) -> dict[str, Any]:
    """Write sync_delay.json and cam*/imu_aligned.csv. Raw logs stay unchanged."""
    root = Path(session_dir)
    usb_path = root / "imu_stream.csv"
    bt_path = root / "imu_bt.csv"
    usb_t, usb_y = load_imu(usb_path) if usb_path.exists() else (np.zeros(0), np.zeros((0, len(IMU_COLS))))
    bt_t, bt_y = load_imu(bt_path) if bt_path.exists() else (np.zeros(0), np.zeros((0, len(IMU_COLS))))
    usb_g = _gyro_norm(usb_y) if len(usb_t) else np.zeros(0)
    bt_g = _gyro_norm(bt_y) if len(bt_t) else np.zeros(0)
    report: dict[str, Any] = {"cameras": {}}
    for side, folder in (("cam0", "cam0"), ("cam1", "cam1")):
        times_path = root / folder / "times.txt"
        video = root / folder / "video.avi"
        if not times_path.exists():
            continue
        t_frame = load_times(times_path)
        delay_usb = {"delay_s": 0.0, "corr": 0.0, "reliable": 0.0}
        delay_bt = {"delay_s": 0.0, "corr": 0.0, "reliable": 0.0}
        if video.exists() and len(t_frame) >= 2 and len(usb_t):
            t_mid, motion = image_motion(video, t_frame)
            if len(t_mid):
                delay_usb = estimate_delay(t_mid, motion, usb_t, usb_g)
                if len(bt_t):
                    delay_bt = estimate_delay(t_mid, motion, bt_t, bt_g)
        if not delay_usb["reliable"]:
            delay_usb["delay_s"] = 0.0
        if not delay_bt["reliable"]:
            delay_bt["delay_s"] = 0.0
        counts = _write_aligned(
            root / folder / "imu_aligned.csv",
            t_frame,
            float(delay_usb["delay_s"]),
            float(delay_bt["delay_s"]),
            usb_t, usb_y, bt_t, bt_y,
        )
        report["cameras"][side] = {
            "delay_usb_s": delay_usb["delay_s"],
            "delay_usb_corr": delay_usb["corr"],
            "delay_usb_reliable": bool(delay_usb["reliable"]),
            "delay_bt_s": delay_bt["delay_s"],
            "delay_bt_corr": delay_bt["corr"],
            "delay_bt_reliable": bool(delay_bt["reliable"]),
            **counts,
        }
    out = root / "sync_delay.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
