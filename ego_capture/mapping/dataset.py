#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build dual-IMU windows labeled by chessboard PnP (board_bt sessions)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..calib import _R_of_row, load_camera_k, load_imu_csv
from ..cameras import chessboard_object_points, find_board
from ..geometry import se3

IMU_DIM = 13


def imu_feature(row: np.ndarray) -> np.ndarray:
    feat = np.concatenate([row[1:7], row[10:17]]).astype(np.float32)
    q = feat[9:13]
    n = float(np.linalg.norm(q))
    feat[9:13] = q / n if n > 0.25 else np.array([1.0, 0.0, 0.0, 0.0], np.float32)
    return feat


def _nearest(times: np.ndarray, t: float, max_dt: float) -> int | None:
    i = int(np.searchsorted(times, t))
    cands = [j for j in (i - 1, i, i + 1) if 0 <= j < len(times)]
    if not cands:
        return None
    j = min(cands, key=lambda k: abs(times[k] - t))
    if abs(times[j] - t) > max_dt:
        return None
    return j


def _window(feats: np.ndarray, idx: int, seq_len: int) -> np.ndarray:
    a = idx - seq_len + 1
    if a >= 0:
        return feats[a : idx + 1]
    pad = np.repeat(feats[:1], -a, axis=0)
    return np.vstack([pad, feats[: idx + 1]])


def _load_frames(session: Path) -> tuple[list[float], list[int]]:
    times, flags = [], []
    path = session / "frames.csv"
    if not path.exists():
        return times, flags
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            times.append(float(rec["frame_timestamp"]))
            flags.append(int(float(rec.get("board_left", 0) or 0)))
    return times, flags


def extract_pnp_labels(session: str, max_frames: int = 280) -> dict[str, Any]:
    sess = Path(session)
    times, flags = _load_frames(sess)
    used = [i for i, flag in enumerate(flags) if flag] if flags else list(range(len(times)))
    if len(used) > max_frames:
        sel = np.linspace(0, len(used) - 1, max_frames).astype(int)
        used = [used[i] for i in sel]
    K, dist = load_camera_k(str(sess))
    if K is None:
        raise RuntimeError(f"没有内参，无法从 {sess.name} 做 PnP 标签。")
    objp = chessboard_object_points()
    img_dir = sess / "cam0" / "images"
    Rs, ts, stamps, idxs = [], [], [], []
    image_size = None
    for i in used:
        jpg = img_dir / f"{i:06d}.jpg"
        if not jpg.exists():
            continue
        bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        image_size = (bgr.shape[1], bgr.shape[0])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ok, corners, _pat = find_board(gray)
        if not ok or corners is None:
            continue
        ok, rvec, tvec = cv2.solvePnP(
            objp, corners.reshape(-1, 1, 2).astype(np.float32),
            K, dist, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        Rs.append(R.astype(np.float64))
        ts.append((tvec.reshape(3) / 1000.0).astype(np.float64))
        stamps.append(times[i] if i < len(times) else 0.0)
        idxs.append(i)
    return {
        "R": np.stack(Rs) if Rs else np.zeros((0, 3, 3)),
        "t": np.stack(ts) if ts else np.zeros((0, 3)),
        "t_stamp": np.asarray(stamps, dtype=np.float64),
        "frame_idx": np.asarray(idxs, dtype=np.int32),
        "K": np.asarray(K, dtype=np.float64),
        "dist": np.asarray(dist, dtype=np.float64).reshape(-1),
        "image_size": list(image_size or []),
    }


def build_windows(
    session: str,
    rig_state_path: str,
    seq_len: int = 80,
    max_frames: int = 280,
) -> dict[str, np.ndarray]:
    sess = Path(session)
    state = json.loads(Path(rig_state_path).read_text(encoding="utf-8"))
    node = state.get("T_C0_Ibt") or {}
    R_geo = np.asarray(node["R"], dtype=np.float64)
    t_geo = np.asarray(node.get("t_m", [0, 0, 0]), dtype=np.float64)
    labels = extract_pnp_labels(session, max_frames=max_frames)
    usb = load_imu_csv(str(sess / "imu_stream.csv"))
    bt = load_imu_csv(str(sess / "imu_bt.csv"))
    if usb.size == 0 or bt.size == 0 or len(labels["R"]) == 0:
        raise RuntimeError(f"{sess.name} 无法组窗口：IMU 或 PnP 为空。")
    usb_f = np.stack([imu_feature(r) for r in usb])
    bt_f = np.stack([imu_feature(r) for r in bt])
    rows = []
    for R, t, stamp in zip(labels["R"], labels["t"], labels["t_stamp"]):
        iu = _nearest(usb[:, 0], float(stamp), 0.025)
        ib = _nearest(bt[:, 0], float(stamp), 0.025)
        if iu is None or ib is None:
            continue
        rows.append((R, t, iu, ib))
    if len(rows) < 4:
        raise RuntimeError("窗口为空：IMU 与棋盘时间对不齐。")
    usb_w, bt_w, R_lab, t_lab, R_pr, t_pr = [], [], [], [], [], []
    for k in range(1, len(rows)):
        R, t, iu, ib = rows[k]
        R0, t0, _iu0, ib0 = rows[k - 1]
        R_I_rel = _R_of_row(bt[ib]) @ _R_of_row(bt[ib0]).T
        R_delta = R_geo @ R_I_rel @ R_geo.T
        usb_w.append(_window(usb_f, iu, seq_len))
        bt_w.append(_window(bt_f, ib, seq_len))
        R_lab.append(R)
        t_lab.append(t)
        R_pr.append(R_delta @ R0)
        t_pr.append(R_delta @ t0)
    return {
        "usb": np.stack(usb_w).astype(np.float32),
        "bt": np.stack(bt_w).astype(np.float32),
        "R_label": np.stack(R_lab).astype(np.float32),
        "t_label": np.stack(t_lab).astype(np.float32),
        "R_prior": np.stack(R_pr).astype(np.float32),
        "t_prior": np.stack(t_pr).astype(np.float32),
        "R_geo": R_geo.astype(np.float32),
        "t_geo": t_geo.astype(np.float32),
        "seq_len": np.array([seq_len], dtype=np.int32),
    }


def time_split(n: int, train: float = 0.70, val: float = 0.15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_train = max(int(n * train), 1)
    n_val = max(int(n * val), 1)
    n_test = max(n - n_train - n_val, 1)
    n_train = n - n_val - n_test
    idx = np.arange(n)
    return idx[:n_train], idx[n_train : n_train + n_val], idx[n_train + n_val :]


class WindowPack(Dataset):
    def __init__(self, pack: dict[str, np.ndarray], indices: np.ndarray, mean=None, std=None):
        self.usb = pack["usb"][indices]
        self.bt = pack["bt"][indices]
        self.R = pack["R_label"][indices]
        self.t = pack["t_label"][indices]
        self.R_prior = pack["R_prior"][indices]
        self.t_prior = pack["t_prior"][indices]
        if mean is None:
            both = np.concatenate([self.usb.reshape(-1, IMU_DIM), self.bt.reshape(-1, IMU_DIM)], 0)
            self.mean = both.mean(0).astype(np.float32)
            self.std = np.clip(both.std(0), 1e-3, None).astype(np.float32)
        else:
            self.mean = mean
            self.std = std

    def __len__(self) -> int:
        return int(self.usb.shape[0])

    def __getitem__(self, i: int):
        usb = (self.usb[i] - self.mean) / self.std
        bt = (self.bt[i] - self.mean) / self.std
        return {
            "usb": torch.from_numpy(usb),
            "bt": torch.from_numpy(bt),
            "R": torch.from_numpy(self.R[i]),
            "t": torch.from_numpy(self.t[i]),
            "R_prior": torch.from_numpy(self.R_prior[i]),
            "t_prior": torch.from_numpy(self.t_prior[i]),
        }
