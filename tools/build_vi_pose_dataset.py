#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build visual-inertial 6-DoF samples from recorded left images and USB IMU.

One sample is the resized left frame plus the one-second IMU window that ends
at that frame. The label is the chessboard PnP pose of the camera. Sessions
keep the split in trajectory_split_20260924.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
SPLIT = DATA / "trajectory_split_20260924.json"
OUT = DATA / "vi_pose"
H, W = 96, 128
IMU_LEN = 200
G = 9.80665


def _session_names(split: dict) -> list[tuple[str, str]]:
    rows = []
    for key in ("train", "val", "test", "extra_train"):
        for name in split.get(key, []):
            rows.append((key, name))
    return rows


def _one(group: str, name: str) -> int:
    raw_path = DATA / "pose_gt_raw" / f"{name}.npz"
    pose_path = DATA / "pose_gt" / f"{name}.npz"
    images = DATA / name / "cam0" / "images"
    imu_path = DATA / name / "imu_stream.csv"
    if not raw_path.is_file() or not images.is_dir() or not imu_path.is_file():
        print(f"skip {name}", flush=True)
        return 0
    raw = np.load(raw_path)
    delay = float(np.load(pose_path)["delay_usb"][0]) if pose_path.is_file() else 0.0
    times = raw["t"].astype(np.float64) - delay
    usable = np.flatnonzero(raw["usable"] == 1)
    imu_t, imu = load_imu(imu_path)
    acc = imu[:, 0:3] * G
    gyro = np.radians(imu[:, 3:6])
    frames, windows, rotations, positions, stamps = [], [], [], [], []
    for index in usable:
        jpg = images / f"{int(index):06d}.jpg"
        if not jpg.is_file():
            continue
        end = int(np.searchsorted(imu_t, times[index], side="right"))
        start = end - IMU_LEN
        if start < 0:
            continue
        gray = cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        small = cv2.resize(gray, (W, H), interpolation=cv2.INTER_AREA)
        window = np.concatenate((acc[start:end], gyro[start:end]), axis=1).astype(np.float32)
        frames.append(small)
        windows.append(window)
        rotations.append(raw["R"][index].astype(np.float32))
        positions.append(raw["p"][index].astype(np.float32))
        stamps.append(times[index])
    if not frames:
        print(f"empty {name}", flush=True)
        return 0
    dest = OUT / group
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / f"{name}.npz",
        image=np.stack(frames),
        imu=np.stack(windows),
        R=np.stack(rotations),
        p=np.stack(positions),
        t=np.asarray(stamps, dtype=np.float64),
    )
    print(f"{group} {name} {len(frames)}", flush=True)
    return len(frames)


def main() -> None:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    counts = {}
    for group, name in _session_names(split):
        counts.setdefault(group, 0)
        counts[group] += _one(group, name)
    (OUT / "index.json").write_text(json.dumps({
        "image_hw": [H, W],
        "imu_len": IMU_LEN,
        "imu_channels": ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"],
        "pose_frame": "chessboard",
        "counts": counts,
    }, indent=2), encoding="utf-8")
    print(counts, flush=True)


if __name__ == "__main__":
    main()
