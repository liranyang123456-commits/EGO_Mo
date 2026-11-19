#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persist independent-board IMU sequences. No images.

Seeds below 1000 are train, 1000-1099 val, 1100-1199 test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim import SimConfig, build_sequence


def split_of(seed: int) -> str:
    if seed >= 1100:
        return "test"
    if seed >= 1000:
        return "val"
    return "train"


def one(seed: int, duration: float, board_mm: float, board_deg: float, out: Path) -> None:
    motion = ("random_spline", "mixed", "aggressive_6dof", "stop_go", "spiral")[seed % 5]
    sequence = build_sequence(SimConfig(
        duration_s=duration,
        camera_fps=10.0,
        imu_hz=200.0,
        seed=seed,
        motion=motion,
        translation_mm=50.0,
        depth_mm=180.0,
        depth_change_mm=40.0,
        rotation_deg=15.0,
        tracking_gain=0.05,
        speed=1.5,
        noise_scale=1.0,
        board_motion=True,
        board_motion_mode="independent",
        board_motion_mm=board_mm,
        board_rotation_deg=board_deg,
        render_images=False,
        gravity_board=(0.0, 0.87, 0.49),
    ))
    poses = sequence.poses
    dest = out / split_of(seed)
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dest / f"seq_{seed:04d}.npz",
        t=poses.t.astype(np.float32),
        usb=sequence.imu_usb[:, :19].astype(np.float32),
        ble=sequence.imu_bt[:, :19].astype(np.float32),
        R_W_C=poses.R_W_C.astype(np.float32),
        p_W_C=poses.p_W_C.astype(np.float32),
        R_W_B=poses.R_W_B.astype(np.float32),
        p_W_B=poses.p_W_B.astype(np.float32),
        seed=np.int32(seed),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--board-mm", type=float, default=40.0)
    parser.add_argument("--board-deg", type=float, default=20.0)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "decouple_corpus")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for seed in range(args.seed_start, args.seed_start + args.count):
        one(seed, args.duration, args.board_mm, args.board_deg, args.out)
        if (seed - args.seed_start + 1) % 10 == 0:
            print(f"exported {seed - args.seed_start + 1}/{args.count}", flush=True)
    (args.out / "manifest.json").write_text(json.dumps({
        "seed_start": args.seed_start,
        "count": args.count,
        "duration_s": args.duration,
        "board_mm": args.board_mm,
        "board_deg": args.board_deg,
        "board_motion_mode": "independent",
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
