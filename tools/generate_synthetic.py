#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headless synthetic dataset generation using the same core as the GUI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim import SimConfig, build_sequence, export_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=12.0)
    ap.add_argument("--motion", choices=("mixed", "lateral", "depth", "circle", "random_spline"), default="mixed")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--imu-hz", type=float, default=200.0)
    ap.add_argument("--noise", type=float, default=1.0)
    ap.add_argument("--board-motion", action="store_true")
    ap.add_argument("--backend", choices=("opencv", "blender"), default="opencv")
    ap.add_argument("--output", type=Path, default=ROOT / "datasets" / "synthetic")
    args = ap.parse_args()
    config = SimConfig(
        duration_s=args.duration,
        motion=args.motion,
        seed=args.seed,
        width=args.width,
        height=args.height,
        camera_fps=args.fps,
        imu_hz=args.imu_hz,
        noise_scale=args.noise,
        board_motion=args.board_motion,
        render_backend=args.backend,
    )
    sequence = build_sequence(config)
    out = export_sequence(
        sequence,
        args.output,
        lambda done, total: print(f"\rframes {done}/{total}", end="", flush=True),
    )
    print(f"\n{out}")


if __name__ == "__main__":
    main()
