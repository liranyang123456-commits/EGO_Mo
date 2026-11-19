#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a split, domain-randomized synthetic corpus for IMU pretraining."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim import SimConfig, build_sequence, export_sequence
from ego_sim.core import load_replay_trajectory

MOTIONS = (
    "lateral", "depth", "circle", "mixed", "random_spline",
    "slow_far", "aggressive_6dof", "pure_rotation",
    "translation_xyz", "pitch_sweep", "stop_go", "spiral",
    "handheld_pause", "handheld_free",
)
# Measured on the real rig from chessboard poses + accelerometer (board frame).
GRAVITY_REAL = (0.0, 0.87, 0.49)


def _replay_config(seed: int, session: str, duration: float, rng: np.random.Generator,
                   gravity, tries: int = 40) -> SimConfig | None:
    """Real trajectory replayed at a random speed/amplitude over a gap-free span."""
    t_lo, t_hi, _pos, _rot, gaps = load_replay_trajectory(session)
    for _ in range(tries):
        speed = float(np.exp(rng.uniform(np.log(0.6), np.log(1.6))))
        span = duration * speed
        if t_hi - t_lo - 1.0 < span:
            continue
        offset = float(rng.uniform(t_lo + 0.5, t_hi - 0.5 - span))
        if any(g0 < offset + span and g1 > offset for g0, g1 in gaps):
            continue
        return SimConfig(
            duration_s=duration, camera_fps=15.0, imu_hz=200.0, seed=seed, motion="replay",
            noise_scale=float(rng.uniform(0.6, 1.7)), render_images=False,
            gravity_board=gravity, replay_session=session, replay_offset_s=offset,
            replay_speed=speed, replay_amp=float(np.exp(rng.uniform(np.log(0.5), np.log(1.6)))),
        )
    return None


def _config(seed: int, motion: str, duration: float, render: bool, backend: str, ood: bool,
            gravity=None) -> SimConfig:
    rng = np.random.default_rng(seed)
    rotation = (
        rng.uniform(15, 28)
        if rng.random() < (0.20 if ood else 0.12)
        else rng.uniform(0.5, 4.0)
    )
    config = SimConfig(
        duration_s=duration,
        camera_fps=10.0 if render else 15.0,
        imu_hz=200.0,
        width=1920 if backend == "blender" else 1280,
        height=1080 if backend == "blender" else 720,
        seed=seed,
        motion=motion,
        translation_mm=float(rng.uniform(30, 90)),
        depth_mm=float(rng.uniform(135, 270)),
        depth_change_mm=float(rng.uniform(15, 80)),
        rotation_deg=float(rotation),
        tracking_gain=float(rng.uniform(0.02, 0.15)),
        speed=float(rng.uniform(1.2, 3.0 if not ood else 3.8)),
        light_level=float(rng.uniform(0.65, 1.35)),
        light_flicker=float(rng.uniform(0.0, 0.18 if not ood else 0.35)),
        noise_scale=float(rng.uniform(0.6, 1.7 if not ood else 2.8)),
        image_noise_std=float(rng.uniform(0.5, 5.0 if not ood else 9.0)),
        motion_blur=float(rng.uniform(0.05, 0.8 if not ood else 1.5)),
        board_motion=ood,
        board_motion_mm=float(rng.uniform(5, 18)),
        baseline_mm=46.5,
        render_backend=backend,
        render_images=render,
        gravity_board=gravity,
    )
    if motion == "slow_far":
        config.duration_s = max(duration, 45.0)
        config.translation_mm = float(rng.uniform(180, 500))
        config.depth_mm = float(rng.uniform(300, 800))
        config.depth_change_mm = float(rng.uniform(150, 500))
        config.rotation_deg = float(rng.uniform(1, 6))
        config.tracking_gain = float(rng.uniform(0.03, 0.12))
        config.speed = float(rng.uniform(0.05, 0.25))
    elif motion == "aggressive_6dof":
        config.translation_mm = float(rng.uniform(70, 180))
        config.depth_change_mm = float(rng.uniform(60, 180))
        config.rotation_deg = float(rng.uniform(35, 100))
        config.tracking_gain = float(rng.uniform(0.1, 0.4))
        config.speed = float(rng.uniform(2.5, 5.0))
    elif motion == "pure_rotation":
        config.translation_mm = float(rng.uniform(0, 8))
        config.depth_change_mm = 0.0
        config.rotation_deg = float(rng.uniform(30, 120))
        config.tracking_gain = 0.0
        config.speed = float(rng.uniform(0.6, 2.0))
    elif motion == "translation_xyz":
        config.translation_mm = float(rng.uniform(40, 150))
        config.depth_change_mm = float(rng.uniform(40, 180))
        config.rotation_deg = float(rng.uniform(0.5, 5))
        config.tracking_gain = float(rng.uniform(0.02, 0.12))
    elif motion == "pitch_sweep":
        config.translation_mm = float(rng.uniform(0, 15))
        config.depth_change_mm = 0.0
        config.rotation_deg = float(rng.uniform(20, 70))
        config.tracking_gain = 0.0
        config.speed = float(rng.uniform(0.4, 1.5))
    elif motion == "stop_go":
        config.translation_mm = float(rng.uniform(40, 130))
        config.speed = float(rng.uniform(0.5, 1.8))
    elif motion in ("handheld_pause", "handheld_free"):
        # Ranges of the real recordings: 10--20 cm travel, 15--40 cm distance,
        # mixed rotation of a few tens of degrees, measured IMU noise level.
        config.translation_mm = float(rng.uniform(12, 40))
        config.depth_mm = float(rng.uniform(150, 350))
        config.depth_change_mm = float(rng.uniform(20, 70))
        config.rotation_deg = float(rng.uniform(1.5, 6))
        config.tracking_gain = float(rng.uniform(0.05, 0.3))
        config.noise_scale = float(rng.uniform(0.8, 1.4))
    elif motion == "spiral":
        config.translation_mm = float(rng.uniform(40, 140))
        config.depth_change_mm = float(rng.uniform(30, 150))
        config.speed = float(rng.uniform(0.4, 1.4))
    return config


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-per-motion", type=int, default=12)
    ap.add_argument("--val-per-motion", type=int, default=3)
    ap.add_argument("--test-per-motion", type=int, default=3)
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--visual-duration", type=float, default=6.0)
    ap.add_argument("--blender-duration", type=float, default=2.0)
    ap.add_argument("--skip-visual", action="store_true")
    ap.add_argument("--skip-blender", action="store_true")
    ap.add_argument("--seed", type=int, default=10000)
    ap.add_argument("--output", type=Path, default=ROOT / "datasets" / "synthetic_corpus")
    ap.add_argument("--motions", default=",".join(MOTIONS))
    ap.add_argument("--gravity", choices=("real", "legacy"), default="real",
                    help="board-frame gravity: measured rig tilt (real) or flat board (legacy)")
    ap.add_argument("--replay-per-session", type=int, default=0,
                    help="replays of each real *training* trajectory at random speed/amplitude")
    ap.add_argument("--replay-val-per-session", type=int, default=0)
    ap.add_argument("--split-file", default="trajectory_split_paper.json")
    ap.add_argument("--camera-fps", type=float, default=15.0,
                    help="reference (label) frame rate of the IMU-only sequences")
    ap.add_argument("--pause-interval", type=float, default=12.5,
                    help="handheld_pause: mean seconds between full stops")
    ap.add_argument("--pause-duration", type=float, default=1.5,
                    help="handheld_pause: mean length of one stop in seconds")
    args = ap.parse_args()

    motions = tuple(m.strip() for m in args.motions.split(",")
                    if m.strip() and m.strip() != "none")
    unknown = set(motions) - set(MOTIONS)
    if unknown:
        raise ValueError(f"unknown motions: {sorted(unknown)}")
    gravity = GRAVITY_REAL if args.gravity == "real" else None

    def imu_config(config: SimConfig) -> SimConfig:
        config.camera_fps = args.camera_fps
        config.pause_interval_s = args.pause_interval
        config.pause_duration_s = args.pause_duration
        return config

    corpus = args.output / f"corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    manifest = {
        "version": 2,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "motions": list(motions),
        "camera_fps": args.camera_fps,
        "pause_interval_s": args.pause_interval,
        "pause_duration_s": args.pause_duration,
        "split_file": args.split_file,
        "gravity_board": list(gravity) if gravity else None,
        "imu": {"train": [], "val": [], "test": [], "ood": []},
        "visual": {"train": [], "val": [], "test": [], "ood": []},
        "blender": [],
    }
    counts = {
        "train": args.train_per_motion,
        "val": args.val_per_motion,
        "test": args.test_per_motion,
    }
    serial = 0
    for split, count in counts.items():
        for motion in motions:
            for _ in range(count):
                seed = args.seed + serial
                serial += 1
                config = imu_config(_config(seed, motion, args.duration, False, "opencv", False,
                                            gravity))
                path = export_sequence(build_sequence(config), corpus / "imu")
                manifest["imu"][split].append({
                    "name": path.name,
                    "path": str(path.relative_to(corpus)),
                    "motion": motion,
                    "config": asdict(config),
                })
                print(f"IMU {split} {motion} {path.name}", flush=True)

    # Real training trajectories replayed through the digital twin. Only the
    # training split of the real data is used so validation/test motion never
    # leaks into pretraining.
    if args.replay_per_session or args.replay_val_per_session:
        split_real = json.loads((ROOT / "datasets" / args.split_file).read_text(encoding="utf-8"))
        rng = np.random.default_rng(args.seed + 777)
        for split, per in (("train", args.replay_per_session), ("val", args.replay_val_per_session)):
            for session in split_real["train"]:
                made = 0
                while made < per:
                    seed = args.seed + serial
                    serial += 1
                    config = _replay_config(seed, session, args.duration, rng, gravity)
                    if config is None:
                        print(f"REPLAY {split} {session}: no gap-free span", flush=True)
                        break
                    config = imu_config(config)
                    path = export_sequence(build_sequence(config), corpus / "imu")
                    manifest["imu"][split].append({
                        "name": path.name, "path": str(path.relative_to(corpus)),
                        "motion": "replay", "source_session": session, "config": asdict(config),
                    })
                    made += 1
                    print(f"REPLAY {split} {session} x{config.replay_speed:.2f} a{config.replay_amp:.2f} {path.name}",
                          flush=True)

    for motion in motions:
        seed = args.seed + serial
        serial += 1
        config = imu_config(_config(seed, motion, args.duration, False, "opencv", True, gravity))
        path = export_sequence(build_sequence(config), corpus / "imu")
        manifest["imu"]["ood"].append({
            "name": path.name, "path": str(path.relative_to(corpus)),
            "motion": motion, "config": asdict(config),
        })

    if not args.skip_visual:
        for split in ("train", "val", "test"):
            for motion in motions:
                seed = args.seed + serial
                serial += 1
                config = _config(seed, motion, args.visual_duration, True, "opencv", False, gravity)
                path = export_sequence(build_sequence(config), corpus / "visual")
                manifest["visual"][split].append({
                    "name": path.name, "path": str(path.relative_to(corpus)),
                    "motion": motion, "config": asdict(config),
                })
                print(f"VIS {split} {motion} {path.name}", flush=True)
        for motion in motions:
            seed = args.seed + serial
            serial += 1
            config = _config(seed, motion, args.visual_duration, True, "opencv", True, gravity)
            path = export_sequence(build_sequence(config), corpus / "visual")
            manifest["visual"]["ood"].append({
                "name": path.name, "path": str(path.relative_to(corpus)),
                "motion": motion, "config": asdict(config),
            })

    if not args.skip_blender:
        for motion in motions:
            seed = args.seed + serial
            serial += 1
            config = _config(seed, motion, args.blender_duration, True, "blender", False, gravity)
            config.camera_fps = 2.0
            path = export_sequence(build_sequence(config), corpus / "blender")
            manifest["blender"].append({
                "name": path.name, "path": str(path.relative_to(corpus)),
                "motion": motion, "config": asdict(config),
            })
            print(f"BLENDER {motion} {path.name}", flush=True)

    manifest_path = corpus / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "latest.txt").write_text(str(corpus), encoding="utf-8")
    print(manifest_path)


if __name__ == "__main__":
    main()
