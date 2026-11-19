#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optical-flow stillness detection on a handheld_pause twin sequence.

Renders one sequence with scene texture, computes dense optical flow between
consecutive frames, and compares the median flow magnitude with the true
camera speed from the pose stream. The question: does flow separate the
1--2 s stops from motion, without any depth estimate?
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim.core import SimConfig, generate_poses, StereoRenderer  # noqa: E402


def main() -> None:
    config = SimConfig(duration_s=40.0, camera_fps=15.0, imu_hz=200.0, seed=7,
                       motion="handheld_pause", render_images=True,
                       width=640, height=360, scene_texture=True,
                       translation_mm=25.0, depth_mm=220.0, rotation_deg=4.0,
                       tracking_gain=0.15)
    t = np.arange(0.0, config.duration_s, 1.0 / config.imu_hz)
    poses = generate_poses(config, t)
    renderer = StereoRenderer(config)

    frame_idx = np.arange(0, len(t), int(config.imu_hz / config.camera_fps))
    frames = []
    for number, k in enumerate(frame_idx):
        frames.append(renderer.render(poses.R_W_C[k], poses.p_W_C[k],
                                      poses.R_W_B[k], poses.p_W_B[k], number,
                                      False, 0.0))
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]

    # True camera speed at each frame time.
    speed = np.linalg.norm(np.gradient(poses.p_W_C, t, axis=0), axis=1)
    frame_speed = speed[frame_idx]

    flow_med = []
    for a, b in zip(gray[:-1], gray[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_med.append(float(np.median(np.linalg.norm(f, axis=2))))
    flow_med = np.array(flow_med)
    mid_speed = 0.5 * (frame_speed[:-1] + frame_speed[1:])

    still = mid_speed < 0.003
    moving = mid_speed > 0.02
    print(f"frames {len(flow_med)}, still {still.sum()}, moving {moving.sum()}")
    print(f"flow median px: still {np.median(flow_med[still]):.4f} "
          f"(p90 {np.percentile(flow_med[still], 90):.4f}), "
          f"moving {np.median(flow_med[moving]):.4f} "
          f"(p10 {np.percentile(flow_med[moving], 10):.4f})")
    # Separation: threshold that maximizes accuracy
    best = max(((np.concatenate([(flow_med < thr) == still,
                                 ])), thr) for thr in np.linspace(0.01, 1.0, 100))
    acc = (flow_med < 0.1) == still
    print(f"at 0.1 px: still recall {(flow_med[still] < 0.1).mean():.3f}, "
          f"moving false-alarm {(flow_med[moving] < 0.1).mean():.3f}")
    np.savez(Path(r"C:\Users\lry\AppData\Local\Temp\flow_still.npz"),
             flow=flow_med, speed=mid_speed)


if __name__ == "__main__":
    main()
