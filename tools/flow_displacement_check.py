#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optical flow as an independent displacement estimator at short horizons.

At 0.5 s the translation is small, so the essential matrix is ill-conditioned,
but the flow magnitude itself carries the scale. This measures how well the
median flow, converted to displacement by the known depth of the ground plane,
recovers the true displacement at 0.5, 1 and 3 s.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim.core import SimConfig, generate_poses, StereoRenderer  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"


def flow_displacement(f0, f1, K, depth):
    """Median flow converted to displacement by the ground-plane depth."""
    flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    med = np.median(flow.reshape(-1, 2), axis=0)
    # x = u * Z / f, y = v * Z / f
    f = K[0, 0]
    return np.array([med[0] * depth / f, med[1] * depth / f, 0.0])


def evaluate(corpus: str, horizon: float, seed: int) -> dict:
    root = Path((PROTO / corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    config = SimConfig(duration_s=40.0, camera_fps=15.0, imu_hz=200.0, seed=seed,
                       motion="handheld_pause", render_images=True,
                       width=640, height=360, scene_texture=True,
                       translation_mm=25.0, depth_mm=220.0, rotation_deg=4.0,
                       tracking_gain=0.15)
    t = np.arange(0.0, config.duration_s, 1.0 / config.imu_hz)
    poses = generate_poses(config, t)
    renderer = StereoRenderer(config)
    K = renderer.K0

    frame_idx = np.arange(0, len(t), int(config.imu_hz / config.camera_fps))
    frames = [cv2.cvtColor(renderer.render(poses.R_W_C[k], poses.p_W_C[k],
                                           poses.R_W_B[k], poses.p_W_B[k], n,
                                           False, 0.0), cv2.COLOR_BGR2GRAY)
              for n, k in enumerate(frame_idx)]
    t_frame = t[frame_idx]

    win = int(horizon * config.camera_fps)
    errs = []
    for a in range(0, len(frames) - win, 15):
        b = a + win
        ka, kb = frame_idx[a], frame_idx[b]
        true_d = poses.p_W_C[kb] - poses.p_W_C[ka]
        # ground plane at z = -1.2 m; camera looks along -z
        depth = 1.2
        est_d = flow_displacement(frames[a], frames[b], K, depth)
        errs.append(np.linalg.norm(est_d - true_d) * 1000)
    return {"horizon_s": horizon, "n": len(errs),
            "err_mm": round(float(np.mean(errs)), 2),
            "median_mm": round(float(np.median(errs)), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    for horizon in (0.5, 1.0, 3.0):
        r = evaluate(args.corpus, horizon, args.seed)
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
