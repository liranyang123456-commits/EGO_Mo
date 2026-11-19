#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Translation direction from optical flow with gyroscope rotation.

With rotation known (from the gyroscope), the epipolar constraint is linear in
the translation direction. On a textured ground plane, dense flow gives one
equation per pixel. This measures how well the flow direction recovers the
true 3-s translation direction, which is what pure IMU cannot observe.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim.core import SimConfig, generate_poses, StereoRenderer  # noqa: E402


def flow_translation_dir(f0, f1, K, R_rel):
    """Translation direction from dense flow, rotation known.

    For normalized coords x0, x1 and y = R_rel x0, the epipolar constraint
    x1' [t]_x y = 0 is linear in t: (y x x1) . t = 0 up to sign.
    """
    flow = cv2.calcOpticalFlowFarneback(f0, f1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    h, w = f0.shape
    yy, xx = np.mgrid[0:h, 0:w]
    x0 = np.stack([(xx - K[0, 2]) / K[0, 0], (yy - K[1, 2]) / K[1, 1],
                   np.ones_like(xx)], axis=-1)
    x1 = x0.copy()
    x1[..., 0] += flow[..., 0] / K[0, 0]
    x1[..., 1] += flow[..., 1] / K[1, 1]
    y = x0 @ R_rel.T
    A = np.cross(y.reshape(-1, 3), x1.reshape(-1, 3))
    _, _, vh = np.linalg.svd(A, full_matrices=False)
    t = vh[-1]
    return t / np.linalg.norm(t)


def main() -> None:
    config = SimConfig(duration_s=40.0, camera_fps=15.0, imu_hz=200.0, seed=7,
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

    # 3-s windows: 45 frames at 15 fps.
    win = 45
    errs = []
    for a in range(0, len(frames) - win, 15):
        b = a + win
        ka, kb = frame_idx[a], frame_idx[b]
        R_rel = poses.R_W_C[ka].T @ poses.R_W_C[kb]
        t_true = poses.R_W_C[ka].T @ (poses.p_W_C[kb] - poses.p_W_C[ka])
        n_true = np.linalg.norm(t_true)
        if n_true < 0.005:
            continue
        t_true = t_true / n_true
        t_est = flow_translation_dir(frames[a], frames[b], K, R_rel)
        ang = np.degrees(np.arccos(np.clip(abs(np.dot(t_est, t_true)), -1, 1)))
        errs.append((n_true * 1000, ang))
    errs = np.array(errs)
    print(f"windows {len(errs)}")
    print(f"direction error deg: median {np.median(errs[:, 1]):.1f}, "
          f"p25 {np.percentile(errs[:, 1], 25):.1f}, "
          f"p75 {np.percentile(errs[:, 1], 75):.1f}")
    print(f"true displacement mm: median {np.median(errs[:, 0]):.1f}")
    # error vs displacement
    for lo, hi in ((0, 15), (15, 40), (40, 200)):
        m = (errs[:, 0] >= lo) & (errs[:, 0] < hi)
        if m.sum():
            print(f"  {lo}-{hi} mm: n {m.sum()}, direction error median "
                  f"{np.median(errs[m, 1]):.1f} deg")


if __name__ == "__main__":
    main()
