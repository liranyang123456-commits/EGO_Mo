#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU versus optical-flow stillness detection, and stop-anchored integration.

On one handheld_pause twin sequence, compare:
  * IMU stillness: gyro < 3 deg/s and |f|-g < 0.01 g over 0.5 s
  * Flow stillness: median dense flow < 0.1 px between 15-fps frames
against the true speed from the pose stream. Then run stop-anchored strapdown
integration with each detector and compare displacement errors.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim.core import SimConfig, generate_poses, StereoRenderer, simulate_imu  # noqa: E402
from ego_sim.core import _profile  # noqa: E402

G0 = 9.80665


def imu_still(t_imu, usb, win_s=0.5, rate_max=3.0, acc_tol=0.01):
    from scipy.ndimage import uniform_filter1d, maximum_filter1d
    hz = 1.0 / np.median(np.diff(t_imu))
    w = max(3, int(round(win_s * hz)))
    rate = np.linalg.norm(usb[:, 3:6], axis=1)
    mag = np.linalg.norm(usb[:, 0:3], axis=1)
    mean = uniform_filter1d(mag, w, mode="nearest")
    std = np.sqrt(np.maximum(uniform_filter1d(mag ** 2, w, mode="nearest") - mean ** 2, 0.0))
    still = (maximum_filter1d(rate, w, mode="nearest") < rate_max) & (std < acc_tol)
    return still


def flow_still(frames, thr=0.1):
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    med = [0.0]
    for a, b in zip(gray[:-1], gray[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        med.append(float(np.median(np.linalg.norm(f, axis=2))))
    return np.array(med) < thr


def stop_anchored(t_imu, acc_w, still, ta, tb):
    """Integrate between ta and tb with v=0 at still samples, drift removed."""
    m = (t_imu >= ta) & (t_imu <= tb)
    if m.sum() < 50:
        return None
    tt = t_imu[m]
    a = acc_w[m]
    s = still[m]
    if s.sum() < 10:
        return None
    g = a[s].mean(0)
    lin = a - g
    dt = np.gradient(tt)
    v = np.cumsum(lin * dt[:, None], axis=0)
    idx = np.flatnonzero(s)
    v = v - np.column_stack([np.interp(tt, tt[idx], v[idx, c]) for c in range(3)])
    d = np.cumsum(v * dt[:, None], axis=0)
    return d[-1] - d[0]


def main() -> None:
    config = SimConfig(duration_s=40.0, camera_fps=15.0, imu_hz=200.0, seed=7,
                       motion="handheld_pause", render_images=True,
                       width=640, height=360, scene_texture=True,
                       translation_mm=25.0, depth_mm=220.0, rotation_deg=4.0,
                       tracking_gain=0.15)
    t = np.arange(0.0, config.duration_s, 1.0 / config.imu_hz)
    poses = generate_poses(config, t)
    renderer = StereoRenderer(config)

    # IMU stream
    profile = _profile()
    rng = np.random.default_rng(config.seed)
    from ego_sim.core import G
    g_dir = np.array([0.0, 0.87, 0.49])
    g_dir = g_dir / np.linalg.norm(g_dir)
    R_W_I = poses.R_W_C  # IMU at camera for this check
    usb = simulate_imu(t, R_W_I, poses.p_W_C, profile, rng, 1.0,
                       gravity_dir=tuple(g_dir))
    # usb columns: accel (g), gyro (deg/s), mag, pressure, temp, ...
    usb6 = np.column_stack([usb[:, 0:3], usb[:, 3:6]])

    # Frames
    frame_idx = np.arange(0, len(t), int(config.imu_hz / config.camera_fps))
    frames = [renderer.render(poses.R_W_C[k], poses.p_W_C[k], poses.R_W_B[k],
                              poses.p_W_B[k], n, False, 0.0)
              for n, k in enumerate(frame_idx)]
    t_frame = t[frame_idx]

    # True speed
    speed = np.linalg.norm(np.gradient(poses.p_W_C, t, axis=0), axis=1)
    true_still_imu = speed < 0.003
    true_still_frame = speed[frame_idx] < 0.003

    # IMU stillness at 200 Hz, then at frame times
    still_imu = imu_still(t, usb6)
    still_imu_frame = still_imu[frame_idx]
    still_flow = flow_still(frames)

    def score(pred, true, name):
        tp = (pred & true).sum()
        fp = (pred & ~true).sum()
        fn = (~pred & true).sum()
        print(f"{name}: recall {tp / max(true.sum(), 1):.3f}, "
              f"false-alarm {fp / max((~true).sum(), 1):.3f}, "
              f"precision {tp / max(pred.sum(), 1):.3f}")

    score(still_imu_frame, true_still_frame, "IMU stillness")
    score(still_flow, true_still_frame, "Flow stillness")

    # Stop-anchored integration on 3-s windows with a stop inside.
    # World-frame acceleration from true poses (gravity removed by still mean).
    acc_w = np.gradient(np.gradient(poses.p_W_C, t, axis=0), t, axis=0)
    win_s = 3.0
    errs = {"imu": [], "flow": [], "true": []}
    for a in range(0, len(frame_idx) - 45, 15):
        ta = t_frame[a]
        tb = t_frame[a + 45]
        true_d = poses.p_W_C[frame_idx[a + 45]] - poses.p_W_C[frame_idx[a]]
        for name, still in (("imu", still_imu), ("flow", None), ("true", true_still_imu)):
            if name == "flow":
                # flow stillness at frame rate; upsample to IMU rate
                s_frame = still_flow
                still_up = np.interp(t, t_frame, s_frame.astype(float)) > 0.5
                d = stop_anchored(t, acc_w, still_up, ta, tb)
            else:
                d = stop_anchored(t, acc_w, still, ta, tb)
            if d is not None:
                errs[name].append(np.linalg.norm(d - true_d) * 1000)
    for name, e in errs.items():
        if e:
            print(f"stop-anchored ({name}): n {len(e)}, "
                  f"median {np.median(e):.1f} mm, mean {np.mean(e):.1f} mm")


if __name__ == "__main__":
    main()
