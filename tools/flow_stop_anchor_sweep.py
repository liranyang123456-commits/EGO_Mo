#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop-anchored integration with flow versus IMU stillness, across pause rates.

For each protocol corpus (pause_4 ... pause_inf), render the test sequences,
detect still phases with dense optical flow and with the IMU, and integrate
3-s windows that contain a stop. Reports displacement error and coverage.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim.core import StereoRenderer, simulate_imu, _profile, G  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"
G0 = 9.80665


def imu_still(t_imu, usb, win_s=0.5, rate_max=3.0, acc_tol=0.01):
    from scipy.ndimage import uniform_filter1d, maximum_filter1d
    hz = 1.0 / np.median(np.diff(t_imu))
    w = max(3, int(round(win_s * hz)))
    rate = np.linalg.norm(usb[:, 3:6], axis=1)
    mag = np.linalg.norm(usb[:, 0:3], axis=1)
    mean = uniform_filter1d(mag, w, mode="nearest")
    std = np.sqrt(np.maximum(uniform_filter1d(mag ** 2, w, mode="nearest") - mean ** 2, 0.0))
    return (maximum_filter1d(rate, w, mode="nearest") < rate_max) & (std < acc_tol)


def flow_still(frames, thr=0.1):
    gray = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    med = [0.0]
    for a, b in zip(gray[:-1], gray[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        med.append(float(np.median(np.linalg.norm(f, axis=2))))
    return np.array(med) < thr


def stop_anchored(t_imu, acc_w, still, ta, tb):
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


def evaluate_sequence(seq_dir: Path, renderer_cache: dict):
    gt = np.load(seq_dir / "ground_truth.npz")
    t = gt["imu_t"].astype(np.float64)
    p_W_C = gt["imu_p_W_C"].astype(np.float64)
    R_W_C = gt["imu_R_W_C"].astype(np.float64)
    config = json.loads((seq_dir / "session_meta.json").read_text(encoding="utf-8"))["config"]
    from ego_sim.core import SimConfig
    cfg = SimConfig(**{k: v for k, v in config.items() if k in SimConfig.__dataclass_fields__})
    cfg.render_images = True
    cfg.scene_texture = True
    cfg.width, cfg.height = 640, 360

    # IMU stream
    profile = _profile()
    rng = np.random.default_rng(cfg.seed)
    g_dir = np.array(cfg.gravity_board) if cfg.gravity_board else np.array([0.0, 0.0, -1.0])
    g_dir = g_dir / np.linalg.norm(g_dir)
    usb = simulate_imu(t, R_W_C, p_W_C, profile, rng, cfg.noise_scale,
                       gravity_dir=tuple(g_dir))
    usb6 = np.column_stack([usb[:, 0:3], usb[:, 3:6]])

    # Frames at 15 fps
    step = int(cfg.imu_hz / cfg.camera_fps)
    frame_idx = np.arange(0, len(t), step)
    key = (cfg.seed, cfg.width, cfg.height)
    if key not in renderer_cache:
        renderer_cache[key] = StereoRenderer(cfg)
    renderer = renderer_cache[key]
    frames = [renderer.render(R_W_C[k], p_W_C[k], np.eye(3), np.zeros(3), n,
                              False, 0.0) for n, k in enumerate(frame_idx)]
    t_frame = t[frame_idx]

    # True speed and stillness
    speed = np.linalg.norm(np.gradient(p_W_C, t, axis=0), axis=1)
    true_still = speed < 0.003

    # Detectors
    still_imu = imu_still(t, usb6)
    still_flow_frame = flow_still(frames)
    still_flow = np.interp(t, t_frame, still_flow_frame.astype(float)) > 0.5

    # World-frame acceleration from true poses
    acc_w = np.gradient(np.gradient(p_W_C, t, axis=0), t, axis=0)

    # 3-s windows at 15 fps
    win = 45
    out = {"imu": [], "flow": [], "true": []}
    for a in range(0, len(frame_idx) - win, 15):
        ta = t_frame[a]
        tb = t_frame[a + win]
        true_d = p_W_C[frame_idx[a + win]] - p_W_C[frame_idx[a]]
        for name, still in (("imu", still_imu), ("flow", still_flow), ("true", true_still)):
            d = stop_anchored(t, acc_w, still, ta, tb)
            if d is not None:
                out[name].append(np.linalg.norm(d - true_d) * 1000)
    return out


def main() -> None:
    corpora = ["pause_4", "pause_8", "pause_12.5", "pause_20", "pause_inf"]
    renderer_cache = {}
    report = {}
    for name in corpora:
        root = Path((PROTO / name / "latest.txt").read_text(encoding="utf-8").strip())
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        test_seqs = [root / e["path"] for e in manifest["imu"]["test"]]
        all_err = {"imu": [], "flow": [], "true": []}
        for seq in test_seqs[:8]:  # cap for time
            r = evaluate_sequence(seq, renderer_cache)
            for k in all_err:
                all_err[k].extend(r[k])
        report[name] = {}
        for k, e in all_err.items():
            if e:
                report[name][k] = {"n": len(e), "median_mm": round(float(np.median(e)), 2),
                                   "mean_mm": round(float(np.mean(e)), 2)}
        print(name, json.dumps(report[name]), flush=True)
    (DATA / "flow_stop_anchor.json").write_text(json.dumps(report, indent=2),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()
