#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PhysNet with flow-based stillness gating on a twin sequence.

A synthetic-only PhysNet predicts 3-s displacements from the IMU stream. A
dense optical-flow stillness detector then zeroes the prediction on windows
whose start is still. Compares PhysNet alone, PhysNet with the learned IMU
gate, and PhysNet with the flow gate, against the true displacement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet_v3 as v3  # noqa: E402
import tools.train_seq as api  # noqa: E402
from ego_sim.core import StereoRenderer  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"


def flow_still_flags(seq_dir: Path, data: dict, config) -> np.ndarray:
    """Per-window flag: the flow at the window start is below 0.1 px."""
    gt = np.load(seq_dir / "ground_truth.npz")
    t = gt["imu_t"].astype(np.float64)
    R_W_C = gt["imu_R_W_C"].astype(np.float64)
    p_W_C = gt["imu_p_W_C"].astype(np.float64)
    step = int(config.imu_hz / config.camera_fps)
    frame_idx = np.arange(0, len(t), step)
    renderer = StereoRenderer(config)
    frames = [cv2.cvtColor(renderer.render(R_W_C[k], p_W_C[k], np.eye(3),
                                           np.zeros(3), n, False, 0.0),
                           cv2.COLOR_BGR2GRAY)
              for n, k in enumerate(frame_idx)]
    med = [0.0]
    for a, b in zip(frames[:-1], frames[1:]):
        f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        med.append(float(np.median(np.linalg.norm(f, axis=2))))
    med = np.array(med)
    t_frame = t[frame_idx]
    # window start time for each pair
    pair_t = data["t_pair"][:, 0]
    flow_at_start = np.interp(pair_t, t_frame, med)
    return flow_at_start < 0.1


def stop_anchored(t_imu, acc_w, still, ta, tb):
    """Integrate from the last still sample before tb to tb, with v=0 there."""
    m = (t_imu >= ta) & (t_imu <= tb)
    if m.sum() < 50:
        return np.array([np.nan, np.nan, np.nan])
    tt = t_imu[m]
    a = acc_w[m]
    s = still[m]
    if s.sum() < 10:
        return np.array([np.nan, np.nan, np.nan])
    # last still sample before tb
    idx = np.flatnonzero(s)
    last = idx[-1]
    g = a[s].mean(0)
    lin = a - g
    dt = np.gradient(tt)
    v = np.cumsum(lin * dt[:, None], axis=0)
    v = v - v[last]
    d = np.cumsum(v * dt[:, None], axis=0)
    return d[-1] - d[last]


def main() -> None:
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.BIAS_MODE = "none"
    models = [v3.load_model(DATA / "physnet_proto" / f"physnet_pause_4_nb_s{s}.pt", device)
              for s in (0, 1)]

    root = Path((PROTO / "pause_4" / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))

    P, Y, S_flow, S_true, D_flow, D_true = [], [], [], [], [], []
    for e in manifest["imu"]["test"][:6]:
        seq = (root / e["path"]).resolve()
        data = v3.build_group([seq], context, horizon, length, cap=160,
                              rng=np.random.default_rng(123))
        dt = v3.to_device(data, device)
        pred = np.mean([m.predict(dt) for m in models], axis=0)
        P.append(pred)
        Y.append(data["y"])
        cfg = __import__("ego_sim.core", fromlist=["SimConfig"]).SimConfig(
            **{k: v for k, v in e["config"].items()
               if k in __import__("ego_sim.core", fromlist=["SimConfig"]).SimConfig.__dataclass_fields__})
        cfg.render_images = True
        cfg.scene_texture = True
        cfg.width, cfg.height = 640, 360
        S_flow.append(flow_still_flags(seq, data, cfg))
        # true stillness at window start from the pose stream
        gt = np.load(seq / "ground_truth.npz")
        t = gt["imu_t"].astype(np.float64)
        speed = np.linalg.norm(np.gradient(gt["imu_p_W_C"].astype(np.float64), t, axis=0), axis=1)
        pair_t = data["t_pair"][:, 0]
        S_true.append(np.interp(pair_t, t, speed) < 0.003)
        # stop-anchored displacement for windows with a stop inside
        acc_w = np.gradient(np.gradient(gt["imu_p_W_C"].astype(np.float64), t, axis=0), t, axis=0)
        true_still = speed < 0.003
        d_flow = []
        d_true = []
        for k in range(len(data["pair"])):
            ta, tb = data["t_pair"][k]
            m = (t >= ta) & (t <= tb)
            if m.sum() < 50:
                d_true.append(np.array([np.nan, np.nan, np.nan]))
                d_flow.append(np.array([np.nan, np.nan, np.nan]))
                continue
            tt = t[m]
            a = acc_w[m]
            s = true_still[m]
            if s.sum() < 10:
                d_true.append(np.array([np.nan, np.nan, np.nan]))
                d_flow.append(np.array([np.nan, np.nan, np.nan]))
                continue
            idx = np.flatnonzero(s)
            last = idx[-1]
            g = a[s].mean(0)
            lin = a - g
            dt = np.gradient(tt)
            v = np.cumsum(lin * dt[:, None], axis=0)
            v = v - v[last]
            d = np.cumsum(v * dt[:, None], axis=0)
            d_true.append(d[-1] - d[last])
            d_flow.append(d[-1] - d[last])
        D_true.append(np.array(d_true))
        D_flow.append(np.array(d_flow))
    P = np.concatenate(P)
    Y = np.concatenate(Y)
    S_flow = np.concatenate(S_flow)
    S_true = np.concatenate(S_true)
    D_true = np.concatenate(D_true)
    D_flow = np.concatenate(D_flow)

    def err(p, m):
        return round(float(np.linalg.norm(p[m] - Y[m], axis=1).mean() * 1000), 2)

    allm = np.ones(len(Y), bool)
    print(f"windows {len(Y)}, flow-still {S_flow.sum()}, true-still {S_true.sum()}")
    print(f"PhysNet all: {err(P, allm)} mm")
    print(f"PhysNet on true-still windows: {err(P, S_true)} mm")
    print(f"PhysNet on moving windows: {err(P, ~S_true)} mm")
    P_flow = P.copy()
    P_flow[S_flow] = 0.0
    print(f"PhysNet + flow gate (zero on flow-still): {err(P_flow, allm)} mm")
    print(f"  on true-still: {err(P_flow, S_true)} mm")
    print(f"  on moving: {err(P_flow, ~S_true)} mm")
    # oracle gate
    P_true = P.copy()
    P_true[S_true] = 0.0
    print(f"PhysNet + true gate: {err(P_true, allm)} mm")
    # stop-anchored replacement on windows with a stop inside
    has_stop = ~np.isnan(D_true[:, 0])
    print(f"windows with true stop inside: {has_stop.sum()}")
    if has_stop.sum():
        P_anchor = P.copy()
        P_anchor[has_stop] = D_true[has_stop]
        print(f"PhysNet + stop-anchored (true stops): {err(P_anchor, allm)} mm")
        print(f"  on windows with stop: {err(P_anchor, has_stop)} mm")
        print(f"  on windows without stop: {err(P_anchor, ~has_stop)} mm")


if __name__ == "__main__":
    main()
