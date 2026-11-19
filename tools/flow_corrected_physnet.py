#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Flow-corrected PhysNet: scale the prediction by the flow magnitude.

PhysNet shrinks toward zero. The median flow is a measure of how much the
camera actually moved. This scales the PhysNet prediction by a factor learned
from the flow, and by zero on windows the flow detector calls still.
"""

from __future__ import annotations

import argparse
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
from ego_sim.core import StereoRenderer, SimConfig  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"


def flow_series(seq_dir: Path, config: SimConfig) -> tuple[np.ndarray, np.ndarray]:
    cache = seq_dir / "flow_median.npz"
    if cache.is_file():
        d = np.load(cache)
        return d["t"], d["flow"]
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
    flow = np.array(med)
    np.savez(cache, t=t[frame_idx], flow=flow)
    return t[frame_idx], flow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.BIAS_MODE = "none"
    models = [v3.load_model(DATA / "physnet_proto" / f"physnet_{args.corpus}_nb_s{s}.pt", device)
              for s in args.seeds]

    root = Path((PROTO / args.corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))

    P, Y, F = [], [], []
    for e in manifest["imu"]["test"][:6]:
        seq = (root / e["path"]).resolve()
        data = v3.build_group([seq], context, horizon, length, cap=160,
                              rng=np.random.default_rng(123))
        dt = v3.to_device(data, device)
        pred = np.mean([m.predict(dt) for m in models], axis=0)
        P.append(pred)
        Y.append(data["y"])
        cfg = SimConfig(**{k: v for k, v in e["config"].items()
                           if k in SimConfig.__dataclass_fields__})
        cfg.render_images = True
        cfg.scene_texture = True
        cfg.width, cfg.height = 640, 360
        t_frame, flow = flow_series(seq, cfg)
        # mean flow over the window
        flow_win = []
        for ta, tb in data["t_pair"]:
            m = (t_frame >= ta) & (t_frame <= tb)
            flow_win.append(float(flow[m].mean()) if m.any() else 0.0)
        F.append(np.array(flow_win))
    P = np.concatenate(P)
    Y = np.concatenate(Y)
    F = np.concatenate(F)

    def err(p, m):
        return round(float(np.linalg.norm(p[m] - Y[m], axis=1).mean() * 1000), 2)

    allm = np.ones(len(Y), bool)
    print(f"windows {len(Y)}")
    print(f"PhysNet: {err(P, allm)} mm")
    # scale by flow: fit alpha on the same data (in-sample)
    norm_P = np.linalg.norm(P, axis=1)
    norm_Y = np.linalg.norm(Y, axis=1)
    # alpha = sum(F * norm_Y) / sum(F * norm_P)
    alpha = (F * norm_Y).sum() / max((F * norm_P).sum(), 1e-9)
    P_scaled = P * (F[:, None] * alpha)
    print(f"PhysNet scaled by flow (alpha {alpha:.3f}): {err(P_scaled, allm)} mm")
    # zero on still
    still = F < 0.1
    P_zero = P.copy()
    P_zero[still] = 0.0
    print(f"PhysNet zero on flow-still ({still.sum()} windows): {err(P_zero, allm)} mm")
    # scale + zero
    P_both = P_scaled.copy()
    P_both[still] = 0.0
    print(f"PhysNet scaled + zero: {err(P_both, allm)} mm")


if __name__ == "__main__":
    main()
