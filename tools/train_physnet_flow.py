#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PhysNet with an optical-flow stillness channel.

Adds one input channel to the IMU windows: the median dense optical flow at
each IMU sample, computed from the rendered frames and interpolated to 200 Hz.
The network can then learn when the flow says the camera is still, instead of
relying only on the gyroscope and accelerometer.
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
    """Median dense flow at each frame, and the frame times. Cached to .npz."""
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


def build_group_with_flow(names, context, horizon, length, cap=0, rng=None):
    """Like v3.build_group, with one extra channel: median flow at each sample."""
    groups = []
    for name in names:
        got = v3.build_session(name, context, horizon, length, cap=cap, rng=rng)
        if got is None:
            continue
        cache = Path(name) / "flow_median.npz"
        if not cache.is_file():
            continue
        d = np.load(cache)
        t_frame, flow = d["t"], d["flow"]
        n = len(got["y"])
        flow_ch = np.zeros((n, length), np.float32)
        for k in range(n):
            ta, tb = got["t_pair"][k]
            grid = ta - context + np.arange(length) / v3.HZ
            flow_ch[k] = np.interp(grid, t_frame, flow, left=flow[0], right=flow[-1])
        got["flow"] = flow_ch
        groups.append(got)
    if not groups:
        return None
    merged = v3._pad_dense(groups)
    merged["flow"] = np.concatenate([g["flow"] for g in groups])
    return merged


class FlowPhysNet(v3.PhysNet):
    """PhysNet with one extra input channel for optical flow."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The stem takes in_ch + 1 channels; we rebuild it.
        in_ch = self.base_ch + 1
        width = self.stem[0].out_channels
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, width, 7, stride=2, padding=3), torch.nn.GELU(),
            torch.nn.Conv1d(width, width, 5, stride=2, padding=2), torch.nn.GELU(),
        )


class FlowModel(v3.Model):
    def __init__(self, net, feat, cfeat, anchor):
        super().__init__(net, feat, cfeat, anchor)

    def stream_full(self, batch, aug=None):
        feats, R, v_pre = self.feat(batch, aug)
        flow = batch["flow"]
        # Binary stillness from flow: below 0.1 px is still.
        # This is a cleaner signal than the raw flow magnitude.
        still = (flow < 0.1).float()
        feats = torch.cat((feats, still[..., None]), dim=-1)
        cfeats = cvalid = None
        if self.cfeat is not None:
            cfeats, _, _ = self.cfeat(batch, aug)
            cvalid = batch["cvalid"]
        return self.net(feats, R, self.anchor, v_pre, batch["valid"], cfeats, cvalid)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--output", default="physnet_flow")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.BIAS_MODE = "none"

    root = Path((PROTO / args.corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))

    train_names = [(root / e["path"]).resolve() for e in manifest["imu"]["train"]]
    val_names = [(root / e["path"]).resolve() for e in manifest["imu"]["val"]]
    test_names = [(root / e["path"]).resolve() for e in manifest["imu"]["test"]]

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        train = build_group_with_flow(train_names, context, horizon, length, cap=120, rng=rng)
        val = build_group_with_flow(val_names, context, horizon, length, cap=120, rng=rng)
        test = build_group_with_flow(test_names, context, horizon, length, cap=160,
                                     rng=np.random.default_rng(123))
        train_t = v3.to_device(train, device)
        val_t = v3.to_device(val, device)
        test_t = v3.to_device(test, device)

        anchor = int(round(context * v3.HZ))
        feat = v3.Featurizer(anchor, horizon, use_phys=True)
        feat.fit(v3.take(train_t, torch.arange(0, len(train["y"]),
                                               max(1, len(train["y"]) // 2000), device=device)))
        net = FlowPhysNet(width=96, lever=True, still=True, gate=True).to(device)
        model = FlowModel(net, feat, None, anchor)

        # training loop (simplified from v3.run_phase)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
        best = None
        best_err = np.inf
        bad = 0
        n = len(train["y"])
        for epoch in range(args.epochs):
            net.train()
            order = torch.randperm(n, device=device)
            total = 0.0
            for i in range(0, n, 32):
                idx = order[i:i + 32]
                batch = v3.take(train_t, idx)
                pred = model.stream(batch)
                y = torch.from_numpy(train["y"][idx.cpu().numpy()]).to(device)
                loss = torch.nn.functional.smooth_l1_loss(pred[:, -1], y, beta=0.5)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                total += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                pred = model.predict(val_t)
                err = np.linalg.norm(pred - val["y"], axis=1).mean() * 1000
            if err < best_err:
                best_err = err
                best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                bad = 0
            else:
                bad += 1
            if bad >= 10:
                break
            print(f"epoch {epoch} train {total / n:.4f} val {err:.2f} mm", flush=True)
        net.load_state_dict(best)
        net.eval()
        pred = model.predict(test_t)
        err = np.linalg.norm(pred - test["y"], axis=1).mean() * 1000
        print(f"seed {seed} test err {err:.2f} mm", flush=True)
        out = DATA / args.output
        out.mkdir(exist_ok=True)
        torch.save({"model": best, "featurizer": feat.state(), "width": 96,
                    "lever": True, "context": context, "horizon": horizon,
                    "still": True, "gate": True, "zupt": False, "logvar": False,
                    "bias_mode": "none", "beta": 0.5},
                   out / f"physnet_flow_{args.corpus}_s{seed}.pt")


if __name__ == "__main__":
    main()
