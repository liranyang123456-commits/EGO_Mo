#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Oracle experiment: does the network have enough capacity?

Feeds the true initial velocity as an extra input channel. If the error drops
sharply, the plateau is caused by missing information, not by network capacity
or architecture. If the error stays high, the architecture is the bottleneck.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet_v3 as v3  # noqa: E402
import tools.train_seq as api  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"


class OraclePhysNet(v3.PhysNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_ch = self.base_ch + 3
        width = self.stem[0].out_channels
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, width, 7, stride=2, padding=3), torch.nn.GELU(),
            torch.nn.Conv1d(width, width, 5, stride=2, padding=2), torch.nn.GELU(),
        )


class OracleModel(v3.Model):
    def stream_full(self, batch, aug=None):
        feats, R, v_pre = self.feat(batch, aug)
        v0 = batch["v0"]  # (B, 3) true initial velocity in the anchor frame
        T = feats.shape[1]
        v0e = v0[:, None, :].expand(-1, T, -1)
        feats = torch.cat((feats, v0e), dim=-1)
        cfeats = cvalid = None
        if self.cfeat is not None:
            cfeats, _, _ = self.cfeat(batch, aug)
            cvalid = batch["cvalid"]
        return self.net(feats, R, self.anchor, v_pre, batch["valid"], cfeats, cvalid)


def add_v0(data: dict) -> dict:
    """True initial velocity at t_a, resolved in the anchor frame."""
    v0 = np.zeros((len(data["y"]), 3), np.float32)
    for k, name in enumerate(data["session"]):
        gt = np.load(Path(name) / "ground_truth.npz")
        t = gt["t"].astype(np.float64)
        p = gt["p"].astype(np.float64)
        R = gt["R"].astype(np.float64)
        a, b = data["pair"][k]
        ta = t[a]
        # velocity at t_a from a local linear fit over +-0.2 s
        lo = np.searchsorted(t, ta - 0.2)
        hi = np.searchsorted(t, ta + 0.2, side="right")
        if hi - lo < 3:
            continue
        A = np.column_stack((t[lo:hi] - ta, np.ones(hi - lo)))
        v = np.linalg.lstsq(A, p[lo:hi], rcond=None)[0][0]
        v0[k] = R[a].T @ v
    data["v0"] = v0
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=15)
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
        train = add_v0(v3.build_group(train_names, context, horizon, length, cap=120, rng=rng))
        val = add_v0(v3.build_group(val_names, context, horizon, length, cap=120, rng=rng))
        test = add_v0(v3.build_group(test_names, context, horizon, length, cap=160,
                                     rng=np.random.default_rng(123)))
        train_t = v3.to_device(train, device)
        val_t = v3.to_device(val, device)
        test_t = v3.to_device(test, device)

        anchor = int(round(context * v3.HZ))
        feat = v3.Featurizer(anchor, horizon, use_phys=True)
        feat.fit(v3.take(train_t, torch.arange(0, len(train["y"]),
                                               max(1, len(train["y"]) // 2000), device=device)))
        net = OraclePhysNet(width=96, lever=True, still=True, gate=True).to(device)
        model = OracleModel(net, feat, None, anchor)

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
        print(f"seed {seed} test err {err:.2f} mm (oracle v0)", flush=True)


if __name__ == "__main__":
    main()
