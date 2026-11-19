#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cascade: short-window estimates as a velocity prior for the long window.

A 0.5-s window has a small unobservable initial-velocity term. This uses the
0.5-s estimate at the start of a 3-s window as a prior for the 3-s window's
initial velocity, and measures whether the 3-s error drops.
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


class CascadePhysNet(v3.PhysNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        in_ch = self.base_ch + 3
        width = self.stem[0].out_channels
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(in_ch, width, 7, stride=2, padding=3), torch.nn.GELU(),
            torch.nn.Conv1d(width, width, 5, stride=2, padding=2), torch.nn.GELU(),
        )


class CascadeModel(v3.Model):
    def stream_full(self, batch, aug=None):
        feats, R, v_pre = self.feat(batch, aug)
        v0 = batch["v0"]
        T = feats.shape[1]
        v0e = v0[:, None, :].expand(-1, T, -1)
        feats = torch.cat((feats, v0e), dim=-1)
        cfeats = cvalid = None
        if self.cfeat is not None:
            cfeats, _, _ = self.cfeat(batch, aug)
            cvalid = batch["cvalid"]
        return self.net(feats, R, self.anchor, v_pre, batch["valid"], cfeats, cvalid)


def train_short(horizon, seed, corpus, device):
    """Train a short-window model and return it."""
    api.LABELS, api.TARGET_S = "pose_gt_raw", horizon
    v3.BIAS_MODE = "none"
    root = Path((PROTO / corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    context = 2.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))
    train_names = [(root / e["path"]).resolve() for e in manifest["imu"]["train"]]
    val_names = [(root / e["path"]).resolve() for e in manifest["imu"]["val"]]
    rng = np.random.default_rng(seed)
    train = v3.build_group(train_names, context, horizon, length, cap=120, rng=rng)
    val = v3.build_group(val_names, context, horizon, length, cap=120, rng=rng)
    train_t = v3.to_device(train, device)
    val_t = v3.to_device(val, device)
    anchor = int(round(context * v3.HZ))
    feat = v3.Featurizer(anchor, horizon, use_phys=True)
    feat.fit(v3.take(train_t, torch.arange(0, len(train["y"]),
                                           max(1, len(train["y"]) // 2000), device=device)))
    net = v3.PhysNet(width=96, lever=True, still=True, gate=True).to(device)
    model = v3.Model(net, feat, None, anchor)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    best = None
    best_err = np.inf
    bad = 0
    n = len(train["y"])
    for epoch in range(15):
        net.train()
        order = torch.randperm(n, device=device)
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
    net.load_state_dict(best)
    net.eval()
    return model, feat, anchor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Train a 0.5-s model
    short_model, short_feat, short_anchor = train_short(0.5, args.seeds[0], args.corpus, device)

    # Now train a 3-s model with the 0.5-s estimate as v0 prior
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    v3.BIAS_MODE = "none"
    root = Path((PROTO / args.corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    context = 2.0
    horizon = 3.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))
    train_names = [(root / e["path"]).resolve() for e in manifest["imu"]["train"]]
    val_names = [(root / e["path"]).resolve() for e in manifest["imu"]["val"]]
    test_names = [(root / e["path"]).resolve() for e in manifest["imu"]["test"]]

    rng = np.random.default_rng(args.seeds[0])
    train = v3.build_group(train_names, context, horizon, length, cap=120, rng=rng)
    val = v3.build_group(val_names, context, horizon, length, cap=120, rng=rng)
    test = v3.build_group(test_names, context, horizon, length, cap=160,
                          rng=np.random.default_rng(123))

    # For each window, estimate the 0.5-s displacement at the start and convert to velocity
    def add_v0_prior(data):
        v0 = np.zeros((len(data["y"]), 3), np.float32)
        # group by session
        for name in np.unique(data["session"]):
            sel = data["session"] == name
            # build short windows for this session
            short_data = v3.build_group([name], context, 0.5,
                                        int(round((1.5 * 0.5 + 2 * context) * v3.HZ)),
                                        cap=0)
            short_t = v3.to_device(short_data, device)
            short_pred = short_model.predict(short_t)
            # short_pred is displacement over 0.5 s; velocity = displacement / 0.5
            # map to the long windows by time
            t_short = short_data["t_pair"][:, 0]
            v_short = short_pred / 0.5
            for k in np.flatnonzero(sel):
                ta = data["t_pair"][k, 0]
                # find the short window ending at ta
                j = np.argmin(np.abs(t_short - ta))
                v0[k] = v_short[j]
        data["v0"] = v0
        return data

    train = add_v0_prior(train)
    val = add_v0_prior(val)
    test = add_v0_prior(test)
    train_t = v3.to_device(train, device)
    val_t = v3.to_device(val, device)
    test_t = v3.to_device(test, device)

    anchor = int(round(context * v3.HZ))
    feat = v3.Featurizer(anchor, horizon, use_phys=True)
    feat.fit(v3.take(train_t, torch.arange(0, len(train["y"]),
                                           max(1, len(train["y"]) // 2000), device=device)))
    net = CascadePhysNet(width=96, lever=True, still=True, gate=True).to(device)
    model = CascadeModel(net, feat, None, anchor)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    best = None
    best_err = np.inf
    bad = 0
    n = len(train["y"])
    for epoch in range(15):
        net.train()
        order = torch.randperm(n, device=device)
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
        print(f"epoch {epoch} val {err:.2f} mm", flush=True)
    net.load_state_dict(best)
    net.eval()
    pred = model.predict(test_t)
    err = np.linalg.norm(pred - test["y"], axis=1).mean() * 1000
    print(f"test err {err:.2f} mm (cascade 0.5s -> 3s)", flush=True)


if __name__ == "__main__":
    main()
