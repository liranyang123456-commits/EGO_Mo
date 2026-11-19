#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Protocol-matched reimplementations of public inertial architectures.

Architectural motifs follow official RoNIN, TLIO and IMUNet repositories.
Input/output layers are adapted from pedestrian 2-D velocity to this study's
3-D, 3-s displacement protocol. Results must be labeled "adapted/retrained",
not official pretrained-model scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
from tools.train_hybrid import _synthetic_build

DATA = ROOT / "datasets"


class ResBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 3, stride, 1, bias=False),
            nn.BatchNorm1d(cout), nn.ReLU(),
            nn.Conv1d(cout, cout, 3, 1, 1, bias=False),
            nn.BatchNorm1d(cout),
        )
        self.skip = (
            nn.Identity() if cin == cout and stride == 1
            else nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        )

    def forward(self, x):
        return F.relu(self.net(x) + self.skip(x))


class RoNINResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(3, 2, 1),
        )
        layers = []
        cin = 64
        for stage, cout in enumerate((64, 128, 256, 512)):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResBlock(cin, cout, stride))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(512, 3)

    def forward(self, x):
        x = F.interpolate(x[:, :, :6].transpose(1, 2), 400, mode="linear", align_corners=False)
        return self.head(F.adaptive_avg_pool1d(self.body(self.stem(x)), 1).squeeze(-1))


class RoNINLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(6, 100, 3, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(100, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        x = F.interpolate(x[:, :, :6].transpose(1, 2), 400, mode="linear", align_corners=False)
        _sequence, (hidden, _cell) = self.lstm(x.transpose(1, 2))
        return self.head(hidden[-1])


class TLIOResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RoNINResNet()
        self.encoder.head = nn.Identity()
        self.mean = nn.Linear(512, 3)
        self.logstd = nn.Linear(512, 3)

    def forward(self, x):
        feature = self.encoder(x)
        return self.mean(feature)


class DepthwiseBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm1d(cin), nn.ELU(),
            nn.Conv1d(cin, cout, 1, bias=False),
            nn.BatchNorm1d(cout), nn.ELU(),
        )
        self.skip = (
            nn.Identity() if cin == cout and stride == 1
            else nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        )

    def forward(self, x):
        return F.elu(self.net(x) + self.skip(x))


class IMUNetAdapted(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(6, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(3, 2, 1),
        )
        self.body = nn.Sequential(
            DepthwiseBlock(64, 64), DepthwiseBlock(64, 64),
            DepthwiseBlock(64, 128, 2), DepthwiseBlock(128, 128),
            DepthwiseBlock(128, 256, 2), DepthwiseBlock(256, 256),
            DepthwiseBlock(256, 512, 2), DepthwiseBlock(512, 512),
            DepthwiseBlock(512, 1024, 2), DepthwiseBlock(1024, 1024),
        )
        self.head = nn.Linear(1024, 3)

    def forward(self, x):
        x = F.interpolate(x[:, :, :6].transpose(1, 2), 200, mode="linear", align_corners=False)
        feature = F.adaptive_avg_pool1d(self.body(self.stem(x)), 1).squeeze(-1)
        return self.head(feature)


def make_model(name):
    return {
        "ronin_resnet": RoNINResNet,
        "ronin_lstm": RoNINLSTM,
        "tlio_resnet": TLIOResNet,
        "imunet": IMUNetAdapted,
    }[name]()


def predict(model, X, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(model(torch.from_numpy(X[i:i + 128]).to(device)).cpu().numpy())
    return np.concatenate(out) * 0.01


def train_phase(model, X, y, Xval, yval, device, epochs, lr, seed, patience):
    rng = np.random.default_rng(seed)
    Xt, yt = torch.from_numpy(X), torch.from_numpy(y / 0.01)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    best, state, bad = None, None, 0
    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(X))
        for i in range(0, len(order), 64):
            idx = torch.from_numpy(order[i:i + 64])
            xb = Xt[idx].clone()
            xb[:, :, :3] += 0.01 * torch.randn(len(idx), 1, 3)
            xb[:, :, 3:6] += 0.01 * torch.randn(len(idx), 1, 3)
            xb, yb = xb.to(device), yt[idx].to(device)
            loss = F.smooth_l1_loss(model(xb), yb, beta=0.2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        metric = api._metrics(predict(model, Xval, device), yval)
        if best is None or metric["err_mm"] < best:
            best = metric["err_mm"]
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(state)
    return metric


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=("ronin_resnet", "ronin_lstm", "tlio_resnet", "imunet"), required=True)
    ap.add_argument("--corpus", type=Path, default=None, help="synthetic corpus (not needed for --only real_only)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="external_benchmark")
    ap.add_argument("--only", choices=("all", "real_only"), default="all")
    ap.add_argument("--holdout", default="",
                    help="real_only: leave-one-session-out fold over all train+val sessions")
    ap.add_argument("--extra-train", nargs="*", default=[],
                    help="additional real sessions always added to training (never validated); "
                         "default: split file 'extra_train'")
    ap.add_argument("--no-extra", action="store_true", help="ignore extra_train sessions")
    ap.add_argument("--stem", default="", help="override output file stem")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context = 2.0
    length = int(round((1.5 * 3.0 + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    groups = {"train": split["train"], "val": split["val"]}
    if args.holdout:
        pool = split["train"] + split["val"]
        assert args.holdout in pool
        groups = {"train": [n for n in pool if n != args.holdout], "val": [args.holdout]}
    extra = [] if args.no_extra else (args.extra_train or split.get("extra_train", []))
    groups["train"] = groups["train"] + [n for n in extra if n not in groups["train"] + groups["val"]]
    real = {}
    for group, names in groups.items():
        parts = [api.build(name, context, length) for name in names]
        real[group] = (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
        )
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    report = {"architecture": args.arch, "seed": args.seed, "protocol": "adapted 6-axis to 3-D 3-s displacement"}

    if args.only == "real_only":
        model = make_model(args.arch).to(device)
        train_phase(model, *real["train"], *real["val"], device, 50, 1e-3, args.seed + 200, 12)
        report["real_only_real_val"] = api._metrics(
            predict(model, real["val"][0], device), real["val"][1]
        )
        report["holdout"] = args.holdout
        report["train_sessions"] = groups["train"]
        stem = args.stem or (f"{args.arch}_cv_{args.holdout}_s{args.seed}" if args.holdout
                             else f"{args.arch}_real_s{args.seed}")
        torch.save(model.state_dict(), out / f"{stem}.pt")
        (out / f"{stem}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False))
        return

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    synth = {}
    for group in ("train", "val"):
        parts = [
            _synthetic_build(args.corpus / e["path"], context, length, 3.0, 80, rng)
            for e in manifest["imu"][group]
        ]
        synth[group] = (
            np.concatenate([p[0] for p in parts if p is not None]),
            np.concatenate([p[1] for p in parts if p is not None]),
        )

    # Synthetic direct model.
    zero = make_model(args.arch).to(device)
    train_phase(zero, *synth["train"], *synth["val"], device, 25, 1e-3, args.seed, 8)
    report["synthetic_zero_shot_real_val"] = api._metrics(
        predict(zero, real["val"][0], device), real["val"][1]
    )
    torch.save(zero.state_dict(), out / f"{args.arch}_zero_s{args.seed}.pt")

    # Fine-tune the synthetic model.
    fine = make_model(args.arch).to(device)
    fine.load_state_dict(zero.state_dict())
    train_phase(fine, *real["train"], *real["val"], device, 50, 5e-4, args.seed + 100, 12)
    report["fine_tuned_real_val"] = api._metrics(
        predict(fine, real["val"][0], device), real["val"][1]
    )
    torch.save(fine.state_dict(), out / f"{args.arch}_fine_s{args.seed}.pt")

    # Same architecture trained on real data only.
    real_model = make_model(args.arch).to(device)
    train_phase(real_model, *real["train"], *real["val"], device, 50, 1e-3, args.seed + 200, 12)
    report["real_only_real_val"] = api._metrics(
        predict(real_model, real["val"][0], device), real["val"][1]
    )
    torch.save(real_model.state_dict(), out / f"{args.arch}_real_s{args.seed}.pt")

    dummy = torch.from_numpy(real["val"][0][:16]).to(device)
    start = time.perf_counter()
    for _ in range(20):
        _ = fine(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    report["parameters"] = sum(p.numel() for p in fine.parameters())
    report["inference_ms_per_window"] = (time.perf_counter() - start) * 1000 / (20 * len(dummy))
    (out / f"{args.arch}_s{args.seed}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    from tools.train_hybrid import _synthetic_build
    main()
