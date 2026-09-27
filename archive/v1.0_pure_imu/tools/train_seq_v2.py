#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Balanced long-context IMU displacement regression with temporal attention.

This trainer never loads the sealed test sessions. It supports:
  none     -- uniform windows (v1 behavior)
  session  -- each recording contributes equal total loss
  motion   -- slow/mid/fast thirds contribute equal total loss
  both     -- equal recordings and equal motion thirds within each recording
  speed    -- continuously upweight larger displacements
  session_speed -- equal recordings with larger displacements upweighted
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as data_api

DATA = ROOT / "datasets"


class ResidualTCN(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(width, width, 5, padding=2 * dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(width, width, 3, padding=dilation, dilation=dilation),
            nn.GroupNorm(8, width),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class AttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Linear(channels, 1)

    def forward(self, h: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(h).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e4)
        weight = torch.softmax(logits, dim=1)
        return torch.sum(h * weight.unsqueeze(-1), dim=1)


class AttentiveStepNet(nn.Module):
    def __init__(self, width: int = 96, hidden: int = 64, dropout: float = 0.12) -> None:
        super().__init__()
        self.inp = nn.Conv1d(8, width, 5, padding=2)
        self.blocks = nn.ModuleList([
            ResidualTCN(width, dilation, dropout)
            for dilation in (1, 2, 4, 8, 16)
        ])
        self.down = nn.Conv1d(width, width, 5, stride=2, padding=2)
        self.gru = nn.GRU(
            width,
            hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        channels = 2 * hidden
        self.full_pool = AttentionPool(channels)
        self.inside_pool = AttentionPool(channels)
        self.head = nn.Sequential(
            nn.Linear(4 * channels, 2 * hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = x[:, :, 6] > 0.5
        h = self.inp(x.transpose(1, 2))
        for block in self.blocks:
            h = block(h)
        h = self.down(h)
        mask = F.max_pool1d(mask.float().unsqueeze(1), 2, stride=2).squeeze(1) > 0.5
        h, _ = self.gru(h.transpose(1, 2))
        start = mask.float().argmax(dim=1)
        end = h.shape[1] - 1 - torch.flip(mask, dims=(1,)).float().argmax(dim=1)
        batch = torch.arange(h.shape[0], device=h.device)
        features = torch.cat(
            (
                self.full_pool(h),
                self.inside_pool(h, mask),
                h[batch, start],
                h[batch, end],
            ),
            dim=-1,
        )
        return self.head(features)


def _weights(groups: list[tuple[str, np.ndarray]], mode: str) -> np.ndarray:
    all_y = np.concatenate([y for _name, y in groups])
    speed = np.linalg.norm(all_y, axis=1)
    cuts = np.quantile(speed, (1 / 3, 2 / 3))
    global_bins = np.digitize(speed, cuts)
    global_counts = np.array([(global_bins == b).sum() for b in range(3)])
    out = []
    for _name, y in groups:
        n = len(y)
        w = np.ones(n, dtype=np.float32)
        bins = np.digitize(np.linalg.norm(y, axis=1), cuts)
        if mode == "session":
            w /= max(n, 1)
        elif mode == "motion":
            for b in range(3):
                w[bins == b] /= max(int(global_counts[b]), 1)
        elif mode == "both":
            for b in range(3):
                m = bins == b
                if m.any():
                    w[m] /= int(m.sum())
        elif mode in {"speed", "session_speed"}:
            local_speed = np.linalg.norm(y, axis=1)
            median = max(float(np.median(speed)), 1e-6)
            w = 0.5 + np.minimum(local_speed / median, 3.0)
            if mode == "session_speed":
                w /= max(float(w.sum()), 1e-6)
        out.append(w)
    weight = np.concatenate(out)
    return (weight / weight.mean()).astype(np.float32)


def _predict(net, X, device):
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 128):
            out.append(net(torch.from_numpy(X[k:k + 128]).to(device)).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--balance",
        choices=("none", "session", "motion", "both", "speed", "session_speed"),
        default="none",
    )
    ap.add_argument("--labels", default="pose_gt_raw")
    ap.add_argument("--out", default="traj_run_v11")
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    data_api.LABELS = args.labels
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    length = int(round((0.30 + 2 * args.context) * data_api.HZ))
    built = {}
    for name in split["train"] + split["val"]:
        got = data_api.build(name, args.context, length)
        if got is None:
            raise RuntimeError(f"no pairs in {name}")
        built[name] = got
        print(json.dumps({"built": name, "n": len(got[0])}), flush=True)

    train_groups = [(name, built[name][1]) for name in split["train"]]
    Xtr = np.concatenate([built[name][0] for name in split["train"]])
    Ytr = np.concatenate([built[name][1] for name in split["train"]])
    Wtr = _weights(train_groups, args.balance)
    Xval = np.concatenate([built[name][0] for name in split["val"]])
    Yval = np.concatenate([built[name][1] for name in split["val"]])

    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(Ytr / 0.01)
    Wt = torch.from_numpy(Wtr)
    net = AttentiveStepNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best_key, best_state, best_epoch, bad = None, None, None, 0
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        weight_sum = 0.0
        for k in range(0, len(perm), 64):
            idx = perm[k:k + 64]
            x = Xt[idx].clone()
            # Bias is constant over a window; sample noise changes every point.
            x[:, :, 0:3] += 0.01 * torch.randn(len(idx), 1, 3)
            x[:, :, 3:6] += 0.01 * torch.randn(len(idx), 1, 3)
            x[:, :, 0:6] += 0.003 * torch.randn_like(x[:, :, 0:6])
            x, y, w = x.to(device), Yt[idx].to(device), Wt[idx].to(device)
            per_axis = F.smooth_l1_loss(net(x), y, beta=0.2, reduction="none")
            per = per_axis.sum(dim=1)
            loss = torch.sum(per * w) / torch.sum(w).clamp(min=1e-6)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(torch.sum(per.detach() * w)) 
            weight_sum += float(torch.sum(w))
        sched.step()
        pred = _predict(net, Xval, device) * 0.01
        val = data_api._metrics(pred, Yval)
        # Fast error prevents selection of a model that collapses to zero.
        key = val["err_mm"] + 0.15 * val["fast_mm"]
        print(json.dumps({
            "epoch": epoch,
            "loss": round(total / max(weight_sum, 1e-6), 4),
            "selection_key": round(key, 4),
            "val": val,
        }), flush=True)
        if best_key is None or key < best_key:
            best_key = key
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 12:
                break

    net.load_state_dict(best_state)
    pred = _predict(net, Xval, device) * 0.01
    val = data_api._metrics(pred, Yval)
    val_by_session = {}
    offset = 0
    for name in split["val"]:
        n = len(built[name][0])
        val_by_session[name] = data_api._metrics(pred[offset:offset + n], Yval[offset:offset + n])
        offset += n
    report = {
        "architecture": "attentive_tcn_bigru",
        "context_s": args.context,
        "seed": args.seed,
        "balance": args.balance,
        "best_epoch": best_epoch,
        "selection_key": best_key,
        "train_sessions": split["train"],
        "val_sessions": split["val"],
        "train_pairs": int(len(Xtr)),
        "val_pairs": int(len(Xval)),
        "val": val,
        "val_by_session": val_by_session,
    }
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"ctx{args.context:g}_{args.balance}_s{args.seed}"
    torch.save(best_state, out / f"model_{tag}.pt")
    np.savez(out / f"pred_{tag}.npz", val_pred=pred, val_y=Yval)
    (out / f"metrics_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"final": report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
