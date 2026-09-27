#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step model: linear map plus a small residual on gyro, acc, and rotation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_contiguous import _chain, _keep, _rmse
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"


def _imu(pack, device):
    acc = torch.from_numpy(pack["acc"]).to(device)
    gyro = torch.from_numpy(pack["gyro"]).to(device)
    with torch.no_grad():
        dR, dp = integrate_aligned(
            acc, gyro,
            torch.tensor(pack["bg"], device=device),
            torch.tensor(pack["ba"], device=device),
            torch.tensor(pack["R0"], device=device),
        )
    rotvec = torch.stack([
        dR[:, 2, 1] - dR[:, 1, 2],
        dR[:, 0, 2] - dR[:, 2, 0],
        dR[:, 1, 0] - dR[:, 0, 1],
    ], dim=1)
    feat = torch.cat([
        dp,
        gyro.mean(dim=1),
        acc.mean(dim=1),
        rotvec,
    ], dim=1)
    return feat.cpu().numpy().astype(np.float32), dp.cpu().numpy().astype(np.float32)


class Residual(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat, coef):
        ones = torch.ones(feat.shape[0], 1, device=feat.device, dtype=feat.dtype)
        linear = torch.cat([feat[:, :3], ones], dim=1) @ coef
        return linear + self.net(feat)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = load_split(names)
    built = {name: build_pairs(packs[name]) for name in names}
    train_names = [name for name in names if name not in (TEST_NAME, VAL_NAME)]
    feats, targets, imu_dp = {}, {}, {}
    for name, pack in built.items():
        feat, dp = _imu(pack, device)
        feats[name] = feat
        imu_dp[name] = dp
        targets[name] = pack["dp"]
    src = np.concatenate([imu_dp[name] for name in train_names])
    dst = np.concatenate([targets[name] for name in train_names])
    design = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    coef_np, *_ = np.linalg.lstsq(design, dst, rcond=None)
    coef = torch.tensor(coef_np, dtype=torch.float32, device=device)
    x = torch.from_numpy(np.concatenate([feats[name] for name in train_names])).to(device)
    y = torch.from_numpy(np.concatenate([targets[name] for name in train_names])).to(device)
    model = Residual().to(device)
    opt = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(1, 41):
        model.train()
        perm = torch.randperm(x.shape[0], device=device)
        total = 0.0
        for start in range(0, x.shape[0], 256):
            batch = perm[start:start + 256]
            pred = model(x[batch], coef)
            loss = (pred - y[batch]).norm(dim=1).mean() * 1000.0
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.item()) * batch.shape[0]
        if epoch % 10 == 0:
            print(json.dumps({"epoch": epoch, "train_mm": round(total / x.shape[0], 2)}), flush=True)
    model.eval()
    report = {}
    with torch.no_grad():
        for name in (VAL_NAME, TEST_NAME):
            feat = torch.from_numpy(feats[name]).to(device)
            pred = model(feat, coef).cpu().numpy()
            base = (np.concatenate([imu_dp[name], np.ones((len(pred), 1))], axis=1) @ coef_np)
            gt = targets[name]
            keep = _keep(built[name]["t_start"], built[name]["t_end"])
            dR = built[name]["dR"]
            report[name] = {
                "step_net_mm": round(float(np.linalg.norm(pred - gt, axis=1).mean() * 1000.0), 2),
                "step_linear_mm": round(float(np.linalg.norm(base - gt, axis=1).mean() * 1000.0), 2),
                "path_net_mm": round(_rmse(_chain(pred, dR, keep), _chain(gt, dR, keep)), 1),
                "path_linear_mm": round(_rmse(_chain(base, dR, keep), _chain(gt, dR, keep)), 1),
            }
            print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v4"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "coef": coef_np}, out / "step_residual.pt")
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
