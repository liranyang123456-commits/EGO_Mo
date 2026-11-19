#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direct 0.2 s displacement from the IMU window, not a correction of the integral."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import align_R_from_acc, exp_so3
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"
G_UP = np.array([0.0, 0.0, 9.81])


def _level(acc: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """(T,3) specific force and gyro, returned in a gravity-level frame as (T,6)."""
    R = align_R_from_acc(acc[0])
    out = np.zeros((len(acc), 6), dtype=np.float32)
    for i in range(len(acc)):
        if i:
            R = R @ exp_so3(gyro[i] * 0.005)
        leveled = R @ acc[i] - G_UP
        out[i, :3] = leveled
        out[i, 3:] = R @ gyro[i]
    return out


def _sequences(pack) -> np.ndarray:
    acc = pack["acc"]
    gyro = pack["gyro"]
    seq = np.zeros((len(acc), acc.shape[1], 6), dtype=np.float32)
    for i in range(len(acc)):
        seq[i] = _level(acc[i], gyro[i])
    return seq


def _r2(pred, truth) -> float:
    resid = np.sum((pred - truth) ** 2)
    base = np.sum((truth - truth.mean(0)) ** 2)
    return float(1.0 - resid / max(base, 1e-12))


def _metrics(pred, truth) -> dict:
    err = np.linalg.norm(pred - truth, axis=1)
    speed = np.linalg.norm(truth, axis=1)
    fast = speed >= np.quantile(speed, 0.67)
    return {
        "step_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "slow_mm": round(float(err[~fast].mean() * 1000.0), 2),
        "pred_mm": round(float(np.median(np.linalg.norm(pred, axis=1)) * 1000.0), 2),
        "truth_mm": round(float(np.median(speed) * 1000.0), 2),
        "r2": round(_r2(pred, truth), 3),
    }


class DirectStep(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(6, 64, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.GELU(),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.GELU(),
        )
        self.head = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x.transpose(1, 2))
        return self.head(h.mean(dim=-1))


def _loader(seq, y, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(seq), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=shuffle)


def _predict(net, seq, device):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(seq), 256):
            batch = torch.from_numpy(seq[i:i + 256]).to(device)
            out.append(net(batch).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, imu, seq = {}, {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        acc = torch.from_numpy(built[name]["acc"]).to(device)
        gyro = torch.from_numpy(built[name]["gyro"]).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(built[name]["bg"], device=device),
                torch.tensor(built[name]["ba"], device=device),
                torch.tensor(built[name]["R0"], device=device),
            )
        imu[name] = dp.cpu().numpy()
        seq[name] = _sequences(built[name])
        print(json.dumps({"built": name, "n": int(len(seq[name]))}), flush=True)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu[n] for n in train])
    dst = np.concatenate([built[n]["dp"].astype(np.float32) for n in train])
    coef, *_ = np.linalg.lstsq(src, dst, rcond=None)
    y = {name: built[name]["dp"].astype(np.float32) for name in names}
    report = {
        "zero": {name: _metrics(np.zeros_like(y[name]), y[name]) for name in (VAL_NAME, TEST_NAME)},
        "linear": {
            name: _metrics(imu[name] @ coef, y[name]) for name in (VAL_NAME, TEST_NAME)
        },
    }
    print(json.dumps({"zero": report["zero"], "linear": report["linear"]}, ensure_ascii=False), flush=True)

    x_train = np.concatenate([seq[n] for n in train])
    y_train = np.concatenate([y[n] for n in train])
    net = DirectStep().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = _loader(x_train, y_train, True)
    best, best_state, wait = 1e9, None, 0
    for epoch in range(1, 41):
        net.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)
            loss = (pred - yb).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        val = _metrics(_predict(net, seq[VAL_NAME], device), y[VAL_NAME])
        print(json.dumps({"epoch": epoch, "val": val}), flush=True)
        if val["step_mm"] < best:
            best = val["step_mm"]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 6:
                break
    net.load_state_dict(best_state)
    report["direct"] = {name: _metrics(_predict(net, seq[name], device), y[name]) for name in (VAL_NAME, TEST_NAME)}
    report["target_step_mm"] = 5.0
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    out = ROOT / "datasets" / "traj_run_v5"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "metrics": report}, out / "direct_step.pt")
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
