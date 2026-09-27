#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Add the 0.2 s rotation to the displacement map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
    rot = torch.stack([
        dR[:, 2, 1] - dR[:, 1, 2],
        dR[:, 0, 2] - dR[:, 2, 0],
        dR[:, 1, 0] - dR[:, 0, 1],
    ], dim=1)
    return dp.cpu().numpy(), rot.cpu().numpy()


def _metrics(pred, truth):
    err = np.linalg.norm(pred - truth, axis=1)
    speed = np.linalg.norm(truth, axis=1)
    fast = speed >= np.quantile(speed, 0.67)
    return {
        "step_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "slow_mm": round(float(err[~fast].mean() * 1000.0), 2),
    }


def _r2(pred, truth):
    resid = np.sum((pred - truth) ** 2)
    base = np.sum((truth - truth.mean(axis=0)) ** 2)
    return round(float(1.0 - resid / max(base, 1e-12)), 3)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, dp, rot = {}, {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        dp[name], rot[name] = _imu(built[name], device)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    y = np.concatenate([built[n]["dp"] for n in train])
    x0 = np.concatenate([dp[n] for n in train])
    x1 = np.concatenate([np.concatenate([dp[n], rot[n]], 1) for n in train])
    c0, *_ = np.linalg.lstsq(x0, y, rcond=None)
    c1, *_ = np.linalg.lstsq(x1, y, rcond=None)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        y_n = built[name]["dp"]
        p0 = dp[name] @ c0
        p1 = np.concatenate([dp[name], rot[name]], 1) @ c1
        report[name] = {
            "base": _metrics(p0, y_n),
            "with_rot": _metrics(p1, y_n),
            "r2_base": _r2(p0, y_n),
            "r2_rot": _r2(p1, y_n),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "rot_feature.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
