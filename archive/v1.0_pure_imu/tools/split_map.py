#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate maps for small and large raw IMU steps."""

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
        _dR, dp = integrate_aligned(
            acc, gyro,
            torch.tensor(pack["bg"], device=device),
            torch.tensor(pack["ba"], device=device),
            torch.tensor(pack["R0"], device=device),
        )
    return dp.cpu().numpy()


def _fit(src, dst):
    coef, *_ = np.linalg.lstsq(src, dst, rcond=None)
    return coef


def _metrics(pred, truth):
    err = np.linalg.norm(pred - truth, axis=1)
    speed = np.linalg.norm(truth, axis=1)
    cut = np.quantile(speed, 0.67) if len(speed) else 0.0
    fast = speed >= cut
    return {
        "step_mm": round(float(err.mean() * 1000.0), 2),
        "slow_mm": round(float(err[~fast].mean() * 1000.0), 2) if np.any(~fast) else None,
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2) if np.any(fast) else None,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, imu = {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        imu[name] = _imu(built[name], device)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    mag = np.linalg.norm(src, axis=1)
    cut = float(np.quantile(mag, 0.67))
    slow, fast = mag < cut, mag >= cut
    coef_slow = _fit(src[slow], dst[slow]) if slow.sum() > 10 else _fit(src, dst)
    coef_fast = _fit(src[fast], dst[fast]) if fast.sum() > 10 else coef_slow
    coef_all = _fit(src, dst)
    report = {"raw_cutoff_m": round(cut, 4)}
    for name in (VAL_NAME, TEST_NAME):
        x = imu[name]
        y = built[name]["dp"]
        base = x @ coef_all
        use_fast = np.linalg.norm(x, axis=1) >= cut
        split = np.zeros_like(base)
        if np.any(~use_fast):
            split[~use_fast] = x[~use_fast] @ coef_slow
        if np.any(use_fast):
            split[use_fast] = x[use_fast] @ coef_fast
        report[name] = {"base": _metrics(base, y), "split": _metrics(split, y)}
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "split_map.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
