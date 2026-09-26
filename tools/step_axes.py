#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the 0.2 s error by camera axis and by step size."""

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


def _bins(err, speed):
    edges = np.quantile(speed, [0.0, 0.33, 0.66, 1.0])
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    labels = ["slow", "mid", "fast"]
    out = {}
    idx = np.digitize(speed, edges[1:-1])
    for i, name in enumerate(labels):
        sel = err[idx == i]
        if len(sel) == 0:
            continue
        out[name] = {
            "n": int(len(sel)),
            "mm": round(float(np.linalg.norm(sel, axis=1).mean() * 1000.0), 2),
            "axis_mm": [round(float(np.sqrt(np.mean(sel[:, k] ** 2)) * 1000.0), 2) for k in range(3)],
        }
    return out


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
    coef, *_ = np.linalg.lstsq(src, dst, rcond=None)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        pred = imu[name] @ coef
        err = pred - built[name]["dp"]
        speed = np.linalg.norm(built[name]["dp"], axis=1)
        report[name] = {
            "step_mm": round(float(np.linalg.norm(err, axis=1).mean() * 1000.0), 2),
            "axis_rms_mm": [round(float(np.sqrt(np.mean(err[:, k] ** 2)) * 1000.0), 2) for k in range(3)],
            "mean_mm": [round(float(v) * 1000.0, 2) for v in err.mean(axis=0)],
            "by_speed": _bins(err, speed),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "step_axes.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
