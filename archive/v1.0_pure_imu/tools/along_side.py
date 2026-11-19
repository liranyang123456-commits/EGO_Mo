#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split the fast-step error into along-track scale and sideways direction."""

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


def _split(err, truth):
    speed = np.linalg.norm(truth, axis=1)
    keep = speed > np.quantile(speed, 0.67)
    e = err[keep]
    t = truth[keep]
    n = t / speed[keep, None]
    along = np.sum(e * n, axis=1)
    side = np.linalg.norm(e - along[:, None] * n, axis=1)
    return {
        "n": int(keep.sum()),
        "along_rms_mm": round(float(np.sqrt(np.mean(along ** 2)) * 1000.0), 2),
        "side_rms_mm": round(float(np.sqrt(np.mean(side ** 2)) * 1000.0), 2),
        "scale": round(float(np.dot(np.linalg.norm(truth[keep] + err[keep], axis=1), speed[keep]) / np.dot(speed[keep], speed[keep])), 3),
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
    coef, *_ = np.linalg.lstsq(src, dst, rcond=None)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        pred = imu[name] @ coef
        err = pred - built[name]["dp"]
        report[name] = _split(err, built[name]["dp"])
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "along_side.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
