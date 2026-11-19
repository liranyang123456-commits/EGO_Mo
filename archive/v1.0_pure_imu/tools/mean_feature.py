#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Predict the step from mean accel and gyro, without the double integral."""

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


def _pack_feat(pack, device):
    acc = pack["acc"]
    gyro = pack["gyro"]
    acc_t = torch.from_numpy(acc).to(device)
    gyro_t = torch.from_numpy(gyro).to(device)
    with torch.no_grad():
        dR, dp = integrate_aligned(
            acc_t, gyro_t,
            torch.tensor(pack["bg"], device=device),
            torch.tensor(pack["ba"], device=device),
            torch.tensor(pack["R0"], device=device),
        )
    rot = torch.stack([
        dR[:, 2, 1] - dR[:, 1, 2],
        dR[:, 0, 2] - dR[:, 2, 0],
        dR[:, 1, 0] - dR[:, 0, 1],
    ], dim=1).cpu().numpy()
    return {
        "dp": dp.cpu().numpy(),
        "mean": np.concatenate([acc.mean(1), gyro.mean(1)], 1),
        "rot": rot,
    }


def _r2(pred, truth):
    resid = np.sum((pred - truth) ** 2)
    base = np.sum((truth - truth.mean(0)) ** 2)
    return round(float(1.0 - resid / max(base, 1e-12)), 3)


def _metrics(pred, truth):
    err = np.linalg.norm(pred - truth, axis=1)
    speed = np.linalg.norm(truth, axis=1)
    fast = speed >= np.quantile(speed, 0.67)
    return {
        "step_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "r2": _r2(pred, truth),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, feat = {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        feat[name] = _pack_feat(built[name], device)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    y = np.concatenate([built[n]["dp"] for n in train])
    kinds = {
        "integral": lambda n: feat[n]["dp"],
        "mean": lambda n: feat[n]["mean"],
        "mean_rot": lambda n: np.concatenate([feat[n]["mean"], feat[n]["rot"]], 1),
    }
    report = {}
    for kind, take in kinds.items():
        x = np.concatenate([take(n) for n in train])
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        report[kind] = {}
        for name in (VAL_NAME, TEST_NAME):
            pred = take(name) @ coef
            report[kind][name] = _metrics(pred, built[name]["dp"])
        print(json.dumps({kind: report[kind]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "mean_feature.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
