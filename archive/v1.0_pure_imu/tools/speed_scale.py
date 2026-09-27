#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Speed-dependent scale on the predicted step, fit on training data only."""

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


def _apply(pred, a, b):
    speed = np.linalg.norm(pred, axis=1, keepdims=True)
    return pred * (a + b * speed)


def _metrics(pred, truth):
    err = pred - truth
    step = float(np.linalg.norm(err, axis=1).mean() * 1000.0)
    speed = np.linalg.norm(truth, axis=1)
    fast = speed > np.quantile(speed, 0.67)
    n = truth[fast] / speed[fast, None]
    along = np.sum(err[fast] * n, axis=1)
    return {
        "step_mm": round(step, 2),
        "fast_mm": round(float(np.linalg.norm(err[fast], axis=1).mean() * 1000.0), 2),
        "along_mean_mm": round(float(along.mean() * 1000.0), 2),
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
    pred = {name: imu[name] @ coef for name in names}
    # Fit pred * (a + b ||pred||) ~= truth on train, in least squares on the vectors.
    p = np.concatenate([pred[n] for n in train])
    y = np.concatenate([built[n]["dp"] for n in train])
    speed = np.linalg.norm(p, axis=1, keepdims=True)
    design = np.concatenate([p, p * speed], axis=1)
    # Solve y ≈ p * a + p * speed * b, per coordinate shared a,b via stacked scalar rows.
    rows = []
    targets = []
    for i in range(len(p)):
        if speed[i, 0] < 1e-6:
            continue
        for k in range(3):
            rows.append([p[i, k], p[i, k] * speed[i, 0]])
            targets.append(y[i, k])
    ab, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(targets), rcond=None)
    a, b = float(ab[0]), float(ab[1])
    report = {"a": round(a, 4), "b_per_m": round(b, 3)}
    for name in (VAL_NAME, TEST_NAME):
        base = _metrics(pred[name], built[name]["dp"])
        scaled = _metrics(_apply(pred[name], a, b), built[name]["dp"])
        report[name] = {"base": base, "scaled": scaled}
        print(json.dumps({name: report[name], "a": report["a"], "b_per_m": report["b_per_m"]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "speed_scale.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
