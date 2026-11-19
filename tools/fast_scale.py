#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One scale for fast steps, fit only on the fast training steps."""

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


def _metrics(pred, truth, cutoff):
    err = pred - truth
    speed = np.linalg.norm(truth, axis=1)
    fast = speed >= cutoff
    along = np.zeros(int(fast.sum()))
    if np.any(fast):
        n = truth[fast] / speed[fast, None]
        along = np.sum(err[fast] * n, axis=1)
    return {
        "step_mm": round(float(np.linalg.norm(err, axis=1).mean() * 1000.0), 2),
        "fast_mm": round(float(np.linalg.norm(err[fast], axis=1).mean() * 1000.0), 2) if np.any(fast) else None,
        "along_mean_mm": round(float(along.mean() * 1000.0), 2) if np.any(fast) else None,
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
    y = np.concatenate([built[n]["dp"] for n in train])
    p = np.concatenate([pred[n] for n in train])
    speed = np.linalg.norm(y, axis=1)
    cutoff = float(np.quantile(speed, 0.67))
    fast = speed >= cutoff
    # pred * s ~= truth on fast steps. s = dot(pred, truth) / dot(pred, pred).
    s = float(np.sum(p[fast] * y[fast]) / max(np.sum(p[fast] * p[fast]), 1e-12))
    report = {"scale": round(s, 3), "cutoff_mm": round(cutoff * 1000.0, 1)}
    for name in (VAL_NAME, TEST_NAME):
        base = pred[name]
        scaled = base.copy()
        use = np.linalg.norm(built[name]["dp"], axis=1) >= cutoff
        # At test time the truth speed is unknown. Scale when the prediction itself is long.
        pred_cut = float(np.quantile(np.linalg.norm(p, axis=1), 0.67))
        use_pred = np.linalg.norm(base, axis=1) >= pred_cut
        scaled[use_pred] *= s
        report[name] = {
            "base": _metrics(base, built[name]["dp"], cutoff),
            "scaled": _metrics(scaled, built[name]["dp"], cutoff),
            "pred_cutoff_mm": round(pred_cut * 1000.0, 1),
        }
        _ = use
        print(json.dumps({name: report[name], "scale": report["scale"]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "fast_scale.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
