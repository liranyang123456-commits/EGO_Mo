#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Carry the last 0.2 s velocity across board-loss gaps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _nearest, _nodes, _solve
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
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)

    for name in (VAL_NAME, TEST_NAME):
        pred = np.concatenate([imu[name], np.ones((len(imu[name]), 1))], 1) @ coef
        step = float(np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0)
        times, rotations, positions = _nodes(name)
        a = _nearest(times, built[name]["t_start"])
        b = _nearest(times, built[name]["t_end"])
        edges = []
        end_at = {}
        for k in range(len(pred)):
            if abs(times[a[k]] - built[name]["t_start"][k]) > 0.03 or abs(times[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            meas = rotations[a[k]] @ pred[k]
            edges.append((int(a[k]), int(b[k]), meas))
            end_at[int(b[k])] = (meas, float(built[name]["span"][k]))
        ratios, errs = [], []
        coast_edges = list(edges)
        for i in range(len(times) - 1):
            dt = float(times[i + 1] - times[i])
            if dt <= 0.30 or i not in end_at:
                continue
            meas, span = end_at[i]
            coast = meas * (dt / max(span, 1e-3))
            truth = positions[i + 1] - positions[i]
            if np.linalg.norm(truth) < 0.01:
                continue
            ratios.append(np.linalg.norm(coast) / np.linalg.norm(truth))
            errs.append(np.linalg.norm(coast - truth) * 1000.0)
            coast_edges.append((i, i + 1, coast))
        plain = _solve(len(times), edges)
        plain = plain - plain[0] + positions[0]
        coast_path = _solve(len(times), coast_edges)
        coast_path = coast_path - coast_path[0] + positions[0]
        def rmse(path):
            return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)
        report = {
            "step_mm": round(step, 2),
            "gaps": len(ratios),
            "coast_scale_p50": round(float(np.median(ratios)), 2) if ratios else None,
            "coast_err_mm_p50": round(float(np.median(errs)), 1) if errs else None,
            "fused_mm": round(rmse(plain), 1),
            "coast_path_mm": round(rmse(coast_path), 1),
        }
        print(json.dumps({name: report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
