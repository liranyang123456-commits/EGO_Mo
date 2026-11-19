#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Where the fused path error sits: axis and time."""

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


def _fused_path(name, pack, pred):
    times, rotations, positions = _nodes(name)
    a = _nearest(times, pack["t_start"])
    b = _nearest(times, pack["t_end"])
    edges = []
    for k in range(len(pred)):
        if abs(times[a[k]] - pack["t_start"][k]) > 0.03 or abs(times[b[k]] - pack["t_end"][k]) > 0.03:
            continue
        if a[k] >= b[k]:
            continue
        edges.append((int(a[k]), int(b[k]), rotations[a[k]] @ pred[k]))
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    return times, path, positions


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
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        pred = np.concatenate([imu[name], np.ones((len(imu[name]), 1))], 1) @ coef
        times, path, positions = _fused_path(name, built[name], pred)
        err = path - positions
        t = times - times[0]
        quint = []
        for q in range(5):
            lo, hi = np.quantile(t, [q / 5, (q + 1) / 5])
            sel = (t >= lo) & (t <= hi if q == 4 else t < hi)
            if not np.any(sel):
                continue
            quint.append(round(float(np.sqrt(np.mean(np.sum(err[sel] ** 2, axis=1))) * 1000.0), 1))
        report[name] = {
            "axis_rms_mm": [round(float(v) * 1000.0, 1) for v in np.sqrt(np.mean(err ** 2, axis=0))],
            "rmse_mm": round(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) * 1000.0), 1),
            "by_time_quintile_mm": quint,
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "error_shape.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
