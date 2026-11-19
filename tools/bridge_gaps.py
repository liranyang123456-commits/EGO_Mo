#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge gaps where the board leaves the image, using IMU integration."""

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


def _windows(pack, device):
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


def _gap_dp(usb_t, usb, t0, t1, bg, ba, R0, device):
    sel = np.flatnonzero((usb_t >= t0) & (usb_t <= t1))
    if len(sel) < 8:
        return None
    acc = torch.from_numpy(usb[sel, 0:3] * 9.81).to(device).unsqueeze(0)
    gyro = torch.from_numpy(usb[sel, 3:6] * (np.pi / 180.0)).to(device).unsqueeze(0)
    with torch.no_grad():
        _dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
    return dp.cpu().numpy()[0]


def _edges(name, pack, pred, imu_t, imu, coef, device, bridge: bool):
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
    n_bridge = 0
    if bridge:
        bg = torch.tensor(pack["bg"], device=device)
        ba = torch.tensor(pack["ba"], device=device)
        R0 = torch.tensor(pack["R0"], device=device)
        for i in range(len(times) - 1):
            dt = float(times[i + 1] - times[i])
            if dt <= 0.30:
                continue
            raw = _gap_dp(imu_t, imu, float(times[i]), float(times[i + 1]), bg, ba, R0, device)
            if raw is None:
                continue
            mapped = np.concatenate([raw, [1.0]]) @ coef
            edges.append((i, i + 1, rotations[i] @ mapped))
            n_bridge += 1
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    rmse = float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)
    return rmse, n_bridge


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, imu_dp = {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        imu_dp[name] = _windows(built[name], device)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu_dp[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        pred = np.concatenate([imu_dp[name], np.ones((len(imu_dp[name]), 1))], 1) @ coef
        step = float(np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0)
        plain, _ = _edges(name, built[name], pred, raw[name]["usb_t"], raw[name]["usb"], coef, device, False)
        bridged, n_bridge = _edges(name, built[name], pred, raw[name]["usb_t"], raw[name]["usb"], coef, device, True)
        report[name] = {
            "step_mm": round(step, 2),
            "fused_mm": round(plain, 1),
            "bridged_mm": round(bridged, 1),
            "bridges": n_bridge,
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "bridged_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
