#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Chain every consecutive chessboard pose with an IMU step."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _solve
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"


def _integrate(usb_t, usb, t0, t1, bg, ba, R0, device):
    sel = np.flatnonzero((usb_t >= t0) & (usb_t <= t1))
    if len(sel) < 4:
        return None
    acc = torch.from_numpy(usb[sel, 0:3] * 9.81).to(device).unsqueeze(0)
    gyro = torch.from_numpy(usb[sel, 3:6] * (np.pi / 180.0)).to(device).unsqueeze(0)
    with torch.no_grad():
        _dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
    return dp.cpu().numpy()[0]


def _rmse(path, positions):
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {name: build_pairs(raw[name]) for name in names}
    imu_dp = {}
    for name in names:
        acc = torch.from_numpy(built[name]["acc"]).to(device)
        gyro = torch.from_numpy(built[name]["gyro"]).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(built[name]["bg"], device=device),
                torch.tensor(built[name]["ba"], device=device),
                torch.tensor(built[name]["R0"], device=device),
            )
        imu_dp[name] = dp.cpu().numpy()
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu_dp[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)

    for name in (VAL_NAME, TEST_NAME):
        data = np.load(GT / f"{name}.npz")
        delay = float(data["delay_usb"][0])
        ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
        times = data["t"][ok] - delay
        rotations = data["R"][ok]
        positions = data["p"][ok]
        bg = torch.tensor(built[name]["bg"], device=device)
        ba = torch.tensor(built[name]["ba"], device=device)
        R0 = torch.tensor(built[name]["R0"], device=device)
        edges = []
        ratios = []
        for i in range(len(times) - 1):
            raw_dp = _integrate(
                raw[name]["usb_t"], raw[name]["usb"],
                float(times[i]), float(times[i + 1]), bg, ba, R0, device,
            )
            if raw_dp is None:
                continue
            mapped = np.concatenate([raw_dp, [1.0]]) @ coef
            truth = positions[i + 1] - positions[i]
            pred = rotations[i] @ mapped
            if np.linalg.norm(truth) > 0.01:
                ratios.append(np.linalg.norm(pred) / np.linalg.norm(truth))
            edges.append((i, i + 1, pred))
        path = _solve(len(times), edges)
        path = path - path[0] + positions[0]
        pred_w = np.concatenate([imu_dp[name], np.ones((len(imu_dp[name]), 1))], 1) @ coef
        step = float(np.linalg.norm(pred_w - built[name]["dp"], axis=1).mean() * 1000.0)
        report = {
            "step_mm": round(step, 2),
            "chain_edges": len(edges),
            "chain_mm": round(_rmse(path, positions), 1),
            "gap_scale_p50": round(float(np.median(ratios)), 2) if ratios else None,
        }
        print(json.dumps({name: report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
