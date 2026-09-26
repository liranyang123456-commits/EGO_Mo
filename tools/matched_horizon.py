#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrate only the IMU samples that cover the pose interval."""

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


def _integrate(pack, device, matched: bool) -> np.ndarray:
    usb = pack["usb"]
    bg = torch.tensor(pack["bg"], device=device)
    ba = torch.tensor(pack["ba"], device=device)
    R0 = torch.tensor(pack["R0"], device=device)
    if not matched:
        acc = torch.from_numpy(usb[:, -40:, 0:3] * 9.81).to(device)
        gyro = torch.from_numpy(usb[:, -40:, 3:6] * (np.pi / 180.0)).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
        return dp.cpu().numpy()
    spans = pack["span"]
    ns = np.clip(np.rint(spans / 0.005), 8, usb.shape[1]).astype(int)
    out = np.zeros((len(usb), 3), dtype=np.float32)
    with torch.no_grad():
        for n in np.unique(ns):
            sel = np.flatnonzero(ns == n)
            acc = torch.from_numpy(usb[sel, -n:, 0:3] * 9.81).to(device)
            gyro = torch.from_numpy(usb[sel, -n:, 3:6] * (np.pi / 180.0)).to(device)
            _dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
            out[sel] = dp.cpu().numpy()
    return out


def _fused(name, pack, pred) -> float:
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
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def _map(src, dst, pred_src):
    design = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    test = np.concatenate([pred_src, np.ones((len(pred_src), 1))], axis=1)
    return test @ coef


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {name: build_pairs(raw[name]) for name in names}
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    report = {}
    for matched, kind in ((False, "fixed"), (True, "matched")):
        imu = {name: _integrate(built[name], device, matched) for name in names}
        src = np.concatenate([imu[n] for n in train])
        dst = np.concatenate([built[n]["dp"] for n in train])
        report[kind] = {}
        for name in (VAL_NAME, TEST_NAME):
            pred = _map(src, dst, imu[name])
            err = np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0
            report[kind][name] = {
                "step_mm": round(float(err), 2),
                "fused_mm": round(_fused(name, built[name], pred), 1),
            }
        print(json.dumps({kind: report[kind]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "matched_horizon.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
