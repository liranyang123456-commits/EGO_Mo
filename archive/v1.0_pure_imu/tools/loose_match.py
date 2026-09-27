#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Reconnect the pose graph with a looser time match. Step error is unchanged."""

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


def _reachable(n, edges):
    seen = {0}
    changed = True
    while changed:
        changed = False
        for a, b, _m in edges:
            if a in seen and b not in seen:
                seen.add(b)
                changed = True
            elif b in seen and a not in seen:
                seen.add(a)
                changed = True
    return seen


def _nearest(times, query):
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = idx - 1
    choose_left = np.abs(times[left] - query) <= np.abs(times[idx] - query)
    return np.where(choose_left, left, idx)


def _eval(times, rotations, positions, t_start, t_end, pred, tol):
    a = _nearest(times, t_start)
    b = _nearest(times, t_end)
    edges = []
    for k in range(len(pred)):
        if abs(times[a[k]] - t_start[k]) > tol or abs(times[b[k]] - t_end[k]) > tol:
            continue
        if a[k] >= b[k]:
            continue
        edges.append((int(a[k]), int(b[k]), rotations[a[k]] @ pred[k]))
    seen = _reachable(len(times), edges)
    if not edges:
        return {"edges": 0, "reachable": 1}
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    mask = np.zeros(len(times), dtype=bool)
    mask[list(seen)] = True

    def rmse(sel):
        if not np.any(sel):
            return None
        err = path[sel] - positions[sel]
        return round(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) * 1000.0), 1)

    return {
        "edges": len(edges),
        "reachable": int(mask.sum()),
        "reachable_mm": rmse(mask),
        "all_mm": rmse(np.ones(len(times), dtype=bool)),
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, imu = {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        acc = torch.from_numpy(built[name]["acc"]).to(device)
        gyro = torch.from_numpy(built[name]["gyro"]).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(built[name]["bg"], device=device),
                torch.tensor(built[name]["ba"], device=device),
                torch.tensor(built[name]["R0"], device=device),
            )
        imu[name] = dp.cpu().numpy()
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        data = np.load(GT / f"{name}.npz")
        delay = float(data["delay_usb"][0])
        ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
        times = data["t"][ok] - delay
        pred = np.concatenate([imu[name], np.ones((len(imu[name]), 1))], 1) @ coef
        report[name] = {}
        for tol in (0.03, 0.08, 0.15):
            report[name][str(tol)] = _eval(
                times, data["R"][ok], data["p"][ok],
                built[name]["t_start"], built[name]["t_end"], pred, tol,
            )
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "loose_match.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
