#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Path error on the poses that 0.2 s edges actually reach."""

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
        pred = np.concatenate([imu[name], np.ones((len(imu[name]), 1))], 1) @ coef
        times, rotations, positions = _nodes(name)
        a = _nearest(times, built[name]["t_start"])
        b = _nearest(times, built[name]["t_end"])
        edges = []
        for k in range(len(pred)):
            if abs(times[a[k]] - built[name]["t_start"][k]) > 0.03 or abs(times[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            edges.append((int(a[k]), int(b[k]), rotations[a[k]] @ pred[k]))
        seen = _reachable(len(times), edges)
        path = _solve(len(times), edges)
        path = path - path[0] + positions[0]
        mask = np.zeros(len(times), dtype=bool)
        mask[list(seen)] = True
        def rmse(sel):
            if not np.any(sel):
                return None
            err = path[sel] - positions[sel]
            return round(float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) * 1000.0), 1)
        cut = int(np.argmin(mask)) if not mask.all() else len(mask)
        # first False after a True, else first False
        missing = np.flatnonzero(~mask)
        cut = int(missing[0]) if len(missing) else len(mask)
        report[name] = {
            "poses": int(len(times)),
            "reachable": int(mask.sum()),
            "reachable_mm": rmse(mask),
            "unreachable_mm": rmse(~mask),
            "cut_t_s": None if cut >= len(times) else round(float(times[cut] - times[0]), 1),
            "all_mm": rmse(np.ones(len(times), dtype=bool)),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "reachable_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
