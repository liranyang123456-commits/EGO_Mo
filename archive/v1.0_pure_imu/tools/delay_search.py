#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pick an extra IMU time shift on the training sessions only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _nearest, _nodes, _solve
from tools.prop_integrate import _body_dp, _propagate
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split

GT = ROOT / "datasets" / "pose_gt"


def _feat(traj, usb_t, t0, t1, shift):
    cols = []
    for a, b in zip(t0, t1):
        dp = _body_dp(traj[0], traj[1], usb_t, float(a) + shift, float(b) + shift)
        cols.append(np.zeros(3) if dp is None else dp)
    return np.stack(cols)


def _step(pred, gt):
    return float(np.linalg.norm(pred - gt, axis=1).mean() * 1000.0)


def _fused(name, pack, pred):
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
    if not edges:
        return None
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {name: build_pairs(raw[name]) for name in names}
    traj = {
        name: _propagate(raw[name]["usb_t"], raw[name]["usb"], raw[name]["bg"], raw[name]["ba"], raw[name]["R0"])
        for name in names
    }
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    best = None
    for shift in np.linspace(-0.04, 0.04, 9):
        feats = {name: _feat(traj[name], raw[name]["usb_t"], built[name]["t_start"], built[name]["t_end"], shift) for name in names}
        src = np.concatenate([feats[n] for n in train])
        dst = np.concatenate([built[n]["dp"] for n in train])
        coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)
        err = np.mean([
            _step(np.concatenate([feats[n], np.ones((len(feats[n]), 1))], 1) @ coef, built[n]["dp"])
            for n in train
        ])
        if best is None or err < best[0]:
            best = (err, float(shift), coef, feats)
        print(json.dumps({"shift_ms": round(float(shift) * 1000.0, 1), "train_step_mm": round(float(err), 2)}), flush=True)
    _err, shift, coef, feats = best
    report = {"shift_ms": round(shift * 1000.0, 1), "train_step_mm": round(float(_err), 2)}
    for name in (VAL_NAME, TEST_NAME):
        pred = np.concatenate([feats[name], np.ones((len(feats[name]), 1))], 1) @ coef
        report[name] = {
            "step_mm": round(_step(pred, built[name]["dp"]), 2),
            "fused_mm": round(_fused(name, built[name], pred), 1),
        }
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "delay_search.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
