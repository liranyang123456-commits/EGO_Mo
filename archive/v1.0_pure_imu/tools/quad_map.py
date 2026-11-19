#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train-only speed term on top of the linear displacement map."""

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


def _design(src: np.ndarray, kind: str) -> np.ndarray:
    ones = np.ones((len(src), 1))
    if kind == "linear":
        return np.concatenate([src, ones], axis=1)
    scale = np.linalg.norm(src, axis=1, keepdims=True)
    return np.concatenate([src, scale * src, ones], axis=1)


def _fit(src, dst, kind):
    coef, *_ = np.linalg.lstsq(_design(src, kind), dst, rcond=None)
    return coef


def _apply(coef, src, kind):
    return _design(src, kind) @ coef


def _step(a, b):
    return float(np.linalg.norm(a - b, axis=1).mean() * 1000.0)


def _fused(pack, pred):
    times, rotations, positions = _nodes(pack["name"])
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


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {}
    imu = {}
    for name in names:
        built[name] = build_pairs(raw[name])
        built[name]["name"] = name
        imu[name] = _imu(built[name], device)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    report = {}
    for kind in ("linear", "quad"):
        coef = _fit(src, dst, kind)
        report[kind] = {}
        for name in (VAL_NAME, TEST_NAME):
            pred = _apply(coef, imu[name], kind)
            report[kind][name] = {
                "step_mm": round(_step(pred, built[name]["dp"]), 2),
                "fused_mm": round(_fused(built[name], pred), 1),
            }
        print(json.dumps({kind: report[kind]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "quad_map.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
