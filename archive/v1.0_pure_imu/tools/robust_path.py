#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Down-weight displacement edges that disagree with the fused path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _nearest, _nodes
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


def _solve(n, edges):
    dim = (n - 1) * 3
    ata = np.zeros((dim, dim))
    atb = np.zeros(dim)
    for a, b, meas, w in edges:
        if a == 0 and b == 0:
            continue
        row_b = None if b == 0 else (b - 1) * 3
        row_a = None if a == 0 else (a - 1) * 3
        if row_b is not None:
            ata[row_b:row_b + 3, row_b:row_b + 3] += w * np.eye(3)
            atb[row_b:row_b + 3] += w * meas
        if row_a is not None:
            ata[row_a:row_a + 3, row_a:row_a + 3] += w * np.eye(3)
            atb[row_a:row_a + 3] -= w * meas
        if row_a is not None and row_b is not None:
            ata[row_a:row_a + 3, row_b:row_b + 3] -= w * np.eye(3)
            ata[row_b:row_b + 3, row_a:row_a + 3] -= w * np.eye(3)
    delta = np.linalg.solve(ata + np.eye(dim) * 1e-6, atb)
    path = np.zeros((n, 3))
    path[1:] = delta.reshape(n - 1, 3)
    return path


def _edges(name, pack, pred):
    times, rotations, positions = _nodes(name)
    a = _nearest(times, pack["t_start"])
    b = _nearest(times, pack["t_end"])
    edges = []
    for k in range(len(pred)):
        if abs(times[a[k]] - pack["t_start"][k]) > 0.03 or abs(times[b[k]] - pack["t_end"][k]) > 0.03:
            continue
        if a[k] >= b[k]:
            continue
        edges.append([int(a[k]), int(b[k]), rotations[a[k]] @ pred[k], 1.0])
    return times, positions, edges


def _rmse(path, positions):
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def _irls(n, positions, edges):
    path = _solve(n, edges)
    path = path - path[0] + positions[0]
    base = _rmse(path, positions)
    for _ in range(4):
        resid = []
        for a, b, meas, _w in edges:
            resid.append(np.linalg.norm((path[b] - path[a]) - meas))
        scale = max(float(np.median(resid)), 1e-4)
        for edge, r in zip(edges, resid):
            edge[3] = 1.0 if r <= scale else scale / r
        path = _solve(n, edges)
        path = path - path[0] + positions[0]
    return base, _rmse(path, positions)


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
        step = float(np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0)
        _t, positions, edges = _edges(name, built[name], pred)
        equal, robust = _irls(len(_t), positions, edges)
        report[name] = {
            "step_mm": round(step, 2),
            "fused_mm": round(equal, 1),
            "robust_mm": round(robust, 1),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "robust_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
