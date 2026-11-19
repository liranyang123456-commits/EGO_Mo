#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Down-weight fast steps when fusing the path. The step metric stays unweighted."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import MotionTrajectoryCorrector
from tools.eval_contiguous import _fit, _predict
from tools.fuse_path import _nearest, _nodes
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split

GT = ROOT / "datasets" / "pose_gt"


def _solve(n: int, edges: list[tuple[int, int, np.ndarray, float]]) -> np.ndarray:
    dim = (n - 1) * 3
    ata = np.zeros((dim, dim))
    atb = np.zeros(dim)
    for a, b, meas, weight in edges:
        if a == 0 and b == 0:
            continue
        w = float(weight)
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


def _pairs(pack, pred, times, rotations, positions):
    a = _nearest(times, pack["t_start"])
    b = _nearest(times, pack["t_end"])
    gyro = np.linalg.norm(pack["gyro"].mean(axis=1), axis=1)
    rows = []
    for k in range(len(pred)):
        if abs(times[a[k]] - pack["t_start"][k]) > 0.03:
            continue
        if abs(times[b[k]] - pack["t_end"][k]) > 0.03:
            continue
        if a[k] >= b[k]:
            continue
        truth = rotations[a[k]].T @ (positions[b[k]] - positions[a[k]])
        board = rotations[a[k]] @ pred[k]
        rows.append((int(a[k]), int(b[k]), pred[k], truth, board, float(gyro[k])))
    return rows


def _var_lookup(rows):
    speed = np.array([np.linalg.norm(pred - truth) for _a, _b, pred, truth, _board, _g in rows])
    gyro = np.array([g for *_rest, g in rows])
    edges = np.quantile(gyro, [0.0, 0.33, 0.66, 1.0])
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    bins = np.digitize(gyro, edges[1:-1])
    table = []
    for b in range(3):
        sel = speed[bins == b]
        table.append(float(np.var(sel)) if len(sel) > 5 else float(np.var(speed)))
    return edges, np.array(table, dtype=np.float64)


def _weight(gyro: float, edges, table) -> float:
    bins = np.digitize([gyro], edges[1:-1])[0]
    return 1.0 / max(table[bins], 1e-8)


def _fused(rows, weights, n, positions):
    edges = [(a, b, board, w) for (a, b, _pred, _truth, board, _g), w in zip(rows, weights)]
    path = _solve(n, edges)
    path = path - path[0] + positions[0]
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = load_split(names)
    built = {name: build_pairs(packs[name]) for name in names}
    train_names = [name for name in names if name not in (TEST_NAME, VAL_NAME)]
    coef = _fit([built[name] for name in train_names], device)
    net = MotionTrajectoryCorrector().to(device)
    ckpt = torch.load(ROOT / "datasets" / "traj_run_v3" / "corrector.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    train_rows = []
    cached = {}
    for name in names:
        _pred, base, _dR = _predict(net, built[name], coef, device)
        times, rotations, positions = _nodes(name)
        rows = _pairs(built[name], base, times, rotations, positions)
        cached[name] = (rows, times, positions)
        if name in train_names:
            train_rows.extend(rows)
    edges, table = _var_lookup(train_rows)
    report = {"gyro_edges_dps": [round(float(v) * 180.0 / np.pi, 2) for v in edges], "var_m2": [float(v) for v in table]}
    for name in (VAL_NAME, TEST_NAME):
        rows, times, positions = cached[name]
        ones = np.ones(len(rows))
        weights = np.array([_weight(row[-1], edges, table) for row in rows])
        step = float(np.mean([np.linalg.norm(pred - truth) for _a, _b, pred, truth, _board, _g in rows]) * 1000.0)
        report[name] = {
            "step_mm": round(step, 2),
            "fused_equal_mm": round(_fused(rows, ones, len(times), positions), 1),
            "fused_weighted_mm": round(_fused(rows, weights, len(times), positions), 1),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "weighted_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
