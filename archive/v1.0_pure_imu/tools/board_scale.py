#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-axis scale in the chessboard frame, fit on the training sessions only."""

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


def _rows(name, pack, pred):
    times, rotations, positions = _nodes(name)
    a = _nearest(times, pack["t_start"])
    b = _nearest(times, pack["t_end"])
    rows = []
    for k in range(len(pred)):
        if abs(times[a[k]] - pack["t_start"][k]) > 0.03 or abs(times[b[k]] - pack["t_end"][k]) > 0.03:
            continue
        if a[k] >= b[k]:
            continue
        meas = rotations[a[k]] @ pred[k]
        truth = positions[b[k]] - positions[a[k]]
        rows.append((int(a[k]), int(b[k]), meas, truth, rotations[a[k]], pred[k]))
    return times, positions, rows


def _scale(rows):
    meas = np.stack([row[2] for row in rows])
    truth = np.stack([row[3] for row in rows])
    scale = np.ones(3)
    for axis in range(3):
        denom = float(np.dot(meas[:, axis], meas[:, axis]))
        if denom > 1e-8:
            scale[axis] = float(np.dot(meas[:, axis], truth[:, axis]) / denom)
    return scale


def _score(name, pack, pred, scale):
    times, positions, rows = _rows(name, pack, pred)
    step = []
    edges = []
    for a, b, meas, truth, rotation, body in rows:
        corrected = meas * scale
        step.append(np.linalg.norm(corrected - truth))
        edges.append((a, b, corrected))
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    rmse = float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)
    return {
        "step_mm": round(float(np.mean(step) * 1000.0), 2),
        "fused_mm": round(rmse, 1),
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
    train_rows = []
    pred = {}
    for name in names:
        pred[name] = np.concatenate([imu[name], np.ones((len(imu[name]), 1))], 1) @ coef
        if name in train:
            _t, _p, rows = _rows(name, built[name], pred[name])
            train_rows.extend(rows)
    scale = _scale(train_rows)
    report = {"board_scale": [round(float(v), 4) for v in scale]}
    for name in (VAL_NAME, TEST_NAME):
        raw_score = _score(name, built[name], pred[name], np.ones(3))
        scaled = _score(name, built[name], pred[name], scale)
        report[name] = {"raw": raw_score, "scaled": scaled}
        print(json.dumps({name: report[name], "scale": report["board_scale"]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "board_scale.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
