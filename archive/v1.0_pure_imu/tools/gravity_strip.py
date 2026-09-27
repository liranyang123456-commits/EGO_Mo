#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove the gravity axis from the integrated displacement and refit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _nearest, _nodes, _solve
from tools.gap_map import _gaps
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _strip(dp, gravity):
    g = gravity / max(np.linalg.norm(gravity), 1e-8)
    return dp - np.outer(dp @ g, g)


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
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, imu, gravity = {}, {}, {}
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
        # Session gravity in the integration frame: R0 maps body to world, world up is +z.
        gravity[name] = raw[name]["R0"].T @ np.array([0.0, 0.0, 1.0])
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    cached = {name: _gaps(name, raw[name], device) for name in (VAL_NAME, TEST_NAME)}
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        src, dst, *_rest = cached[name]
        g = gravity[name]
        raw_dp = src[:, :3]
        ang_raw, ang_g_pred, ang_g_truth = [], [], []
        for dp, truth in zip(raw_dp, dst):
            ang_raw.append(_angle(dp, truth))
            ang_g_pred.append(_angle(dp, g))
            ang_g_truth.append(_angle(truth, g))
        report[name] = {
            "raw_vs_truth_deg": round(float(np.nanmedian(ang_raw)), 1),
            "raw_vs_gravity_deg": round(float(np.nanmedian(ang_g_pred)), 1),
            "truth_vs_gravity_deg": round(float(np.nanmedian(ang_g_truth)), 1),
        }
    src = np.concatenate([_strip(imu[n], gravity[n]) for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)
    for name in (VAL_NAME, TEST_NAME):
        feat = _strip(imu[name], gravity[name])
        pred = np.concatenate([feat, np.ones((len(feat), 1))], 1) @ coef
        report[name]["step_mm"] = round(float(np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0), 2)
        report[name]["fused_mm"] = round(_fused(name, built[name], pred), 1)
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "gravity_strip.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
