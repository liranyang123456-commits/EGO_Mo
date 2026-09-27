#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-recording rotation from the first fast steps, scored on the rest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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


def _kabsch(src, dst):
    h = src.T @ dst
    u, _s, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1] *= -1
        r = vt.T @ u.T
    return r


def _score(name, src, dst):
    speed = np.linalg.norm(dst, axis=1)
    order = np.flatnonzero(speed > np.quantile(speed, 0.5))
    take = order[:40]
    rest = np.setdiff1d(np.arange(len(dst)), take)
    rot = _kabsch(src[take], dst[take])
    pred = src @ rot.T
    err = np.linalg.norm(pred[rest] - dst[rest], axis=1)
    base = np.linalg.norm(dst[rest], axis=1)
    fast = np.linalg.norm(dst[rest], axis=1) >= np.quantile(speed, 0.67)
    return {
        "held_out": int(len(rest)),
        "zero_mm": round(float(base.mean() * 1000.0), 2),
        "aligned_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2) if np.any(fast) else None,
    }


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        pack = build_pairs(raw[name])
        acc = torch.from_numpy(pack["acc"]).to(device)
        gyro = torch.from_numpy(pack["gyro"]).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(pack["bg"], device=device),
                torch.tensor(pack["ba"], device=device),
                torch.tensor(pack["R0"], device=device),
            )
        report[name] = _score(name, dp.cpu().numpy(), pack["dp"])
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v5" / "within_session.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
