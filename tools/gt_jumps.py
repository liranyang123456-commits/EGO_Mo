#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Find the largest jump in the held-out chessboard trajectory."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NAME = "traj_20260923_023422"
GT = ROOT / "datasets" / "pose_gt" / f"{NAME}.npz"


def main() -> None:
    data = np.load(GT)
    ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
    t = data["t"][ok]
    p = data["p"][ok]
    t = t - t[0]
    dp = np.linalg.norm(np.diff(p, axis=0), axis=1)
    dt = np.diff(t)
    speed = dp / np.clip(dt, 1e-3, None)
    order = np.argsort(dp)[::-1][:8]
    jumps = []
    for i in order:
        jumps.append({
            "t_s": round(float(t[i]), 2),
            "dt_s": round(float(dt[i]), 3),
            "jump_mm": round(float(dp[i]) * 1000.0, 1),
            "dz_mm": round(float(p[i + 1, 2] - p[i, 2]) * 1000.0, 1),
        })
    # Error shape was flat after the first fifth. Report pose span in each fifth.
    quint = []
    for q in range(5):
        lo, hi = np.quantile(t, [q / 5.0, (q + 1) / 5.0])
        sel = (t >= lo) & (t <= hi if q == 4 else t < hi)
        if not np.any(sel):
            continue
        quint.append({
            "t0": round(float(lo), 1),
            "z_mm": [round(float(v) * 1000.0, 1) for v in (p[sel, 2].min(), p[sel, 2].max())],
            "span_mm": round(float(np.linalg.norm(p[sel].max(0) - p[sel].min(0))) * 1000.0, 1),
        })
    report = {
        "poses": int(ok.sum()),
        "duration_s": round(float(t[-1]), 1),
        "largest_jumps": jumps,
        "median_step_mm": round(float(np.median(dp)) * 1000.0, 2),
        "quintiles": quint,
        "fast_steps": int(np.sum(speed > 0.5)),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    out = ROOT / "datasets" / "traj_run_v3" / "gt_jumps.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
