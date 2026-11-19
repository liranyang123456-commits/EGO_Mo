#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-point pose-graph ATE of the CV-calibrated blends on both test recordings.

The stored per-window predictions of the fold ensembles
(datasets/session_cv_matched_{cvx,cvg}.npz) already contain the calibrated
blend, but its trajectory error had only been solved for the non-gated blend
on the first held-out recording. This script solves the same end-point graph
(one edge per scored window) for every stored predictor, so the calibrated
blends get a trajectory error on both recordings, with and without the gate.
The other predictors are solved as a check against the stored values.

    python tools/calibrated_blend_ate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
from tools.build_trajectory_comparison import _metrics, _solve_positions

DATA = ROOT / "datasets"
SESSIONS = {"held_out": "traj_20260923_023422", "t1": "traj_20260926_143837"}
KEYS = ("physnet", "imunet", "equal_blend", "cv_calibrated_blend")


def main() -> None:
    out = {}
    for name, ses in SESSIONS.items():
        _t, R, p, usable = phys._load_gt(ses)
        node = {int(f): i for i, f in enumerate(usable)}
        out[name] = {"recording": ses}
        for tag in ("cvx", "cvg"):
            z = np.load(DATA / f"session_cv_matched_{tag}.npz", allow_pickle=True)
            edges = np.array([(node[int(a)], node[int(b)]) for a, b in z[f"{ses}__pair"]], np.int64)
            y = z[f"{ses}__y"]
            res = {}
            for key in KEYS:
                pred = z[f"{ses}__{key}"].astype(np.float64)
                traj = _solve_positions(p[usable], R[usable], edges, pred)
                res[key] = {"err_mm": round(float(np.linalg.norm(pred - y, axis=1).mean() * 1000), 2),
                            **_metrics(traj, p[usable])}
            out[name][tag] = {"edges": int(len(edges)), **res}
    (DATA / "calibrated_blend_ate.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for name, v in out.items():
        for tag in ("cvx", "cvg"):
            print(name, tag, {k: (r["err_mm"], r["ate_rmse_mm"]) for k, r in v[tag].items() if isinstance(r, dict)})


if __name__ == "__main__":
    main()
