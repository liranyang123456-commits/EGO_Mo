#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gated + non-gated PhysNet mixture from the existing fold models.

Leave-one-session-out error from the saved out-of-fold predictions of the cvg
and cvx fold models, and historical held-out-test error and ATE from the saved
CV-ensemble predictions. No model is retrained and no sealed recording is read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "datasets"
TEST = "traj_20260923_023422"


def cv_error(tags, sessions):
    tot, n = 0.0, 0
    for ses in sessions:
        preds, y = [], None
        for tag in tags:
            for s in (0, 1):
                z = np.load(DATA / "physnet_cv" / f"pred_{tag}_{ses}_s{s}.npz")
                preds.append(z["val_pred"])
                y = z["val_y"]
        e = np.linalg.norm(np.mean(preds, axis=0) - y, axis=1) * 1000
        tot, n = tot + e.sum(), n + len(e)
    return float(tot / n)


def main():
    import tools.train_seq as api
    import tools.train_physnet as phys
    from tools.eval_session_cv import trajectory_ate
    from tools.recheck_metrics import metrics
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    sessions = split["train"] + split["val"]
    out = {"cv_mm": {"gate": cv_error(["cvg"], sessions), "nogate": cv_error(["cvx"], sessions),
                     "mix": cv_error(["cvg", "cvx"], sessions)}}
    g = np.load(DATA / "session_cv_gate_benchmark.npz")
    n = np.load(DATA / "session_cv_benchmark.npz")
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    length = int(round((1.5 * 3.0 + 2 * 2.0) * api.HZ))
    data = phys.build_group([TEST], 2.0, 3.0, length)
    mix = 0.5 * (g["physnet"] + n["physnet"])
    out["test"] = {}
    for name, p in (("gate", g["physnet"]), ("nogate", n["physnet"]), ("mix", mix),
                    ("imunet", g["imunet"])):
        out["test"][name] = metrics(p, g["y"]) | {
            "ate_mm": float(trajectory_ate(TEST, data, p, None)["single_edge"]["ate_rmse_mm"])}
    (DATA / "physnet_v3" / "mixture.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
