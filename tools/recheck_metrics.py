#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent re-computation of the held-out test metrics from saved predictions.

Re-implements error, multivariate R^2, fastest-third error and the paired 3-s
block bootstrap without importing the training code, and compares them with
the stored benchmark JSON files.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "datasets"
KEYS = {"physnet": "physnet_cv_ensemble", "imunet": "imunet_cv_ensemble",
        "equal_blend": "equal_blend_physnet_imunet", "cv_calibrated_blend": "cv_calibrated_blend"}


def metrics(p, y):
    e = np.linalg.norm(p - y, axis=1) * 1000
    mag = np.linalg.norm(y, axis=1) * 1000
    fast = mag >= np.quantile(mag, 0.67)
    r2 = 1 - ((p - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()
    return {"err_mm": round(float(e.mean()), 2), "r2": round(float(r2), 3),
            "fast_mm": round(float(e[fast].mean()), 2)}


def boot(ea, eb, t, n=4000, seed=1):
    """95% CI of mean(eb - ea) in mm; positive means method a is better."""
    rng = np.random.default_rng(seed)
    b = np.floor((t - t.min()) / 3.0).astype(int)
    ids = np.unique(b)
    d = eb - ea
    out = []
    for _ in range(n):
        pick = rng.choice(ids, len(ids))
        sel = np.concatenate([np.flatnonzero(b == i) for i in pick])
        out.append(d[sel].mean())
    return [round(float(v) * 1000, 2) for v in np.quantile(out, [0.025, 0.975])]


def main():
    report = {}
    for name in ("session_cv_benchmark", "session_cv_gate_benchmark"):
        z = np.load(DATA / f"{name}.npz")
        test = json.loads((DATA / f"{name}.json").read_text(encoding="utf-8"))["test"]
        y, t = z["y"], z["t_pair"][:, 0]
        rows = {"zero_motion": metrics(np.zeros_like(y), y)}
        mismatch = []
        for k, jk in KEYS.items():
            rows[k] = metrics(z[k], y)
            js = test.get(jk, {})
            for f in ("err_mm", "r2", "fast_mm"):
                if f in js and abs(js[f] - rows[k][f]) > (0.011 if f != "r2" else 0.0011):
                    mismatch.append((k, f, rows[k][f], js[f]))
        err = {k: np.linalg.norm(z[k] - y, axis=1) for k in KEYS}
        zero = np.linalg.norm(y, axis=1)
        ci = {
            "physnet_vs_imunet": boot(err["physnet"], err["imunet"], t),
            "calibrated_blend_vs_imunet": boot(err["cv_calibrated_blend"], err["imunet"], t),
            "equal_blend_vs_imunet": boot(err["equal_blend"], err["imunet"], t),
            "physnet_vs_zero": boot(err["physnet"], zero, t),
            "imunet_vs_zero": boot(err["imunet"], zero, t),
            "equal_blend_vs_zero": boot(err["equal_blend"], zero, t),
            "calibrated_blend_vs_zero": boot(err["cv_calibrated_blend"], zero, t),
        }
        report[name] = {"n": int(len(y)), "metrics": rows, "ci_mm": ci,
                        "stored_ci_mm": {k: test.get(k) for k in test if "95ci" in k},
                        "mismatches_vs_json": mismatch}
    out = DATA / "recheck_metrics.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
