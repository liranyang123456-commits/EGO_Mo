#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stricter versions of the initial-velocity proxy analysis.

The published proxy is the mean reference velocity over [t_a-2, t_b+2], which
contains the target displacement itself, and its affine correction is fitted
and scored on the same validation pairs. This script adds
  * a context-only proxy: velocity from [t_a-2, t_a] and [t_b, t_b+2] only;
  * an oracle initial-velocity term v(t_a)*dt of the decomposition, with
    v(t_a) from the reference over t_a +- 0.3 s (reported separately, since it
    needs reference frames next to t_a);
  * a cross-session fit: the affine map is fitted on one validation recording
    and applied to the other.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.make_paper_figures import _gt  # noqa: E402

DATA = ROOT / "datasets"


def _vel(t, pg, us, t0, t1):
    idx = us[(t[us] >= t0) & (t[us] <= t1)]
    if len(idx) < 2 or t[idx[-1]] - t[idx[0]] < 0.6 * (t1 - t0):
        return None
    return (pg[idx[-1]] - pg[idx[0]]) / (t[idx[-1]] - t[idx[0]])


def _score(X, e, groups):
    base = float(np.linalg.norm(e, axis=1).mean())
    A = np.column_stack((X, np.ones(len(X))))
    coef = np.linalg.lstsq(A, e, rcond=None)[0]
    in_sample = float(np.linalg.norm(e - A @ coef, axis=1).mean())
    cross = np.empty_like(e)
    for g in np.unique(groups):
        fit, test = groups != g, groups == g
        c = np.linalg.lstsq(A[fit], e[fit], rcond=None)[0]
        cross[test] = e[test] - A[test] @ c
    cross_err = float(np.linalg.norm(cross, axis=1).mean())
    return {
        "n": int(len(e)), "mean_error_mm": base,
        "per_axis_correlation": [float(np.corrcoef(X[:, i], e[:, i])[0, 1]) for i in range(3)],
        "in_sample_mm": in_sample,
        "in_sample_reduction_percent": 100 * (1 - in_sample / base),
        "cross_session_mm": cross_err,
        "cross_session_reduction_percent": 100 * (1 - cross_err / base),
        "mean_proxy_magnitude_mm": float(np.linalg.norm(X, axis=1).mean()),
    }


def main(eval_npz="physnet_v1/eval_wd1.npz"):
    ev = np.load(DATA / eval_npz, allow_pickle=True)
    pred, y, pairs, t_pair, sess = (ev["val_pred"], ev["val_y"], ev["val_pair"],
                                    ev["val_t_pair"], ev["val_session"])
    err = (pred - y) * 1000
    full, ctx, keep, init, keep_init = [], [], [], [], []
    for k in range(len(y)):
        t, Rg, pg, us = _gt(str(sess[k]))
        a = pairs[k][0]
        ta, tb = t_pair[k]
        dt = tb - ta
        v_full = _vel(t, pg, us, ta - 2.0, tb + 2.0)
        v_pre, v_post = _vel(t, pg, us, ta - 2.0, ta), _vel(t, pg, us, tb, tb + 2.0)
        if v_full is not None and v_pre is not None and v_post is not None:
            full.append(Rg[a].T @ v_full * dt * 1000)
            ctx.append(Rg[a].T @ (0.5 * (v_pre + v_post)) * dt * 1000)
            keep.append(k)
        v_init = _vel(t, pg, us, ta - 0.3, ta + 0.3)
        if v_init is not None:
            init.append(Rg[a].T @ v_init * dt * 1000)
            keep_init.append(k)
    keep, keep_init = np.asarray(keep), np.asarray(keep_init)
    groups = np.asarray([str(s) for s in sess])
    report = {
        "model": eval_npz, "variants": {
            "full": _score(np.asarray(full), err[keep], groups[keep]),
            "context": _score(np.asarray(ctx), err[keep], groups[keep]),
            "oracle_initial": _score(np.asarray(init), err[keep_init], groups[keep_init]),
        },
        "oracle_reference_displacement_mm": float(np.linalg.norm(y[keep_init] * 1000, axis=1).mean()),
    }
    out = DATA / "velocity_proxy_check.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
