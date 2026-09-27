#!/usr/bin/env python3
"""Validation-select a blend of real-only and synthetic-pretrained ensembles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def metrics(pred, y, rng):
    err = np.linalg.norm(pred - y, axis=1) * 1000
    zero = np.linalg.norm(y, axis=1) * 1000
    gain = zero - err
    resid = np.sum((pred - y) ** 2)
    base = np.sum((y - y.mean(0)) ** 2)
    fast = zero >= np.quantile(zero, 0.67)
    draws = [
        float(gain[rng.integers(0, len(gain), len(gain))].mean())
        for _ in range(5000)
    ]
    return {
        "n": len(y),
        "zero_mm": round(float(zero.mean()), 2),
        "err_mm": round(float(err.mean()), 2),
        "fast_zero_mm": round(float(zero[fast].mean()), 2),
        "fast_mm": round(float(err[fast].mean()), 2),
        "r2": round(float(1 - resid / max(base, 1e-12)), 3),
        "gain_mm": round(float(gain.mean()), 2),
        "gain_percent": round(float(100 * gain.mean() / zero.mean()), 1),
        "gain_95ci_mm": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
    }


def main() -> None:
    real = np.load(DATA / "traj_run_v14" / "final_ensemble_ctx2_h3.npz")
    hybrid = np.load(DATA / "traj_run_v17" / "final_hybrid_predictions.npz")
    if not np.allclose(real["val_y"], hybrid["real_val_y"]):
        raise RuntimeError("validation labels do not align")
    candidates = np.linspace(0, 1, 201)
    losses = []
    for alpha in candidates:
        pred = (1 - alpha) * real["val_pred"] + alpha * hybrid["real_val_pred"]
        losses.append(float(np.linalg.norm(pred - real["val_y"], axis=1).mean()))
    alpha = float(candidates[int(np.argmin(losses))])
    rng = np.random.default_rng(0)
    val_pred = (1 - alpha) * real["val_pred"] + alpha * hybrid["real_val_pred"]
    test_pred = (1 - alpha) * real["test_pred"] + alpha * hybrid["real_test_pred"]
    report = {
        "alpha_hybrid": alpha,
        "alpha_real_only": 1 - alpha,
        "selection": "alpha selected only by aggregate real validation mean error",
        "val": metrics(val_pred, real["val_y"], rng),
        "test": metrics(test_pred, real["test_y"], rng),
    }
    out = DATA / "traj_run_v17"
    (out / "final_blended_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez(
        out / "final_blended_predictions.npz",
        val_pred=val_pred,
        val_y=real["val_y"],
        test_pred=test_pred,
        test_y=real["test_y"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
