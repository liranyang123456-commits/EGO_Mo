#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare baselines, zero-shot, fine-tuned and blended models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
from tools.train_hybrid import _synthetic_build

DATA = ROOT / "datasets"


def _ensemble(paths, device):
    models = []
    for path in paths:
        net = api.SeqNet().to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        models.append(net)
    return models


def _predict(models, X, device):
    return np.mean([api._predict(model, X, device) * 0.01 for model in models], axis=0)


def _strapdown(X):
    out = np.zeros((len(X), 3), np.float32)
    dt = 1.0 / api.HZ
    for i, x in enumerate(X):
        mask = x[:, 6] > 0.5
        acc = x[mask, :3] * 9.80665
        v = np.zeros(3)
        p = np.zeros(3)
        for a in acc:
            p += v * dt + 0.5 * a * dt * dt
            v += a * dt
        out[i] = p
    return out


def _summary(X):
    features = []
    dt = 1.0 / api.HZ
    for x in X:
        mask = x[:, 6] > 0.5
        acc = x[mask, :3] * 9.80665
        gyro = x[mask, 3:6] * 100.0
        v = np.zeros(3)
        p = np.zeros(3)
        for a in acc:
            p += v * dt + 0.5 * a * dt * dt
            v += a * dt
        features.append(np.concatenate((
            acc.mean(0), acc.std(0), gyro.mean(0), gyro.std(0), p,
            [mask.sum() * dt],
        )))
    return np.asarray(features)


def _ridge_fit(X, y, lam=1e-2):
    f = _summary(X)
    mean, std = f.mean(0), f.std(0) + 1e-6
    z = (f - mean) / std
    design = np.column_stack((z, np.ones(len(z))))
    reg = np.eye(design.shape[1]) * lam
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ y)
    return mean, std, coef


def _ridge_predict(model, X):
    mean, std, coef = model
    z = (_summary(X) - mean) / std
    return np.column_stack((z, np.ones(len(z)))) @ coef


def _metrics(pred, y):
    return api._metrics(pred.astype(np.float32), y.astype(np.float32))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=DATA / "method_benchmark_20260924.json")
    args = ap.parse_args()
    api.LABELS = "pose_gt_raw"
    api.TARGET_S = 3.0
    context = 2.0
    length = int(round((1.5 * api.TARGET_S + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))

    def real(group):
        parts = [api.build(name, context, length) for name in split[group]]
        return (
            np.concatenate([part[0] for part in parts if part is not None]),
            np.concatenate([part[1] for part in parts if part is not None]),
        )

    real_train = real("train")
    real_val = real("val")
    real_test = real("test")
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(123)

    def synthetic(group):
        parts = [
            _synthetic_build(args.corpus / entry["path"], context, length, 3.0, 160, rng)
            for entry in manifest["imu"][group]
        ]
        return (
            np.concatenate([part[0] for part in parts if part is not None]),
            np.concatenate([part[1] for part in parts if part is not None]),
        )

    syn_test = synthetic("test")
    syn_ood = synthetic("ood")
    seeds = range(5)
    real_models = _ensemble(
        [DATA / "traj_run_v14" / f"seq_ctx2_h3_s{s}.pt" for s in seeds],
        device,
    )
    hybrid_models = _ensemble(
        [DATA / "traj_run_v17" / f"seq_ctx2_h3_r0_p35_lr0.001_s{s}.pt" for s in seeds],
        device,
    )
    zero_models = _ensemble(
        [DATA / "traj_run_v19" / f"pretrained_ctx2_h3_p35_preonly_s{s}.pt" for s in seeds],
        device,
    )
    ridge = _ridge_fit(*real_train)

    methods = {}
    for dataset_name, (X, y) in {
        "synthetic_test": syn_test,
        "synthetic_ood": syn_ood,
        "real_validation": real_val,
        "real_test": real_test,
    }.items():
        pred_real = _predict(real_models, X, device)
        pred_hybrid = _predict(hybrid_models, X, device)
        predictions = {
            "zero_motion": np.zeros_like(y),
            "strapdown_zero_velocity": _strapdown(X),
            "ridge_real_fitted": _ridge_predict(ridge, X),
            "synthetic_zero_shot": _predict(zero_models, X, device),
            "real_only_network": pred_real,
            "synthetic_pretrain_real_finetune": pred_hybrid,
            "validation_blend": 0.2 * pred_real + 0.8 * pred_hybrid,
        }
        methods[dataset_name] = {
            name: _metrics(pred, y)
            for name, pred in predictions.items()
        }
        print(json.dumps({dataset_name: methods[dataset_name]}, ensure_ascii=False), flush=True)
    payload = {
        "protocol": {
            "horizon_s": 3.0,
            "context_each_side_s": 2.0,
            "real_test_sealed_during_selection": True,
            "public_model_note": (
                "RoNIN/TLIO published numbers are not inserted as direct scores: "
                "their public weights target pedestrian motion and incompatible frames. "
                "synthetic_zero_shot is the direct-run comparison under this sensor protocol."
            ),
        },
        "datasets": methods,
    }
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
