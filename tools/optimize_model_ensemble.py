#!/usr/bin/env python3
"""Validation-only nonnegative blend across adapted public and proposed models."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
from tools.train_external_architectures import make_model, predict

DATA = ROOT / "datasets"


def _ours(paths, X, device):
    models = []
    for path in paths:
        net = api.SeqNet().to(device)
        net.load_state_dict(torch.load(path, map_location=device))
        net.eval()
        models.append(net)
    return np.mean([api._predict(net, X, device) * 0.01 for net in models], axis=0)


def _external(arch, suffix, X, device):
    model = make_model(arch).to(device)
    model.load_state_dict(torch.load(
        DATA / "external_benchmark" / f"{arch}_{suffix}_s0.pt",
        map_location=device,
    ))
    model.eval()
    return predict(model, X, device)


def _external_ensemble(arch, suffix, seeds, X, device):
    outputs = []
    for seed in seeds:
        model = make_model(arch).to(device)
        model.load_state_dict(torch.load(
            DATA / "external_benchmark" / f"{arch}_{suffix}_s{seed}.pt",
            map_location=device,
        ))
        model.eval()
        outputs.append(predict(model, X, device))
    return np.mean(outputs, axis=0)


def _metrics(pred, y):
    return api._metrics(pred.astype(np.float32), y.astype(np.float32))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--physnet-eval", type=Path, default=None,
                    help="eval_*.npz from tools/train_physnet.py eval --test; adds a physnet candidate")
    ap.add_argument("--tag", default="optimized_ensemble")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context = 2.0
    length = int(round((1.5 * 3.0 + 2 * context) * api.HZ))
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def build(group):
        parts = [api.build(name, context, length) for name in split[group]]
        return (
            np.concatenate([p[0] for p in parts]),
            np.concatenate([p[1] for p in parts]),
        )

    Xv, yv = build("val")
    Xt, yt = build("test")
    seeds = range(5)
    candidates = {
        "ours_real": (
            _ours([DATA / "traj_run_v14" / f"seq_ctx2_h3_s{s}.pt" for s in seeds], Xv, device),
            _ours([DATA / "traj_run_v14" / f"seq_ctx2_h3_s{s}.pt" for s in seeds], Xt, device),
        ),
        "ours_hybrid": (
            _ours([DATA / "traj_run_v17" / f"seq_ctx2_h3_r0_p35_lr0.001_s{s}.pt" for s in seeds], Xv, device),
            _ours([DATA / "traj_run_v17" / f"seq_ctx2_h3_r0_p35_lr0.001_s{s}.pt" for s in seeds], Xt, device),
        ),
        "ronin_fine": (
            _external("ronin_resnet", "fine", Xv, device),
            _external("ronin_resnet", "fine", Xt, device),
        ),
        "tlio_fine": (
            _external("tlio_resnet", "fine", Xv, device),
            _external("tlio_resnet", "fine", Xt, device),
        ),
        "imunet_real_single": (
            _external("imunet", "real", Xv, device),
            _external("imunet", "real", Xt, device),
        ),
        "imunet_real_ensemble": (
            _external_ensemble("imunet", "real", range(5), Xv, device),
            _external_ensemble("imunet", "real", range(5), Xt, device),
        ),
    }
    if args.physnet_eval is not None:
        phys = np.load(args.physnet_eval, allow_pickle=True)
        # Both builders enumerate pairs identically; assert before mixing predictions.
        assert np.allclose(phys["val_y"], yv, atol=1e-6) and np.allclose(phys["test_y"], yt, atol=1e-6)
        candidates["physnet"] = (phys["val_pred"].astype(np.float32), phys["test_pred"].astype(np.float32))
    names = list(candidates)
    Pv = np.stack([candidates[name][0] for name in names], axis=0)
    Pt = np.stack([candidates[name][1] for name in names], axis=0)

    def objective(w):
        pred = np.einsum("m,mni->ni", w, Pv)
        return float(np.linalg.norm(pred - yv, axis=1).mean())

    result = minimize(
        objective,
        np.full(len(names), 1 / len(names)),
        method="SLSQP",
        bounds=[(0, 1)] * len(names),
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        options={"ftol": 1e-12, "maxiter": 500},
    )
    weights = np.maximum(result.x, 0)
    weights /= weights.sum()
    val_pred = np.einsum("m,mni->ni", weights, Pv)
    test_pred = np.einsum("m,mni->ni", weights, Pt)
    report = {
        "selection": "nonnegative weights summing to one, optimized on real validation only",
        "weights": {name: round(float(weight), 6) for name, weight in zip(names, weights)},
        "validation": _metrics(val_pred, yv),
        "test": _metrics(test_pred, yt),
        "candidate_validation": {name: _metrics(pred[0], yv) for name, pred in candidates.items()},
        "candidate_test": {name: _metrics(pred[1], yt) for name, pred in candidates.items()},
    }
    out = DATA / "external_benchmark"
    (out / f"{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.savez(out / f"{args.tag}_predictions.npz",
             val_pred=val_pred, val_y=yv, test_pred=test_pred, test_y=yt)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
