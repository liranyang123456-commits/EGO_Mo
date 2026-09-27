#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final evaluation of the validation-selected hybrid ensemble."""

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


def _predict(models, X, device):
    return np.mean([api._predict(model, X, device) * 0.01 for model in models], axis=0)


def _extra(pred, y, rng):
    err = np.linalg.norm(pred - y, axis=1) * 1000.0
    zero = np.linalg.norm(y, axis=1) * 1000.0
    gain = zero - err
    draws = [
        float(gain[rng.integers(0, len(gain), len(gain))].mean())
        for _ in range(5000)
    ]
    return {
        "gain_mm": round(float(gain.mean()), 2),
        "gain_percent": round(float(100 * gain.mean() / max(zero.mean(), 1e-9)), 1),
        "gain_95ci_mm": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "error_median_mm": round(float(np.median(err)), 2),
        "error_p90_mm": round(float(np.quantile(err, 0.9)), 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--run-dir", default="traj_run_v17")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    args = ap.parse_args()

    api.LABELS = "pose_gt_raw"
    api.TARGET_S = args.horizon
    seeds = [int(x) for x in args.seeds.split(",")]
    length = int(round((1.5 * args.horizon + 2 * args.context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = DATA / args.run_dir
    models = []
    for seed in seeds:
        model = api.SeqNet().to(device)
        model.load_state_dict(torch.load(
            run / f"seq_ctx2_h3_r0_p35_lr0.001_s{seed}.pt",
            map_location=device,
        ))
        model.eval()
        models.append(model)
    rng = np.random.default_rng(0)
    report = {
        "selection": "tuned synthetic pretrain selected on real validation; sealed real test opened here",
        "seeds": seeds,
        "context_s": args.context,
        "horizon_s": args.horizon,
    }
    saved = {}

    real_split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    for group in ("val", "test"):
        predictions, labels = [], []
        per_session = {}
        for name in real_split[group]:
            got = api.build(name, args.context, length)
            if got is None:
                continue
            X, y = got
            pred = _predict(models, X, device)
            metrics = api._metrics(pred, y)
            metrics.update(_extra(pred, y, rng))
            per_session[name] = metrics
            predictions.append(pred)
            labels.append(y)
        pred, y = np.concatenate(predictions), np.concatenate(labels)
        metrics = api._metrics(pred, y)
        metrics.update(_extra(pred, y, rng))
        report[f"real_{group}"] = metrics
        report[f"real_{group}_by_session"] = per_session
        saved[f"real_{group}_pred"] = pred
        saved[f"real_{group}_y"] = y
        print(json.dumps({f"real_{group}": metrics}), flush=True)

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    for group in ("test", "ood"):
        predictions, labels = [], []
        by_motion = {}
        for motion in sorted({entry["motion"] for entry in manifest["imu"][group]}):
            motion_pred, motion_y = [], []
            entries = [entry for entry in manifest["imu"][group] if entry["motion"] == motion]
            for entry in entries:
                got = _synthetic_build(
                    args.corpus / entry["path"],
                    args.context, length, args.horizon,
                    160, rng,
                )
                if got is None:
                    continue
                X, y = got
                pred = _predict(models, X, device)
                motion_pred.append(pred)
                motion_y.append(y)
            mp, my = np.concatenate(motion_pred), np.concatenate(motion_y)
            by_motion[motion] = api._metrics(mp, my)
            predictions.append(mp)
            labels.append(my)
        pred, y = np.concatenate(predictions), np.concatenate(labels)
        metrics = api._metrics(pred, y)
        metrics.update(_extra(pred, y, rng))
        report[f"synthetic_{group}"] = metrics
        report[f"synthetic_{group}_by_motion"] = by_motion
        print(json.dumps({f"synthetic_{group}": metrics}), flush=True)

    (run / "final_hybrid_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    np.savez(run / "final_hybrid_predictions.npz", **saved)


if __name__ == "__main__":
    main()
