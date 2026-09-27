#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate a validation-selected IMU ensemble once on sealed test sessions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import tools.train_seq as train_seq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def _extra(pred: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    err = np.linalg.norm(pred - y, axis=1) * 1000.0
    zero = np.linalg.norm(y, axis=1) * 1000.0
    gain = zero - err
    draws = []
    for _ in range(5000):
        idx = rng.integers(0, len(gain), len(gain))
        draws.append(float(gain[idx].mean()))
    axis_rms = np.sqrt(np.mean((pred - y) ** 2, axis=0)) * 1000.0
    return {
        "error_median_mm": round(float(np.median(err)), 2),
        "error_p90_mm": round(float(np.quantile(err, 0.9)), 2),
        "gain_mm": round(float(gain.mean()), 2),
        "gain_percent": round(float(100.0 * gain.mean() / max(zero.mean(), 1e-9)), 1),
        "gain_95ci_mm": [
            round(float(np.quantile(draws, 0.025)), 2),
            round(float(np.quantile(draws, 0.975)), 2),
        ],
        "axis_rms_mm": [round(float(v), 2) for v in axis_rms],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=float, required=True)
    ap.add_argument("--horizon", type=float, default=0.2)
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--labels", default="pose_gt_raw")
    ap.add_argument("--run-dir", default="traj_run_v10")
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    ap.add_argument("--camera-aligned", action="store_true")
    args = ap.parse_args()

    train_seq.LABELS = args.labels
    train_seq.TARGET_S = args.horizon
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    length = int(round((1.50 * args.horizon + 2 * args.context) * train_seq.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run = DATA / args.run_dir
    models = []
    horizon_tag = "" if abs(args.horizon - 0.2) < 1e-9 else f"_h{args.horizon:g}"
    tag_context = f"{args.context:g}{horizon_tag}{'_cam' if args.camera_aligned else ''}"
    for seed in seeds:
        net = train_seq.SeqNet().to(device)
        state = torch.load(run / f"seq_ctx{tag_context}_s{seed}.pt", map_location=device)
        net.load_state_dict(state)
        net.eval()
        models.append(net)

    rng = np.random.default_rng(0)
    report = {
        "context_s": args.context,
        "horizon_s": args.horizon,
        "seeds": seeds,
        "labels": args.labels,
        "camera_aligned": args.camera_aligned,
        "selection": "maximum aggregate validation R2; test sealed until this evaluation",
        "val_by_session": {},
        "test_by_session": {},
    }
    saved = {}
    for group in ("val", "test"):
        all_pred, all_y = [], []
        for name in split[group]:
            got = train_seq.build(
                name,
                args.context,
                length,
                camera_aligned=args.camera_aligned,
            )
            if got is None:
                continue
            X, y = got
            pred = np.mean(
                [train_seq._predict(net, X, device) * 0.01 for net in models],
                axis=0,
            )
            metrics = train_seq._metrics(pred, y)
            metrics.update(_extra(pred, y, rng))
            report[f"{group}_by_session"][name] = metrics
            all_pred.append(pred)
            all_y.append(y)
        pred = np.concatenate(all_pred)
        y = np.concatenate(all_y)
        metrics = train_seq._metrics(pred, y)
        metrics.update(_extra(pred, y, rng))
        report[group] = metrics
        saved[f"{group}_pred"] = pred
        saved[f"{group}_y"] = y
        print(json.dumps({group: metrics}, ensure_ascii=False), flush=True)

    np.savez(run / f"final_ensemble_ctx{tag_context}.npz", **saved)
    (run / "final_evaluation.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
