#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Calibration of the predicted displacement uncertainty.

A pseudo-measurement needs an uncertainty statement per prediction, not only
an aggregate error. The heteroscedastic head of PhysNet predicts a per-axis
variance for the displacement stream; this script checks whether that variance
is usable as a standard uncertainty: interval coverage, sharpness, the
correlation between predicted and realized error, and the error of the most
and least confident deciles. A single scale factor fitted on validation is
reported as a recalibration constant, as is done for ensemble variance.
"""

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
import tools.train_physnet as phys

DATA = ROOT / "datasets"


def collect(models, names, context, horizon, length, device):
    data = phys.build_group(names, context, horizon, length)
    data_t = phys.to_device(data, device)
    per_model, per_var = [], []
    for m in models:
        st = m.predict_streams(data_t)
        # end-of-pair index inside the dense arrays is not guaranteed, so take
        # the stream value at tau_b directly from `end` and the variance at the
        # dense frame closest to tau_b.
        j = np.argmin(np.abs(data["dense_tau"] - data["tau_b"][:, None])
                      + 1e6 * (data["dense_valid"] <= 0), axis=1)
        rows = np.arange(len(j))
        per_model.append(st["end"])
        per_var.append(st["var"][rows, j])
    pred = np.mean(per_model, axis=0)
    aleatoric = np.mean(per_var, axis=0)                      # predicted variance, m^2
    epistemic = np.var(per_model, axis=0) if len(models) > 1 else np.zeros_like(aleatoric)
    return data, pred, aleatoric, epistemic


def stats(pred, y, var, label):
    err = pred - y
    sigma = np.sqrt(np.maximum(var, 1e-12))
    z = err / sigma
    out = {
        "n": int(len(y)),
        "err_mm": round(float(np.linalg.norm(err, axis=1).mean() * 1000), 2),
        "sigma_mm_median": round(float(np.median(sigma) * 1000), 2),
        "coverage_1sigma": round(float((np.abs(z) <= 1).mean()), 3),
        "coverage_2sigma": round(float((np.abs(z) <= 2).mean()), 3),
        "rms_z": round(float(np.sqrt((z ** 2).mean())), 3),
        "scale_to_calibrate": round(float(np.sqrt((z ** 2).mean())), 3),
    }
    # Does a larger predicted sigma really mean a larger error?
    mag = np.linalg.norm(err, axis=1)
    s = np.linalg.norm(sigma, axis=1)
    out["spearman_sigma_error"] = round(float(
        np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(mag)))[0, 1]), 3)
    order = np.argsort(s)
    d = max(len(order) // 10, 1)
    out["err_mm_most_confident_decile"] = round(float(mag[order[:d]].mean() * 1000), 2)
    out["err_mm_least_confident_decile"] = round(float(mag[order[-d:]].mean() * 1000), 2)
    out["label"] = label
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    ap.add_argument("--test", action="store_true", help="also open the sealed test sessions")
    ap.add_argument("--output", type=Path, default=DATA / "uncertainty_calibration.json")
    ap.add_argument("--cross-session", action="store_true",
                    help="fit the scale on one validation recording, score the other")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", args.horizon
    length = int(round((1.5 * args.horizon + 2 * args.context) * phys.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    models = [phys.load_model(Path(p), device) for p in args.models]
    report = {"models": args.models}
    groups = {"val": split["val"]}
    if args.test:
        groups["test"] = split["test"]
    scale = None
    for group, names in groups.items():
        data, pred, alea, epi = collect(models, names, args.context, args.horizon, length, device)
        y = data["y"]
        report[group] = {
            "aleatoric": stats(pred, y, alea, "predicted variance"),
            "ensemble": stats(pred, y, epi + 1e-12, "ensemble spread"),
            "total": stats(pred, y, alea + epi, "predicted + ensemble"),
        }
        if group == "val":
            scale = report["val"]["total"]["scale_to_calibrate"]
        report[group]["total_rescaled"] = stats(pred, y, (alea + epi) * scale ** 2,
                                                f"rescaled by {scale} (fitted on validation)")
        print(json.dumps({group: report[group]}, ensure_ascii=False, indent=2), flush=True)
    report["validation_scale"] = scale
    if args.cross_session and len(split["val"]) > 1:
        # Out-of-sample check: the scale is fitted on one validation recording
        # and the coverage is scored on the other.
        parts = {n: collect(models, [n], args.context, args.horizon, length, device)
                 for n in split["val"]}
        z1, z2, cross = [], [], {}
        for held in split["val"]:
            fit = [n for n in split["val"] if n != held]
            zs = []
            for n in fit:
                d, p, a, e = parts[n]
                zs.append(((p - d["y"]) / np.sqrt(np.maximum(a + e, 1e-12))).ravel())
            k = float(np.sqrt(np.mean(np.concatenate(zs) ** 2)))
            d, p, a, e = parts[held]
            z = (p - d["y"]) / np.sqrt(np.maximum((a + e) * k ** 2, 1e-12))
            z1.append((np.abs(z) <= 1).ravel())
            z2.append((np.abs(z) <= 2).ravel())
            cross[held] = {"scale_from_other": round(k, 3),
                           "coverage_1sigma": round(float((np.abs(z) <= 1).mean()), 3),
                           "coverage_2sigma": round(float((np.abs(z) <= 2).mean()), 3)}
        report["cross_session"] = {
            "per_session": cross,
            "pooled_coverage_1sigma": round(float(np.concatenate(z1).mean()), 3),
            "pooled_coverage_2sigma": round(float(np.concatenate(z2).mean()), 3),
        }
        print(json.dumps({"cross_session": report["cross_session"]}, indent=2), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
