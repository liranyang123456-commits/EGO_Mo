#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Remove a train-only displacement bias and remeasure the held-out path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_contiguous import _chain, _fit, _keep, _predict, _rmse
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split

GT = ROOT / "datasets" / "pose_gt"
OUT = ROOT / "datasets" / "traj_run_v3" / "debias_path.json"


def _step(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean() * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = load_split(names)
    built = {name: build_pairs(packs[name]) for name in names}
    train_names = [name for name in names if name not in (TEST_NAME, VAL_NAME)]
    coef = _fit([built[name] for name in train_names], device)
    net_on = False
    preds = {}
    for name in names:
        _net_p, base, _dR = _predict_linear(built[name], coef, device)
        preds[name] = base
    residuals = [preds[name] - built[name]["dp"] for name in train_names]
    bias = np.mean(np.concatenate(residuals), axis=0)
    report = {"bias_mm": [round(float(v) * 1000.0, 2) for v in bias]}
    for name in (VAL_NAME, TEST_NAME):
        gt = built[name]["dp"]
        raw = preds[name]
        debiased = raw - bias
        keep = _keep(built[name]["t_start"], built[name]["t_end"])
        dR = built[name]["dR"]
        report[name] = {
            "step_linear_mm": round(_step(raw, gt), 2),
            "step_debias_mm": round(_step(debiased, gt), 2),
            "path_linear_mm": round(_rmse(_chain(raw, dR, keep), _chain(gt, dR, keep)), 1),
            "path_debias_mm": round(_rmse(_chain(debiased, dR, keep), _chain(gt, dR, keep)), 1),
            "residual_mean_mm": [round(float(v) * 1000.0, 2) for v in (raw - gt).mean(axis=0)],
            "residual_std_mm": [round(float(v) * 1000.0, 2) for v in (raw - gt).std(axis=0)],
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    print(json.dumps({"bias_mm": report["bias_mm"]}, ensure_ascii=False), flush=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _ = net_on


def _predict_linear(pack, coef, device):
    from tools.eval_contiguous import _predict
    from ego_capture.mapping.inertial import MotionTrajectoryCorrector
    net = MotionTrajectoryCorrector().to(device)
    ckpt = torch.load(ROOT / "datasets" / "traj_run_v3" / "corrector.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    return _predict(net, pack, coef, device)


if __name__ == "__main__":
    main()
