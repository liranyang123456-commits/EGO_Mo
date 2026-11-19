#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gate-bypass control for the stillness-gate trajectory trade-off.

The 16 frozen gated PhysNet fold models are evaluated on the held-out
recording twice: as trained, and with the multiplicative stillness gate
switched off at inference only (weights untouched). The second condition is
outside the training condition of these networks, so it cannot attribute the
gated/non-gated difference to a training component; it only tests whether
removing the gate's action at inference recovers the non-gated trajectory.

    python tools/analyze_gate_bypass.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
from tools.build_trajectory_comparison import _solve_positions

DATA = ROOT / "datasets"
TEST = "traj_20260923_023422"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _t, Rg, pg, usable = phys._load_gt(TEST)
    data = phys.build_group([TEST], 2.0, 3.0, int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ)))
    data_t = phys.to_device(data, device)
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds = split["train"] + split["val"]
    y = data["y"]
    node = {int(f): i for i, f in enumerate(usable)}
    edges = np.array([(node[int(a)], node[int(b)]) for a, b in data["pair"]], np.int64)

    def score(pred: np.ndarray) -> dict:
        traj = _solve_positions(pg[usable], Rg[usable], edges, pred.astype(np.float64))
        ok = np.isfinite(traj).all(1)
        diff = (traj[ok] - pg[usable][ok]) * 1000
        return {"err_mm": round(float(np.linalg.norm(pred - y, axis=1).mean() * 1000), 2),
                "per_axis_rmse_mm_xyz": [round(float(np.sqrt(np.mean(diff[:, a] ** 2))), 1) for a in range(3)],
                "ate_mm": round(float(np.sqrt(np.mean((diff ** 2).sum(1)))), 2)}

    def ensemble(tag: str, bypass: bool) -> np.ndarray:
        preds = []
        for s in folds:
            for k in (0, 1):
                m = phys.load_model(DATA / "physnet_cv" / f"physnet_{tag}_{s}_s{k}.pt", device)
                if bypass:
                    assert m.net.use_gate, "bypass is defined for gated models only"
                    m.net.use_gate = False
                preds.append(m.predict(data_t))
        return np.mean(preds, axis=0)

    report = {
        "recording": TEST, "windows": int(len(y)), "models_per_ensemble": 2 * len(folds),
        "non_gated": score(ensemble("cvx", False)),
        "gated": score(ensemble("cvg", False)),
        "gated_weights_gate_bypassed_at_inference": score(ensemble("cvg", True)),
        "note": "The bypassed condition is outside the training condition of the gated networks.",
    }
    (DATA / "gate_bypass_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
