#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How strongly the data constrain the learned lever arm, component by component.

PhysNet adds (R_{a<-k} - I) lambda to the integrated displacement. The
component lambda_j is multiplied by column j of (R_{a<-b} - I), so its effect on
the 3-s displacement is that column times the change of lambda_j. This script
reports, for the leave-one-session-out models of the manuscript:

* the lever arm of the 16 non-gated (cvx) and 16 gated (cvg) fold models;
* the RMS norm of each column of (R_{a<-b} - I) over the windows these models
  were trained and validated on (eight trajectory and four rigidity
  recordings), with the gyroscope attitude that the network itself uses;
* the displacement change caused by moving lambda_z by the difference between
  the two families, per axis and as a 3-D vector;
* the error of the frozen non-gated ensemble on both test recordings with
  lambda_z of every model moved by that difference, against the stored models.

    python tools/analyze_lever_excitation.py
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

DATA = ROOT / "datasets"
TESTS = {"held_out": "traj_20260923_023422", "t1": "traj_20260926_143837"}
LENGTH = int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ))


def end_rotation(data: dict) -> np.ndarray:
    """R_{a<-b} of every window: the gyroscope attitude at the end of the interval."""
    R = phys.quat_to_mat(torch.from_numpy(data["quat"])).numpy()
    idx = np.clip(np.rint(data["tau_b"]).astype(int), 0, R.shape[1] - 1)
    return R[np.arange(len(idx)), idx]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds = split["train"] + split["val"]
    # The fold models were trained with the four rigidity recordings of 24 September.
    rigid = [n for n in split["extra_train"] if n.startswith("rigid_20260924")]
    models = {tag: [phys.load_model(DATA / "physnet_cv" / f"physnet_{tag}_{s}_s{k}.pt", device)
                    for s in folds for k in (0, 1)] for tag in ("cvx", "cvg")}
    lever = {tag: np.array([m.net.lever.detach().cpu().numpy() * 1000 for m in ms]) for tag, ms in models.items()}
    shift = lever["cvg"].mean(0) - lever["cvx"].mean(0)

    M, counts = [], {}
    for name in folds + rigid:
        data = phys.build_group([name], 2.0, 3.0, LENGTH)
        M.append(end_rotation(data) - np.eye(3))
        counts[name] = int(len(data["y"]))
    M = np.concatenate(M)
    col_rms = np.sqrt((M ** 2).sum(1).mean(0))
    change = M[:, :, 2] * shift[2]                       # mm, in the anchor frame
    angle = np.degrees(np.arccos(np.clip((np.trace(M, axis1=1, axis2=2) + 2.0) / 2.0, -1.0, 1.0)))

    tests = {}
    for key, name in TESTS.items():
        data = phys.build_group([name], 2.0, 3.0, LENGTH)
        data_t = phys.to_device(data, device)
        y = data["y"]

        def err(pred):
            return round(float(np.linalg.norm(pred - y, axis=1).mean() * 1000), 2)

        stored = np.mean([m.predict(data_t) for m in models["cvx"]], axis=0)
        for m in models["cvx"]:
            m.net.lever.data[2] += float(shift[2]) / 1000.0
        moved = np.mean([m.predict(data_t) for m in models["cvx"]], axis=0)
        for m in models["cvx"]:
            m.net.lever.data[2] -= float(shift[2]) / 1000.0
        tests[key] = {"recording": name, "windows": int(len(y)),
                      "err_stored_mm": err(stored), "err_lambda_z_moved_mm": err(moved),
                      "prediction_change_rms_mm": round(float(np.sqrt(((moved - stored) ** 2).sum(1).mean()) * 1000), 2)}

    report = {
        # Standard deviation over the 16 fold models with ddof=0, as in the manuscript.
        "lever_mm": {tag: {"models": int(len(v)), "mean": [round(float(x), 1) for x in v.mean(0)],
                           "sd": [round(float(x), 1) for x in v.std(0)]} for tag, v in lever.items()},
        "shift_gated_minus_nongated_mm": [round(float(x), 1) for x in shift],
        "windows": {"recordings": counts, "trajectory": int(sum(counts[n] for n in folds)),
                    "rigidity": int(sum(counts[n] for n in rigid)), "total": int(len(M))},
        "rotation_deg": {"median": round(float(np.median(angle)), 1), "p90": round(float(np.quantile(angle, 0.9)), 1)},
        "column_rms_norm_xyz": [round(float(x), 3) for x in col_rms],
        "lambda_z_shift_effect_mm": {"per_axis_rms_xyz": [round(float(x), 2) for x in np.sqrt((change ** 2).mean(0))],
                                     "vector_rms": round(float(np.sqrt((change ** 2).sum(1).mean())), 2)},
        "nongated_ensemble_with_lambda_z_moved": tests,
    }
    (DATA / "lever_excitation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
