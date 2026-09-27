#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-session-out with a cleaned training pool (v3, post-freeze).

Same configuration as the frozen gated PhysNet (cvg), but the two recordings
with the poorest camera-IMU rigidity (traj_20260923_015507: 11.6 deg rotation
residual, traj_20260923_015629: 7.8 deg) are removed from the pool. Each of the
remaining six recordings is held out in turn; its error is compared with the
frozen cvg fold model on the same recording (which was trained with the two
recordings included). The fold ensemble is also scored on the historical
held-out test recording.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "datasets"
SPLIT = "trajectory_split_clean.json"
OUT = "physnet_v3cv"
CFG = ["--weight-decay", "0.1", "--still-weight", "1.0", "--gate"]
TEST = "traj_20260923_023422"


def main(seeds=(0, 1)):
    split = json.loads((DATA / SPLIT).read_text(encoding="utf-8"))
    pool = split["train"] + split["val"]
    for ses in pool:
        for s in seeds:
            ck = DATA / OUT / f"physnet_cvclean_{ses}_s{s}.pt"
            if ck.is_file():
                continue
            (DATA / OUT).mkdir(exist_ok=True)
            cmd = [sys.executable, "-u", str(ROOT / "tools" / "train_physnet_v3.py"), "train",
                   "--split-file", SPLIT, "--out", OUT, "--tag", f"cvclean_{ses}", "--seed", str(s),
                   "--holdout", ses, *CFG]
            with (DATA / OUT / f"log_cvclean_{ses}_s{s}.txt").open("w", encoding="utf-8") as f:
                subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)
            print("trained", ses, s, flush=True)

    import tools.train_physnet_v3 as v3
    import tools.train_seq as api
    from tools.eval_session_cv import trajectory_ate
    from tools.recheck_metrics import metrics
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    length = int(round((1.5 * 3.0 + 2 * 2.0) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"folds": {}}
    tot_c = tot_f = n_tot = 0.0
    for ses in pool:
        clean = [json.loads((DATA / OUT / f"metrics_cvclean_{ses}_s{s}.json").read_text(encoding="utf-8"))
                 for s in seeds]
        frozen = [json.loads((DATA / "physnet_cv" / f"metrics_cvg_{ses}_s{s}.json").read_text(encoding="utf-8"))
                  for s in seeds]
        # two-seed fold ensembles, from the saved out-of-fold predictions
        pc = np.mean([np.load(DATA / OUT / f"pred_cvclean_{ses}_s{s}.npz")["val_pred"] for s in seeds], 0)
        pf = np.mean([np.load(DATA / "physnet_cv" / f"pred_cvg_{ses}_s{s}.npz")["val_pred"] for s in seeds], 0)
        y = np.load(DATA / OUT / f"pred_cvclean_{ses}_s0.npz")["val_y"]
        ec, ef = metrics(pc, y), metrics(pf, y)
        report["folds"][ses] = {"clean": ec, "frozen": ef, "zero_mm": float(np.linalg.norm(y, axis=1).mean() * 1000),
                                "single_clean": [c["val"]["err_mm"] for c in clean],
                                "single_frozen": [f["val"]["err_mm"] for f in frozen]}
        tot_c += ec["err_mm"] * len(y); tot_f += ef["err_mm"] * len(y); n_tot += len(y)
        print(f"{ses:24s} zero {report['folds'][ses]['zero_mm']:5.1f}  frozen {ef['err_mm']:6.2f}  clean {ec['err_mm']:6.2f}", flush=True)
    report["cv_mm"] = {"clean": tot_c / n_tot, "frozen": tot_f / n_tot}
    models = [v3.load_model(DATA / OUT / f"physnet_cvclean_{ses}_s{s}.pt", device) for ses in pool for s in seeds]
    data = v3.build_group([TEST], 2.0, 3.0, length)
    dt = v3.to_device(data, device)
    p = np.mean([m.predict(dt) for m in models], axis=0)
    g = np.load(DATA / "session_cv_gate_benchmark.npz")
    assert np.allclose(g["y"], data["y"])
    report["test"] = {
        "clean": metrics(p, data["y"]) | {"ate_mm": float(trajectory_ate(TEST, data, p, None)["single_edge"]["ate_rmse_mm"])},
        "frozen_gate": metrics(g["physnet"], g["y"]),
        "imunet": metrics(g["imunet"], g["y"]),
    }
    from tools.recheck_metrics import boot
    e = lambda q: np.linalg.norm(q - g["y"], axis=1)
    report["test"]["ci_clean_vs_imunet_mm"] = boot(e(p), e(g["imunet"]), g["t_pair"][:, 0])
    report["test"]["ci_clean_vs_frozen_mm"] = boot(e(p), e(g["physnet"]), g["t_pair"][:, 0])
    (DATA / OUT / "cv_clean.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"cv_mm": report["cv_mm"], "test": report["test"]}, indent=2))


if __name__ == "__main__":
    main()
