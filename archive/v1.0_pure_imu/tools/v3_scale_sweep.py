#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does undoing the shrinkage help the trajectory? Validation only.

Multiplies the seed-ensemble predictions of a v3 configuration by a global
factor alpha and reports window error and pose-graph ATE per validation
recording, so the trade-off between local error and trajectory error of the
conditional-mean shrinkage can be read directly.
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
DATA = ROOT / "datasets"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["base", "nogate"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 1.25, 1.5, 1.75, 2.0])
    args = ap.parse_args()
    import tools.train_physnet_v3 as v3
    import tools.train_seq as api
    from tools.eval_session_cv import trajectory_ate
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    length = int(round((1.5 * 3.0 + 2 * 2.0) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    report = {}
    for name in args.configs:
        models = [v3.load_model(DATA / "physnet_v3" / f"physnet_{name}_s{s}.pt", device)
                  for s in args.seeds]
        cache = {}
        for ses in split["val"]:
            data = v3.build_group([ses], 2.0, 3.0, length)
            dt = v3.to_device(data, device)
            cache[ses] = (data, np.mean([m.predict(dt) for m in models], axis=0))
        rows = {}
        for a in args.alphas:
            errs, ates, n = [], [], []
            for ses, (data, pred) in cache.items():
                p = a * pred
                errs.append(np.linalg.norm(p - data["y"], axis=1).sum() * 1000)
                n.append(len(p))
                ates.append(trajectory_ate(ses, data, p, None)["single_edge"]["ate_rmse_mm"])
            rows[a] = {"err_mm": float(sum(errs) / sum(n)), "ate_mm": float(np.mean(ates)),
                       "ate_per_session": ates}
            print(f"{name:8s} alpha {a:4.2f}  err {rows[a]['err_mm']:6.2f}  ATE {rows[a]['ate_mm']:6.1f}  "
                  f"{np.round(ates, 1)}", flush=True)
        report[name] = rows
    (DATA / "physnet_v3" / "scale_sweep.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
