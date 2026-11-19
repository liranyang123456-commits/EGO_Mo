#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 method-development loop on the fixed split (never touches sealed data).

Trains each configuration with several seeds via tools/train_physnet_v3.py on
datasets/trajectory_split_paper.json, then scores the seed ensemble on the two
validation recordings: mean error, R^2, fastest-third error, per-axis slope of
prediction on reference (shrinkage), median predicted/reference magnitude, and
pose-graph ATE per validation recording.

    python tools/v3_experiments.py --configs base bias_local --seeds 0 1
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets"
SPLIT = "trajectory_split_paper.json"
OUT = "physnet_v3"
BASE = ["--weight-decay", "0.1", "--still-weight", "1.0", "--gate"]
CONFIGS = {
    "base": BASE,
    "bias_local": BASE + ["--bias-mode", "local"],
    "beta5": BASE + ["--beta", "5.0"],
    "nll_main": BASE + ["--nll-weight", "1.0", "--end-weight", "0.3", "--dense-weight", "0.3"],
    "nogate_beta5": ["--weight-decay", "0.1", "--beta", "5.0"],
    "nogate": ["--weight-decay", "0.1"],
    "base_clean": BASE + ["--split-file", "trajectory_split_clean.json"],
    "base_0924": BASE + ["--split-file", "trajectory_split_0924only.json"],
}
TWIN = DATA / "synthetic_realistic"
for _mix in ("realistic", "handheld", "replay", "periodic_twin"):
    CONFIGS[f"pre_{_mix}"] = BASE + ["--pretrain-corpus", str(TWIN / f"mix_{_mix}")]
CONFIGS["pre_realistic_bal"] = CONFIGS["pre_realistic"] + ["--pretrain-balance", "class"]
CONFIGS["bias_none"] = BASE + ["--bias-mode", "none"]
CONFIGS["bias_strict"] = BASE + ["--bias-mode", "strict"]
CONFIGS["bias_none_pre_bal"] = CONFIGS["pre_realistic_bal"] + ["--bias-mode", "none"]


def train(name, seed, extra):
    ck = DATA / OUT / f"physnet_{name}_s{seed}.pt"
    if ck.is_file():
        return
    split = [] if "--split-file" in extra else ["--split-file", SPLIT]
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "train_physnet_v3.py"), "train",
           *split, "--out", OUT, "--tag", name, "--seed", str(seed), *extra]
    log = DATA / OUT / f"log_{name}_s{seed}.txt"
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def evaluate(name, seeds):
    import tools.train_physnet_v3 as v3
    import tools.train_seq as api
    from tools.eval_session_cv import trajectory_ate
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [v3.load_model(DATA / OUT / f"physnet_{name}_s{s}.pt", device) for s in seeds]
    split = json.loads((DATA / SPLIT).read_text(encoding="utf-8"))
    res = {"config": name, "seeds": seeds, "sessions": {}}
    P, Y = [], []
    for ses in split["val"]:
        data = v3.build_group([ses], context, horizon, length)
        dt = v3.to_device(data, device)
        pred = np.mean([m.predict(dt) for m in models], axis=0)
        P.append(pred); Y.append(data["y"])
        ate = trajectory_ate(ses, data, pred, None)["single_edge"]["ate_rmse_mm"]
        res["sessions"][ses] = {"err_mm": api._metrics(pred, data["y"])["err_mm"], "ate_mm": ate}
    p, y = np.concatenate(P), np.concatenate(Y)
    m = api._metrics(p, y)
    res.update({
        "err_mm": m["err_mm"], "r2": m["r2"], "fast_mm": m["fast_mm"],
        "slope": [float(np.polyfit(y[:, k], p[:, k], 1)[0]) for k in range(3)],
        "median_ratio": float(np.median(np.linalg.norm(p, axis=1)) /
                              np.median(np.linalg.norm(y, axis=1))),
        "ate_mm_mean": float(np.mean([v["ate_mm"] for v in res["sessions"].values()])),
    })
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--output", type=Path, default=DATA / OUT / "results.json")
    args = ap.parse_args()
    results = json.loads(args.output.read_text(encoding="utf-8")) if args.output.is_file() else {}
    for name in args.configs:
        for s in args.seeds:
            train(name, s, CONFIGS[name])
        results[name] = evaluate(name, args.seeds)
        r = results[name]
        print(f"{name:14s} err {r['err_mm']:6.2f}  R2 {r['r2']:.3f}  fast {r['fast_mm']:6.2f}  "
              f"slope {np.round(r['slope'], 2)}  med {r['median_ratio']:.2f}  "
              f"ATE {r['ate_mm_mean']:6.1f}", flush=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
