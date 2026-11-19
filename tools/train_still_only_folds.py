#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Separate the auxiliary stillness loss from the stillness gate.

The stored fold ensembles differ in two things at once: the non-gated models
("cvx") have neither the stillness loss nor the gate, the gated models
("cvg") have both. This script trains the missing configuration, stillness
loss without the gate ("cvs"), under the same leave-one-session-out protocol
(8 folds x 2 seeds, same recordings and options as tools/retrain_loso_all.py),
and scores all three configurations with one code path on the held-out
recording and on T1: local error, end-point pose-graph ATE and its per-axis
RMSE. The stored cvx/cvg models and their second training
(datasets/reproducibility_check/loso_retrained) are scored alongside, so the
new configuration can be read against the spread between two trainings.

The frozen training script is called with its output flag pointing inside
datasets/reproducibility_check/loso_still_only/; no frozen file is written.
Existing models are skipped, so a stopped run resumes.

    python tools/train_still_only_folds.py --runs 2
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
from tools.build_trajectory_comparison import _solve_positions
from tools.retrain_loso_all import EXTRA_TRAIN, SEEDS, check_stored_protocol

DATA = ROOT / "datasets"
BASE_REL = "reproducibility_check/loso_still_only"
OUT = DATA / "gate_ablation_still_only.json"


def train_run(folds: list[str], run: int) -> str:
    rel = f"{BASE_REL}/run{run}/physnet_cv"
    for ses in folds:
        for seed in SEEDS:
            if (DATA / rel / f"physnet_cvs_{ses}_s{seed}.pt").is_file():
                continue
            start = time.time()
            subprocess.run([sys.executable, "tools/train_physnet.py", "train", "--out", rel,
                            "--tag", f"cvs_{ses}", "--seed", str(seed), "--holdout", ses,
                            "--weight-decay", "0.1", "--extra-train", *EXTRA_TRAIN,
                            "--still-weight", "1.0"],          # stillness loss, no --gate
                           cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
            print(f"run {run}: cvs {ses} s{seed} ({time.time() - start:.0f} s)", flush=True)
    return rel


def load_models(rel: str, tag: str, folds: list[str], device):
    models = [phys.load_model(DATA / rel / f"physnet_{tag}_{ses}_s{seed}.pt", device)
              for ses in folds for seed in SEEDS]
    flags = {(bool(m.net.use_still), bool(m.net.use_gate)) for m in models}
    if len(flags) != 1:
        raise AssertionError(f"{rel}/{tag}: mixed model configurations {flags}")
    return models, flags.pop()


def score(models, folds: list[str], tests: list[str], device, cache: dict) -> dict:
    length = int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ))
    out = {}
    # out-of-fold cross-validation error: each recording by the two models that did not see it
    errs = []
    for k, ses in enumerate(folds):
        if ses not in cache:
            cache[ses] = phys.build_group([ses], 2.0, 3.0, length)
        data = cache[ses]
        data_t = phys.to_device(data, device)
        pred = np.mean([m.predict(data_t) for m in models[k * len(SEEDS):(k + 1) * len(SEEDS)]], axis=0)
        errs.append(np.linalg.norm(pred - data["y"], axis=1))
    out["cv_err_mm"] = round(float(np.concatenate(errs).mean() * 1000), 2)
    for ses in tests:
        if ses not in cache:
            cache[ses] = phys.build_group([ses], 2.0, 3.0, length)
        data = cache[ses]
        data_t = phys.to_device(data, device)
        pred = np.mean([m.predict(data_t) for m in models], axis=0)
        _t, R, p, usable = phys._load_gt(ses)
        node = {int(f): i for i, f in enumerate(usable)}
        edges = np.array([(node[int(a)], node[int(b)]) for a, b in data["pair"]], np.int64)
        traj = _solve_positions(p[usable], R[usable], edges, pred.astype(np.float64))
        ok = np.isfinite(traj).all(1)
        diff = (traj[ok] - p[usable][ok]) * 1000
        y = data["y"]
        out[ses] = {
            "windows": int(len(y)),
            "err_mm": round(float(np.linalg.norm(pred - y, axis=1).mean() * 1000), 2),
            "zero_mm": round(float(np.linalg.norm(y, axis=1).mean() * 1000), 2),
            "ate_mm": round(float(np.sqrt(np.mean((diff ** 2).sum(1)))), 2),
            "per_axis_rmse_mm_xyz": [round(float(np.sqrt(np.mean(diff[:, a] ** 2))), 1) for a in range(3)],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2, help="independent trainings of the 16 stillness-loss-only models")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds, tests = split["train"] + split["val"], split["test"]
    check_stored_protocol(folds)
    rels = []
    for run in range(1, args.runs + 1):
        rel = f"{BASE_REL}/run{run}/physnet_cv"
        if not args.eval_only:
            rel = train_run(folds, run)
        rels.append(rel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache: dict = {}
    configs = [
        ("no stillness loss, no gate", "first training", "physnet_cv", "cvx"),
        ("no stillness loss, no gate", "second training", "reproducibility_check/loso_retrained/physnet_cv", "cvx"),
        *[("stillness loss, no gate", f"training {i + 1}", rel, "cvs") for i, rel in enumerate(rels)],
        ("stillness loss and gate", "first training", "physnet_cv", "cvg"),
        ("stillness loss and gate", "second training", "reproducibility_check/loso_retrained/physnet_cv", "cvg"),
    ]
    expected = {"cvx": (False, False), "cvs": (True, False), "cvg": (True, True)}
    report = {"folds": folds, "test_recordings": tests, "models_per_ensemble": len(folds) * len(SEEDS), "results": []}
    for name, training, rel, tag in configs:
        models, flags = load_models(rel, tag, folds, device)
        if flags != expected[tag]:
            raise AssertionError(f"{rel}/{tag}: (stillness loss, gate) = {flags}, expected {expected[tag]}")
        row = {"configuration": name, "training": training, "folder": rel, "tag": tag,
               "stillness_loss": flags[0], "gate": flags[1], **score(models, folds, tests, device, cache)}
        report["results"].append(row)
        print(name, "|", training, "| CV", row["cv_err_mm"],
              {s[-15:]: (row[s]["err_mm"], row[s]["ate_mm"], row[s]["per_axis_rmse_mm_xyz"]) for s in tests}, flush=True)
        del models
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
