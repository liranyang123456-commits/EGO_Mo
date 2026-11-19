#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train all 80 leave-one-session-out fold models again into a separate folder,
then score them on CV, on the held-out recording and on T1.

The frozen scripts (train_physnet.py, train_external_architectures.py,
eval_session_cv.py) are called with their output flags pointing inside
datasets/reproducibility_check/loso_retrained/. Neither datasets/physnet_cv/,
datasets/external_benchmark/, nor any frozen file is written. Existing fresh
models are skipped, so a stopped run resumes where it left off.

    python tools/retrain_loso_all.py
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

import tools.eval_session_cv as esc
import tools.train_physnet as phys
import tools.train_seq as api
from tools.train_external_architectures import make_model, predict as predict_external

DATA = ROOT / "datasets"
CHECK = DATA / "reproducibility_check" / "loso_retrained"
PHYS_REL = "reproducibility_check/loso_retrained/physnet_cv"
EXT_REL = "reproducibility_check/loso_retrained/external_benchmark"
# The four rigidity recordings used by every stored fold model (checked below).
EXTRA_TRAIN = [
    "rigid_20260924_123958",
    "rigid_20260924_124243",
    "rigid_20260924_140510",
    "rigid_20260924_142902",
]
BASELINES = ("imunet", "ronin_resnet", "tlio_resnet")
SEEDS = (0, 1)


def _run(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def check_stored_protocol(folds: list[str]) -> None:
    for path in (
        DATA / "physnet_cv" / f"metrics_cvx_{folds[0]}_s0.json",
        DATA / "physnet_cv" / f"metrics_cvg_{folds[0]}_s0.json",
        *(DATA / "external_benchmark" / f"{a}_cvx_{folds[0]}_s0.json" for a in BASELINES),
    ):
        rig = sorted(n for n in json.loads(path.read_text(encoding="utf-8"))["train_sessions"]
                     if n.startswith("rigid_"))
        if rig != sorted(EXTRA_TRAIN):
            raise SystemExit(f"{path.name} used {rig}, expected {EXTRA_TRAIN}")


def train_physnet_folds(folds: list[str], gated: bool) -> None:
    tag = "cvg" if gated else "cvx"
    out = DATA / PHYS_REL
    for ses in folds:
        for seed in SEEDS:
            if (out / f"physnet_{tag}_{ses}_s{seed}.pt").is_file():
                continue
            start = time.time()
            cmd = ["tools/train_physnet.py", "train", "--out", PHYS_REL,
                   "--tag", f"{tag}_{ses}", "--seed", str(seed), "--holdout", ses,
                   "--weight-decay", "0.1", "--extra-train", *EXTRA_TRAIN]
            if gated:
                cmd += ["--still-weight", "1.0", "--gate"]
            _run(cmd)
            print(f"physnet {tag} {ses} s{seed} ({time.time() - start:.0f} s)", flush=True)


def train_baseline_folds(folds: list[str], arch: str) -> None:
    out = DATA / EXT_REL
    for ses in folds:
        for seed in SEEDS:
            stem = f"{arch}_cvx_{ses}_s{seed}"
            if (out / f"{stem}.pt").is_file():
                continue
            start = time.time()
            _run(["tools/train_external_architectures.py", "--arch", arch, "--only", "real_only",
                  "--holdout", ses, "--seed", str(seed), "--out", EXT_REL, "--stem", stem,
                  "--extra-train", *EXTRA_TRAIN])
            print(f"{stem} ({time.time() - start:.0f} s)", flush=True)


def evaluate_fresh_folds(folds: list[str], test_sessions: list[str]) -> Path:
    """Same computation as eval_session_cv.py, reading baselines from EXT_REL.

    eval_session_cv.py has no flag for the baseline folder; calling its helpers
    here avoids touching that frozen file or datasets/external_benchmark/.
    """
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext_nets = {
        arch: [
            _load_ext(arch, DATA / EXT_REL / f"{arch}_cvx_{ses}_s{seed}.pt", device)
            for ses in folds for seed in SEEDS
        ]
        for arch in BASELINES
    }
    for tag in ("cvx", "cvg"):
        models = [phys.load_model(DATA / PHYS_REL / f"physnet_{tag}_{ses}_s{seed}.pt", device)
                  for ses in folds for seed in SEEDS]
        cv_phys, cv_imu, cv_y = [], [], []
        for k, ses in enumerate(folds):
            X_ses, y_ses = api.build(ses, context, length)
            data_t = phys.to_device(phys.build_group([ses], context, horizon, length), device)
            sl = slice(k * len(SEEDS), (k + 1) * len(SEEDS))
            cv_phys.append(np.mean([m.predict(data_t) for m in models[sl]], axis=0))
            cv_imu.append(np.mean([predict_external(n, X_ses, device) for n in ext_nets["imunet"][sl]], axis=0))
            cv_y.append(y_ses)
        cv_phys, cv_imu, cv_y = (np.concatenate(v) for v in (cv_phys, cv_imu, cv_y))

        def cv_err(w, alpha):
            return float(np.linalg.norm(alpha * (w * cv_phys + (1 - w) * cv_imu) - cv_y, axis=1).mean() * 1000)

        _, w_best, a_best = min(
            ((cv_err(w, a), w, a) for w in np.linspace(0, 1, 21) for a in np.linspace(0.6, 1.2, 25)),
            key=lambda r: r[0])
        test_out, arrays_all, pooled = {}, {}, {}
        for test_name in test_sessions:
            test_phys = phys.build_group([test_name], context, horizon, length)
            test_t = phys.to_device(test_phys, device)
            phys_pred = np.mean([m.predict(test_t) for m in models], axis=0)
            dense_pred = np.mean([m.predict_dense(test_t) for m in models], axis=0)
            y = test_phys["y"]
            X_test, _ = api.build(test_name, context, length)
            err_phys = np.linalg.norm(phys_pred - y, axis=1)
            res = {
                "n": int(len(y)),
                "zero_motion": api._metrics(np.zeros_like(y), y),
                "physnet_cv_ensemble": api._metrics(phys_pred, y),
                "trajectory_physnet": esc.trajectory_ate(test_name, test_phys, phys_pred, dense_pred),
                "physnet_minus_zero_mm_95ci_blocks": esc._bootstrap_ci(
                    err_phys, np.linalg.norm(y, axis=1), test_phys["t_pair"]),
            }
            arrays = {"y": y, "t_pair": test_phys["t_pair"], "pair": test_phys["pair"], "physnet": phys_pred}
            for arch, nets in ext_nets.items():
                pred = np.mean([predict_external(n, X_test, device) for n in nets], axis=0)
                arrays[arch] = pred
                res[f"{arch}_cv_ensemble"] = api._metrics(pred, y)
                res[f"trajectory_{arch}"] = esc.trajectory_ate(test_name, test_phys, pred, None)
                res[f"physnet_minus_{arch}_mm_95ci_blocks"] = esc._bootstrap_ci(
                    err_phys, np.linalg.norm(pred - y, axis=1), test_phys["t_pair"])
            blend = float(a_best) * (float(w_best) * phys_pred + (1.0 - float(w_best)) * arrays["imunet"])
            equal = 0.5 * (phys_pred + arrays["imunet"])
            arrays.update(cv_calibrated_blend=blend, equal_blend=equal)
            res["equal_blend_physnet_imunet"] = api._metrics(equal, y)
            res["trajectory_equal_blend"] = esc.trajectory_ate(test_name, test_phys, equal, None)
            res["cv_calibrated_blend"] = api._metrics(blend, y)
            res["cv_calibrated_blend_minus_imunet_mm_95ci_blocks"] = esc._bootstrap_ci(
                np.linalg.norm(blend - y, axis=1), np.linalg.norm(arrays["imunet"] - y, axis=1),
                test_phys["t_pair"])
            test_out[test_name] = res
            for k, v in arrays.items():
                pooled.setdefault(k, []).append(v)
                arrays_all[f"{test_name}__{k}"] = v
        y_all = np.concatenate(pooled["y"])
        test_pooled = {"sessions": test_sessions, "n": int(len(y_all)),
                       "zero_motion": api._metrics(np.zeros_like(y_all), y_all),
                       **{k: api._metrics(np.concatenate(pooled[k]), y_all)
                          for k in ("physnet", *BASELINES, "equal_blend", "cv_calibrated_blend")}}
        report = {
            "cv_calibration": {
                "blend_weight_physnet": round(float(w_best), 3), "shrink": round(float(a_best), 3),
                "cv_err_mm_physnet": round(cv_err(1.0, 1.0), 2), "cv_err_mm_imunet": round(cv_err(0.0, 1.0), 2),
                "cv_err_mm_equal_blend": round(cv_err(0.5, 1.0), 2),
                "cv_err_mm_calibrated": round(cv_err(w_best, a_best), 2),
            },
            "test_sessions": test_out,
            "test_pooled": test_pooled,
        }
        (CHECK / f"session_cv_matched_{tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez(CHECK / f"session_cv_matched_{tag}.npz", **arrays_all)
    summary = CHECK / "matched_benchmark_summary.json"
    _run(["tools/summarize_matched_benchmark.py", "--cv-dir", str(CHECK),
          "--physnet-dir", PHYS_REL, "--ext-dir", EXT_REL, "--output", str(summary)])
    return summary


def _load_ext(arch: str, path: Path, device):
    net = make_model(arch).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    return net


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds = split["train"] + split["val"]
    check_stored_protocol(folds)
    CHECK.mkdir(parents=True, exist_ok=True)
    if not args.eval_only:
        train_physnet_folds(folds, gated=False)
        train_physnet_folds(folds, gated=True)
        for arch in BASELINES:
            train_baseline_folds(folds, arch)
    summary = evaluate_fresh_folds(folds, split["test"])
    print("wrote", summary, flush=True)


if __name__ == "__main__":
    main()
