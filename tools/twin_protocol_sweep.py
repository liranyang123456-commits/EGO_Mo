#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Acquisition-protocol study in the digital twin (synthetic data only).

The initial velocity of a 3-s window is not observable from the IMU unless the
rig was at rest shortly before. This sweep trains synthetic-only PhysNet
models on handheld_pause corpora that differ only in the acquisition protocol
and scores them on synthetic test sequences:

* pause interval: one 1--2 s full stop every 4 / 8 / 12.5 / 20 s, or never;
* look-back: 2 s context (base) or an additional 8 s coarse context;
* reference frame rate of the labels: 10 / 15 / 30 fps (pause every 12.5 s),
  all scored on the same 15 fps test set.

    python tools/twin_protocol_sweep.py train --jobs pause_4:short pause_4:long
    python tools/twin_protocol_sweep.py eval
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
PROTO = DATA / "synthetic_protocol"
OUT = "physnet_proto"
BASE = ["--weight-decay", "0.1", "--still-weight", "1.0", "--gate"]
VARIANTS = {"short": [], "long": ["--coarse", "8.0"], "nb": ["--bias-mode", "none"]}
JOBS = ([f"{c}:short" for c in ("pause_4", "pause_8", "pause_12.5", "pause_20", "pause_inf",
                                "fps_10", "fps_30")]
        + [f"{c}:long" for c in ("pause_4", "pause_8", "pause_12.5", "pause_20", "pause_inf")]
        + [f"ideal_{c}:short" for c in ("pause_4", "pause_inf")]
        + [f"{c}:nb" for c in ("pause_4", "pause_8", "pause_12.5", "pause_20", "pause_inf",
                               "ideal_pause_4", "ideal_pause_inf")])
SINCE_BINS = (0.0, 1.0, 3.0, 6.0, 10.0, np.inf)


def corpus(name: str) -> Path:
    return Path((PROTO / name / "latest.txt").read_text(encoding="utf-8").strip())


def train(job: str, seed: int) -> None:
    name, variant = job.split(":")
    tag = f"{name}_{variant}"
    ck = DATA / OUT / f"physnet_{tag}_s{seed}.pt"
    if ck.is_file():
        return
    cmd = [sys.executable, "-u", str(ROOT / "tools" / "train_physnet_v3.py"), "train",
           "--split-file", "trajectory_split_paper.json", "--out", OUT, "--tag", tag,
           "--seed", str(seed), "--pretrain-corpus", str(corpus(name)), "--synthetic-only",
           *BASE, *VARIANTS[variant]]
    log = DATA / OUT / f"log_{tag}_s{seed}.txt"
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True)


def evaluate(seeds) -> dict:
    import tools.train_physnet_v3 as v3
    import tools.train_seq as api
    from tools.twin_motion_class_eval import window_features
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = {}

    def test_set(name, coarse):
        key = (name, coarse, v3.BIAS_MODE)
        if key not in cache:
            manifest = json.loads((corpus(name) / "manifest.json").read_text(encoding="utf-8"))
            paths = [v3._corpus_path(corpus(name), e["path"]) for e in manifest["imu"]["test"]]
            data = v3.build_group(paths, context, horizon, length, coarse=coarse, cap=160,
                                  rng=np.random.default_rng(123))
            feats = {"since_stop": [], "imu_still": [], "stop_class": []}
            for p in paths:
                sel = data["session"] == str(p)
                sub = {"pair": data["pair"][sel]}
                f = window_features(p, sub)
                for key in feats:
                    feats[key].append(f[key])
            cache[key] = (data, {k: np.concatenate(v) for k, v in feats.items()})
        return cache[key]

    def score(pred, y, feats):
        e = np.linalg.norm(pred - y, axis=1) * 1000
        z = np.linalg.norm(y, axis=1) * 1000
        row = {"n": int(len(y)), "err_mm": round(float(e.mean()), 2),
               "zero_mm": round(float(z.mean()), 2),
               "slope": round(float((pred * y).sum() / (y * y).sum()), 3),
               "r2": round(float(1 - ((pred - y) ** 2).sum() / ((y - y.mean(0)) ** 2).sum()), 3),
               "by_since_stop": {}, "by_stop_position": {}}
        for cls in ("free", "context", "window"):
            m = feats["stop_class"] == cls
            if m.sum() >= 30:
                row["by_stop_position"][cls] = {
                    "n": int(m.sum()), "share": round(float(m.mean()), 3),
                    "err_mm": round(float(e[m].mean()), 2), "zero_mm": round(float(z[m].mean()), 2),
                    "slope": round(float((pred[m] * y[m]).sum() / (y[m] * y[m]).sum()), 3)}
        ss = feats["since_stop"]
        for lo, hi in zip(SINCE_BINS[:-1], SINCE_BINS[1:]):
            m = (ss >= lo) & (ss < hi)
            if m.sum() >= 30:
                row["by_since_stop"][f"{lo:g}-{hi:g}"] = {
                    "n": int(m.sum()), "err_mm": round(float(e[m].mean()), 2),
                    "zero_mm": round(float(z[m].mean()), 2),
                    "slope": round(float((pred[m] * y[m]).sum() / (y[m] * y[m]).sum()), 3)}
        return row

    report = {"seeds": seeds, "models": {}}
    for job in JOBS:
        name, variant = job.split(":")
        tag = f"{name}_{variant}"
        paths = [DATA / OUT / f"physnet_{tag}_s{s}.pt" for s in seeds]
        if not all(p.is_file() for p in paths):
            continue
        models = [v3.load_model(p, device) for p in paths]
        coarse = 8.0 if variant == "long" else 0.0
        targets = [name, "pause_12.5"] if name.startswith("fps") else [name]
        entry = {}
        for target in targets:
            data, feats = test_set(target, coarse)
            dt = v3.to_device(data, device)
            pred = np.mean([m.predict(dt) for m in models], axis=0)
            entry[f"test_{target}"] = score(pred, data["y"], feats)
            del dt
            torch.cuda.empty_cache()
        report["models"][tag] = entry
        own = entry[f"test_{targets[-1]}"]
        print(f"{tag:18s} err {own['err_mm']:6.2f} zero {own['zero_mm']:6.2f} "
              f"slope {own['slope']:.3f} R2 {own['r2']:.3f}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--jobs", nargs="*", default=JOBS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--output", type=Path, default=DATA / "twin_protocol_sweep.json")
    args = ap.parse_args()
    if args.cmd == "train":
        for job in args.jobs:
            for s in args.seeds:
                train(job, s)
                print("done", job, s, flush=True)
    else:
        report = evaluate(args.seeds)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
