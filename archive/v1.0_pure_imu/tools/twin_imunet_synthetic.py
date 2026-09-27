#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapted IMUNet on the acquisition-protocol corpora, with and without switching.

Trains synthetic-only IMUNet models like the synthetic-direct step of the
frozen tools/train_external_architectures.py (25 epochs, lr 1e-3, patience 8),
with 120 pairs per sequence as for PhysNet, and scores them on the same test windows as
PhysNet (identical pair enumeration and subsampling), alone and with the
stop-anchored switching of tools/stop_anchor_check.py. The switching threshold
is chosen on the synthetic validation split for IMUNet as well.

    python tools/twin_imunet_synthetic.py train --corpora pause_4 pause_inf
    python tools/twin_imunet_synthetic.py eval --corpora pause_4 pause_12.5 pause_inf
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

import tools.train_physnet_v3 as v3  # noqa: E402
import tools.train_seq as api  # noqa: E402
from tools.train_external_architectures import make_model, predict, train_phase  # noqa: E402
from tools.train_hybrid import _synthetic_build  # noqa: E402
from tools.stop_anchor_check import SPANS, predictions, summarize  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"
OUT = DATA / "imunet_proto"
CONTEXT, HORIZON = 2.0, 3.0
LENGTH = int(round((1.5 * HORIZON + 2 * CONTEXT) * api.HZ))
CAP = 120  # pairs per training sequence, as for the synthetic-only PhysNet models


def _paths(corpus: str, split: str):
    root = Path((PROTO / corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    return [v3._corpus_path(root, e["path"]) for e in manifest["imu"][split]]


def _build(paths, cap, rng):
    parts = [_synthetic_build(p, CONTEXT, LENGTH, HORIZON, cap, rng) for p in paths]
    parts = [p for p in parts if p is not None]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def train(corpus: str, seed: int, device, epochs=25, patience=8, tag="") -> None:
    ck = OUT / f"imunet{tag}_{corpus}_s{seed}.pt"
    if ck.is_file():
        return
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Xtr, ytr = _build(_paths(corpus, "train"), CAP, rng)
    Xva, yva = _build(_paths(corpus, "val"), CAP, rng)
    model = make_model("imunet").to(device)
    train_phase(model, Xtr, ytr, Xva, yva, device, epochs, 1e-3, seed, patience)
    OUT.mkdir(exist_ok=True)
    torch.save(model.state_dict(), ck)
    print("trained", corpus, seed, flush=True)


def _scored(corpus, split, nets, device):
    """PhysNet-aligned windows with IMUNet as the learned estimate."""
    v3.BIAS_MODE = "none"
    names = _paths(corpus, split)
    P = predictions(names, [], cap=160)
    X, y = _build(names, 160, np.random.default_rng(123))
    assert np.allclose(y, P["y"], atol=1e-6)
    P["learned"] = np.mean([predict(n, X, device) for n in nets], axis=0)
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--corpora", nargs="+", default=["pause_4", "pause_12.5", "pause_inf"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--tag", default="", help="checkpoint name suffix, e.g. _e50")
    ap.add_argument("--max-span", type=float, default=None,
                    help="use this switching threshold instead of selecting one on validation")
    ap.add_argument("--output", type=Path, default=DATA / "twin_imunet_synthetic.json")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.cmd == "train":
        for c in args.corpora:
            for s in args.seeds:
                train(c, s, device, args.epochs, args.patience, args.tag)
        return
    report = {"seeds": args.seeds, "tag": args.tag, "corpora": {}}
    val, test = {}, {}
    for c in args.corpora:
        nets = []
        for s in args.seeds:
            net = make_model("imunet").to(device)
            net.load_state_dict(torch.load(OUT / f"imunet{args.tag}_{c}_s{s}.pt",
                                           map_location=device))
            net.eval()
            nets.append(net)
        val[c] = _scored(c, "val", nets, device)
        test[c] = _scored(c, "test", nets, device)
    from tools.stop_anchor_check import pooled_err
    curve = {s: round(pooled_err(list(val.values()), s), 2) for s in SPANS}
    best = args.max_span if args.max_span is not None else min(curve, key=curve.get)
    report["val_pooled_err_by_max_span"] = curve
    report["chosen_max_span_s"] = best
    for c, P in test.items():
        report["corpora"][c] = summarize(P, best)
        r = report["corpora"][c]["all"]
        print(f"{c:10s} IMUNet {r['learned']:6.2f}  +switch {r['hybrid']:6.2f}  zero {r['zero']:6.2f}",
              flush=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
