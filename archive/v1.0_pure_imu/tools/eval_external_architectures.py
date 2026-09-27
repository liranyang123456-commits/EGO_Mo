#!/usr/bin/env python3
"""Evaluate protocol-adapted public architectures on common test splits."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
from tools.train_external_architectures import make_model, predict
from tools.train_hybrid import _synthetic_build

DATA = ROOT / "datasets"
ARCHES = ("ronin_resnet", "ronin_lstm", "tlio_resnet", "imunet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--run-dir", default="external_benchmark")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context = 2.0
    length = int(round((1.5 * 3.0 + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    real_parts = [api.build(name, context, length) for name in split["test"]]
    real = (
        np.concatenate([p[0] for p in real_parts]),
        np.concatenate([p[1] for p in real_parts]),
    )
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    rng = np.random.default_rng(123)
    synth_parts = [
        _synthetic_build(args.corpus / e["path"], context, length, 3.0, 160, rng)
        for e in manifest["imu"]["test"]
    ]
    synth = (
        np.concatenate([p[0] for p in synth_parts if p is not None]),
        np.concatenate([p[1] for p in synth_parts if p is not None]),
    )
    out = {}
    root = DATA / args.run_dir
    for arch in ARCHES:
        out[arch] = {}
        for mode, suffix in (("zero_shot", "zero"), ("fine_tuned", "fine"), ("real_only", "real")):
            model = make_model(arch).to(device)
            model.load_state_dict(torch.load(root / f"{arch}_{suffix}_s0.pt", map_location=device))
            model.eval()
            out[arch][mode] = {
                "synthetic_test": api._metrics(predict(model, synth[0], device), synth[1]),
                "real_test": api._metrics(predict(model, real[0], device), real[1]),
            }
        print(json.dumps({arch: out[arch]}, ensure_ascii=False), flush=True)
    (root / "test_benchmark.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
