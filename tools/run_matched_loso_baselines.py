#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Leave-one-session-out fold models for the adapted public baselines.

Trains each architecture with exactly the protocol used for the IMUNet fold
models (``imunet_cvx_*``): the eight non-test trajectory recordings as folds,
the four rigidity recordings that pass the mount test as training-only data,
and two seeds per fold (sixteen models per architecture). Existing outputs
are skipped, so the script can be resumed.

    python tools/run_matched_loso_baselines.py --arch ronin_resnet tlio_resnet
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"

PAPER_EXTRA_TRAIN = [
    "rigid_20260924_123958",
    "rigid_20260924_124243",
    "rigid_20260924_140510",
    "rigid_20260924_142902",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", nargs="+", default=["ronin_resnet", "tlio_resnet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--tag", default="cvx")
    args = ap.parse_args()
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds = split["train"] + split["val"]
    reference = json.loads((DATA / "external_benchmark" / f"imunet_{args.tag}_{folds[0]}_s0.json")
                           .read_text(encoding="utf-8"))
    expected = sorted(n for n in reference["train_sessions"] if n.startswith("rigid_"))
    assert expected == sorted(PAPER_EXTRA_TRAIN), f"IMUNet fold models used {expected}"

    for arch in args.arch:
        for ses in folds:
            for seed in args.seeds:
                stem = f"{arch}_{args.tag}_{ses}_s{seed}"
                if (DATA / "external_benchmark" / f"{stem}.pt").is_file():
                    print(f"skip {stem}", flush=True)
                    continue
                cmd = [sys.executable, str(ROOT / "tools" / "train_external_architectures.py"),
                       "--arch", arch, "--only", "real_only", "--holdout", ses,
                       "--seed", str(seed), "--stem", stem,
                       "--extra-train", *PAPER_EXTRA_TRAIN]
                start = time.time()
                subprocess.run(cmd, check=True, cwd=ROOT)
                print(f"done {stem} in {time.time() - start:.0f} s", flush=True)


if __name__ == "__main__":
    main()
