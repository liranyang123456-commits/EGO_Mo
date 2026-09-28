#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precompute median dense optical flow for every sequence in a corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim.core import SimConfig  # noqa: E402
from tools.train_physnet_flow import flow_series  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    args = ap.parse_args()
    root = Path((PROTO / args.corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    cfg = SimConfig()
    cfg.width, cfg.height = 640, 360
    cfg.scene_texture = True
    for split in ("train", "val", "test"):
        for e in manifest["imu"][split]:
            seq = (root / e["path"]).resolve()
            cache = seq / "flow_median.npz"
            if cache.is_file():
                print("cached", seq.name, flush=True)
                continue
            t, flow = flow_series(seq, cfg)
            print("done", seq.name, len(flow), flush=True)


if __name__ == "__main__":
    main()
