#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Assemble realistic digital-twin pretraining corpora with motion classes.

Merges generator corpora into manifests with absolute paths (the format
tools/train_physnet_v3.py and tools/train_external_architectures.py accept)
and labels every sequence with a motion class:

    pause   handheld_pause: non-periodic handheld motion, 1--2 s full stop every 10--15 s
    free    handheld_free:  the same motion without full stops
    spline  random_spline:  faster random-knot spline (legacy family)
    replay  real training trajectories replayed at random speed/amplitude
    periodic  sinusoid-based families of the earlier twin (lateral, circle, ...)

    python tools/twin_realistic_mix.py --corpora A B --name realistic
    python tools/twin_realistic_mix.py --corpora A B --name handheld --classes pause free
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "synthetic_realistic"
CLASS = {"handheld_pause": "pause", "handheld_free": "free", "random_spline": "spline",
         "replay": "replay"}


def motion_class(motion: str) -> str:
    return CLASS.get(motion, "periodic")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="+", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--classes", nargs="*", default=[],
                    help="keep only these motion classes (default: all)")
    args = ap.parse_args()
    merged = {"version": 3, "sources": [], "imu": {"train": [], "val": [], "test": [], "ood": []},
              "visual": {"train": [], "val": [], "test": [], "ood": []}, "blender": []}
    for corpus in args.corpora:
        corpus = corpus.resolve()
        manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
        merged["sources"].append({"corpus": str(corpus), "motions": manifest.get("motions"),
                                  "camera_fps": manifest.get("camera_fps"),
                                  "pause_interval_s": manifest.get("pause_interval_s")})
        for split, entries in manifest["imu"].items():
            for e in entries:
                cls = motion_class(e["motion"])
                if args.classes and cls not in args.classes:
                    continue
                path = Path(e["path"])
                merged["imu"][split].append(e | {"path": str(path if path.is_absolute()
                                                             else corpus / path),
                                                 "motion_class": cls})
    merged["counts"] = {split: dict(collections.Counter(e["motion_class"] for e in v))
                        for split, v in merged["imu"].items()}
    out = OUT / f"mix_{args.name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(out, json.dumps(merged["counts"]))


if __name__ == "__main__":
    main()
