#!/usr/bin/env python3
"""Compose a curriculum manifest without duplicating sequence files."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

NEW_MOTIONS = {
    "slow_far", "aggressive_6dof", "pure_rotation", "translation_xyz",
    "pitch_sweep", "stop_go", "spiral",
}


def _absolute_entries(root: Path, entries: list[dict]) -> list[dict]:
    result = []
    for entry in entries:
        copy = dict(entry)
        copy["path"] = str((root / entry["path"]).resolve())
        result.append(copy)
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("expanded", type=Path)
    ap.add_argument("--new-train-per-motion", type=int, default=2)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()
    base = json.loads((args.base / "manifest.json").read_text(encoding="utf-8"))
    expanded = json.loads((args.expanded / "manifest.json").read_text(encoding="utf-8"))
    out = args.output_root / f"curriculum_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": 1,
        "kind": "cross-corpus curriculum",
        "motions": base["motions"] + sorted(NEW_MOTIONS),
        "imu": {"train": [], "val": [], "test": [], "ood": []},
        "visual": {"train": [], "val": [], "test": [], "ood": []},
        "blender": [],
    }
    manifest["imu"]["train"] = _absolute_entries(args.base, base["imu"]["train"])
    for motion in sorted(NEW_MOTIONS):
        entries = [e for e in expanded["imu"]["train"] if e["motion"] == motion]
        manifest["imu"]["train"] += _absolute_entries(
            args.expanded, entries[:args.new_train_per_motion]
        )
    for split in ("val", "test", "ood"):
        manifest["imu"][split] = _absolute_entries(args.base, base["imu"][split])
        for motion in sorted(NEW_MOTIONS):
            entries = [e for e in expanded["imu"][split] if e["motion"] == motion]
            if entries:
                manifest["imu"][split] += _absolute_entries(args.expanded, entries[:1])
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(out)


if __name__ == "__main__":
    main()
