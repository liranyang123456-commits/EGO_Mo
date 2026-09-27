#!/usr/bin/env python3
"""Create the minimal source/data archive needed for remote training."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = DATA / "fleet_train_bundle.tgz"


def main() -> None:
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    names = split["train"] + split["val"]
    with tarfile.open(OUT, "w:gz") as archive:
        archive.add(ROOT / "ego_capture", arcname="ego_capture")
        for rel in ("tools/train_seq.py", "tools/train_trajectory.py"):
            archive.add(ROOT / rel, arcname=rel)
        for rel in ("pose_gt_raw", "pose_gt"):
            archive.add(DATA / rel, arcname=f"datasets/{rel}")
        archive.add(
            DATA / "trajectory_split_20260924.json",
            arcname="datasets/trajectory_split_20260924.json",
        )
        for name in names:
            archive.add(
                DATA / name / "imu_stream.csv",
                arcname=f"datasets/{name}/imu_stream.csv",
            )
    print(f"{OUT} {OUT.stat().st_size / 1024 / 1024:.1f} MiB")


if __name__ == "__main__":
    main()
