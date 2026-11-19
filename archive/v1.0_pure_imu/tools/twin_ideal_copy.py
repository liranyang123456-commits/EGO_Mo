#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Copy a twin corpus with the ideal IMU stream in place of the noisy one.

Same trajectories, no white noise, bias, random walk or quantization, so the
difference to the original corpus isolates the sensor error from the motion
(initial-velocity) limit.

    python tools/twin_ideal_copy.py pause_inf pause_4
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / "datasets" / "synthetic_protocol"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="+")
    args = ap.parse_args()
    for name in args.names:
        src = Path((PROTO / name / "latest.txt").read_text(encoding="utf-8").strip())
        dst = PROTO / f"ideal_{name}" / src.name
        for seq in sorted((src / "imu").iterdir()):
            out = dst / "imu" / seq.name
            out.mkdir(parents=True, exist_ok=True)
            for f in seq.iterdir():
                if f.is_file() and f.name != "imu_stream.csv":
                    shutil.copy2(f, out / f.name)
            shutil.copy2(seq / "imu_stream_ideal.csv", out / "imu_stream.csv")
            for cam in ("cam0", "cam1"):
                if (seq / cam).is_dir():
                    shutil.copytree(seq / cam, out / cam, dirs_exist_ok=True)
        shutil.copy2(src / "manifest.json", dst / "manifest.json")
        (PROTO / f"ideal_{name}" / "latest.txt").write_text(str(dst), encoding="utf-8")
        print(dst)


if __name__ == "__main__":
    main()
