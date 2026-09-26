#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Cleaned raw chessboard PnP in the same layout as the locked reference.

A frame is a label when the board is still (BLE gyro < 8 deg/s), the solve
reprojects within 1.5 px and the jump filter kept it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = DATA / "pose_gt_raw"


def main() -> None:
    index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
    OUT.mkdir(parents=True, exist_ok=True)
    names = [row["name"] for row in index]
    # Rigidity-check recordings (rigid_*) also carry chessboard poses; export
    # them too so they can serve as extra rotation-rich training data.
    names += sorted(p.stem for p in (DATA / "pose_gt").glob("*.npz") if p.stem not in names)
    rows = []
    for name in names:
        gt = np.load(DATA / "pose_gt" / f"{name}.npz")
        usable = ((gt["ok"] == 1) & (gt["reproj"] < 1.5) & (gt["ble_gyro"] < 8.0)).astype(np.uint8)
        np.savez_compressed(OUT / f"{name}.npz", t=gt["t"], R=gt["R"].astype(np.float32),
                            p=gt["p"].astype(np.float32), usable=usable, px=gt["reproj"].astype(np.float32))
        rows.append({"name": name, "frames": int(len(usable)), "usable": int(usable.sum())})
        print(json.dumps(rows[-1]), flush=True)
    (OUT / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
