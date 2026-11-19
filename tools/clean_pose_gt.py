#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Drop PnP poses that jump relative to the previous kept pose."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GT = ROOT / "datasets" / "pose_gt"


def _angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    rel = Ra.T @ Rb
    cos = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def clean(path: Path) -> dict:
    data = np.load(path)
    t = data["t"]
    ok = data["ok"].astype(np.uint8).copy()
    R = data["R"]
    p = data["p"]
    last = None
    dropped = 0
    for i in range(len(t)):
        if ok[i] == 0:
            continue
        if last is None:
            last = i
            continue
        dt = float(t[i] - t[last])
        if dt > 0.5:
            last = i
            continue
        dang = _angle(R[last], R[i])
        dmm = float(np.linalg.norm(p[i] - p[last]) * 1000.0)
        # A handheld camera cannot turn 25 deg or jump 40 mm between nearby frames.
        if dang > 25.0 or (dt < 0.2 and dmm > 40.0):
            ok[i] = 0
            dropped += 1
            continue
        last = i
    still = (ok == 1) & (data["ble_gyro"] < 8.0)
    payload = {key: data[key] for key in data.files}
    payload["ok"] = ok
    np.savez_compressed(path, **payload)
    return {
        "name": path.stem,
        "kept": int(ok.sum()),
        "kept_still": int(still.sum()),
        "dropped_jumps": dropped,
    }


def main() -> None:
    rows = []
    for path in sorted(GT.glob("traj_*.npz")):
        rows.append(clean(path))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    index_path = GT / "index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        by = {row["name"]: row for row in rows}
        for row in index:
            extra = by.get(row["name"])
            if extra:
                row["pnp_clean"] = extra["kept"]
                row["pnp_still"] = extra["kept_still"]
                row["dropped_jumps"] = extra["dropped_jumps"]
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
