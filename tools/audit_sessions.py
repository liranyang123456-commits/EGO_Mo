#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score captured sessions for trajectory training."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def _imu_gaps(path: Path) -> dict:
    if not path.is_file():
        return {"n": 0}
    ts = []
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            try:
                ts.append(float(rec["timestamp"]))
            except (KeyError, ValueError):
                continue
    if len(ts) < 5:
        return {"n": len(ts)}
    d = np.diff(np.asarray(ts))
    return {
        "n": len(ts),
        "hz": (len(ts) - 1) / max(ts[-1] - ts[0], 1e-6),
        "dt_p50_ms": float(np.percentile(d, 50) * 1000),
        "dt_p95_ms": float(np.percentile(d, 95) * 1000),
        "gaps_gt_20ms": int((d > 0.02).sum()),
        "gaps_gt_50ms": int((d > 0.05).sum()),
    }


def _count(path: Path, suffix: str) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() == suffix)


def _size(path: Path) -> list[int]:
    if not path.is_file():
        return []
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return []
    return [int(im.shape[1]), int(im.shape[0])]


def _lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def main() -> None:
    rows = []
    for summary_path in sorted(DATA.glob("*/capture_summary.json")):
        sess = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        meta_path = sess / "session_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        n0 = _count(sess / "cam0" / "images", ".jpg")
        n1 = _count(sess / "cam1" / "images", ".jpg")
        size = _size(next((sess / "cam0" / "images").glob("*.jpg"), Path("missing")))
        usb = _imu_gaps(sess / "imu_stream.csv")
        bt = _imu_gaps(sess / "imu_bt.csv")
        sync = (summary.get("sync") or {}).get("cameras") or {}
        cam0 = sync.get("cam0") or {}
        usable = (
            sess.name.startswith("traj_")
            and float(summary.get("seconds") or 0) >= 40
            and usb.get("n", 0) > 1000
            and usb.get("hz", 0) >= 150
            and usb.get("gaps_gt_50ms", 99) <= 5
            and bt.get("n", 0) > 1000
            and bt.get("hz", 0) >= 120
            and n0 >= 200
            and n0 >= 0.85 * max(int(summary.get("frames") or 0), 1)
        )
        rows.append({
            "name": sess.name,
            "step": meta.get("step"),
            "seconds": round(float(summary.get("seconds") or 0), 1),
            "frames": summary.get("frames"),
            "size": size,
            "requested": meta.get("requested"),
            "jpg0": n0,
            "jpg1": n1,
            "times0": _lines(sess / "cam0" / "times.txt"),
            "times1": _lines(sess / "cam1" / "times.txt"),
            "drops": summary.get("drops"),
            "usb": usb,
            "bt": bt,
            "usb_delay_ms": None if cam0.get("delay_usb_s") is None else round(float(cam0["delay_usb_s"]) * 1000, 1),
            "usb_corr": None if cam0.get("delay_usb_corr") is None else round(float(cam0["delay_usb_corr"]), 3),
            "usb_reliable": cam0.get("delay_usb_reliable"),
            "calib_l": summary.get("calib_left"),
            "calib_r": summary.get("calib_right"),
            "usable": usable,
        })
    text = json.dumps(rows, ensure_ascii=False, indent=2)
    out = DATA / "audit_sessions.json"
    out.write_text(text, encoding="utf-8")
    print(text)
    print("usable", sum(1 for r in rows if r["usable"]), "of", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
