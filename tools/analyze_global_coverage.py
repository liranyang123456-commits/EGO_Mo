#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Survey two-board detection coverage in a 4K global-view session.

Runs the two-board detector on every saved cam2 frame and reports where both
boards are visible together -- those are the segments from which global ground
truth can be estimated. Writes a per-frame CSV and prints a segment table.

Usage:
    python tools/analyze_global_coverage.py <gtraj_session_dir> [--min-segment-s 3]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.global_cam import TwoBoardDetector  # noqa: E402


def analyze(session: str, min_segment_s: float = 3.0) -> dict:
    root = Path(session)
    cam2 = root / "cam2"
    times = np.loadtxt(cam2 / "times.txt", dtype=np.float64)
    images = sorted((cam2 / "images").glob("*.jpg"))
    n = min(len(times), len(images))
    if n == 0:
        raise SystemExit(f"{session}: 没有 4K 图像")

    detector = TwoBoardDetector()
    found_a = np.zeros(n, dtype=bool)
    found_b = np.zeros(n, dtype=bool)
    for k in range(n):
        frame = cv2.imread(str(images[k]))
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = detector.detect(gray)
        found_a[k] = bool(res["A"].get("found"))
        found_b[k] = bool(res["B"].get("found"))
        if (k + 1) % 200 == 0:
            print(f"  {k+1}/{n}", flush=True)

    t = times[:n] - times[0]
    both = found_a & found_b

    # Contiguous both-visible segments (allow short gaps <= 5 frames).
    gap = 5
    segments = []
    start = None
    last_good = None
    for k in range(n):
        if both[k]:
            if start is None:
                start = k
            last_good = k
        else:
            if start is not None and (k - last_good) > gap:
                segments.append((start, last_good))
                start = None
    if start is not None:
        segments.append((start, last_good))

    seg_rows = []
    for (s0, s1) in segments:
        dur = float(t[s1] - t[s0])
        if dur >= min_segment_s:
            seg_rows.append({
                "start_frame": int(s0),
                "end_frame": int(s1),
                "start_s": round(float(t[s0]), 2),
                "end_s": round(float(t[s1]), 2),
                "duration_s": round(dur, 2),
                "frames": int(s1 - s0 + 1),
            })

    out_csv = root / "coverage_cam2.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["frame_idx", "t_s", "boardA_found", "boardB_found", "both"])
        for k in range(n):
            writer.writerow([k, f"{t[k]:.3f}", int(found_a[k]), int(found_b[k]), int(both[k])])

    summary = {
        "session": root.name,
        "frames": int(n),
        "duration_s": round(float(t[-1]), 2),
        "coverage_A": round(float(found_a.mean()), 3),
        "coverage_B": round(float(found_b.mean()), 3),
        "coverage_both": round(float(both.mean()), 3),
        "segments_ge_min": seg_rows,
        "min_segment_s": min_segment_s,
        "output": str(out_csv),
    }
    (root / "coverage_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--min-segment-s", type=float, default=3.0)
    args = ap.parse_args()
    s = analyze(args.session, args.min_segment_s)
    print(f"\n=== {s['session']} ===")
    print(f"帧数 {s['frames']}  时长 {s['duration_s']}s")
    print(f"覆盖 A {s['coverage_A']:.0%}  B {s['coverage_B']:.0%}  同时 {s['coverage_both']:.0%}")
    print(f"≥{s['min_segment_s']}s 的可用片段：{len(s['segments_ge_min'])} 段")
    for seg in s["segments_ge_min"]:
        print(f"  {seg['start_s']:>7.2f}s - {seg['end_s']:>7.2f}s  "
              f"({seg['duration_s']:>5.1f}s, {seg['frames']} 帧)")


if __name__ == "__main__":
    main()
