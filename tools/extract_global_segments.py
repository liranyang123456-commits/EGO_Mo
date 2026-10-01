#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Package the usable segments of a 4K global-view session into clean
ground-truth bundles.

Reads global_pose_gt.csv (already restricted to the both-boards-visible
segments by build_global_pose_gt.py --only-segments) and, when the rig
transform T_C0_A is available, derives the left camera's global pose
T_G_C0(t) = T_G_A(t) @ inv(T_C0_A) and the cross-check T_C0_B(t) =
inv(T_G_C0(t)) @ T_G_B(t) (the GP050 board in the left-camera frame, which the
stereo chain also measures directly).

Writes one enriched CSV per segment under <session>/segments/ and a summary.

Usage:
    python tools/extract_global_segments.py <gtraj_session_dir>
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.geometry import se3_inv  # noqa: E402

POSES = ("G_A", "G_B", "G_C0", "C0_B")


def _read_pose_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            rows.append(rec)
    return rows


def _parse_T(row: dict[str, Any], name: str) -> np.ndarray | None:
    if row.get(f"{name}_tx_mm") in (None, ""):
        return None
    R = np.array([float(row[f"{name}_r{r}{c}"]) for r in range(3) for c in range(3)]).reshape(3, 3)
    t = np.array([float(row[f"{name}_t{ax}_mm"]) for ax in "xyz"]) / 1000.0
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _T_to_fields(T: np.ndarray) -> list[float]:
    R = T[:3, :3]
    t = T[:3, 3] * 1000.0
    return [float(v) for v in R.reshape(-1)] + [float(v) for v in t]


def _motion_stats(times: np.ndarray, Ts: list[np.ndarray]) -> dict[str, float]:
    """Total path length and rotation range over the segment."""
    if len(Ts) < 2:
        return {"path_mm": 0.0, "rot_range_deg": 0.0}
    pts = np.stack([T[:3, 3] for T in Ts])
    path = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)) * 1000.0)
    # rotation range: max pairwise angle from the first frame
    R0 = Ts[0][:3, :3]
    angs = []
    for T in Ts:
        Rd = R0.T @ T[:3, :3]
        angs.append(float(np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) * 0.5, -1, 1)))))
    return {"path_mm": round(path, 1), "rot_range_deg": round(max(angs), 1)}


def extract(session: str) -> dict[str, Any]:
    root = Path(session)
    pose_csv = root / "global_pose_gt.csv"
    cov_path = root / "coverage_summary.json"
    if not pose_csv.is_file() or not cov_path.is_file():
        raise SystemExit("先运行 analyze_global_coverage.py 和 build_global_pose_gt.py --only-segments")

    rig_T = None
    rig_state = ROOT / "datasets" / "global_rig_state.json"
    if rig_state.is_file():
        state = json.loads(rig_state.read_text(encoding="utf-8"))
        rig_T = np.asarray(state["T_C0_A"]["matrix"], dtype=np.float64)

    cov = json.loads(cov_path.read_text(encoding="utf-8"))
    segments = cov.get("segments_ge_min", [])
    rows = _read_pose_csv(pose_csv)
    by_frame = {int(r["frame_idx"]): r for r in rows}

    out_dir = root / "segments"
    out_dir.mkdir(exist_ok=True)
    seg_summaries = []
    for si, seg in enumerate(segments):
        s0, s1 = int(seg["start_frame"]), int(seg["end_frame"])
        seg_rows = []
        times = []
        Ts_A = []
        for fi in range(s0, s1 + 1):
            row = by_frame.get(fi)
            if row is None:
                continue
            T_G_A = _parse_T(row, "G_A")
            T_G_B = _parse_T(row, "G_B")
            T_G_C0 = None
            T_C0_B = None
            if T_G_A is not None and rig_T is not None:
                T_G_C0 = T_G_A @ se3_inv(rig_T)
                if T_G_B is not None:
                    T_C0_B = se3_inv(T_G_C0) @ T_G_B
            out = {
                "frame_idx": fi,
                "stamp": row["stamp"],
                "A_reproj_px": row.get("A_reproj_px", ""),
                "B_reproj_px": row.get("B_reproj_px", ""),
            }
            for name, T in (("G_A", T_G_A), ("G_B", T_G_B), ("G_C0", T_G_C0), ("C0_B", T_C0_B)):
                fields = _T_to_fields(T) if T is not None else [""] * 12
                for ax, v in zip(("x", "y", "z"), fields[9:12]):
                    out[f"{name}_t{ax}_mm"] = v
            seg_rows.append(out)
            times.append(float(row["stamp"]))
            if T_G_A is not None:
                Ts_A.append(T_G_A)

        if not seg_rows:
            continue
        seg_csv = out_dir / f"segment_{si:02d}_{seg['start_s']:.0f}s_{seg['end_s']:.0f}s.csv"
        fieldnames = list(seg_rows[0].keys())
        with seg_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(seg_rows)

        stats = _motion_stats(np.asarray(times), Ts_A)
        seg_summaries.append({
            "segment": si,
            "start_s": seg["start_s"],
            "end_s": seg["end_s"],
            "duration_s": seg["duration_s"],
            "frames_written": len(seg_rows),
            "rigA_path_mm": stats["path_mm"],
            "rigA_rot_range_deg": stats["rot_range_deg"],
            "csv": str(seg_csv),
        })

    summary = {
        "session": root.name,
        "has_rig_transform": rig_T is not None,
        "segments": seg_summaries,
        "total_usable_s": round(sum(s["duration_s"] for s in seg_summaries), 1),
        "output_dir": str(out_dir),
    }
    (root / "segments_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python tools/extract_global_segments.py <gtraj会话目录>")
        raise SystemExit(1)
    s = extract(sys.argv[1])
    print(f"=== {s['session']} ===")
    print(f"含刚体外参: {s['has_rig_transform']}   可用总时长 {s['total_usable_s']}s")
    for seg in s["segments"]:
        print(f"  段{seg['segment']}: {seg['start_s']:.1f}-{seg['end_s']:.1f}s "
              f"({seg['duration_s']:.1f}s, {seg['frames_written']}帧)  "
              f"刚体A行程 {seg['rigA_path_mm']:.0f}mm 转角范围 {seg['rigA_rot_range_deg']:.0f}°")
    print(f"输出 {s['output_dir']}")


if __name__ == "__main__":
    main()
