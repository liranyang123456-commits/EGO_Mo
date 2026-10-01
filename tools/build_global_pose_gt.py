#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Global-pose ground truth for a 4K global-view trajectory session.

For every frame of the fixed 4K camera (cam2), both chessboards are detected
and solved by PnP against the 4K intrinsics:

* board A (7x5 squares of 5 mm, 6x4 inner corners) rides on the
  stereo+USB-IMU rig, so T_G_A(t) is the rig's global pose;
* board B (GP050, 11x8 inner corners) carries the BLE IMU, so T_G_B(t) is the
  second rig's global pose.

With the constant T_C0_A from tools/calibrate_global_rig.py, the left camera's
global pose follows as T_G_C0(t) = T_G_A(t) @ inv(T_C0_A), and the GP050 board
in the left-camera frame is T_C0_B(t) = inv(T_G_C0(t)) @ T_G_B(t) -- the same
quantity the stereo view measures directly, which gives a cross-check.

Usage:
    python tools/build_global_pose_gt.py <gtraj_session_dir> [--write-images]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.geometry import se3, se3_inv  # noqa: E402
from ego_capture.session import INNER, INNER2, SQUARE2_MM, SQUARE_MM  # noqa: E402
from tools.calibrate_global_rig import (  # noqa: E402
    board_object_points,
    detect_full,
    load_intrinsics,
    _latest_file,
)


def _solve(gray, inner, square_mm, K, dist, prev_corners):
    corners = detect_full(gray, inner)
    if corners is None:
        return None, None
    pts = corners.reshape(-1, 2)
    if prev_corners is not None and len(prev_corners) == len(pts):
        d_same = float(np.mean(np.linalg.norm(pts - prev_corners, axis=1)))
        d_flip = float(np.mean(np.linalg.norm(pts[::-1] - prev_corners, axis=1)))
        if d_flip < d_same:
            pts = pts[::-1].copy()
    objp = board_object_points(inner, square_mm)
    ok, rvec, tvec = cv2.solvePnP(
        objp, pts.reshape(-1, 1, 2).astype(np.float64), K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None, None
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    reproj = float(np.sqrt(np.mean((proj.reshape(-1, 2) - pts) ** 2)))
    R, _ = cv2.Rodrigues(rvec)
    return (R, tvec.reshape(3).astype(np.float64), reproj), pts


def _rt_to_row(T: np.ndarray) -> list[float]:
    R = T[:3, :3]
    t = T[:3, 3] * 1000.0  # m -> mm
    return [float(v) for v in R.reshape(-1)] + [float(v) for v in t]


def evaluate(session: str, write_images: bool = False,
             only_segments: bool = False) -> dict[str, Any]:
    root = Path(session)
    cam2 = root / "cam2"
    times_path = cam2 / "times.txt"
    if not times_path.is_file():
        raise SystemExit(f"缺少 {times_path}（该段不含 4K 通道）")
    K4_path = _latest_file(ROOT / "datasets", "calib_g4k_", "camera_calibration_4k.json")
    if K4_path is None:
        raise SystemExit("找不到 4K 内参（先完成第 13 步或运行 tools/selfcalib_4k.py）。")
    K4, dist4 = load_intrinsics(K4_path)

    # Optionally restrict to the usable (both-boards-visible) segments.
    keep: Optional[set[int]] = None
    cov_path = root / "coverage_summary.json"
    if only_segments and cov_path.is_file():
        cov = json.loads(cov_path.read_text(encoding="utf-8"))
        keep = set()
        for seg in cov.get("segments_ge_min", []):
            keep.update(range(int(seg["start_frame"]), int(seg["end_frame"]) + 1))
        print(f"只处理 {len(keep)} 帧（{len(cov.get('segments_ge_min', []))} 个可用片段）")

    rig_state_path = ROOT / "datasets" / "global_rig_state.json"
    T_C0_A: Optional[np.ndarray] = None
    if rig_state_path.is_file():
        state = json.loads(rig_state_path.read_text(encoding="utf-8"))
        T_C0_A = np.asarray(state["T_C0_A"]["matrix"], dtype=np.float64)

    times = np.loadtxt(times_path, dtype=np.float64)
    images = sorted((cam2 / "images").glob("*.jpg")) if (cam2 / "images").is_dir() else []
    use_video = not images
    cap = None
    if use_video:
        cap = cv2.VideoCapture(str(cam2 / "video.avi"))

    out_csv = root / "global_pose_gt.csv"
    header = ["frame_idx", "stamp",
              "A_found", "A_reproj_px", "B_found", "B_reproj_px"]
    for name in ("G_A", "G_B", "G_C0", "C0_B"):
        header += [f"{name}_r{r}{c}" for r in range(3) for c in range(3)]
        header += [f"{name}_tx_mm", f"{name}_ty_mm", f"{name}_tz_mm"]

    n_found_a = n_found_b = n_written = 0
    reproj_a: list[float] = []
    reproj_b: list[float] = []
    prev_a = prev_b = None
    rows_out: list[list[Any]] = []

    n_frames = len(times)
    for idx in range(n_frames):
        if keep is not None and idx not in keep:
            continue
        if use_video:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
        else:
            if idx >= len(images):
                break
            frame = cv2.imread(str(images[idx]))
            if frame is None:
                continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        pa, pts_a = _solve(gray, INNER2, SQUARE2_MM, K4, dist4, prev_a)
        pb, pts_b = _solve(gray, INNER, SQUARE_MM, K4, dist4, prev_b)
        if pa is not None:
            prev_a = pts_a
            n_found_a += 1
            reproj_a.append(pa[2])
        if pb is not None:
            prev_b = pts_b
            n_found_b += 1
            reproj_b.append(pb[2])

        row: list[Any] = [idx, f"{times[idx]:.6f}",
                          int(pa is not None), f"{pa[2]:.3f}" if pa else "",
                          int(pb is not None), f"{pb[2]:.3f}" if pb else ""]
        T_G_A = T_G_B = T_G_C0 = T_C0_B = None
        if pa is not None:
            T_G_A = se3(pa[0], pa[1] / 1000.0)
        if pb is not None:
            T_G_B = se3(pb[0], pb[1] / 1000.0)
        if T_G_A is not None and T_C0_A is not None:
            T_G_C0 = T_G_A @ se3_inv(T_C0_A)
            if T_G_B is not None:
                T_C0_B = se3_inv(T_G_C0) @ T_G_B
        for T in (T_G_A, T_G_B, T_G_C0, T_C0_B):
            row += _rt_to_row(T) if T is not None else [""] * 12
        rows_out.append(row)
        n_written += 1

    if cap is not None:
        cap.release()

    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows_out)

    summary = {
        "session": root.name,
        "frames": n_written,
        "boardA_coverage": n_found_a / max(n_written, 1),
        "boardB_coverage": n_found_b / max(n_written, 1),
        "boardA_reproj_px_median": float(np.median(reproj_a)) if reproj_a else None,
        "boardB_reproj_px_median": float(np.median(reproj_b)) if reproj_b else None,
        "has_rig_transform": T_C0_A is not None,
        "intrinsics_4k": str(K4_path),
        "output": str(out_csv),
    }
    (root / "global_pose_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Global-pose ground truth for a 4K session")
    parser.add_argument("session", help="gtraj_ 会话目录")
    parser.add_argument("--write-images", action="store_true", help="（保留，无操作）")
    parser.add_argument("--only-segments", action="store_true",
                        help="只处理 coverage_summary.json 里两板同见的可用片段")
    args = parser.parse_args()
    summary = evaluate(args.session, write_images=args.write_images,
                       only_segments=args.only_segments)
    print(f"帧数 {summary['frames']}")
    print(f"棋盘覆盖：刚体板 {summary['boardA_coverage']:.0%}，GP050 {summary['boardB_coverage']:.0%}")
    print(f"重投影中位数 (px)：A {summary['boardA_reproj_px_median']}, B {summary['boardB_reproj_px_median']}")
    print(f"含刚体外参：{summary['has_rig_transform']}")
    print(f"输出 {summary['output']}")


if __name__ == "__main__":
    main()
