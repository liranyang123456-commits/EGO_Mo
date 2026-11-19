#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dual-view calibration of the fixed transform between the rig board and the
left camera (T_C0_A), for the 4K global-view setup.

The rig (stereo camera + USB IMU) carries the small board A (7x5 squares of
5 mm, 6x4 inner corners). The fixed 4K camera sees board A and the shared
GP050 board B; the stereo left camera sees board B. At each held pose:

    T_C0_A = T_C0_B * inv(T_G_B) * T_G_A

with T_C0_B from the stereo PnP, T_G_B and T_G_A from the 4K PnP. Because the
rig is rigid, T_C0_A is constant across poses; it is averaged robustly and
checked on held-out poses. The result lets the 4K camera report the left
camera's global pose:  T_G_C0(t) = T_G_A(t) * inv(T_C0_A).

Usage:
    python tools/calibrate_global_rig.py <calib_grig_session_dir>
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.geometry import orthonormalize, se3, se3_inv, se3_to_dict  # noqa: E402
from ego_capture.session import INNER, INNER2, SQUARE2_MM, SQUARE_MM  # noqa: E402

DETECT_MAX_SIDE = 1280
STILL_VEL_PX = 2.0     # bbox centroid speed below which a pose counts as held
STILL_MIN_S = 1.5      # hold duration


def board_object_points(inner: tuple[int, int], square_mm: float) -> np.ndarray:
    cols, rows = inner
    objp = np.zeros((cols * rows, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_mm)
    return objp


def _find(gray: np.ndarray, inner: tuple[int, int]):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    for pattern in (inner, (inner[1], inner[0])):
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if ok and corners is not None and len(corners) == pattern[0] * pattern[1]:
            return corners.astype(np.float32), pattern
    return None


def detect_full(gray: np.ndarray, inner: tuple[int, int]) -> Optional[np.ndarray]:
    """Detect at reduced resolution, refine at full resolution."""
    height, width = gray.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / float(max(height, width)))
    small = gray
    if scale < 1.0:
        small = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    found = _find(small, inner)
    if found is None:
        # Second pass with local contrast for dim or low-contrast frames.
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(small)
        except Exception:
            clahe = None
        if clahe is not None:
            found = _find(clahe, inner)
    if found is None:
        return None
    corners, _pattern = found
    pts = corners.reshape(-1, 1, 2).astype(np.float32).copy()
    if scale < 1.0:
        pts[:, :, 0] /= scale
        pts[:, :, 1] /= scale
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    try:
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
    except cv2.error:
        pass
    return pts


def solve_pose(
    gray: np.ndarray,
    inner: tuple[int, int],
    square_mm: float,
    K: np.ndarray,
    dist: np.ndarray,
    prev_corners: Optional[np.ndarray] = None,
) -> Optional[dict[str, Any]]:
    corners = detect_full(gray, inner)
    if corners is None:
        return None
    pts = corners.reshape(-1, 2)
    if prev_corners is not None:
        prev = np.asarray(prev_corners, dtype=np.float32).reshape(-1, 2)
        if len(prev) == len(pts):
            d_same = float(np.mean(np.linalg.norm(pts - prev, axis=1)))
            d_flip = float(np.mean(np.linalg.norm(pts[::-1] - prev, axis=1)))
            if d_flip < d_same:
                pts = pts[::-1].copy()
    objp = board_object_points(inner, square_mm)
    ok, rvec, tvec = cv2.solvePnP(
        objp, pts.reshape(-1, 1, 2).astype(np.float64), K, dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    reproj = float(np.sqrt(np.mean((proj.reshape(-1, 2) - pts) ** 2)))
    R, _ = cv2.Rodrigues(rvec)
    return {
        "R": orthonormalize(R),
        "t_mm": tvec.reshape(3).astype(np.float64),
        "reproj_px": reproj,
        "corners": pts,
    }


def _latest_file(root: Path, prefix: str, filename: str) -> Optional[Path]:
    best: Optional[Path] = None
    best_mtime = -1.0
    if not root.is_dir():
        return None
    for name in os.listdir(root):
        path = root / name / filename
        if name.startswith(prefix) and path.is_file() and path.stat().st_mtime > best_mtime:
            best = path
            best_mtime = path.stat().st_mtime
    return best


def load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    K = np.asarray(payload["camera_matrix"], dtype=np.float64)
    dist = np.asarray(payload["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
    return K, dist


def _read_boards_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            try:
                bbox_a = [float(v) for v in (rec.get("boardA_bbox") or "0;0;0;0").split(";")]
                rows.append({
                    "idx": int(rec["frame_idx"]),
                    "stamp": float(rec["stamp"]),
                    "A_found": int(rec.get("boardA_found") or 0),
                    "B_found": int(rec.get("boardB_found") or 0),
                    "A_bbox": bbox_a,
                })
            except (KeyError, ValueError):
                continue
    return rows


def _still_frames(rows: list[dict[str, Any]], fps: float) -> list[int]:
    """Indices of frames in the middle of held poses (board A still)."""
    if not rows:
        return []
    min_len = max(2, int(round(STILL_MIN_S * max(fps, 1.0))))
    centroid = [
        ((r["A_bbox"][0] + r["A_bbox"][2]) * 0.5, (r["A_bbox"][1] + r["A_bbox"][3]) * 0.5)
        for r in rows
    ]
    still = []
    for i, r in enumerate(rows):
        if not r["A_found"]:
            still.append(False)
            continue
        if i == 0:
            still.append(True)
            continue
        dx = centroid[i][0] - centroid[i - 1][0]
        dy = centroid[i][1] - centroid[i - 1][1]
        still.append((dx * dx + dy * dy) ** 0.5 < STILL_VEL_PX)
    chosen: list[int] = []
    run: list[int] = []
    for i, ok in enumerate(still + [False]):
        if ok:
            run.append(i)
        else:
            if len(run) >= min_len:
                chosen.append(run[len(run) // 2])
            run = []
    return chosen


def _nearest_index(times: np.ndarray, stamp: float) -> int:
    return int(np.argmin(np.abs(times - stamp)))


def _quat_of(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    q[0] = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) * 0.5
    q[1] = np.copysign(np.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) * 0.5, R[2, 1] - R[1, 2])
    q[2] = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) * 0.5, R[0, 2] - R[2, 0])
    q[3] = np.copysign(np.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) * 0.5, R[1, 0] - R[0, 1])
    n = np.linalg.norm(q)
    return q / n if n > 0 else np.array([1.0, 0.0, 0.0, 0.0])


def _mean_rotation(rotations: list[np.ndarray]) -> np.ndarray:
    qs = np.stack([_quat_of(R) for R in rotations])
    ref = qs[0]
    qs = np.array([q if np.dot(q, ref) >= 0 else -q for q in qs])
    q = qs.mean(axis=0)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return orthonormalize(np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]))


def evaluate(session: str) -> dict[str, Any]:
    root = Path(session)
    boards_csv = root / "cam2" / "boards.csv"
    if not boards_csv.is_file():
        raise SystemExit(f"缺少 {boards_csv}（该段不是全局采集，或 4K 未录制）")
    K4_path = _latest_file(ROOT / "datasets", "calib_g4k_", "camera_calibration_4k.json")
    K0_path = _latest_file(ROOT / "datasets", "calib_intrinsics_", "camera_calibration_cam0.json")
    if K4_path is None or K0_path is None:
        raise SystemExit("找不到 4K 或双目左目内参。先完成第 4 步和第 13 步。")
    K4, dist4 = load_intrinsics(K4_path)
    K0, dist0 = load_intrinsics(K0_path)

    rows = _read_boards_csv(boards_csv)
    times2 = np.loadtxt(root / "cam2" / "times.txt", dtype=np.float64)
    fps = (len(times2) - 1) / (times2[-1] - times2[0]) if len(times2) > 1 else 30.0
    chosen = _still_frames(rows, fps)
    if len(chosen) < 8:
        raise SystemExit(f"静止姿态太少（{len(chosen)}）。每个姿态静止 2 秒、采 25 个以上。")

    times0_path = root / "cam0" / "times.txt"
    times0 = np.loadtxt(times0_path, dtype=np.float64) if times0_path.is_file() else None

    poses: list[dict[str, Any]] = []
    prev_a = None
    for idx in chosen:
        img2 = root / "cam2" / "images" / f"{idx:06d}.jpg"
        if not img2.is_file():
            continue
        gray2 = cv2.imread(str(img2), cv2.IMREAD_GRAYSCALE)
        if gray2 is None:
            continue
        pa = solve_pose(gray2, INNER2, SQUARE2_MM, K4, dist4, prev_corners=prev_a)
        if pa is None:
            continue
        prev_a = pa["corners"]
        pb = solve_pose(gray2, INNER, SQUARE_MM, K4, dist4)
        if pb is None:
            continue
        # Corresponding stereo-left frame by nearest timestamp.
        if times0 is None:
            continue
        j = _nearest_index(times0, float(times2[idx]))
        img0 = root / "cam0" / "images" / f"{j:06d}.jpg"
        if not img0.is_file():
            continue
        gray0 = cv2.imread(str(img0), cv2.IMREAD_GRAYSCALE)
        if gray0 is None:
            continue
        p0b = solve_pose(gray0, INNER, SQUARE_MM, K0, dist0)
        if p0b is None:
            continue
        T_G_A = se3(pa["R"], pa["t_mm"] / 1000.0)
        T_G_B = se3(pb["R"], pb["t_mm"] / 1000.0)
        T_C0_B = se3(p0b["R"], p0b["t_mm"] / 1000.0)
        T_C0_A = T_C0_B @ se3_inv(T_G_B) @ T_G_A
        poses.append({
            "idx": idx,
            "T_C0_A": T_C0_A,
            "reproj": (pa["reproj_px"], pb["reproj_px"], p0b["reproj_px"]),
        })
    if len(poses) < 8:
        raise SystemExit(f"可用姿态太少（{len(poses)}）：需要 4K 同时看到两板且左目看到 GP050。")

    rotations = [p["T_C0_A"][:3, :3] for p in poses]
    translations = np.stack([p["T_C0_A"][:3, 3] for p in poses])
    R_mean = _mean_rotation(rotations)
    t_mean = np.median(translations, axis=0)
    T_mean = se3(R_mean, t_mean)

    # Held-out consistency: fit on even poses, measure on odd ones.
    even = poses[::2]
    odd = poses[1::2]
    holdout_rot = holdout_t = float("nan")
    if len(even) >= 4 and len(odd) >= 4:
        R_e = _mean_rotation([p["T_C0_A"][:3, :3] for p in even])
        t_e = np.median(np.stack([p["T_C0_A"][:3, 3] for p in even]), axis=0)
        T_e = se3(R_e, t_e)
        rot_errs = []
        t_errs = []
        for p in odd:
            T_diff = se3_inv(T_e) @ p["T_C0_A"]
            tr = float(np.degrees(np.arccos(np.clip((np.trace(T_diff[:3, :3]) - 1) * 0.5, -1, 1))))
            rot_errs.append(tr)
            t_errs.append(float(np.linalg.norm(T_diff[:3, 3]) * 1000.0))
        holdout_rot = float(np.mean(rot_errs))
        holdout_t = float(np.mean(t_errs))

    spread_t = translations.std(axis=0) * 1000.0
    result = {
        "session": root.name,
        "poses_used": len(poses),
        "T_C0_A": se3_to_dict(T_mean),
        "T_A_C0": se3_to_dict(se3_inv(T_mean)),
        "translation_spread_mm_std": spread_t.tolist(),
        "holdout_rotation_err_deg": holdout_rot,
        "holdout_translation_err_mm": holdout_t,
        "reproj_px_median": {
            "boardA_4k": float(np.median([p["reproj"][0] for p in poses])),
            "boardB_4k": float(np.median([p["reproj"][1] for p in poses])),
            "boardB_cam0": float(np.median([p["reproj"][2] for p in poses])),
        },
        "intrinsics": {"cam_4k": str(K4_path), "cam0": str(K0_path)},
        "note": "T_C0_A: rig board A in the left-camera frame (constant). "
                "T_G_C0(t) = T_G_A(t) @ inv(T_C0_A).",
    }
    out = root / "global_rig_result.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    state = ROOT / "datasets" / "global_rig_state.json"
    state.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python tools/calibrate_global_rig.py <calib_grig_会话目录>")
        raise SystemExit(1)
    result = evaluate(sys.argv[1])
    t = result["T_C0_A"]["t_mm"]
    print(f"姿态数 {result['poses_used']}")
    print(f"T_C0_A 平移 (mm): {[round(v, 2) for v in t]}")
    print(f"T_C0_A 姿态 RPY (deg): {[round(v, 2) for v in result['T_C0_A']['rpy_deg']]}")
    print(f"留出检查：旋转 {result['holdout_rotation_err_deg']:.3f} deg，"
          f"平移 {result['holdout_translation_err_mm']:.2f} mm")
    print(f"重投影中位数 (px): {result['reproj_px_median']}")
    print("已写 global_rig_result.json 与 datasets/global_rig_state.json")


if __name__ == "__main__":
    main()
