#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate the constant rig transform T_C0_A (rig board A -> left camera)
in-line from a global-view trajectory session, without a dedicated
calibration recording.

At any frame where the 4K camera sees both boards and the stereo left camera
sees the GP050 board, the rigid-rig transform follows as

    T_C0_A = T_C0_B * inv(T_G_B) * T_G_A

with T_C0_B from the stereo PnP and T_G_A, T_G_B from the 4K PnP. The estimate
is averaged robustly over slow-motion frames (so the residual stereo/4K timing
mismatch is negligible) and written to datasets/global_rig_state.json.

Usage:
    python tools/estimate_rig_transform_inline.py <gtraj_session_dir> [<more...>]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.geometry import orthonormalize, se3, se3_inv, se3_to_dict  # noqa: E402
from ego_capture.session import INNER, INNER2, SQUARE2_MM, SQUARE_MM  # noqa: E402
from tools.calibrate_global_rig import (  # noqa: E402
    board_object_points,
    detect_full,
    load_intrinsics,
    _latest_file,
    _mean_rotation,
)


def _pnp(gray, inner, square_mm, K, dist, prev=None):
    corners = detect_full(gray, inner)
    if corners is None:
        return None, None
    pts = corners.reshape(-1, 2)
    if prev is not None and len(prev) == len(pts):
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
        return None, None
    R, _ = cv2.Rodrigues(rvec)
    return (orthonormalize(R), tvec.reshape(3).astype(np.float64)), pts


def _load_times(path: Path) -> np.ndarray:
    return np.loadtxt(path, dtype=np.float64)


def estimate(sessions: list[str]) -> dict[str, Any]:
    K4_path = _latest_file(ROOT / "datasets", "calib_g4k_", "camera_calibration_4k.json")
    K0_path = _latest_file(ROOT / "datasets", "calib_intrinsics_", "camera_calibration_cam0.json")
    if K4_path is None or K0_path is None:
        raise SystemExit("找不到 4K 或双目左目内参。")
    K4, dist4 = load_intrinsics(K4_path)
    K0, dist0 = load_intrinsics(K0_path)

    samples: list[np.ndarray] = []
    max_samples = 80  # the constant needs only a few dozen good samples
    for session in sessions:
        if len(samples) >= max_samples:
            break
        root = Path(session)
        cam2 = root / "cam2"
        cam0 = root / "cam0"
        if not (cam2 / "times.txt").is_file() or not (cam0 / "times.txt").is_file():
            continue
        times2 = _load_times(cam2 / "times.txt")
        times0 = _load_times(cam0 / "times.txt")
        imgs2 = sorted((cam2 / "images").glob("*.jpg"))
        imgs0 = sorted((cam0 / "images").glob("*.jpg"))
        n2 = min(len(times2), len(imgs2))
        n0 = min(len(times0), len(imgs0))
        prev_a = prev_b = prev_0 = None
        prev_centroid: Optional[np.ndarray] = None
        for k in range(n2):
            frame2 = cv2.imread(str(imgs2[k]))
            if frame2 is None:
                continue
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            pa, pts_a = _pnp(gray2, INNER2, SQUARE2_MM, K4, dist4, prev_a)
            pb, pts_b = _pnp(gray2, INNER, SQUARE_MM, K4, dist4, prev_b)
            if pa is not None:
                prev_a = pts_a
            if pb is not None:
                prev_b = pts_b
            if pa is None or pb is None:
                continue
            # Slow-motion gate: board A centroid barely moved since last frame.
            centroid = pts_a.mean(axis=0)
            slow = True
            if prev_centroid is not None:
                slow = float(np.linalg.norm(centroid - prev_centroid)) < 3.0
            prev_centroid = centroid
            if not slow:
                continue
            # Time-matched stereo-left frame.
            j = int(np.argmin(np.abs(times0[:n0] - times2[k])))
            if abs(float(times0[j]) - float(times2[k])) > 0.08:
                continue
            frame0 = cv2.imread(str(imgs0[j]))
            if frame0 is None:
                continue
            gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
            p0b, pts0 = _pnp(gray0, INNER, SQUARE_MM, K0, dist0, prev_0)
            if p0b is None:
                continue
            prev_0 = pts0
            T_G_A = se3(pa[0], pa[1] / 1000.0)
            T_G_B = se3(pb[0], pb[1] / 1000.0)
            T_C0_B = se3(p0b[0], p0b[1] / 1000.0)
            samples.append(T_C0_B @ se3_inv(T_G_B) @ T_G_A)
            if len(samples) >= max_samples:
                break
        print(f"  {root.name}: 累计 {len(samples)} 个样本")

    if len(samples) < 20:
        raise SystemExit(f"可用样本太少（{len(samples)}）。需要 4K 同见两板且左目见 GP050 的帧。")

    rotations = [T[:3, :3] for T in samples]
    translations = np.stack([T[:3, 3] for T in samples])
    R_mean = _mean_rotation(rotations)
    t_mean = np.median(translations, axis=0)
    T_mean = se3(R_mean, t_mean)

    # Spread and held-out style consistency.
    t_spread = translations.std(axis=0) * 1000.0
    rot_errs = []
    for T in samples:
        Td = se3_inv(T_mean) @ T
        ang = float(np.degrees(np.arccos(np.clip((np.trace(Td[:3, :3]) - 1) * 0.5, -1, 1))))
        rot_errs.append(ang)
    result = {
        "method": "in-line dual-view composition on trajectory frames",
        "sessions": [str(Path(s).name) for s in sessions],
        "samples": len(samples),
        "T_C0_A": se3_to_dict(T_mean),
        "T_A_C0": se3_to_dict(se3_inv(T_mean)),
        "translation_spread_mm_std": t_spread.tolist(),
        "rotation_err_deg_mean": float(np.mean(rot_errs)),
        "rotation_err_deg_p90": float(np.percentile(rot_errs, 90)),
        "note": "T_C0_A: rig board A in the left-camera frame (constant). "
                "T_G_C0(t) = T_G_A(t) @ inv(T_C0_A).",
    }
    state = ROOT / "datasets" / "global_rig_state.json"
    state.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    if len(sys.argv) < 2:
        print("用法: python tools/estimate_rig_transform_inline.py <gtraj会话目录> [更多...]")
        raise SystemExit(1)
    result = estimate(sys.argv[1:])
    t = result["T_C0_A"]["t_mm"]
    print(f"\n样本数 {result['samples']}")
    print(f"T_C0_A 平移 (mm): {[round(v, 2) for v in t]}")
    print(f"T_C0_A 姿态 RPY (deg): {[round(v, 2) for v in result['T_C0_A']['rpy_deg']]}")
    print(f"旋转一致性 均值 {result['rotation_err_deg_mean']:.3f} deg, "
          f"p90 {result['rotation_err_deg_p90']:.3f} deg")
    print(f"平移散布 std (mm): {[round(v, 2) for v in result['translation_spread_mm_std']]}")
    print("已写 datasets/global_rig_state.json")


if __name__ == "__main__":
    main()
