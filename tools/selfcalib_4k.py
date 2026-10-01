#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Self-calibrate the fixed-focus 4K global camera from recorded footage.

The GP050 board (11x8 inner corners, 3 mm pitch) appears throughout the
global-view sessions at many positions, distances and tilts. Because the 4K
focal length was locked, a single intrinsics set applies, and Zhang's method
recovers it from these diverse views without a dedicated calibration session.

Detections are collected at full 4K resolution (downscaled search, then
full-resolution cornerSubPix refinement), subsampled for diversity across
image regions and apparent sizes, then fed to cv2.calibrateCamera.

Usage:
    python tools/selfcalib_4k.py <gtraj_session_dir> [<more sessions...>]
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.session import INNER, SQUARE_MM  # noqa: E402
from tools.calibrate_global_rig import board_object_points, detect_full  # noqa: E402

MAX_VIEWS = 120         # cap on calibration views
GRID = (4, 3)           # frame cells for positional diversity
PER_CELL = 12           # max views kept per cell
STRIDE = 6              # look at every Nth frame


def collect_views(sessions: list[str], cache: str | None = None) -> tuple[list[np.ndarray], tuple[int, int]]:
    if cache and os.path.isfile(cache):
        data = np.load(cache)
        return [data[f"v{k}"] for k in range(len(data.files))], tuple(int(v) for v in data["size"])
    views: list[np.ndarray] = []
    cells: dict[tuple[int, int], int] = {}
    size = None
    for session in sessions:
        cam2 = Path(session) / "cam2"
        images = sorted((cam2 / "images").glob("*.jpg"))
        if not images:
            print(f"  {session}: 无图像，跳过")
            continue
        print(f"  {Path(session).name}: {len(images)} 帧")
        for k in range(0, len(images), STRIDE):
            frame = cv2.imread(str(images[k]))
            if frame is None:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if size is None:
                size = (gray.shape[1], gray.shape[0])
            corners = detect_full(gray, INNER)
            if corners is None:
                continue
            pts = corners.reshape(-1, 2)
            cx = float(pts[:, 0].mean()) / size[0]
            cy = float(pts[:, 1].mean()) / size[1]
            cell = (min(GRID[0] - 1, int(cx * GRID[0])), min(GRID[1] - 1, int(cy * GRID[1])))
            if cells.get(cell, 0) >= PER_CELL:
                continue
            cells[cell] = cells.get(cell, 0) + 1
            views.append(pts)
            if len(views) >= MAX_VIEWS:
                break
    if cache and views:
        np.savez(cache, **{f"v{k}": v for k, v in enumerate(views)},
                 size=np.array(size or (G4K_W, G4K_H)))
    return views, size or (G4K_W, G4K_H)


G4K_W, G4K_H = 3840, 2160


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--cache", default=str(ROOT / "datasets" / "selfcalib_4k_views.npz"),
                    help="视图缓存路径（避免重复检测）")
    ap.add_argument("--distortion", choices=["full", "k1", "zero"], default="k1",
                    help="畸变模型：full 全模型 / k1 只估径向一阶 / zero 零畸变")
    args = ap.parse_args()
    sessions = args.sessions
    print("收集 GP050 棋盘视图…")
    views, size = collect_views(sessions, cache=args.cache)
    print(f"共 {len(views)} 个视图，图像尺寸 {size[0]}×{size[1]}")
    if len(views) < 12:
        raise SystemExit("视图太少，无法标定。换棋盘覆盖更多的录像。")

    objp = board_object_points(INNER, SQUARE_MM).astype(np.float32).reshape(-1, 1, 3)
    obj_pts = [objp.copy() for _ in views]
    img_pts = [v.reshape(-1, 1, 2).astype(np.float32) for v in views]
    flags = 0
    if args.distortion == "k1":
        flags = cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST
    elif args.distortion == "zero":
        flags = (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
                 | cv2.CALIB_ZERO_TANGENT_DIST)
    t0 = time.time()
    rms, K, dist, _r, _t = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None, flags=flags)
    print(f"标定耗时 {time.time()-t0:.1f}s  模型 {args.distortion}")

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    print(f"RMS 重投影 {rms:.3f} px")
    print(f"fx {fx:.1f}  fy {fy:.1f}  (fx/fy {fx/fy:.4f})")
    print(f"cx {cx:.1f}  cy {cy:.1f}  (画面中心 {size[0]/2:.0f},{size[1]/2:.0f})")
    print(f"畸变 {[round(float(v),4) for v in dist.reshape(-1)]}")

    payload = {
        "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "camera": "global_4k",
        "method": f"self-calibration from recorded GP050 views (Zhang, {args.distortion} distortion)",
        "focus_locked": True,
        "chessboard": "GP050-3-12*9",
        "chessboard_size": list(INNER),
        "square_size_mm": SQUARE_MM,
        "image_size": [int(size[0]), int(size[1])],
        "num_images": len(views),
        "camera_matrix": K.tolist(),
        "distortion_coefficients": dist.reshape(-1).tolist(),
        "reprojection_error_pixels": float(rms),
        "parameters": {"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
        "source_sessions": [str(Path(s).name) for s in sessions],
    }
    out_dir = ROOT / "datasets" / f"calib_g4k_self_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "camera_calibration_4k.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"已写 {out}")


if __name__ == "__main__":
    main()
