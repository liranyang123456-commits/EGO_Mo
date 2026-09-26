#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stereo calibrate from chessboard pairs, then SGBM a point cloud from ego frames."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ego_capture.cameras import chessboard_object_points, find_board
from ego_capture.geometry import se3_to_dict
from ego_capture.rig import STEREO_BASELINE_MM
from ego_capture.session import write_json

INTR = ROOT / "datasets" / "calib_intrinsics_20260922_012032"
EGO = ROOT / "datasets" / "ego_seq_20260922_012720"
OUT = ROOT / "datasets" / "recon_from_ego"


def _load_calib(session: Path, name: str):
    data = json.loads((session / f"camera_calibration_{name}.json").read_text(encoding="utf-8"))
    K = np.asarray(data["camera_matrix"], dtype=np.float64)
    dist = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1)
    size = tuple(int(x) for x in data["image_size"])
    return K, dist, size


def _paired_indices(session: Path, limit: int = 40) -> list[int]:
    both = []
    with (session / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if int(float(rec.get("board_left", 0) or 0)) and int(float(rec.get("board_right", 0) or 0)):
                both.append(int(float(rec["frame_idx"])))
    if len(both) > limit:
        sel = np.linspace(0, len(both) - 1, limit).astype(int)
        both = [both[i] for i in sel]
    return both


def recalibrate(session: Path, cam: str, board_key: str) -> tuple[np.ndarray, np.ndarray, tuple[int, int], float]:
    idxs = []
    with (session / "frames.csv").open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            if int(float(rec.get(board_key, 0) or 0)):
                idxs.append(int(float(rec["frame_idx"])))
    if len(idxs) > 50:
        idxs = [idxs[i] for i in np.linspace(0, len(idxs) - 1, 50).astype(int)]
    objp = chessboard_object_points()
    obj_pts, img_pts = [], []
    size = None
    for idx in idxs:
        path = session / cam / "images" / f"{idx:06d}.jpg"
        if not path.exists():
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        size = (bgr.shape[1], bgr.shape[0])
        ok, corners, _ = find_board(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
        if not ok:
            continue
        obj_pts.append(objp)
        img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
    if size is None or len(obj_pts) < 12:
        raise RuntimeError(f"{cam} 重标定样本不足")
    flags = cv2.CALIB_FIX_K3
    rms, K, dist, _, _ = cv2.calibrateCamera(
        obj_pts, img_pts, size, None, None, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 80, 1e-6),
    )
    return K, dist.reshape(-1), size, float(rms)


def stereo_calibrate(session: Path) -> dict:
    K0, d0, size, rms0 = recalibrate(session, "cam0", "board_left")
    K1, d1, _, rms1 = recalibrate(session, "cam1", "board_right")
    print(f"recalib cam0 rms={rms0:.3f} cam1 rms={rms1:.3f}", flush=True)
    objp = chessboard_object_points()
    obj_pts, img0, img1 = [], [], []
    for idx in _paired_indices(session):
        p0 = session / "cam0" / "images" / f"{idx:06d}.jpg"
        p1 = session / "cam1" / "images" / f"{idx:06d}.jpg"
        if not p0.exists() or not p1.exists():
            continue
        g0 = cv2.cvtColor(cv2.imread(str(p0), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        g1 = cv2.cvtColor(cv2.imread(str(p1), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
        ok0, c0, _ = find_board(g0)
        ok1, c1, _ = find_board(g1)
        if not (ok0 and ok1):
            continue
        obj_pts.append(objp)
        img0.append(c0.reshape(-1, 1, 2).astype(np.float32))
        img1.append(c1.reshape(-1, 1, 2).astype(np.float32))
    if len(obj_pts) < 8:
        raise RuntimeError(f"双目同时看到棋盘的帧只有 {len(obj_pts)}，无法 stereoCalibrate。")
    flags = cv2.CALIB_FIX_INTRINSIC
    rms, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        obj_pts, img0, img1, K0, d0, K1, d1, size, flags=flags,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    t_mm = (T.reshape(3) * 1000.0) if np.linalg.norm(T) < 2 else T.reshape(3)
    # OpenCV stereoCalibrate t is in object-point units = mm
    t_m = t_mm / 1000.0
    baseline = float(np.linalg.norm(t_mm))
    return {
        "n_pairs": len(obj_pts),
        "rms_px": float(rms),
        "recalib_rms_cam0": rms0,
        "recalib_rms_cam1": rms1,
        "R": R.tolist(),
        "t_mm": t_mm.tolist(),
        "t_m": t_m.tolist(),
        "baseline_mm": baseline,
        "nominal_baseline_mm": STEREO_BASELINE_MM,
        "baseline_error_mm": baseline - STEREO_BASELINE_MM,
        "image_size": list(size),
        "K0": K0.tolist(),
        "K1": K1.tolist(),
        "d0": d0.tolist(),
        "d1": d1.tolist(),
        "T_C0_C1": se3_to_dict(np.vstack([np.hstack([R, t_m.reshape(3, 1)]), [0, 0, 0, 1]])),
    }


def _write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> int:
    mask = np.isfinite(xyz).all(1)
    z = xyz[:, 2]
    if np.nanmedian(np.abs(z[mask])) > 5:
        xyz = xyz / 1000.0
        z = xyz[:, 2]
    mask &= (z > 0.05) & (z < 1.60)
    xyz, rgb = xyz[mask], rgb[mask]
    if len(xyz) > 400_000:
        sel = np.random.default_rng(0).choice(len(xyz), 400_000, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(xyz)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(xyz, rgb):
            handle.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {int(c[2])} {int(c[1])} {int(c[0])}\n")
    return int(len(xyz))


def reconstruct_ego(calib: dict, ego: Path, out: Path, n_pairs: int = 10) -> dict:
    K0 = np.asarray(calib["K0"], np.float64)
    K1 = np.asarray(calib["K1"], np.float64)
    d0 = np.asarray(calib["d0"], np.float64)
    d1 = np.asarray(calib["d1"], np.float64)
    size = tuple(calib["image_size"])
    # 估计的 R 含约 10°，stereoRectify 会把右目甩出画面。按平行双目 + 测得基线出点云。
    m1l, m1r = cv2.initUndistortRectifyMap(K0, d0, None, K0, size, cv2.CV_32FC1)
    m2l, m2r = cv2.initUndistortRectifyMap(K1, d1, None, K1, size, cv2.CV_32FC1)
    tx = -abs(float(calib.get("baseline_mm") or STEREO_BASELINE_MM)) / 1000.0
    f = 0.5 * (K0[0, 0] + K1[0, 0])
    cx, cy = float(K0[0, 2]), float(K0[1, 2])
    Q = np.array(
        [[1, 0, 0, -cx], [0, 1, 0, -cy], [0, 0, 0, f], [0, 0, -1.0 / tx, 0]],
        dtype=np.float32,
    )
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=16 * 8, blockSize=5,
        P1=8 * 3 * 25, P2=32 * 3 * 25, uniquenessRatio=10,
        speckleWindowSize=80, speckleRange=2, disp12MaxDiff=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    left_dir = ego / "cam0" / "images"
    right_dir = ego / "cam1" / "images"
    names = sorted(p.name for p in left_dir.glob("*.jpg") if (right_dir / p.name).exists())
    if not names:
        raise RuntimeError("ego 没有成对左右图。")
    pick = [names[i] for i in np.linspace(0, len(names) - 1, n_pairs).astype(int)]
    clouds = []
    preview = None
    for name in pick:
        iml = cv2.imread(str(left_dir / name), cv2.IMREAD_COLOR)
        imr = cv2.imread(str(right_dir / name), cv2.IMREAD_COLOR)
        rl = cv2.remap(iml, m1l, m1r, cv2.INTER_LINEAR)
        rr = cv2.remap(imr, m2l, m2r, cv2.INTER_LINEAR)
        disp = sgbm.compute(cv2.cvtColor(rl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(rr, cv2.COLOR_BGR2GRAY))
        disp_f = disp.astype(np.float32) / 16.0
        valid = disp_f > 1.0
        pts = cv2.reprojectImageTo3D(disp_f, Q)[valid]
        col = rl[valid]
        clouds.append((pts, col))
        if preview is None:
            vis = np.hstack([rl, rr])
            preview = vis
            cv2.imwrite(str(out / "rectified_preview.jpg"), vis)
            cv2.imwrite(str(out / "disparity_preview.png"), np.clip(disp_f * 4, 0, 255).astype(np.uint8))
    xyz = np.concatenate([c[0] for c in clouds], 0)
    rgb = np.concatenate([c[1] for c in clouds], 0)
    n = _write_ply(out / "cloud.ply", xyz, rgb)
    return {"pairs": pick, "points": n, "ply": str(out / "cloud.ply")}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("stereoCalibrate ...", flush=True)
    calib = stereo_calibrate(INTR)
    write_json(str(OUT / "stereo_extrinsics.json"), calib)
    print(
        f"pairs={calib['n_pairs']} rms={calib['rms_px']:.3f}px  "
        f"baseline={calib['baseline_mm']:.1f}mm (nominal {STEREO_BASELINE_MM})",
        flush=True,
    )
    print("SGBM recon ...", flush=True)
    rec = reconstruct_ego(calib, EGO, OUT)
    write_json(str(OUT / "recon_summary.json"), {**rec, "stereo": calib})
    print(f"points={rec['points']} -> {rec['ply']}", flush=True)


if __name__ == "__main__":
    main()
