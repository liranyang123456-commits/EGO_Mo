#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Metric stereo extrinsic calibration with held-out verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
DEFAULT_SESSION = DATA / "calib_intrinsics_20260922_123120"
DEFAULT_OUT = DATA / "stereo_extrinsic_measured.json"
PATTERN = (11, 8)
SQUARE_M = 0.003


def load_camera(path: Path):
    p = json.loads(path.read_text(encoding="utf-8"))
    return (np.asarray(p["camera_matrix"], np.float64),
            np.asarray(p["distortion_coefficients"], np.float64),
            tuple(p["image_size"]))


def object_points():
    x, y = np.meshgrid(np.arange(PATTERN[0]), np.arange(PATTERN[1]))
    return np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size))) * SQUARE_M


def detect(session: Path, stride: int, cap: int, max_pair_dt: float):
    left = session / "cam0" / "images"
    right = session / "cam1" / "images"
    t0 = np.loadtxt(session / "cam0" / "times.txt", dtype=np.float64)
    t1 = np.loadtxt(session / "cam1" / "times.txt", dtype=np.float64)
    rows = []
    left_paths = sorted(left.glob("*.jpg"))
    for li in range(0, min(len(left_paths), len(t0)), stride):
        lp = left_paths[li]
        rj = int(np.argmin(np.abs(t1 - t0[li])))
        if abs(float(t1[rj] - t0[li])) > max_pair_dt:
            continue
        rp = right / f"{rj:06d}.jpg"
        if not rp.is_file():
            continue
        images, corners = [], []
        for path in (lp, rp):
            im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            scale = 0.5
            small = cv2.resize(im, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
            ok, c = cv2.findChessboardCorners(
                small, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH |
                cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
            )
            if not ok:
                break
            c = c / scale
            c = cv2.cornerSubPix(
                im, c, (5, 5), (-1, -1),
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
            )
            images.append(im)
            corners.append(c)
        if len(corners) == 2:
            centre = np.mean(corners[0].reshape(-1, 2), axis=0)
            area = cv2.contourArea(cv2.convexHull(corners[0]))
            rows.append((f"{li:06d}_{rj:06d}", corners[0], corners[1],
                         centre, area, float(t1[rj] - t0[li])))
    if len(rows) > cap:
        # Preserve time and viewpoint diversity without looking at residuals.
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, cap).astype(int)]
    return rows


def _find_corners(path: Path):
    if not path.is_file():
        return None
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    scale = 0.5
    small = cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, c = cv2.findChessboardCorners(
        small, PATTERN, cv2.CALIB_CB_ADAPTIVE_THRESH |
        cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    )
    if not ok:
        return None
    c = cv2.cornerSubPix(
        im, c / scale, (5, 5), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1e-3)
    )
    return c.reshape(-1, 2)


def _all_corners(session: Path, cam: str, n: int):
    cache = session / f"{cam}_corners_cache.npz"
    if cache.is_file():
        z = np.load(cache)
        if int(z["n"]) == n:
            return z["corners"], z["found"].astype(bool)
    corners = np.full((n, PATTERN[0] * PATTERN[1], 2), np.nan)
    found = np.zeros(n, bool)
    for i in range(n):
        c = _find_corners(session / cam / "images" / f"{i:06d}.jpg")
        if c is not None:
            corners[i], found[i] = c, True
    np.savez_compressed(cache, corners=corners, found=found, n=n)
    return corners, found


def _frame_motion(corners, found):
    """Max corner displacement (px) to the previous and next frame; inf if unknown."""
    n = len(found)
    motion = np.full(n, np.inf)
    for i in range(1, n - 1):
        if found[i - 1] and found[i] and found[i + 1]:
            d0 = np.linalg.norm(corners[i] - corners[i - 1], axis=1).max()
            d1 = np.linalg.norm(corners[i + 1] - corners[i], axis=1).max()
            motion[i] = max(d0, d1)
    return motion


def estimate_right_lag(c0, f0, c1, f1, max_off: int = 3):
    """Sub-frame lag of right images relative to left, from board-centre motion.

    Uses only image-plane motion (no extrinsic), so it can be estimated before
    the calibration without circularity. Returns the fractional offset ``off``
    such that right-image index ``i + off`` best matches left image ``i``.
    """
    x0, x1 = c0.mean(axis=1), c1.mean(axis=1)
    n = len(f0)
    offs = np.arange(-max_off, max_off + 1)
    score = []
    for off in offs:
        i = np.arange(max_off, n - max_off)
        m = f0[i] & f1[i + off]
        score.append(np.mean([np.corrcoef(x0[i[m], k], x1[i[m] + off, k])[0, 1]
                              for k in (0, 1)]))
    score = np.asarray(score)
    k = int(np.argmax(score))
    k = min(max(k, 1), len(offs) - 2)
    y0, y1, y2 = score[k - 1], score[k], score[k + 1]
    delta = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    best_off = offs[k] + float(np.clip(delta, -1, 1))
    return best_off, {"offsets": offs.tolist(), "corr": score.round(4).tolist(),
                      "peak_offset_frames": best_off}


def detect_quasi_static(session: Path, motion_px: float, max_pair_dt: float,
                        min_spacing_s: float, compensate_lag: bool = True):
    """Stereo pairs restricted to frames where the board is nearly still.

    Left and right JPEGs with the same index are written as a pair by the
    capture loop; cam1/times.txt contains extra lines and does not index the
    right images, so pairing is by image index. The cameras are not
    hardware-synchronised (lag below one frame), so only frames whose corners
    move less than ``motion_px`` to both neighbouring frames, in both cameras,
    are kept.
    """
    t0 = np.loadtxt(session / "cam0" / "times.txt", dtype=np.float64)
    n = len(t0)
    c0, f0 = _all_corners(session, "cam0", n)
    c1, f1 = _all_corners(session, "cam1", n)
    m0, m1 = _frame_motion(c0, f0), _frame_motion(c1, f1)
    off, lag_info = estimate_right_lag(c0, f0, c1, f1)
    if not compensate_lag:
        off = 0.0
    if abs(off) > 1.0:
        raise RuntimeError(f"right-camera lag {off:.2f} frames exceeds one frame")
    rows, last_t = [], -np.inf
    for li in range(1, n - 1):
        if not f0[li] or m0[li] > motion_px or t0[li] - last_t < min_spacing_s:
            continue
        if not f1[li] or m1[li] > motion_px:
            continue
        # m1 <= motion_px guarantees f1 at li-1 and li+1
        j = li - 1 if off < 0 else li + 1
        w = abs(off)
        right = (1 - w) * c1[li] + w * c1[j]
        left = c0[li].astype(np.float32).reshape(-1, 1, 2)
        centre = left.reshape(-1, 2).mean(axis=0)
        area = cv2.contourArea(cv2.convexHull(left))
        rows.append((f"{li:06d}", left,
                     right.astype(np.float32).reshape(-1, 1, 2), centre, area,
                     float(off), float(t0[li])))
        last_t = t0[li]
    stats = {
        "pairing": "same image index",
        "right_offset_frames": float(off), "lag_estimate": lag_info,
        "left_frames": int(n), "left_detected": int(f0.sum()),
        "right_detected": int(f1.sum()), "pairs": len(rows),
        "motion_px_max": motion_px, "min_spacing_s": min_spacing_s,
    }
    return rows, stats


def _pnp(obj, pts, K, d):
    ok, rv, tv = cv2.solvePnP(obj, pts.astype(np.float64), K, d)
    return rv.reshape(3), tv.reshape(3)


def _rotation_average(Rs):
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    return R if np.linalg.det(R) > 0 else U @ np.diag([1, 1, -1]) @ Vt


def fit(rows, K0, d0, K1, d1, size):
    """Joint reprojection fit of the extrinsic and per-pair board poses.

    Intrinsics are fixed. The extrinsic is initialised from the chordal mean
    of per-pair PnP relative poses, because cv2.stereoCalibrate without an
    initial guess diverges with this lens's large higher-order distortion.
    """
    from scipy.optimize import least_squares

    obj = object_points().astype(np.float64)
    init0, rel_R, rel_T = [], [], []
    for r in rows:
        rv0, tv0 = _pnp(obj, r[1], K0, d0)
        rv1, tv1 = _pnp(obj, r[2], K1, d1)
        R0, R1 = cv2.Rodrigues(rv0)[0], cv2.Rodrigues(rv1)[0]
        init0.append(np.r_[rv0, tv0])
        rel_R.append(R1 @ R0.T)
        rel_T.append(tv1 - R1 @ R0.T @ tv0)
    R_init = _rotation_average(np.asarray(rel_R))
    x0 = np.r_[cv2.Rodrigues(R_init)[0].reshape(3), np.median(rel_T, axis=0),
               np.concatenate(init0)]
    pts0 = [r[1].reshape(-1, 2).astype(np.float64) for r in rows]
    pts1 = [r[2].reshape(-1, 2).astype(np.float64) for r in rows]

    def residual(x):
        R = cv2.Rodrigues(x[:3])[0]
        T = x[3:6]
        out = []
        for i in range(len(rows)):
            rv0 = x[6 + 6 * i: 9 + 6 * i]
            tv0 = x[9 + 6 * i: 12 + 6 * i]
            p0 = cv2.projectPoints(obj, rv0, tv0, K0, d0)[0].reshape(-1, 2)
            R1 = R @ cv2.Rodrigues(rv0)[0]
            rv1 = cv2.Rodrigues(R1)[0]
            p1 = cv2.projectPoints(obj, rv1, R @ tv0 + T, K1, d1)[0].reshape(-1, 2)
            out.append((p0 - pts0[i]).ravel())
            out.append((p1 - pts1[i]).ravel())
        return np.concatenate(out)

    from scipy.sparse import lil_matrix
    n_obs = 4 * len(obj)
    sparsity = lil_matrix((n_obs * len(rows), len(x0)), dtype=int)
    for i in range(len(rows)):
        sparsity[i * n_obs:(i + 1) * n_obs, :6] = 1
        sparsity[i * n_obs:(i + 1) * n_obs, 6 + 6 * i:12 + 6 * i] = 1
    sol = least_squares(residual, x0, method="trf", jac_sparsity=sparsity,
                        x_scale="jac", max_nfev=100)
    res = sol.fun.reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(res ** 2, axis=1))))
    R = cv2.Rodrigues(sol.x[:3])[0]
    T = sol.x[3:6]
    tx = np.array([[0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]])
    E = tx @ R
    F = np.linalg.inv(K1).T @ E @ np.linalg.inv(K0)
    return rms, R, T, E, F


def rotation_error(Ra, Rb):
    c = np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(c)))


def holdout_score(rows, test_idx, R, T, K0, d0, K1, d1):
    obj = object_points().astype(np.float64)
    rot, trans, epi = [], [], []
    F = np.linalg.inv(K1).T @ np.array([
        [0, -T[2], T[1]], [T[2], 0, -T[0]], [-T[1], T[0], 0]
    ]) @ R @ np.linalg.inv(K0)
    for i in test_idx:
        row = rows[i]
        ok0, rv0, tv0 = cv2.solvePnP(obj, row[1], K0, d0)
        ok1, rv1, tv1 = cv2.solvePnP(obj, row[2], K1, d1)
        if not (ok0 and ok1):
            continue
        R0 = cv2.Rodrigues(rv0)[0]
        R1 = cv2.Rodrigues(rv1)[0]
        R_obs = R1 @ R0.T
        T_obs = tv1.reshape(3) - R_obs @ tv0.reshape(3)
        rot.append(rotation_error(R, R_obs))
        trans.append(np.linalg.norm(T - T_obs) * 1000)
        x0 = cv2.undistortPoints(row[1], K0, d0, P=K0).reshape(-1, 2)
        x1 = cv2.undistortPoints(row[2], K1, d1, P=K1).reshape(-1, 2)
        h0 = np.column_stack((x0, np.ones(len(x0))))
        h1 = np.column_stack((x1, np.ones(len(x1))))
        lines1 = (F @ h0.T).T
        dist = np.abs(np.sum(lines1 * h1, axis=1)) / np.linalg.norm(lines1[:, :2], axis=1)
        epi.extend(dist.tolist())
    return np.asarray(rot), np.asarray(trans), np.asarray(epi)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--stride", type=int, default=8)
    p.add_argument("--cap", type=int, default=160)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-pair-dt", type=float, default=0.080,
                   help="maximum nearest-frame time difference, seconds")
    p.add_argument("--quasi-static", action="store_true",
                   help="use only frames where the board is nearly still")
    p.add_argument("--no-lag-comp", action="store_true",
                   help="disable sub-frame right-camera lag compensation")
    p.add_argument("--motion-px", type=float, default=5.0,
                   help="max adjacent-frame corner motion in quasi-static mode")
    p.add_argument("--min-spacing", type=float, default=0.0,
                   help="min time between selected pairs, seconds")
    p.add_argument("--block-s", type=float, default=10.0,
                   help="contiguous time block for held-out folds in quasi-static mode")
    args = p.parse_args()
    K0, d0, size0 = load_camera(args.session / "camera_calibration_cam0.json")
    K1, d1, size1 = load_camera(args.session / "camera_calibration_cam1.json")
    if size0 != size1:
        raise RuntimeError(f"camera image sizes differ: {size0}, {size1}")
    selection = {"mode": "stride", "stride": args.stride, "cap": args.cap}
    if args.quasi_static:
        rows, selection = detect_quasi_static(args.session, args.motion_px,
                                              args.max_pair_dt, args.min_spacing,
                                              compensate_lag=not args.no_lag_comp)
        selection["mode"] = "quasi_static"
        selection["fold_block_s"] = args.block_s
        t = np.array([r[6] for r in rows]) if rows else np.zeros(0)
        fold_of = (np.floor((t - t.min()) / args.block_s).astype(int) % args.folds
                   if len(t) else t.astype(int))
    else:
        rows = detect(args.session, args.stride, args.cap, args.max_pair_dt)
        fold_of = np.arange(len(rows)) % args.folds
    if len(rows) < 30:
        raise RuntimeError(f"only {len(rows)} paired detections ({selection})")

    fold_rows, all_rot, all_trans, all_epi = [], [], [], []
    for fold in range(args.folds):
        test = np.flatnonzero(fold_of == fold)
        test_set = set(test.tolist())
        train = [r for i, r in enumerate(rows) if i not in test_set]
        rms, R, T, _E, _F = fit(train, K0, d0, K1, d1, size0)
        re, te, ep = holdout_score(rows, test, R, T, K0, d0, K1, d1)
        all_rot.extend(re.tolist()); all_trans.extend(te.tolist()); all_epi.extend(ep.tolist())
        fold_rows.append({
            "fold": fold, "train": len(train), "test": len(test),
            "fit_rms_px": rms,
            "holdout_rotation_rmse_deg": float(np.sqrt(np.mean(re ** 2))),
            "holdout_translation_rmse_mm": float(np.sqrt(np.mean(te ** 2))),
            "holdout_epipolar_p50_px": float(np.median(ep)),
        })
    rms, R, T, E, F = fit(rows, K0, d0, K1, d1, size0)
    rv = cv2.Rodrigues(R)[0].reshape(3)
    report = {
        "kind": "measured_stereo_extrinsic_fixed_intrinsics",
        "session": args.session.name, "paired_detections": len(rows),
        "selection": selection,
        "pair_offset": (
            {"right_offset_frames": selection["right_offset_frames"]}
            if args.quasi_static else {
                "median_ms": float(np.median([r[5] for r in rows]) * 1000),
                "p95_abs_ms": float(np.quantile(np.abs([r[5] for r in rows]), 0.95) * 1000),
            }),
        "image_size": list(size0), "fit_rms_px": rms,
        "convention": "p_C1 = R_C1_C0 p_C0 + t_C1_C0",
        "R_C1_C0": R.tolist(), "t_C1_C0_m": T.tolist(),
        "baseline_mm": float(np.linalg.norm(T) * 1000),
        "rotation_vector_deg": np.degrees(rv).tolist(),
        "E": E.tolist(), "F": F.tolist(),
        "cross_validation": {
            "folds": fold_rows,
            "rotation_rmse_deg": float(np.sqrt(np.mean(np.asarray(all_rot) ** 2))),
            "translation_rmse_mm": float(np.sqrt(np.mean(np.asarray(all_trans) ** 2))),
            "epipolar_p50_px": float(np.median(all_epi)),
            "epipolar_p95_px": float(np.quantile(all_epi, 0.95)),
        },
        "acceptance": {
            "fit_rms_px_max": 0.6,
            "holdout_epipolar_p95_px_max": 1.0,
            "holdout_rotation_rmse_deg_max": 0.5,
            "holdout_translation_rmse_mm_max": 2.0,
        },
    }
    a = report["acceptance"]; cv = report["cross_validation"]
    report["passed"] = bool(
        rms <= a["fit_rms_px_max"] and
        cv["epipolar_p95_px"] <= a["holdout_epipolar_p95_px_max"] and
        cv["rotation_rmse_deg"] <= a["holdout_rotation_rmse_deg_max"] and
        cv["translation_rmse_mm"] <= a["holdout_translation_rmse_mm_max"]
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({
        "output": str(args.output), "passed": report["passed"],
        "paired": len(rows), "baseline_mm": report["baseline_mm"],
        "fit_rms_px": rms, "cv": cv,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
