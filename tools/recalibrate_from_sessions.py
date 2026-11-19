#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intrinsic calibration from the board views of other recordings made with
the same camera, focus and power-on session.

Frames in which the board is (nearly) at rest are detected again with the
detector of the label pipeline (tools/build_pose_gt.py), a subset that covers
every reached cell of the 6 x 4 image grid and mixes positions and tilts is
calibrated, and the views left out score the new intrinsics against the old
ones, by image region. Recordings that are later scored with the new
intrinsics (the caliper trials) must not be among the sources.

    python tools/recalibrate_from_sessions.py --out calib_intrinsics_20261108_selfcal \
        --compare calib_intrinsics_20261108_003014 calib_intrinsics_20261108_003014 rigid_20261108_003840 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.build_pose_gt as bp
from ego_capture import camera_config, guided
from ego_capture.cameras import calibrate_from_corners, chessboard_object_points
from ego_capture.session import INNER
from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
MAX_BOARD_DPS = 10.0      # board IMU rate at the frame; sharper corners than in motion
HELDOUT_N = 400


def board_rate(session: Path, t: np.ndarray) -> np.ndarray:
    """Angular rate of the IMU on the board at the frame times (0 without that IMU)."""
    path = session / "imu_bt.csv"
    if not path.is_file():
        return np.zeros(len(t))
    bt, by = load_imu(path)
    if len(bt) < 10:
        return np.zeros(len(t))
    return np.interp(t, bt, np.linalg.norm(by[:, 3:6], axis=1))


def collect(name: str, step: int) -> list[dict]:
    session = DATA / name
    t = bp._times(session / "cam0" / "times.txt")
    images = sorted((session / "cam0" / "images").glob("*.jpg"))
    n = min(len(t), len(images))
    rate = board_rate(session, t[:n])
    out = []
    for i in range(0, n, step):
        if rate[i] > MAX_BOARD_DPS:
            continue
        gray = cv2.imread(str(images[i]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        corners, pattern = bp._find(gray)
        if corners is None or tuple(pattern) != INNER:
            continue
        out.append({"session": name, "frame": i, "corners": corners.reshape(-1, 2).astype(np.float64),
                    "size": (gray.shape[1], gray.shape[0])})
    return out


def score(views: list[dict], K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """(RMS reprojection error px, centroid x px) of every view, PnP with the given intrinsics."""
    obj = chessboard_object_points()
    rows = []
    for v in views:
        img = v["corners"].reshape(-1, 1, 2).astype(np.float32)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        rows.append((float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - v["corners"]) ** 2, axis=1)))),
                     float(v["corners"][:, 0].mean())))
    return np.asarray(rows)


def by_region(rows: np.ndarray, width: int) -> dict:
    out = {}
    for name, lo, hi in (("left", 0, width / 3), ("middle", width / 3, 2 * width / 3), ("right", 2 * width / 3, width)):
        m = (rows[:, 1] >= lo) & (rows[:, 1] < hi)
        if m.any():
            out[name] = {"views": int(m.sum()), "rms_px_median": round(float(np.median(rows[m, 0])), 4)}
    out["all"] = {"views": int(len(rows)), "rms_px_median": round(float(np.median(rows[:, 0])), 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    ap.add_argument("--out", required=True, help="new calibration folder under datasets/")
    ap.add_argument("--compare", help="calibration folder whose intrinsics score the same held-out views")
    ap.add_argument("--step", type=int, default=3, help="use every n-th frame")
    ap.add_argument("--max-views", type=int, default=150)
    args = ap.parse_args()

    views = []
    for name in args.sessions:
        got = collect(name, args.step)
        print(f"{name}: {len(got)} board views at rest", flush=True)
        views += got
    size = views[0]["size"]
    corners = [v["corners"] for v in views]
    chosen = guided.select_samples(corners, size, max_n=args.max_views)
    chosen_ids = {id(c) for c in chosen}
    rest = [v for v in views if id(v["corners"]) not in chosen_ids]
    rng = np.random.default_rng(0)
    heldout = [rest[i] for i in sorted(rng.choice(len(rest), size=min(HELDOUT_N, len(rest)), replace=False))]

    payload = calibrate_from_corners(chosen, size)
    K = np.asarray(payload["camera_matrix"])
    dist = np.asarray(payload["distortion_coefficients"])
    report = {"new": by_region(score(heldout, K, dist), size[0])}
    if args.compare:
        old = json.loads(camera_config.intrinsics_json(name=args.compare).read_text(encoding="utf-8"))
        report["old"] = by_region(score(heldout, np.asarray(old["camera_matrix"]),
                                        np.asarray(old["distortion_coefficients"])), size[0])
        report["old_calibration"] = args.compare

    meta = json.loads((DATA / args.sessions[0] / "session_meta.json").read_text(encoding="utf-8"))
    tracker = guided.CoverageTracker(size)
    tracker.update(chosen)
    payload.update({
        "calibration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "method": "board views at rest from other recordings of the same camera, focus and power-on session",
        "sources": {name: sum(v["session"] == name for v in views) for name in args.sessions},
        "selected_views": len(chosen), "heldout": report,
        "tilts_in_selection": tracker.tilts,
        "focus_state": meta.get("focus_state"), "focus_locked": meta.get("focus_locked"),
        "camera_index": meta.get("cam_left_id"),
    })
    folder = DATA / args.out
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "camera_calibration_cam0.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    camera_config.ensure_label_files(folder)
    p = payload["parameters"]
    print(json.dumps({"fx": round(p["fx"], 2), "fy": round(p["fy"], 2), "cx": round(p["cx"], 2), "cy": round(p["cy"], 2),
                      "dist": np.round(dist, 4).tolist(), "rms_px": round(payload["reprojection_error_pixels"], 4),
                      "coverage": payload["coverage"]["fraction"], "tilts": tracker.tilts, "views": len(views),
                      "heldout": report}, ensure_ascii=False, indent=2))
    print("wrote", folder / "camera_calibration_cam0.json")


if __name__ == "__main__":
    main()
