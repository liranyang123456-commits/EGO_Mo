#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uncertainty budget for the gyro-locked chessboard reference.

A  corner noise: Monte Carlo spread of camera position, raw PnP vs locked refit
B  sensitivity of the locked position to a rotation error, per camera axis
C  gyro drift about the optical axis, against PnP roll (its best axis)
D  camera-IMU delay: locked reference re-exported with the delay offset
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import exp_so3
from tools.disambiguate_steps import _fit_t, _k, _obj
from tools.export_gt_locked import export_session

DATA = ROOT / "datasets"
HELD = ("traj_20260923_023241", "traj_20260923_023422")
SIGMAS = (0.10, 0.25, 0.50)
TRIALS = 60
FRAMES = 120
SHIFTS_MS = (-60, -40, -20, -10, 0, 10, 20, 40, 60)


def _so3_log(R):
    c = float(np.clip((np.trace(R) - 1) / 2, -1, 1))
    th = float(np.arccos(c))
    if th < 1e-9:
        return np.zeros(3)
    return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) * th / (2 * np.sin(th))


def _pose_to_cam(R_oc, p_oc):
    R_pnp = R_oc.T
    t_mm = (-R_oc.T @ p_oc) * 1000.0
    return R_pnp, t_mm


def monte_carlo(name, rng):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    lk = np.load(DATA / "pose_gt_locked" / f"{name}.npz")
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    obj = _obj().astype(np.float64)
    idx = np.flatnonzero(lk["usable"] == 1)
    idx = rng.choice(idx, min(FRAMES, len(idx)), replace=False)
    out = {}
    for sigma in SIGMAS:
        raw_std, lock_std, raw_rot, flips = [], [], [], 0
        for i in idx:
            R_pnp, t_mm = _pose_to_cam(lk["R"][i].astype(np.float64), lk["p"][i].astype(np.float64))
            rvec, _ = cv2.Rodrigues(R_pnp)
            clean, _ = cv2.projectPoints(obj, rvec, t_mm, K, dist)
            clean = clean.reshape(-1, 2)
            p_raw, p_lock, w_raw = [], [], []
            for _ in range(TRIALS):
                noisy = clean + rng.normal(0, sigma, clean.shape)
                ok, rv, tv = cv2.solvePnP(obj, noisy.reshape(-1, 1, 2), K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
                if ok:
                    Rn, _ = cv2.Rodrigues(rv)
                    p_raw.append(-Rn.T @ tv.reshape(3))
                    w_raw.append(_so3_log(R_pnp.T @ Rn))
                t_fit, _ = _fit_t(obj.astype(np.float32), noisy.reshape(-1, 1, 2).astype(np.float64), R_pnp, t_mm.copy(), K, dist)
                p_lock.append(-R_pnp.T @ t_fit)
            p_raw = np.asarray(p_raw)
            p_lock = np.asarray(p_lock)
            w_raw = np.degrees(np.asarray(w_raw))
            flips += int(np.sum(np.linalg.norm(w_raw, axis=1) > 5.0))
            raw_std.append(np.sqrt(np.sum(p_raw.var(0))))
            lock_std.append(np.sqrt(np.sum(p_lock.var(0))))
            raw_rot.append(np.sqrt(np.sum(w_raw.var(0))))
        out[f"{sigma:.2f}px"] = {
            "raw_pos_mm_med": round(float(np.median(raw_std)), 3),
            "raw_pos_mm_p90": round(float(np.quantile(raw_std, 0.9)), 3),
            "lock_pos_mm_med": round(float(np.median(lock_std)), 3),
            "lock_pos_mm_p90": round(float(np.quantile(lock_std, 0.9)), 3),
            "raw_rot_deg_med": round(float(np.median(raw_rot)), 3),
            "raw_flip_rate": round(flips / float(len(idx) * TRIALS), 4),
        }
    return out


def rotation_sensitivity(name, rng):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    lk = np.load(DATA / "pose_gt_locked" / f"{name}.npz")
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    obj = _obj()
    idx = np.flatnonzero(lk["usable"] == 1)
    idx = rng.choice(idx, min(FRAMES, len(idx)), replace=False)
    delta = np.radians(0.5)
    per_axis = {0: [], 1: [], 2: []}
    px_axis = {0: [], 1: [], 2: []}
    dist_mm = []
    for i in idx:
        R_oc = lk["R"][i].astype(np.float64)
        p_oc = lk["p"][i].astype(np.float64)
        R_pnp, t_mm = _pose_to_cam(R_oc, p_oc)
        dist_mm.append(float(np.linalg.norm(t_mm)))
        rvec, _ = cv2.Rodrigues(R_pnp)
        corners, _ = cv2.projectPoints(obj, rvec, t_mm, K, dist)
        for ax in range(3):
            w = np.zeros(3)
            w[ax] = delta
            R_oc_err = R_oc @ exp_so3(w)
            t_fit, px = _fit_t(obj, corners, R_oc_err.T, t_mm.copy(), K, dist)
            p_err = -R_oc_err @ (t_fit / 1000.0)
            per_axis[ax].append(np.linalg.norm(p_err - p_oc) * 1000.0 / 0.5)
            px_axis[ax].append(px / 0.5)
    return {
        "distance_mm_med": round(float(np.median(dist_mm)), 1),
        "mm_per_deg": [round(float(np.median(per_axis[a])), 3) for a in range(3)],
        "px_per_deg": [round(float(np.median(px_axis[a])), 3) for a in range(3)],
    }


def gyro_drift(name):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    lk = np.load(DATA / "pose_gt_locked" / f"{name}.npz")
    ok = (gt["ok"] == 1) & (gt["reproj"] < 1.0) & (gt["ble_gyro"] < 8.0)
    idx = np.flatnonzero(ok & (lk["usable"] == 1))
    t = gt["t"]
    gaps, diffs = [], []
    for a_pos in range(0, len(idx), 5):
        a = idx[a_pos]
        for b in idx[a_pos + 1:]:
            span = float(t[b] - t[a])
            if span < 5.0:
                continue
            if span > 30.0:
                break
            raw = _so3_log(gt["R"][a].T @ gt["R"][b])
            lock = _so3_log(lk["R"][a].astype(np.float64).T @ lk["R"][b].astype(np.float64))
            gaps.append(span)
            diffs.append(np.degrees(lock[2] - raw[2]))
            break
    if len(gaps) < 5:
        return None
    gaps = np.asarray(gaps)
    diffs = np.asarray(diffs)
    rate = np.abs(diffs) / gaps
    return {
        "pairs": int(len(gaps)),
        "span_s_med": round(float(np.median(gaps)), 1),
        "roll_diff_deg_med": round(float(np.median(np.abs(diffs))), 3),
        "drift_deg_per_s_med": round(float(np.median(rate)), 4),
        "drift_deg_per_s_p90": round(float(np.quantile(rate, 0.9)), 4),
    }


def delay_sweep(name):
    ref = np.load(DATA / "pose_gt_locked" / f"{name}.npz")
    rows = []
    tmp = DATA / "pose_gt_locked_shift"
    for ms in SHIFTS_MS:
        row = export_session(name, shift=ms / 1000.0, out_dir=tmp)
        got = np.load(tmp / f"{name}.npz")
        both = (got["usable"] == 1) & (ref["usable"] == 1)
        dp = np.linalg.norm(got["p"][both] - ref["p"][both], axis=1) * 1000.0
        rows.append({
            "shift_ms": ms,
            "usable": row["usable"],
            "px_med": row["px_med"],
            "pos_vs_nominal_mm_med": round(float(np.median(dp)), 3) if len(dp) else None,
        })
    return rows


def main() -> None:
    rng = np.random.default_rng(0)
    report = {}
    for name in HELD:
        block = {
            "monte_carlo": monte_carlo(name, rng),
            "rotation_sensitivity": rotation_sensitivity(name, rng),
            "gyro_drift_optical_axis": gyro_drift(name),
            "delay_sweep": delay_sweep(name),
        }
        report[name] = block
        print(json.dumps({name: block}, ensure_ascii=False), flush=True)
    out = DATA / "traj_run_v8" / "gt_uncertainty.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
