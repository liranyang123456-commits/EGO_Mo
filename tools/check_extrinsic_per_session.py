#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Estimate the camera-IMU rotation R_CI independently for every recording.

Rotation increments over 0.8-1.2 s from the bias-corrected gyroscope are
matched to chessboard-PnP increments by Kabsch/Wahba. The estimate is
compared with the two stored extrinsics (OLD for the 2026-09-23 mount, NEW for
2026-09-24), and the per-axis correlation of the increments is reported for
each candidate, so a wrong extrinsic shows up as a large angle and a lower
held-out correlation.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.train_seq as api
from ego_capture.sync import load_imu

DATA = ROOT / "datasets"


def increments(name):
    raw = np.load(DATA / "pose_gt_raw" / f"{name}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{name}.npz")["delay_usb"][0])
    use = raw["usable"] == 1
    t = raw["t"][use] - delay
    R_bc = raw["R"][use].astype(np.float64)
    ti, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = np.radians(usb[:, 3:6])
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    if still.sum() > 50:
        gyro = gyro - gyro[still].mean(0)
    # integrated IMU orientation
    q = Rotation.identity()
    rots = [q]
    for k in range(1, len(ti)):
        q = q * Rotation.from_rotvec(0.5 * (gyro[k] + gyro[k - 1]) * (ti[k] - ti[k - 1]))
        rots.append(q)
    quats = Rotation.concatenate(rots)
    from scipy.spatial.transform import Slerp
    slerp = Slerp(ti, quats)
    phi_i, phi_c = [], []
    for a in range(0, len(t)):
        b = np.searchsorted(t, t[a] + 1.0)
        if b >= len(t) or not (0.8 <= t[b] - t[a] <= 1.2):
            continue
        if t[a] < ti[0] or t[b] > ti[-1]:
            continue
        Ri = (slerp([t[a]]).inv() * slerp([t[b]]))[0].as_rotvec()
        Rc = Rotation.from_matrix(R_bc[a].T @ R_bc[b]).as_rotvec()
        if np.linalg.norm(Rc) < np.radians(1.0):
            continue
        phi_i.append(Ri); phi_c.append(Rc)
    return np.asarray(phi_i), np.asarray(phi_c)


def kabsch(A, B):
    """R minimizing sum ||R a - b||^2."""
    H = A.T @ B
    U, _S, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def angle(Ra, Rb):
    return float(np.degrees(np.linalg.norm(Rotation.from_matrix(Ra @ Rb.T).as_rotvec())))


def corr(R, A, B):
    P = A @ R.T
    return [float(np.corrcoef(P[:, k], B[:, k])[0, 1]) for k in range(3)]


def main():
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    names = split["train"] + split["val"] + split["test"] + split["extra_train"]
    old = api.R_CAMERA_IMU_OLD.astype(np.float64)
    new = api.R_CAMERA_IMU_NEW.astype(np.float64)
    report = {}
    for name in names:
        A, B = increments(name)
        if len(A) < 30:
            print(name, "too few increments", len(A)); continue
        half = np.arange(len(A)) % 2 == 0
        est = kabsch(A[half], B[half])
        res = {
            "n": int(len(A)),
            "angle_to_old_deg": angle(est, old), "angle_to_new_deg": angle(est, new),
            "heldout_corr_est": corr(est, A[~half], B[~half]),
            "heldout_corr_old": corr(old, A[~half], B[~half]),
            "heldout_corr_new": corr(new, A[~half], B[~half]),
            "residual_deg_old": float(np.degrees(np.sqrt(np.mean(np.sum((A[~half] @ old.T - B[~half]) ** 2, 1))))),
            "residual_deg_new": float(np.degrees(np.sqrt(np.mean(np.sum((A[~half] @ new.T - B[~half]) ** 2, 1))))),
            "residual_deg_est": float(np.degrees(np.sqrt(np.mean(np.sum((A[~half] @ est.T - B[~half]) ** 2, 1))))),
            "R_est": est.tolist(),
        }
        report[name] = res
        print(f"{name:24s} n={res['n']:4d}  est->OLD {res['angle_to_old_deg']:5.2f} deg  est->NEW "
              f"{res['angle_to_new_deg']:5.2f} deg  resid deg OLD {res['residual_deg_old']:.2f} "
              f"NEW {res['residual_deg_new']:.2f} est {res['residual_deg_est']:.2f}", flush=True)
    (DATA / "extrinsic_per_session.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
