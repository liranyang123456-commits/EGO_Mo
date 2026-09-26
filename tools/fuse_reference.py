#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gyro-PnP rotation smoother and the translation refit on top of it.

Rotation at camera frame i is G_i exp(e_i), where G is the gyro chain and e a
small correction with a bias state: e' = e + b dt, b a random walk. PnP gives
e_i + noise, with noise that grows with angular rate (measured per axis on
all recordings). A Rauch-Tung-Striebel smoother gives e and its variance at
every frame, board visible or not. Translation is refit to the board
projection under the smoothed rotation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import exp_so3
from ego_capture.sync import load_imu
from tools.chain_lock import BODY_TO_CAM
from tools.disambiguate_steps import _fit_t, _k, _obj
from tools.gt_uncertainty import _so3_log

DATA = ROOT / "datasets"
OUT = DATA / "pose_gt_fused"
D2R = np.pi / 180.0
# PnP frame noise (deg) = base + slope * angular rate (deg/s), capped; tilt, tilt, roll.
PNP_BASE = np.array([0.55, 0.45, 0.08])
PNP_SLOPE = np.array([0.035, 0.035, 0.012])
PNP_CAP = np.array([1.6, 1.6, 0.6])
Q_E = (0.05 * D2R) ** 2
Q_B = (0.02 * D2R) ** 2
B0 = (0.2 * D2R) ** 2
PX_MAX = 1.0


def gyro_chain(name):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * D2R
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    w = (gyro - bg) @ BODY_TO_CAM.T
    t_cam = gt["t"] - float(gt["delay_usb"][0])
    n = len(t_cam)
    k = np.clip(np.searchsorted(usb_t, t_cam), 1, len(usb_t) - 1)
    G = np.empty((n, 3, 3))
    rate = np.zeros(n)
    R = np.eye(3)
    j = 0
    for i in range(n):
        while j + 1 < len(usb_t) and usb_t[j + 1] <= t_cam[i]:
            dt = float(usb_t[j + 1] - usb_t[j])
            if 0.0 < dt <= 0.05:
                R = R @ exp_so3(w[j + 1] * dt)
            j += 1
        G[i] = R
        m = (usb_t > t_cam[i] - 0.05) & (usb_t < t_cam[i] + 0.05)
        rate[i] = float(np.degrees(np.linalg.norm(w[m], axis=1).mean())) if m.any() else 0.0
    return gt, t_cam, G, rate


def smooth(t, G, R_meas, meas_ok, rate, drop=None):
    """Per-axis RTS smoother on [e, b]. Returns e (n,3) and std (n,3), radians."""
    n = len(t)
    use = meas_ok.copy()
    if drop is not None:
        use[drop] = False
    z = np.zeros((n, 3))
    for i in np.flatnonzero(use):
        z[i] = _so3_log(G[i].T @ R_meas[i])
    sig = np.minimum(PNP_BASE + PNP_SLOPE * rate[:, None], PNP_CAP) * D2R
    e_s = np.zeros((n, 3))
    s_s = np.zeros((n, 3))
    for ax in range(3):
        xf = np.zeros((n, 2))
        Pf = np.zeros((n, 2, 2))
        xp = np.zeros((n, 2))
        Pp = np.zeros((n, 2, 2))
        x = np.array([z[np.flatnonzero(use)[0], ax] if use.any() else 0.0, 0.0])
        P = np.diag([(2.0 * D2R) ** 2, B0])
        Fs = []
        for i in range(n):
            if i > 0:
                dt = max(float(t[i] - t[i - 1]), 1e-3)
                F = np.array([[1.0, dt], [0.0, 1.0]])
                Q = np.diag([Q_E * dt, Q_B * dt])
                x = F @ x
                P = F @ P @ F.T + Q
                Fs.append(F)
            xp[i], Pp[i] = x, P
            if use[i]:
                r = sig[i, ax] ** 2
                S = P[0, 0] + r
                K = P[:, 0] / S
                x = x + K * (z[i, ax] - x[0])
                P = P - np.outer(K, P[0, :])
            xf[i], Pf[i] = x, P
        xs, Ps = xf[-1].copy(), Pf[-1].copy()
        e_s[-1, ax], s_s[-1, ax] = xs[0], np.sqrt(max(Ps[0, 0], 0.0))
        for i in range(n - 2, -1, -1):
            F = Fs[i]
            C = Pf[i] @ F.T @ np.linalg.inv(Pp[i + 1])
            xs = xf[i] + C @ (xs - xp[i + 1])
            Ps = Pf[i] + C @ (Ps - Pp[i + 1]) @ C.T
            e_s[i, ax], s_s[i, ax] = xs[0], np.sqrt(max(Ps[0, 0], 0.0))
    return e_s, s_s


def fused_rotations(t, G, R_meas, ok, rate, drop=None, passes=2):
    G_lin = G.copy()
    for _ in range(passes):
        e, s = smooth(t, G_lin, R_meas, ok, rate, drop)
        G_lin = np.stack([G_lin[i] @ exp_so3(e[i]) for i in range(len(t))])
    return G_lin, s


def cross_validate(t, G, R_meas, ok, rate, rng, fold=5):
    idx = np.flatnonzero(ok)
    drop = idx[rng.permutation(len(idx))[: len(idx) // fold]]
    R_fused, _ = fused_rotations(t, G, R_meas, ok, rate, drop)
    fused_err, interp_err, chain_err = [], [], []
    keep = np.setdiff1d(idx, drop)
    # Gyro chain aligned once to all kept PnP (best constant offset), no smoothing.
    offs = np.stack([_so3_log(G[i].T @ R_meas[i]) for i in keep]).mean(0)
    G_const = np.stack([G[i] @ exp_so3(offs) for i in range(len(t))])
    for i in drop:
        fused_err.append(np.degrees(_so3_log(R_fused[i].T @ R_meas[i])))
        chain_err.append(np.degrees(_so3_log(G_const[i].T @ R_meas[i])))
        lo = keep[keep < i]
        hi = keep[keep > i]
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]
            u = (t[i] - t[a]) / max(t[b] - t[a], 1e-6)
            Ri = R_meas[a] @ exp_so3(u * _so3_log(R_meas[a].T @ R_meas[b]))
            interp_err.append(np.degrees(_so3_log(Ri.T @ R_meas[i])))
    rms = lambda x: [round(float(v), 3) for v in np.sqrt(np.mean(np.asarray(x) ** 2, 0))]
    return {"held_frames": int(len(drop)), "fused_rms_deg": rms(fused_err),
            "pnp_interp_rms_deg": rms(interp_err), "gyro_chain_rms_deg": rms(chain_err)}


def export(name, rng):
    gt, t, G, rate = gyro_chain(name)
    ok = (gt["ok"] == 1) & (gt["reproj"] < 1.5) & (gt["ble_gyro"] < 8.0)
    R_meas = gt["R"].astype(np.float64)
    cv = cross_validate(t, G, R_meas, ok, rate, rng)
    R_f, s = fused_rotations(t, G, R_meas, ok, rate)
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    obj = _obj()
    n = len(t)
    p = np.zeros((n, 3))
    px = np.full(n, np.nan)
    usable = np.zeros(n, dtype=np.uint8)
    for i in np.flatnonzero(ok):
        t_raw = (-gt["R"][i].T @ gt["p"][i]) * 1000.0
        rvec, _ = cv2.Rodrigues(gt["R"][i].T)
        corners, _ = cv2.projectPoints(obj, rvec, t_raw, K, dist)
        t_fit, err = _fit_t(obj, corners, R_f[i].T, t_raw.copy(), K, dist)
        p[i] = -R_f[i] @ (t_fit / 1000.0)
        px[i] = err
        usable[i] = int(err <= PX_MAX)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{name}.npz", t=gt["t"], R=R_f.astype(np.float32), p=p.astype(np.float32),
                        rot_std_deg=np.degrees(s).astype(np.float32), px=px.astype(np.float32),
                        usable=usable, board_ok=ok.astype(np.uint8))
    raw_dp = np.linalg.norm(p[usable == 1] - gt["p"][usable == 1], axis=1) * 1000.0
    ang = [np.degrees(np.linalg.norm(_so3_log(R_f[i].T @ R_meas[i]))) for i in np.flatnonzero(usable)]
    return {
        "name": name, "frames": int(n), "board_ok": int(ok.sum()), "usable": int(usable.sum()),
        "px_med": round(float(np.nanmedian(px[usable == 1])), 3) if usable.any() else None,
        "rot_std_deg_med": [round(float(v), 3) for v in np.median(np.degrees(s[usable == 1]), 0)],
        "vs_raw_pos_mm": [round(float(np.median(raw_dp)), 2), round(float(np.quantile(raw_dp, 0.9)), 2)],
        "vs_raw_deg": [round(float(np.median(ang)), 3), round(float(np.quantile(ang, 0.9)), 3)],
        "cross_val": cv,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    rng = np.random.default_rng(0)
    index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    rows = []
    for name in names:
        row = export(name, rng)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
