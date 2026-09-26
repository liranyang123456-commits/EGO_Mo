#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-based camera-IMU lever-arm estimate (no network involved).

For a rigid rig observing a static chessboard (board frame B = world),
with r the IMU position in the left-camera frame C,

    R_CI f_I = R_BC^T (p''_BC - g_B) + ([w']x + [w]x[w]x) r + b_C,

where f_I is the accelerometer specific force, w the gyro rate in C, p_BC and
R_BC the PnP camera pose, g_B gravity in the board frame and b_C a constant
accelerometer bias. The equation is linear in (g_B, r, b_C). Both sides are
low-pass filtered with the same zero-phase filter, because p'' comes from
10-fps PnP. The reported lever arm is lambda = -r, i.e. the vector from the
IMU sensing centre to the camera optical centre in C, which is the quantity
entered in datasets/lever_arm_measurements.csv and learned by PhysNet.

Modes
-----
sim-check  digital-twin sequence with a known lever arm, 10-fps pose samples
           with PnP-like noise and outliers; checks that the estimator
           recovers the injected value.
real       estimate on recorded sessions (default: the rigid rotation
           sessions of the current mounting).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets"
G = 9.80665
HZ = 200.0
DEFAULT_OUT = DATA / "lever_arm_physics_estimate.json"


def _skew(v):
    z = np.zeros(v.shape[:-1] + (3, 3))
    z[..., 0, 1], z[..., 0, 2] = -v[..., 2], v[..., 1]
    z[..., 1, 0], z[..., 1, 2] = v[..., 2], -v[..., 0]
    z[..., 2, 0], z[..., 2, 1] = -v[..., 1], v[..., 0]
    return z


def _spans(t, max_gap):
    cut = np.flatnonzero(np.diff(t) > max_gap) + 1
    return np.split(np.arange(len(t)), cut)


def _despike(p, thresh_m):
    """Drop poses far from the median of their +-2 neighbours."""
    keep = np.ones(len(p), bool)
    for i in range(2, len(p) - 2):
        med = np.median(p[i - 2:i + 3], axis=0)
        keep[i] = np.linalg.norm(p[i] - med) < thresh_m
    return keep


def build_system(t_cam, R_bc, p_bc, t_imu, acc_g, gyro_dps, R_ci,
                 fc_hz=2.0, max_gap_s=0.35, min_span_s=4.0, trim_s=1.0,
                 out_hz=20.0, despike_mm=1.5):
    """Stacked rows (y, H) for y = H @ [g_B, r, b_C]."""
    keep = _despike(p_bc, despike_mm / 1000.0)
    t_cam, R_bc, p_bc = t_cam[keep], R_bc[keep], p_bc[keep]
    gyro = np.radians(gyro_dps) @ R_ci.T
    still = np.linalg.norm(gyro_dps, axis=1) < 1.0
    if still.sum() > 100:
        gyro = gyro - gyro[still].mean(0)
    f_c = (acc_g * G) @ R_ci.T
    sos = butter(4, fc_hz, fs=HZ, output="sos")
    ys, Hs, ts = [], [], []
    for idx in _spans(t_cam, max_gap_s):
        if len(idx) < 8 or t_cam[idx[-1]] - t_cam[idx[0]] < min_span_s:
            continue
        tc = t_cam[idx]
        grid = np.arange(tc[0], tc[-1], 1.0 / HZ)
        acc_p = CubicSpline(tc, p_bc[idx], axis=0)(grid, 2)
        R = Slerp(tc, Rotation.from_matrix(R_bc[idx]))(grid).as_matrix()
        w = np.column_stack([np.interp(grid, t_imu, gyro[:, k]) for k in range(3)])
        f = np.column_stack([np.interp(grid, t_imu, f_c[:, k]) for k in range(3)])
        lp = lambda x: sosfiltfilt(sos, x, axis=0)
        w_lp = lp(w)
        w_dot = np.gradient(w_lp, grid, axis=0)
        A = _skew(w_dot) + _skew(w_lp) @ _skew(w_lp)
        Rt = np.transpose(R, (0, 2, 1))
        y = lp(f) - lp(np.einsum("nij,nj->ni", Rt, acc_p))
        Gm = lp(-Rt.reshape(len(grid), 9)).reshape(-1, 3, 3)
        Am = lp(A.reshape(len(grid), 9)).reshape(-1, 3, 3)
        sel = (grid >= grid[0] + trim_s) & (grid <= grid[-1] - trim_s)
        sel &= (np.arange(len(grid)) % int(round(HZ / out_hz))) == 0
        H = np.concatenate([Gm[sel], Am[sel], np.repeat(np.eye(3)[None], sel.sum(), 0)], axis=2)
        ys.append(y[sel]); Hs.append(H); ts.append(grid[sel])
    if not ys:
        raise RuntimeError("no usable pose span")
    return np.concatenate(ys), np.concatenate(Hs), np.concatenate(ts)


FIX_GRAVITY_NORM = False


def solve(y, H, iters=10, huber=1.345):
    """Robust fit; returns x = [g_B, r, b_C] and the robust row scale.

    Linear Huber IRLS by default. With FIX_GRAVITY_NORM, gravity is
    parameterised by its direction only (|g_B| = G), which helps when small
    rotations let R^T g trade off against the accelerometer bias.
    """
    from scipy.optimize import least_squares

    Y, M = y.reshape(-1), H.reshape(-1, 9)
    if not FIX_GRAVITY_NORM:
        w = np.ones(len(Y))
        for _ in range(iters):
            sw = np.sqrt(w)
            x = np.linalg.lstsq(M * sw[:, None], Y * sw, rcond=None)[0]
            res = Y - M @ x
            s = 1.4826 * np.median(np.abs(res)) + 1e-12
            u = np.abs(res) / (huber * s)
            w = np.where(u <= 1, 1.0, 1.0 / u)
        return x, float(s)
    x_lin = np.linalg.lstsq(M, Y, rcond=None)[0]
    g0 = x_lin[:3] / max(np.linalg.norm(x_lin[:3]), 1e-9)
    theta0 = [np.arccos(np.clip(g0[2], -1, 1)), np.arctan2(g0[1], g0[0])]

    def unpack(q):
        th, ph = q[0], q[1]
        g = G * np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
        return np.r_[g, q[2:8]]

    def fun(q):
        return M @ unpack(q) - Y

    r0 = fun(np.r_[theta0, x_lin[3:9]])
    scale = 1.4826 * np.median(np.abs(r0)) + 1e-9
    sol = least_squares(fun, np.r_[theta0, x_lin[3:9]], loss="huber", f_scale=huber * scale)
    res = fun(sol.x)
    return unpack(sol.x), float(1.4826 * np.median(np.abs(res)))


def estimate(y, H, t, n_boot=300, block_s=3.0, seed=0):
    x, s = solve(y, H)
    rng = np.random.default_rng(seed)
    blocks = np.floor((t - t.min()) / block_s).astype(int)
    ids = np.unique(blocks)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sel = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        boots.append(solve(y[sel], H[sel], iters=5)[0])
    boots = np.asarray(boots)
    sv = np.linalg.svd(H[:, :, 3:6].reshape(-1, 3), compute_uv=False)
    lam = -x[3:6] * 1000
    return {
        "lambda_imu_to_camera_mm": lam.tolist(),
        "lambda_norm_mm": float(np.linalg.norm(lam)),
        "lambda_95ci_mm": np.quantile(-boots[:, 3:6] * 1000, [0.025, 0.975], axis=0).T.tolist(),
        "lambda_std_mm": (boots[:, 3:6].std(0) * 1000).tolist(),
        "gravity_norm_m_s2": float(np.linalg.norm(x[:3])),
        "accel_bias_camera_m_s2": x[6:9].tolist(),
        "residual_robust_sigma_m_s2": float(s),
        "rows": int(len(y)),
        "lever_excitation_singular_values": sv.tolist(),
    }


def sim_check(args):
    from ego_sim.core import R_CAMERA_IMU, SimConfig, build_sequence
    truth_r = np.array(args.sim_lever_mm, float)
    rng = np.random.default_rng(3)
    results = []
    for seed in range(args.sim_runs):
        cfg = SimConfig(duration_s=args.duration, seed=100 + seed, motion="pure_rotation",
                        rotation_deg=args.rotation_deg, speed=args.speed, render_images=False,
                        gravity_board=(0.0, 0.87, 0.49),
                        imu_lever_arm_mm_x=truth_r[0], imu_lever_arm_mm_y=truth_r[1],
                        imu_lever_arm_mm_z=truth_r[2])
        seq = build_sequence(cfg)
        t = seq.poses.t
        cam_t = np.arange(t[0] + 0.05, t[-1] - 0.05, 0.1)
        k = np.searchsorted(t, cam_t)
        R_wc, p_wc = seq.poses.R_W_C[k], seq.poses.p_W_C[k]
        # PnP-like errors plus 2 % 2-mm outliers
        R_obs = np.einsum("nij,njk->nik", R_wc, Rotation.from_rotvec(
            rng.normal(0, np.radians(args.pnp_rot_deg), (len(k), 3))).as_matrix())
        p_obs = p_wc + rng.normal(0, args.pnp_pos_mm / 1000.0, p_wc.shape)
        out = rng.random(len(k)) < 0.02
        p_obs[out] += rng.normal(0, 0.002, (out.sum(), 3))
        acc_g, gyro_dps = seq.imu_usb[:, 0:3], seq.imu_usb[:, 3:6]
        y, H, tt = build_system(t[k], R_obs, p_obs, t, acc_g, gyro_dps, R_CAMERA_IMU,
                                fc_hz=args.fc)
        est = estimate(y, H, tt, n_boot=100, seed=seed)
        est["seed"] = cfg.seed
        results.append(est)
    lam_true = -truth_r
    lam = np.array([r["lambda_imu_to_camera_mm"] for r in results])
    return {
        "mode": "sim-check", "fc_hz": args.fc,
        "motion": {"duration_s": args.duration, "rotation_deg": args.rotation_deg,
                   "speed": args.speed},
        "pnp_noise": {"position_mm": args.pnp_pos_mm, "rotation_deg": args.pnp_rot_deg},
        "injected_lambda_mm": lam_true.tolist(),
        "runs": results,
        "mean_error_mm": (lam.mean(0) - lam_true).tolist(),
        "rmse_per_axis_mm": np.sqrt(((lam - lam_true) ** 2).mean(0)).tolist(),
        "ci_coverage": float(np.mean([
            all(lo <= v <= hi for v, (lo, hi) in zip(lam_true, r["lambda_95ci_mm"]))
            for r in results])),
    }


def _load_real(name):
    from ego_capture.sync import load_imu
    import tools.train_seq as api
    raw = np.load(DATA / "pose_gt_raw" / f"{name}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{name}.npz")["delay_usb"][0])
    use = raw["usable"] == 1
    t_imu, usb = load_imu(DATA / name / "imu_stream.csv")
    R_ci = (api.R_CAMERA_IMU_NEW if name.startswith(("traj_20260924_", "rigid_20260924_14"))
            else api.R_CAMERA_IMU_OLD).astype(np.float64)
    return (raw["t"][use] - delay, raw["R"][use].astype(np.float64),
            raw["p"][use].astype(np.float64), t_imu, usb[:, 0:3], usb[:, 3:6], R_ci)


def real(args):
    per, ys, Hs, ts, offset = {}, [], [], [], 0.0
    for name in args.sessions:
        t_cam, R, p, t_imu, acc, gyro, R_ci = _load_real(name)
        y, H, tt = build_system(t_cam, R, p, t_imu, acc, gyro, R_ci, fc_hz=args.fc)
        per[name] = estimate(y, H, tt, n_boot=args.boot)
        # Gravity in the board frame and accel bias differ between sessions,
        # so the pooled fit keeps them per session and shares only r.
        ys.append(y); Hs.append((H, len(ts))); ts.append(tt + offset)
        offset = ts[-1].max() + 100.0
    n_s = len(ys)
    rows = []
    for k, (H, _i) in enumerate(Hs):
        Hk = np.zeros((len(H), 3, 3 + 6 * n_s))
        Hk[:, :, 0:3] = H[:, :, 3:6]
        Hk[:, :, 3 + 6 * k:6 + 6 * k] = H[:, :, 0:3]
        Hk[:, :, 6 + 6 * k:9 + 6 * k] = H[:, :, 6:9]
        rows.append(Hk)
    Y, HH, T = np.concatenate(ys), np.concatenate(rows), np.concatenate(ts)

    def pooled_solve(yy, hh):
        M = hh.reshape(-1, hh.shape[2]); v = yy.reshape(-1)
        w = np.ones(len(v))
        for _ in range(10):
            sw = np.sqrt(w)
            x = np.linalg.lstsq(M * sw[:, None], v * sw, rcond=None)[0]
            res = v - M @ x
            s = 1.4826 * np.median(np.abs(res)) + 1e-12
            u = np.abs(res) / (1.345 * s)
            w = np.where(u <= 1, 1.0, 1.0 / u)
        return x
    x = pooled_solve(Y, HH)
    rng = np.random.default_rng(0)
    blocks = np.floor(T / 3.0).astype(int)
    ids = np.unique(blocks)
    boots = []
    for _ in range(args.boot):
        pick = rng.choice(ids, len(ids), replace=True)
        sel = np.concatenate([np.flatnonzero(blocks == b) for b in pick])
        boots.append(pooled_solve(Y[sel], HH[sel])[:3])
    boots = -np.asarray(boots) * 1000
    lam = -x[:3] * 1000
    learned = []
    for f in sorted(DATA.glob("physnet_cv/metrics_cvx_*.json")):
        m = json.loads(f.read_text(encoding="utf-8"))
        if "lever_arm_mm" in m:
            learned.append(m["lever_arm_mm"])
    learned = np.asarray(learned) if learned else None
    return {
        "mode": "real", "fc_hz": args.fc, "sessions": args.sessions,
        "per_session": per,
        "pooled": {
            "lambda_imu_to_camera_mm": lam.tolist(),
            "lambda_norm_mm": float(np.linalg.norm(lam)),
            "lambda_95ci_mm": np.quantile(boots, [0.025, 0.975], axis=0).T.tolist(),
            "lambda_std_mm": boots.std(0).tolist(),
        },
        "physnet_learned_lambda_mm": None if learned is None else {
            "n": int(len(learned)), "mean": learned.mean(0).tolist(),
            "std": learned.std(0).tolist()},
    }


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("sim-check")
    s.add_argument("--sim-lever-mm", type=float, nargs=3, default=[22.6, 5.2, 29.7],
                   help="IMU position in the camera frame (r); lambda = -r")
    s.add_argument("--sim-runs", type=int, default=5)
    s.add_argument("--fc", type=float, default=2.0)
    s.add_argument("--duration", type=float, default=90.0)
    s.add_argument("--rotation-deg", type=float, default=20.0)
    s.add_argument("--speed", type=float, default=1.2)
    s.add_argument("--pnp-pos-mm", type=float, default=0.3)
    s.add_argument("--pnp-rot-deg", type=float, default=0.2)
    s.add_argument("--output", type=Path, default=DATA / "lever_arm_sim_check.json")
    r = sub.add_parser("real")
    r.add_argument("--sessions", nargs="+",
                   default=["rigid_20260924_140510", "rigid_20260924_142902"])
    r.add_argument("--fc", type=float, default=2.0)
    r.add_argument("--boot", type=int, default=300)
    r.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    report = sim_check(args) if args.mode == "sim-check" else real(args)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    brief = ({k: report[k] for k in ("injected_lambda_mm", "mean_error_mm", "rmse_per_axis_mm", "ci_coverage")}
             if args.mode == "sim-check" else
             {"pooled": report["pooled"], "learned": report["physnet_learned_lambda_mm"],
              "per_session": {k: (np.round(v["lambda_imu_to_camera_mm"], 1).tolist(),
                                  round(v["gravity_norm_m_s2"], 3)) for k, v in report["per_session"].items()}})
    print(json.dumps(brief, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
