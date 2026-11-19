#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PnP orientation in motion against an IMU fixed to the board.

When the camera stays still and the board, carrying the Bluetooth IMU, is
moved by hand, PnP gives the board orientation in camera coordinates at every
frame and the gyroscope gives its angular velocity without any image. Per
recording the comparison gives

  delay     lateness of the camera stamps against the IMU stream, at the
            maximum correlation of the angular rates;
  rate      PnP angular velocity over 4 frame intervals against the gyroscope;
  rotation  1-s rotations scored as in tools/check_rigidity.py (mount fitted
            on every other window, scored on the others);
  window    relative rotation over 1-s and 3-s windows, the orientation part
            of a label. The integrated gyroscope has errors of its own that
            grow with the rotation it integrates, whereas the PnP error of a
            window depends only on its two endpoints; the difference is
            therefore regressed on the integrated path, and the intercept
            bounds the PnP part.

The camera IMU must stay still (p95 rate below 1 deg/s) and the board IMU move.

    python tools/board_imu_motion_check.py
    python tools/board_imu_motion_check.py rigid_20261108_003840 traj_20261108_023619
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
OUT_JSON = DATA / "board_imu_motion_check.json"
OUT_FIG = ROOT / "paper" / "response_figures" / "board_imu_motion.pdf"
HALF = 2                      # PnP rates over 2 * HALF frame intervals
RATE_RANGE = (3.0, 200.0)     # deg/s used for the delay and the IMU mount
DELAY_GRID = (-0.6, 0.2)      # s, IMU time minus camera time
WINDOWS_S = (1.0, 3.0)
LEVER_MM = 150.0              # median working distance of the trajectory recordings
SIGMA_THETA_DEG = 0.21        # rotation spread of the label budget (Section V-B)


def _kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rotation R with B ~= A R^T (rows are vectors)."""
    U, _s, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def _relative(R: np.ndarray, a, b) -> Rotation:
    """Board rotation from frame a to frame b in board coordinates (R: camera -> board)."""
    return Rotation.from_matrix(np.einsum("...ij,...kj->...ik", R[a], R[b]))


def load(name: str) -> dict:
    g = np.load(DATA / "pose_gt" / f"{name}.npz")
    bt, by = load_imu(DATA / name / "imu_bt.csv")
    ut, uy = load_imu(DATA / name / "imu_stream.csv")
    keep = np.r_[True, np.diff(bt) > 0]
    return {"name": name, "t": g["t"].astype(np.float64), "R": g["R"].astype(np.float64),
            "ok": (g["ok"] == 1) & (g["reproj"] < 1.5),
            "bt": bt[keep], "gyro": by[keep, 3:6], "usb_t": ut, "usb": uy}


def camera_still(s: dict) -> dict:
    gyro = np.linalg.norm(s["usb"][:, 3:6], axis=1)
    acc = s["usb"][:, 0:3]
    sec = np.floor(s["usb_t"] - s["usb_t"][0]).astype(int)
    means = np.array([acc[sec == k].mean(axis=0) for k in np.unique(sec) if (sec == k).sum() > 50])
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    centre = np.median(means, axis=0)
    centre /= np.linalg.norm(centre)
    spread = np.degrees(np.arccos(np.clip(means @ centre, -1.0, 1.0)))
    return {"gyro_p95_dps": float(np.quantile(gyro, 0.95)), "tilt_max_deg": float(spread.max()),
            "still": bool(np.quantile(gyro, 0.95) < 1.0 and spread.max() < 0.5)}


def qualifies(name: str) -> bool:
    folder = DATA / name
    if not ((folder / "imu_bt.csv").is_file() and (folder / "imu_stream.csv").is_file()):
        return False
    g = np.load(DATA / "pose_gt" / f"{name}.npz")
    board = g["ble_gyro"][g["ble_gyro"] < 999.0]
    if len(board) < 100 or np.quantile(board, 0.95) < 8.0 or (g["ok"] == 1).mean() < 0.8:
        return False
    return camera_still(load(name))["still"]


def pnp_rates(s: dict):
    t, idx = s["t"], np.flatnonzero(s["ok"])
    a, b = idx[:-2 * HALF], idx[2 * HALF:]
    step = float(np.median(np.diff(t[idx])))
    keep = (t[b] - t[a]) <= (2 * HALF + 1) * step
    a, b = a[keep], b[keep]
    rv = _relative(s["R"], a, b).as_rotvec(degrees=True)
    return t[a], t[b], rv / (t[b] - t[a])[:, None]


def _cumulative(bt: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """Trapezoidal integral of the rates; gaps are bridged by the mean of their end rates."""
    return np.vstack([np.zeros(3), np.cumsum(0.5 * (gyro[1:] + gyro[:-1]) * np.diff(bt)[:, None], axis=0)])


def gyro_mean(bt, C, t0, t1) -> np.ndarray:
    c0 = np.stack([np.interp(t0, bt, C[:, i]) for i in range(3)], axis=1)
    c1 = np.stack([np.interp(t1, bt, C[:, i]) for i in range(3)], axis=1)
    return (c1 - c0) / (t1 - t0)[:, None]


def delay_and_mount(s: dict, t0, t1, wp):
    bt, C = s["bt"], _cumulative(s["bt"], s["gyro"])
    mag = np.linalg.norm(wp, axis=1)
    moving = (mag > RATE_RANGE[0]) & (mag < RATE_RANGE[1])
    inside = lambda tau: moving & (t0 + tau >= bt[0]) & (t1 + tau <= bt[-1])  # noqa: E731
    coarse = []
    for tau in np.arange(DELAY_GRID[0], DELAY_GRID[1] + 1e-9, 0.005):
        m = inside(tau)
        wb = gyro_mean(bt, C, t0[m] + tau, t1[m] + tau)
        coarse.append((float(np.corrcoef(mag[m], np.linalg.norm(wb, axis=1))[0, 1]), float(tau)))
    tau0 = max(coarse)[1]
    best = None
    for tau in np.arange(tau0 - 0.02, tau0 + 0.02 + 1e-9, 0.001):
        m = inside(tau)
        wb = gyro_mean(bt, C, t0[m] + tau, t1[m] + tau)
        R_IB = _kabsch(wp[m], wb)
        corr = float(np.corrcoef((wp[m] @ R_IB.T).ravel(), wb.ravel())[0, 1])
        if best is None or corr > best[0]:
            best = (corr, float(tau), R_IB)
    return best


def static_bias(s: dict, tau: float):
    """Gyroscope output while PnP sees the board still (rotation below 0.3 deg in 1 s)."""
    t, R, ok, bt = s["t"], s["R"], s["ok"], s["bt"]
    mask = np.zeros(len(bt), dtype=bool)
    for k in np.flatnonzero(ok)[::3]:
        j = int(np.searchsorted(t, t[k] + 1.0))
        if j < len(t) and ok[j] and _relative(R, k, j).magnitude() < np.radians(0.3):
            mask |= (bt >= t[k] + tau) & (bt <= t[j] + tau)
    n = int(mask.sum())
    return (s["gyro"][mask].mean(axis=0) if n >= 200 else np.zeros(3)), n


def rate_metrics(s: dict, t0, t1, wp, tau, R_IB, bias) -> dict:
    bt = s["bt"]
    m = (t0 + tau >= bt[0]) & (t1 + tau <= bt[-1])
    wb = gyro_mean(bt, _cumulative(bt, s["gyro"] - bias), t0[m] + tau, t1[m] + tau)
    pred = wp[m] @ R_IB.T
    mag = np.linalg.norm(pred, axis=1)
    sel = (mag > RATE_RANGE[0]) & (mag < RATE_RANGE[1])
    gmag = np.linalg.norm(wb, axis=1)
    bins = {}
    for lo, hi in ((3, 10), (10, 30), (30, 60), (60, 200)):
        b = sel & (gmag >= lo) & (gmag < hi)
        if b.sum() >= 10:
            bins[f"{lo}-{hi} dps"] = {"n": int(b.sum()), "pnp_over_gyro": round(float(np.median(mag[b] / gmag[b])), 3)}
    return {"n": int(sel.sum()),
            "vector_corr": round(float(np.corrcoef(pred[sel].ravel(), wb[sel].ravel())[0, 1]), 4),
            "scale_gyro_over_pnp": round(float(np.sum(pred[sel] * wb[sel]) / np.sum(pred[sel] ** 2)), 4),
            "residual_rms_dps": round(float(np.sqrt(np.mean(np.sum((wb[sel] - pred[sel]) ** 2, axis=1)))), 3),
            "rate_rms_dps": round(float(np.sqrt(np.mean(mag[sel] ** 2))), 2),
            "by_gyro_rate": bins}


def rotation_1s(s: dict, tau: float, bias) -> tuple[dict, np.ndarray]:
    """1-s windows, 3-80 deg/s, mount fitted on even windows and scored on odd ones."""
    t, R, ok, bt = s["t"], s["R"], np.flatnonzero(s["ok"]), s["bt"]
    C = _cumulative(bt, np.radians(s["gyro"] - bias))
    wi, wc = [], []
    for pos, a in enumerate(ok[:-1]):
        later = ok[pos + 1:]
        b = int(later[np.argmin(np.abs(t[later] - (t[a] + 1.0)))])
        ta, tb = t[a] + tau, t[b] + tau
        if abs(t[b] - t[a] - 1.0) > 0.2 or ta < bt[0] or tb > bt[-1]:
            continue
        g = np.array([np.interp(tb, bt, C[:, i]) - np.interp(ta, bt, C[:, i]) for i in range(3)])
        rate = np.degrees(np.linalg.norm(g)) / (t[b] - t[a])
        if 3.0 <= rate <= 80.0:
            wi.append(g)
            wc.append(_relative(R, a, b).as_rotvec())
    wi, wc = np.degrees(np.asarray(wi)), np.degrees(np.asarray(wc))
    fit = np.arange(len(wi)) % 2 == 0
    R_f = _kabsch(wi[fit], wc[fit])
    pred, truth = wi[~fit] @ R_f.T, wc[~fit]
    corr = [float(np.corrcoef(pred[:, k], truth[:, k])[0, 1]) for k in range(3)]
    out = {"windows": int(len(wi)), "test_windows": int((~fit).sum()),
           "axis_corr": [round(c, 4) for c in corr],
           "magnitude_corr": round(float(np.corrcoef(np.linalg.norm(pred, axis=1),
                                                     np.linalg.norm(truth, axis=1))[0, 1]), 4),
           "scale_pnp_over_gyro": round(float(np.sum(pred * truth) / np.sum(pred * pred)), 4),
           "rms_deg_xyz": [round(float(v), 3) for v in np.sqrt(np.mean((pred - truth) ** 2, axis=0))]}
    return out, np.c_[np.linalg.norm(pred, axis=1), np.linalg.norm(truth, axis=1)]


def windows(s: dict, tau: float, R_IB, bias) -> np.ndarray:
    """Rows (window s, difference deg, gyro path deg, larger endpoint rate deg/s, PnP rotation deg)."""
    t, R, ok, bt = s["t"], s["R"], s["ok"], s["bt"]
    w = np.radians(s["gyro"] - bias)
    mid = 0.5 * (w[1:] + w[:-1])
    inc = Rotation.from_rotvec(mid * np.diff(bt)[:, None]).as_quat()
    q = np.zeros((len(bt), 4))
    q[0] = (0.0, 0.0, 0.0, 1.0)
    for k in range(1, len(bt)):
        x1, y1, z1, w1 = q[k - 1]
        x2, y2, z2, w2 = inc[k - 1]
        q[k] = (w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2)
    orient = Slerp(bt, Rotation.from_quat(q))
    path = np.r_[0.0, np.cumsum(np.degrees(np.linalg.norm(mid, axis=1)) * np.diff(bt))]
    rate = np.degrees(np.linalg.norm(w, axis=1))
    rows = []
    for L in WINDOWS_S:
        for a in np.flatnonzero(ok)[::2]:
            j = int(np.searchsorted(t, t[a] + L))
            cands = [k for k in (j - 1, j) if 0 <= k < len(t) and ok[k] and abs(t[k] - t[a] - L) < 0.06]
            if not cands:
                continue
            b = cands[0]
            ta, tb = t[a] + tau, t[b] + tau
            if ta < bt[0] or tb > bt[-1]:
                continue
            gyro_rot = orient([ta])[0].inv() * orient([tb])[0]
            Q = Rotation.from_matrix(R_IB.T @ gyro_rot.as_matrix() @ R_IB)
            P = _relative(R, a, b)
            rows.append((L, np.degrees((P.inv() * Q).magnitude()),
                         np.interp(tb, bt, path) - np.interp(ta, bt, path),
                         max(np.interp(ta, bt, rate), np.interp(tb, bt, rate)), np.degrees(P.magnitude())))
    return np.asarray(rows)


def decompose(rows: np.ndarray) -> dict:
    """difference^2 = e0^2 + (k path)^2 + (c endpoint rate)^2, robust on a log scale."""
    e, path, rate = rows[:, 1], rows[:, 2], rows[:, 3]
    f = lambda x: np.log(np.sqrt(x[0] ** 2 + (x[1] * path) ** 2 + (x[2] * rate) ** 2) + 1e-6) - np.log(e + 1e-6)  # noqa: E731
    sol = least_squares(f, [0.1, 0.01, 0.005], loss="soft_l1", f_scale=0.5)
    e0, k, c = (abs(float(v)) for v in sol.x)
    return {"endpoint_deg": round(e0, 3), "per_deg_of_path": round(k, 4), "per_dps_endpoint_rate": round(c, 5),
            "endpoint_label_mm_at_150mm": round(LEVER_MM * np.radians(e0), 2),
            "budget_relative_deg": round(np.sqrt(2.0) * SIGMA_THETA_DEG, 3)}


def _summary_windows(rows: np.ndarray) -> dict:
    out = {}
    for L in WINDOWS_S:
        r = rows[rows[:, 0] == L]
        rest = r[:, 3] < 3.0
        out[f"{L:g} s"] = {"n": int(len(r)), "pnp_rotation_median_deg": round(float(np.median(r[:, 4])), 2),
                           "difference_median_deg": round(float(np.median(r[:, 1])), 3),
                           "difference_p95_deg": round(float(np.quantile(r[:, 1], 0.95)), 3),
                           "rest_to_rest_n": int(rest.sum()),
                           "rest_to_rest_median_deg": round(float(np.median(r[rest, 1])), 3) if rest.any() else None}
    return out


def figure(per: dict, rot_pairs: np.ndarray, rows: np.ndarray, dec: dict, path: Path) -> None:
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt
    names = list(per)
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [0.9, 1.0, 1.25]})
    delays = np.array([per[n]["delay_ms"] for n in names])
    a.plot(np.arange(len(names)), delays, "o", color="#1971c2", ms=4)
    a.axhline(np.median(delays), color="black", lw=0.6)
    a.set_xticks(np.arange(len(names)), [n.split("_")[0][:5] + n[-4:] for n in names], rotation=60, fontsize=6)
    a.set_ylabel("camera stamp late by (ms)")
    a.set_ylim(np.median(delays) - 25, np.median(delays) + 25)
    a.set_title(f"(a) delay, median {np.median(delays):.0f} ms", fontsize=8, loc="left")
    a.grid(alpha=0.25, lw=0.5)
    b.plot(rot_pairs[:, 1], rot_pairs[:, 0], ".", ms=1.5, color="#495057", alpha=0.5)
    lim = float(np.quantile(rot_pairs, 0.995)) * 1.05
    b.plot([0, lim], [0, lim], color="#e03131", lw=0.8)
    b.set_xlim(0, lim)
    b.set_ylim(0, lim)
    b.set_xlabel("gyroscope, 1-s rotation (deg)")
    b.set_ylabel("PnP, 1-s rotation (deg)")
    corr = min(min(per[n]["rotation_1s"]["axis_corr"]) for n in names)
    scales = [per[n]["rotation_1s"]["scale_pnp_over_gyro"] for n in names]
    b.set_title(f"(b) axis r >= {corr:.3f}, scale {min(scales):.2f}-{max(scales):.2f}", fontsize=8, loc="left")
    b.grid(alpha=0.25, lw=0.5)
    edges = np.array([0, 1, 2, 5, 10, 20, 40, 80, 160])
    for L, col in zip(WINDOWS_S, ("#2f9e44", "#c2255c")):
        r = rows[rows[:, 0] == L]
        x, y, lo, hi = [], [], [], []
        for k in range(len(edges) - 1):
            m = (r[:, 2] >= edges[k]) & (r[:, 2] < edges[k + 1])
            if m.sum() >= 20:
                x.append(np.median(r[m, 2]))
                y.append(np.median(r[m, 1]))
                lo.append(np.quantile(r[m, 1], 0.25))
                hi.append(np.quantile(r[m, 1], 0.75))
        x, y = np.array(x), np.array(y)
        c.errorbar(x, y, yerr=[y - np.array(lo), np.array(hi) - y], fmt="o", ms=3, color=col, lw=0.8,
                   capsize=1.5, label=f"{L:g}-s windows")
    xs = np.linspace(0.3, edges[-1], 200)
    c.plot(xs, np.sqrt(dec["endpoint_deg"] ** 2 + (dec["per_deg_of_path"] * xs) ** 2), color="black", lw=0.8,
           label=f"fit: {dec['endpoint_deg']:.2f} deg + {100 * dec['per_deg_of_path']:.1f}% of path")
    c.axhline(dec["budget_relative_deg"], color="#e03131", ls="--", lw=0.8,
              label=r"budget $\sqrt{2}\,\sigma_\theta$")
    c.set_xscale("log")
    c.set_xlim(0.3, edges[-1])
    c.set_xlabel("gyroscope path in the window (deg)")
    c.set_ylabel("PnP minus gyroscope (deg)")
    c.set_title("(c) relative rotation, median and IQR", fontsize=8, loc="left")
    c.legend(frameon=False, fontsize=6.2, loc="upper left")
    c.grid(alpha=0.25, lw=0.5, which="both")
    fig.tight_layout(w_pad=1.0)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="*", help="default: every recording that qualifies")
    ap.add_argument("--output", type=Path, default=OUT_JSON)
    ap.add_argument("--figure", type=Path, default=OUT_FIG)
    ap.add_argument("--data-root", type=Path, help="folder with pose_gt/ and the session IMU logs")
    args = ap.parse_args()
    global DATA
    if args.data_root:
        DATA = args.data_root
    names = args.sessions or [p.stem for p in sorted((DATA / "pose_gt").glob("*.npz")) if qualifies(p.stem)]
    per, all_rows, all_rot = {}, [], []
    for name in names:
        s = load(name)
        t0, t1, wp = pnp_rates(s)
        corr, tau, R_IB = delay_and_mount(s, t0, t1, wp)
        bias, n_static = static_bias(s, tau)
        rot, pairs = rotation_1s(s, tau, bias)
        rows = windows(s, tau, R_IB, bias)
        per[name] = {"camera": camera_still(s), "frames": int(len(s["t"])), "pnp_ok": round(float(s["ok"].mean()), 3),
                     "delay_ms": round(-1000.0 * tau, 1), "delay_vector_corr": round(corr, 4),
                     "imu_mount_deg": round(float(np.degrees(Rotation.from_matrix(R_IB).magnitude())), 2),
                     "gyro_static_bias_dps": np.round(bias, 3).tolist(), "static_samples": n_static,
                     "rate": rate_metrics(s, t0, t1, wp, tau, R_IB, bias), "rotation_1s": rot,
                     "windows": _summary_windows(rows)}
        all_rows.append(rows)
        all_rot.append(pairs)
        print(f"{name}: delay {per[name]['delay_ms']:.1f} ms, 1-s axis r {min(rot['axis_corr']):.3f}, "
              f"scale {rot['scale_pnp_over_gyro']:.3f}, rate residual {per[name]['rate']['residual_rms_dps']:.2f} dps",
              flush=True)
    rows, rot_pairs = np.vstack(all_rows), np.vstack(all_rot)
    dec = decompose(rows)
    delays = np.array([v["delay_ms"] for v in per.values()])
    report = {
        "kind": "board_imu_orientation_in_motion",
        "recordings": names,
        "delay_ms": {"median": round(float(np.median(delays)), 1), "min": float(delays.min()), "max": float(delays.max()),
                     "within_2ms_of_median": int(np.sum(np.abs(delays - np.median(delays)) <= 2.0))},
        "rotation_1s": {"axis_corr_min": min(min(v["rotation_1s"]["axis_corr"]) for v in per.values()),
                        "magnitude_corr_min": min(v["rotation_1s"]["magnitude_corr"] for v in per.values()),
                        "scale_range": [min(v["rotation_1s"]["scale_pnp_over_gyro"] for v in per.values()),
                                        max(v["rotation_1s"]["scale_pnp_over_gyro"] for v in per.values())],
                        "test_windows": int(sum(v["rotation_1s"]["test_windows"] for v in per.values()))},
        "windows": _summary_windows(rows),
        "decomposition": dec,
        "per_recording": per,
        "limitations": [
            "The camera is the monocular camera of the caliper standard (13 fps), not the endoscope.",
            "The board is moved by hand; the IMU on it is a separate Bluetooth unit (gyro resolution 0.061 deg/s, "
            "stamps taken on arrival), so its own scale, misalignment and timing errors enter the comparison.",
            "Only orientation is tested; the translation of the board in motion has no independent reference here.",
        ],
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(args.output.with_suffix(".npz"), names=np.asarray(names), delay_ms=delays,
                        rot_pnp_deg=rot_pairs[:, 0], rot_gyro_deg=rot_pairs[:, 1], window_s=rows[:, 0],
                        window_diff_deg=rows[:, 1], window_path_deg=rows[:, 2], window_end_rate_dps=rows[:, 3])
    figure(per, rot_pairs, rows, dec, args.figure)
    print(json.dumps({k: report[k] for k in ("delay_ms", "rotation_1s", "windows", "decomposition")},
                     ensure_ascii=False, indent=2))
    print("wrote", args.output, "and", args.figure)


if __name__ == "__main__":
    main()
