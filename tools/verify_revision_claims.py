#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute the evidence behind the revision claims on stops and on the reference.

Writes datasets/revision_claim_checks.json with
  pause_windows      real windows with and without a reference-confirmed pause,
                     raw bands and a magnitude-matched contrast with a block
                     bootstrap over (recording, 3-s block)
  switched_windows   windows switched to stop-anchored integration on the
                     held-out and validation recordings (where they lie, how far
                     they move, error before and after switching)
  stereo_check       two-view triangulation against monocular PnP on the
                     intrinsic-calibration recording: board centroid and the
                     camera-centre displacement used as the label
  external_4k_check  4K global trajectory against monocular PnP, full
                     distribution per session
  gyro_quantization  fraction of exact-zero gyroscope samples in the static
                     bench logs and in the trajectory recordings

    python tools/verify_revision_claims.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.eval_stereo_and_rolling_shutter as esr
import tools.stop_anchor_check as sac
import tools.train_physnet as phys
import tools.train_physnet_v3 as v3
import tools.zupt_bound as zb

DATA = ROOT / "datasets"
OUT = DATA / "revision_claim_checks.json"
TEST = "traj_20260923_023422"
GYRO_LSB_DPS = 2000.0 / 32768.0


def _pred(ses: str, folds: dict) -> np.ndarray:
    if ses in folds:
        p0 = np.load(DATA / "physnet_cv" / f"pred_cvg_{ses}_s0.npz")["val_pred"]
        p1 = np.load(DATA / "physnet_cv" / f"pred_cvg_{ses}_s1.npz")["val_pred"]
        return 0.5 * (p0 + p1)
    return np.load(DATA / "session_cv_gate_benchmark.npz")["physnet"]


def pause_windows(n_boot: int = 2000) -> dict:
    """Gated PhysNet errors (out-of-fold CV and held-out test) by pause status."""
    folds = json.loads((DATA / "session_cv_gate_benchmark.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    for ses in list(folds) + [TEST]:
        t, _R, p, usable = phys._load_gt(ses)
        vg, vok = phys.gt_velocity(t, p, usable)
        speed = np.linalg.norm(vg, axis=1) * 1000.0
        segs = zb.still_segments(t, vok, speed, 3.0, 0.4)
        data = phys.build_group([ses], 2.0, 3.0, 1700)
        pred, y = _pred(ses, folds), data["y"]
        dt_med = float(np.median(np.diff(t[usable])))
        for k, (ta, tb) in enumerate(data["t_pair"]):
            near = usable[(t[usable] >= ta - 1.5) & (t[usable] <= tb + 1.5) & vok[usable]]
            accumulated = int((speed[near] < 5.0).sum()) * dt_med
            inside = any(t[a] >= ta and t[b] <= tb for a, b in segs)
            rows.append((ses, ta, float(np.linalg.norm(y[k]) * 1000),
                         float(np.linalg.norm(pred[k] - y[k]) * 1000), accumulated, inside))
    ses_ids = np.array([r[0] for r in rows])
    ta = np.array([r[1] for r in rows])
    mag = np.array([r[2] for r in rows])
    err = np.array([r[3] for r in rows])
    acc_still = np.array([r[4] for r in rows])
    inside = np.array([r[5] for r in rows])
    cont = acc_still < 0.1

    def band(sel_pause, lo, hi):
        s = (mag >= lo) & (mag < hi)
        c, q = s & cont, s & sel_pause
        return {"band_mm": [lo, hi],
                "continuous": {"n": int(c.sum()), "mean_disp_mm": round(float(mag[c].mean()), 1),
                               "err_mm": round(float(err[c].mean()), 1)},
                "pause": {"n": int(q.sum()), "mean_disp_mm": round(float(mag[q].mean()), 1),
                          "err_mm": round(float(err[q].mean()), 1)},
                "raw_reduction_pct": round(float(100 * (1 - err[q].mean() / err[c].mean())), 1),
                "err_over_disp": [round(float(err[c].mean() / mag[c].mean()), 3),
                                  round(float(err[q].mean() / mag[q].mean()), 3)]}

    block = np.array([f"{s}:{int(np.floor((a - ta[ses_ids == s].min()) / 3.0))}"
                      for s, a in zip(ses_ids, ta)])
    edges = np.arange(10.0, 82.5, 2.5)

    def matched(sel_pause, idx=None):
        """Mean error difference (pause - continuous) at matched displacement,
        weighting 2.5-mm bins by the pause-window counts."""
        idx = np.arange(len(mag)) if idx is None else idx
        m, e, p, c = mag[idx], err[idx], sel_pause[idx], cont[idx]
        diff, ref, w = 0.0, 0.0, 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (m >= lo) & (m < hi)
            if (s & p).sum() == 0 or (s & c).sum() == 0:
                continue
            k = float((s & p).sum())
            diff += k * (e[s & p].mean() - e[s & c].mean())
            ref += k * e[s & c].mean()
            w += k
        return diff / w, ref / w, int(w)

    def matched_with_ci(sel_pause):
        d, ref, n = matched(sel_pause)
        rng = np.random.default_rng(0)
        ids = np.unique(block)
        members = {b: np.flatnonzero(block == b) for b in ids}
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(ids, len(ids), replace=True)
            boots.append(matched(sel_pause, np.concatenate([members[b] for b in pick]))[0])
        lo, hi = np.quantile(boots, (0.025, 0.975))
        return {"n_pause_matched": n, "diff_mm": round(float(d), 2),
                "matched_continuous_err_mm": round(float(ref), 2),
                "diff_pct": round(float(100 * d / ref), 1),
                "ci95_mm": [round(float(lo), 2), round(float(hi), 2)]}

    accumulated_pause = acc_still >= 0.4
    return {
        "windows": int(len(mag)),
        "definition_in_submission": "accumulated reference speed < 5 mm/s for >= 0.4 s anywhere in "
                                    "[ta-1.5 s, tb+1.5 s]; continuous: < 0.1 s",
        "bands_as_submitted": [band(accumulated_pause, lo, hi)
                               for lo, hi in ((10, 30), (30, 60), (60, 150), (15, 80))],
        "matched_as_submitted_definition": matched_with_ci(accumulated_pause),
        "definition_contiguous_inside": "contiguous reference pause < 3 mm/s for >= 0.4 s inside [ta, tb]",
        "bands_contiguous_inside": [band(inside, lo, hi) for lo, hi in ((10, 30), (30, 60), (15, 80))],
        "matched_contiguous_inside": matched_with_ci(inside),
    }


def switched_windows() -> dict:
    """Windows that the IMU-only stop detector switches to stop-anchored integration."""
    folds = json.loads((DATA / "session_cv_gate_benchmark.json").read_text(encoding="utf-8"))["folds"]
    out = {}
    for ses in (TEST, "traj_20260924_144019"):
        pred = _pred(ses, folds)
        res = {}
        for mode in ("none", "session"):
            v3.BIAS_MODE = mode
            data = v3.build_group([ses], 2.0, 3.0, 1700)
            est, span = sac.stop_anchor(data, 400)
            use = span <= 2.0
            y = data["y"]
            t0 = float(data["t_pair"][:, 0].min())
            res[f"gyro_bias_{mode}"] = {
                "n_switched": int(use.sum()), "n_windows": int(len(y)),
                "anchor_time_range_s": [round(float(data["t_pair"][use, 0].min() - t0), 1),
                                        round(float(data["t_pair"][use, 0].max() - t0), 1)],
                "mean_disp_mm": round(float(np.linalg.norm(y[use], axis=1).mean() * 1000), 2),
                "learned_err_mm": round(float(np.linalg.norm(pred[use] - y[use], axis=1).mean() * 1000), 2),
                "stop_anchored_err_mm": round(float(np.linalg.norm(est[use] - y[use], axis=1).mean() * 1000), 2),
                "session_mean_disp_mm": round(float(np.linalg.norm(y, axis=1).mean() * 1000), 1),
            }
        out[ses] = res
    return out


def stereo_check() -> dict:
    meas = json.loads((DATA / "stereo_extrinsic_measured.json").read_text(encoding="utf-8"))
    ses = DATA / meas["session"]
    cal0 = json.loads((ses / "camera_calibration_cam0.json").read_text())
    cal1 = json.loads((ses / "camera_calibration_cam1.json").read_text())
    K0, d0 = np.asarray(cal0["camera_matrix"], float), np.asarray(cal0["distortion_coefficients"], float)
    K1, d1 = np.asarray(cal1["camera_matrix"], float), np.asarray(cal1["distortion_coefficients"], float)
    R10 = np.asarray(meas["R_C1_C0"], float)
    t10 = np.asarray(meas["t_C1_C0_m"], float) * 1000.0
    w = abs(float(meas["selection"]["right_offset_frames"]))
    z0, z1 = np.load(ses / "cam0_corners_cache.npz"), np.load(ses / "cam1_corners_cache.npz")
    c0, f0, c1, f1 = z0["corners"], z0["found"].astype(bool), z1["corners"], z1["found"].astype(bool)
    times = np.loadtxt(ses / "cam0" / "times.txt")
    obj = esr.object_points_mm()
    cent_p, cent_s, cam_p, cam_s, rot_p, rot_s, ts = [], [], [], [], [], [], []
    for i in range(1, min(len(f0), len(f1)) - 1):
        if not (f0[i] and f1[i] and f1[i - 1]):
            continue
        p0 = c0[i].astype(float)
        p1 = ((1.0 - w) * c1[i] + w * c1[i - 1]).astype(float)
        X, rms, _epi = esr.triangulate_board_mm(p0, p1, K0, d0, K1, d1, R10, t10)
        if rms > 1.5:
            continue
        ok, rv, tv = cv2.solvePnP(obj, p0, K0, d0)
        if not ok:
            continue
        Rm, _ = cv2.Rodrigues(rv)
        tm = tv.reshape(3)
        mu_x, mu_o = X.mean(0), obj.mean(0)
        U, _s, Vt = np.linalg.svd((obj - mu_o).T @ (X - mu_x))
        Rs = Vt.T @ U.T
        if np.linalg.det(Rs) < 0:
            Vt[-1] *= -1
            Rs = Vt.T @ U.T
        tsv = mu_x - Rs @ mu_o
        cent_p.append(((Rm @ obj.T).T + tm).mean(0))
        cent_s.append(mu_x)
        cam_p.append(-Rm.T @ tm)
        cam_s.append(-Rs.T @ tsv)
        rot_p.append(Rm.T)
        rot_s.append(Rs.T)
        ts.append(times[i])
    cent_p, cent_s, cam_p, cam_s, ts = map(np.asarray, (cent_p, cent_s, cam_p, cam_s, ts))
    rot_p, rot_s = np.asarray(rot_p), np.asarray(rot_s)
    dc, dl, mc, ml = [], [], [], []
    for i in range(len(ts)):
        prev = np.flatnonzero((ts < ts[i]) & (ts >= ts[i] - 4.5) & (ts <= ts[i] - 1.8))
        if len(prev) == 0:
            continue
        j = prev[np.argmin(np.abs(ts[prev] - (ts[i] - 3.0)))]
        a, b = cent_p[i] - cent_p[j], cent_s[i] - cent_s[j]
        dc.append(np.linalg.norm(a - b))
        mc.append(np.linalg.norm(b))
        a2 = rot_p[j].T @ (cam_p[i] - cam_p[j])
        b2 = rot_s[j].T @ (cam_s[i] - cam_s[j])
        dl.append(np.linalg.norm(a2 - b2))
        ml.append(np.linalg.norm(b2))
    dc, dl, mc, ml = map(np.asarray, (dc, dl, mc, ml))

    def summary(diff, mag):
        return {"n": int(len(diff)), "mean_disp_mm": round(float(mag.mean()), 2),
                "median_mm": round(float(np.median(diff)), 2), "mean_mm": round(float(diff.mean()), 2),
                "p90_mm": round(float(np.percentile(diff, 90)), 2)}

    return {"session": meas["session"],
            "stereo_extrinsic_fitted_on_same_session": True,
            "stereo_fit_rms_px": round(float(meas["fit_rms_px"]), 3),
            "board_centroid_3s": summary(dc, mc),
            "camera_centre_label_3s": summary(dl, ml)}


def external_4k_check() -> dict:
    out = {}
    for s in ("gtraj_20261002_002134", "gtraj_20261002_003129", "gtraj_20261002_024309"):
        tf, p_bc, R_bc = [], [], []
        for f in sorted((DATA / s / "fused_gt").glob("segment_*_fused.csv")):
            for r in csv.DictReader(f.open(encoding="utf-8")):
                tf.append(float(r["stamp"]))
                R_gc = Rotation.from_quat([float(r[f"C0_q{a}"]) for a in "xyzw"]).as_matrix()
                p_gc = np.array([float(r[f"C0_t{a}"]) for a in "xyz"])
                R_gb = Rotation.from_quat([float(r[f"B_q{a}"]) for a in "xyzw"]).as_matrix()
                p_gb = np.array([float(r[f"B_t{a}"]) for a in "xyz"])
                R_bc.append(R_gb.T @ R_gc)
                p_bc.append(R_gb.T @ (p_gc - p_gb))
        tf, p_bc, R_bc = np.asarray(tf), np.asarray(p_bc), np.asarray(R_bc)
        data = phys.build_group([s], 2.0, 3.0, 1700)
        A, B = [], []
        for k, (ta, tb) in enumerate(data["t_pair"]):
            ia, ib = int(np.argmin(np.abs(tf - ta))), int(np.argmin(np.abs(tf - tb)))
            if abs(tf[ia] - ta) > 0.05 or abs(tf[ib] - tb) > 0.05:
                continue
            A.append(data["y"][k] * 1000)
            B.append(R_bc[ia].T @ (p_bc[ib] - p_bc[ia]) * 1000)
        A, B = np.asarray(A), np.asarray(B)
        e = np.linalg.norm(A - B, axis=1)
        out[s] = {"n": int(len(e)),
                  "mean_disp_pnp_mm": round(float(np.linalg.norm(A, axis=1).mean()), 2),
                  "mean_disp_4k_mm": round(float(np.linalg.norm(B, axis=1).mean()), 2),
                  "median_diff_mm": round(float(np.median(e)), 2),
                  "mean_diff_mm": round(float(e.mean()), 2),
                  "p90_diff_mm": round(float(np.percentile(e, 90)), 2),
                  "r2_4k_vs_pnp": round(float(1 - ((A - B) ** 2).sum() / ((A - A.mean(0)) ** 2).sum()), 3)}
    return out


def gyro_quantization() -> dict:
    out = {"lsb_dps": round(GYRO_LSB_DPS, 6)}
    names = sorted(p.name for p in DATA.glob("align_imu_static_*")) + [TEST, "traj_20260924_144019"]
    for name in names:
        f = DATA / name / "imu_stream.csv"
        if not f.is_file():
            continue
        a = np.genfromtxt(f, delimiter=",", skip_header=1, usecols=(0, 4, 5, 6))
        t, g = a[:, 0] - a[0, 0], a[:, 1:4]
        zero = (np.abs(g) < 1e-9).all(1)
        out[name] = {"seconds": round(float(t[-1]), 1),
                     "exact_zero_fraction": round(float(zero.mean()), 3),
                     "exact_zero_fraction_first_5s": round(float(zero[t < 5.0].mean()), 3)}
    return out


def main() -> None:
    report = {"pause_windows": pause_windows(), "switched_windows": switched_windows(),
              "stereo_check": stereo_check(), "external_4k_check": external_4k_check(),
              "gyro_quantization": gyro_quantization()}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
