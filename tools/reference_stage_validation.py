#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent metric validation of the chessboard-PnP reference.

The external standard is a translation stage, micrometer or gauge block. The
analysis may fit only a rigid transform between stage and board axes; it never
fits scale. Therefore a scale error in PnP remains visible in the held-out
residuals.

Workflow
--------
1. ``init`` writes datasets/reference_stage_plan.csv.
2. Capture one 8--12 s static ``stage_*`` recording per row and enter its
   folder name in the ``session`` column. Read stage coordinates independently.
3. Build pose_gt for those sessions with the normal pipeline.
4. ``analyze`` performs leave-one-trial-out rigid alignment and writes a JSON
   report. No result is valid until every required row is present.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / "datasets"
DEFAULT_PLAN = DATA / "reference_stage_plan.csv"
DEFAULT_REPORT = DATA / "reference_stage_validation.json"


def init_plan(path: Path, repeats: int = 3) -> None:
    positions = [(0.0, 0.0, 0.0, "origin")]
    for axis in range(3):
        label = "XYZ"[axis]
        for value in (-20.0, -10.0, -5.0, 5.0, 10.0, 20.0):
            p = [0.0, 0.0, 0.0]
            p[axis] = value
            positions.append((*p, f"{label}{value:+g}"))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "position_id", "trial", "true_x_mm", "true_y_mm", "true_z_mm",
            "session", "operator", "stage_id", "temperature_c", "notes",
        ])
        writer.writeheader()
        for trial in range(1, repeats + 1):
            # Reverse every second traversal to expose backlash/hysteresis.
            seq = positions if trial % 2 else list(reversed(positions))
            for x, y, z, label in seq:
                writer.writerow({
                    "position_id": label, "trial": trial,
                    "true_x_mm": x, "true_y_mm": y, "true_z_mm": z,
                    "session": "", "operator": "", "stage_id": "",
                    "temperature_c": "", "notes": "",
                })
    print(f"wrote {path} ({len(positions) * repeats} recordings)")


def _median_pose(session: str) -> tuple[np.ndarray, dict]:
    raw = DATA / "pose_gt_raw" / f"{session}.npz"
    if not raw.is_file():
        raise FileNotFoundError(f"missing {raw}; run pose_gt export first")
    blob = np.load(raw)
    use = blob["usable"] == 1
    if int(use.sum()) < 30:
        raise ValueError(f"{session}: only {int(use.sum())} usable frames")
    p = blob["p"][use].astype(np.float64) * 1000.0
    med = np.median(p, axis=0)
    radial = np.linalg.norm(p - med, axis=1)
    return med, {
        "usable_frames": int(use.sum()),
        "static_repeatability_p95_mm": float(np.quantile(radial, 0.95)),
    }


def _rigid_fit(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """dst ~= R @ src + t; no scale parameter."""
    cs, cd = src.mean(0), dst.mean(0)
    u, _s, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    R = vt.T @ u.T
    if np.linalg.det(R) < 0:
        vt[-1] *= -1
        R = vt.T @ u.T
    return R, cd - R @ cs


def analyze(plan: Path, output: Path) -> dict:
    rows = list(csv.DictReader(plan.open(newline="", encoding="utf-8-sig")))
    missing = [f"{r['position_id']}/trial{r['trial']}" for r in rows
               if not r["session"].strip()]
    if missing:
        raise RuntimeError(
            f"plan incomplete: {len(missing)} session names missing; first: {missing[:5]}"
        )
    true, measured, trials, details = [], [], [], {}
    for row in rows:
        session = row["session"].strip()
        med, quality = _median_pose(session)
        true.append([float(row[f"true_{a}_mm"]) for a in "xyz"])
        measured.append(med)
        trials.append(int(row["trial"]))
        details[session] = quality | {
            "position_id": row["position_id"], "trial": int(row["trial"]),
            "true_mm": true[-1], "measured_median_mm": med.tolist(),
        }
    true, measured, trials = map(np.asarray, (true, measured, trials))

    folds, all_err, all_axis = [], [], []
    for holdout in sorted(np.unique(trials)):
        fit, test = trials != holdout, trials == holdout
        R, t = _rigid_fit(measured[fit], true[fit])
        pred = (R @ measured[test].T).T + t
        err_vec = pred - true[test]
        err = np.linalg.norm(err_vec, axis=1)
        all_err.extend(err.tolist())
        all_axis.extend(err_vec.tolist())
        folds.append({
            "held_out_trial": int(holdout),
            "n": int(test.sum()),
            "rmse_mm": float(np.sqrt(np.mean(err ** 2))),
            "mean_mm": float(err.mean()),
            "max_mm": float(err.max()),
        })
    all_err, all_axis = np.asarray(all_err), np.asarray(all_axis)

    # Scale along each stage axis after a rigid fit to all positions. Scale is
    # evaluated, never fitted into the coordinate transform.
    R, t = _rigid_fit(measured, true)
    aligned = (R @ measured.T).T + t
    scales = {}
    cross_axis = {}
    for k, axis in enumerate("xyz"):
        varying = np.any(np.abs(true[:, k:k + 1]) > 0, axis=1)
        varying &= np.max(np.abs(np.delete(true, k, axis=1)), axis=1) < 1e-9
        x, y = true[varying, k], aligned[varying, k]
        slope = float(np.dot(x - x.mean(), y - y.mean()) /
                      max(np.dot(x - x.mean(), x - x.mean()), 1e-12))
        scales[axis] = {
            "scale": slope, "scale_error_percent": 100.0 * (slope - 1.0)
        }
        cross_axis[axis] = float(np.sqrt(np.mean(
            np.delete(aligned[varying] - true[varying], k, axis=1) ** 2
        )))

    # Repeatability across trials at every commanded position.
    rep = []
    for position in sorted(set(r["position_id"] for r in rows)):
        idx = np.array([r["position_id"] == position for r in rows])
        pts = measured[idx]
        centre = pts.mean(0)
        rep.extend(np.linalg.norm(pts - centre, axis=1).tolist())
    rep = np.asarray(rep)

    thresholds = {
        "heldout_rmse_mm_max": 1.0,
        "heldout_max_mm_max": 2.0,
        "axis_scale_error_percent_max_abs": 1.0,
        "cross_axis_rmse_mm_max": 0.5,
        "repeatability_p95_mm_max": 0.5,
    }
    summary = {
        "n_recordings": len(rows),
        "trials": sorted(int(v) for v in np.unique(trials)),
        "heldout_rmse_mm": float(np.sqrt(np.mean(all_err ** 2))),
        "heldout_mean_mm": float(all_err.mean()),
        "heldout_p95_mm": float(np.quantile(all_err, 0.95)),
        "heldout_max_mm": float(all_err.max()),
        "axis_bias_mm": np.mean(all_axis, axis=0).tolist(),
        "axis_rmse_mm": np.sqrt(np.mean(all_axis ** 2, axis=0)).tolist(),
        "axis_scale": scales,
        "cross_axis_rmse_mm": cross_axis,
        "repeatability_p95_mm": float(np.quantile(rep, 0.95)),
    }
    passed = (
        summary["heldout_rmse_mm"] <= thresholds["heldout_rmse_mm_max"] and
        summary["heldout_max_mm"] <= thresholds["heldout_max_mm_max"] and
        max(abs(v["scale_error_percent"]) for v in scales.values()) <=
        thresholds["axis_scale_error_percent_max_abs"] and
        max(cross_axis.values()) <= thresholds["cross_axis_rmse_mm_max"] and
        summary["repeatability_p95_mm"] <= thresholds["repeatability_p95_mm_max"]
    )
    report = {
        "kind": "independent_stage_validation_no_scale_fit",
        "plan": str(plan), "passed": bool(passed),
        "thresholds": thresholds, "summary": summary,
        "folds": folds, "recordings": details,
        "limitations": [
            "Stage or micrometer calibration certificate must be archived separately.",
            "This validates translation scale and linearity, not board pitch/flatness.",
            "No result is valid if the true coordinates were inferred from camera data.",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": passed,
                      "summary": summary}, ensure_ascii=False, indent=2))
    return report


CALIPER_PLAN = DATA / "reference_caliper_plan.csv"
CALIPER_POSITIONS_MM = (0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0)
# Centre of the 11 x 8 inner corners of the GP050 board (3-mm squares).
BOARD_CENTRE_MM = np.array([15.0, 10.5, 0.0])


def frame_positions(blob, quantity: str) -> np.ndarray:
    """Per-frame position in mm. 'camera': camera centre in board coordinates,
    the quantity the labels are made of. 'board': board centre in camera
    coordinates, the translation the caliper sets. With the camera fixed, a
    tilt of the sliding jaw by theta moves the camera centre in board
    coordinates by distance * theta, which the caliper does not read."""
    p = blob["p"].astype(np.float64) * 1000.0
    if quantity == "camera":
        return p
    R = blob["R"].astype(np.float64)
    return np.einsum("nji,nj->ni", R, BOARD_CENTRE_MM - p)


def _mean_rotation(R: np.ndarray) -> np.ndarray:
    U, _s, Vt = np.linalg.svd(R.mean(axis=0))
    return U @ Vt


def _imu_acc(session: str, fname: str):
    """(time, accelerometer) of imu_bt.csv (IMU on the sliding jaw) or
    imu_stream.csv (IMU on the camera), or None."""
    from ego_capture.sync import load_imu
    path = DATA / session / fname
    if not path.is_file():
        return None
    t, y = load_imu(path)
    return (t, y[:, 0:3]) if len(t) else None


def _hold_gravity(imu, t0: float, t1: float):
    if imu is None:
        return None
    a = imu[1][(imu[0] >= t0) & (imu[0] <= t1)]
    if len(a) < 20:
        return None
    g = a.mean(axis=0)
    return g / np.linalg.norm(g)


def init_caliper_plan(path: Path, repeats: int = 3) -> None:
    """One continuous recording per (axis, trial); the board rides on the
    caliper jaw and pauses at every listed reading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "axis", "trial", "readings_mm", "session", "operator",
            "caliper_id", "caliper_resolution_mm", "temperature_c", "notes",
        ])
        writer.writeheader()
        for axis in "XYZ":
            for trial in range(1, repeats + 1):
                seq = (CALIPER_POSITIONS_MM if trial % 2
                       else tuple(reversed(CALIPER_POSITIONS_MM)))
                writer.writerow({
                    "axis": axis, "trial": trial,
                    "readings_mm": " ".join(f"{v:g}" for v in seq),
                    "session": "", "operator": "", "caliper_id": "",
                    "caliper_resolution_mm": "", "temperature_c": "", "notes": "",
                })
    print(f"wrote {path} ({3 * repeats} recordings, "
          f"{len(CALIPER_POSITIONS_MM)} pauses each)")


def _static_segments(session: str, still_step_mm: float, min_s: float, n_expected: int):
    """Median camera-in-board position of the n_expected pauses, in time order.

    Positions are first median-filtered over +-0.7 s, which removes the flips
    of the PnP solution between nearby discrete poses (~1 mm) inside a pause
    while following a move; a frame is stationary when the filtered
    frame-to-frame step is below ``still_step_mm``. The longest n_expected runs
    are kept, so short false pauses during a move are discarded.
    """
    raw = DATA / "pose_gt_raw" / f"{session}.npz"
    if not raw.is_file():
        raise FileNotFoundError(f"missing {raw}; run pose_gt export first")
    blob = np.load(raw)
    use = blob["usable"] == 1
    t = blob["t"][use]
    p = blob["p"][use].astype(np.float64) * 1000.0
    pb = frame_positions(blob, "board")[use]
    R = blob["R"][use].astype(np.float64)
    smooth = np.array([np.median(p[np.abs(t - tk) <= 0.7], axis=0) for tk in t])
    step = np.r_[0.0, np.linalg.norm(np.diff(smooth, axis=0), axis=1)]
    still = step < still_step_mm
    edges = np.flatnonzero(np.diff(np.r_[0, still.astype(int), 0]))
    runs = [(a, b - 1) for a, b in zip(edges[::2], edges[1::2])]
    merged = runs[:1]
    for a, b in runs[1:]:
        a0, b0 = merged[-1]
        same = np.linalg.norm(np.median(smooth[a0:b0 + 1], 0) - np.median(smooth[a:b + 1], 0)) < 0.5
        if t[a] - t[b0] <= 0.8 and same:
            merged[-1] = (a0, b)
        else:
            merged.append((a, b))
    runs = [(a, b) for a, b in merged if t[b] - t[a] >= min_s]
    runs = sorted(sorted(runs, key=lambda r: t[r[1]] - t[r[0]], reverse=True)[:n_expected])
    segs = []
    for i, j in runs:
        q = p[i:j + 1]
        med = np.median(q, axis=0)
        segs.append({
            "t0": float(t[i]), "t1": float(t[j]), "frames": int(j - i + 1),
            "median_mm": med, "board_mm": np.median(pb[i:j + 1], axis=0),
            "R": _mean_rotation(R[i:j + 1]), "board_imu_g": None, "camera_imu_g": None,
            "p95_mm": float(np.quantile(np.linalg.norm(q - med, axis=1), 0.95)),
        })
    return segs


def _event_segments(session: str):
    """Hold windows and readings that the guided capture stored with the
    recording (caliper_events.json), or None for recordings without them."""
    path = DATA / session / "caliper_events.json"
    if not path.is_file():
        return None
    events = json.loads(path.read_text(encoding="utf-8"))
    blob = np.load(DATA / "pose_gt_raw" / f"{session}.npz")
    use = blob["usable"] == 1
    t = blob["t"]
    p = blob["p"].astype(np.float64) * 1000.0
    pb = frame_positions(blob, "board")
    R = blob["R"].astype(np.float64)
    imu, cam_imu = _imu_acc(session, "imu_bt.csv"), _imu_acc(session, "imu_stream.csv")
    segs = []
    for ev in events["pauses"]:
        sel = use & (t >= ev["t_hold_start"]) & (t <= ev["t_hold_end"])
        if sel.sum() < 5:
            continue
        q = p[sel]
        med = np.median(q, axis=0)
        segs.append({
            "t0": float(ev["t_hold_start"]), "t1": float(ev["t_hold_end"]),
            "frames": int(sel.sum()), "median_mm": med, "board_mm": np.median(pb[sel], axis=0),
            "R": _mean_rotation(R[sel]),
            "board_imu_g": _hold_gravity(imu, ev["t_hold_start"], ev["t_hold_end"]),
            "camera_imu_g": _hold_gravity(cam_imu, ev["t_hold_start"], ev["t_hold_end"]),
            "p95_mm": float(np.quantile(np.linalg.norm(q - med, axis=1), 0.95)),
            "reading_mm": float(ev["reading_mm"]), "target_mm": float(ev["target_mm"]),
        })
    return {"segs": segs, "zero_reading_mm": events.get("zero_reading_mm"),
            "caliper": events.get("caliper"), "rejected_holds": len(events.get("rejected_holds", []))}


def caliper_trial(session: str, planned: np.ndarray, still_mm: float, min_s: float) -> dict:
    """Pause segments of one traversal with their readings and targets.

    Recordings made with the guided capture carry the hold windows and the
    reading typed for each of them; the readings are used as they are (the
    zero error cancels because each trial's origin is fitted on its own first
    pause). Older recordings fall back to motion-based segmentation and the
    planned readings."""
    ev = _event_segments(session)
    if ev is not None:
        segs = ev["segs"]
        return {"segs": segs, "readings": np.array([s["reading_mm"] for s in segs]),
                "targets": np.array([s["target_mm"] for s in segs]), "source": "caliper_events.json",
                "zero_reading_mm": ev["zero_reading_mm"], "rejected_holds": ev["rejected_holds"]}
    segs = _static_segments(session, still_mm, min_s, len(planned))
    if len(segs) != len(planned):
        raise RuntimeError(
            f"{session}: found {len(segs)} stationary intervals, "
            f"plan lists {len(planned)} readings; re-record with >=3-s pauses "
            "and moves faster than 3 mm/s")
    return {"segs": segs, "readings": planned.copy(), "targets": planned.copy(),
            "source": "motion_segmentation"}


def _line_fit(s: np.ndarray, m: np.ndarray):
    """m ~= o + s * u with |u| = 1: direction and origin only, no scale."""
    sc = s - s.mean()
    g = (sc[:, None] * (m - m.mean(0))).sum(0) / max(float(sc @ sc), 1e-12)
    scale = float(np.linalg.norm(g))
    u = g / max(scale, 1e-12)
    o = m.mean(0) - s.mean() * u
    return u, o, scale


CALIPER_THRESHOLDS = {
    "heldout_rmse_mm_max": 1.0,
    "heldout_max_mm_max": 2.0,
    "axis_scale_error_percent_max_abs": 1.0,
    "cross_axis_rmse_mm_max": 0.5,
    "repeatability_p95_mm_max": 0.5,
}


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b) / max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-12)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _axis_metrics(per_axis: dict, key: str) -> dict:
    """Held-out scores of one position quantity ('camera' or 'board')."""
    axes, all_err, rep_all = {}, [], []
    for axis, trials in sorted(per_axis.items()):
        s_all = np.concatenate([tr["s"] for tr in trials])
        err_axis, perp_axis = [], []
        # The origin of every trial is refitted on its own first (smallest)
        # reading, which cancels the zero error of the caliper; the direction
        # comes from the other trials only.
        for k, tr in enumerate(trials):
            s, m = tr["s"], tr[key]
            train_s = np.concatenate([o["s"] for j, o in enumerate(trials) if j != k])
            train_m = np.concatenate([o[key] - o[key][np.argmin(o["s"])] for j, o in enumerate(trials) if j != k])
            u, _o, _scale = _line_fit(train_s, train_m)
            origin = m[np.argmin(s)] - s.min() * u
            keep = np.arange(len(s)) != np.argmin(s)
            e = (origin + s[keep, None] * u) - m[keep]
            err_axis.extend(np.linalg.norm(e, axis=1).tolist())
            along = e @ u
            perp_axis.extend(np.linalg.norm(e - along[:, None] * u, axis=1).tolist())
        rel = np.concatenate([tr[key] - tr[key][np.argmin(tr["s"])] for tr in trials])
        u, _o, scale = _line_fit(s_all, rel)
        # Repeatability: spread over trials of the residual from the unit-scale
        # line at the same planned position. Readings differ slightly between
        # trials (the jaw is set by hand), so positions are compared after
        # removing (reading - first reading) * u.
        for target in np.unique(np.concatenate([tr["target"] for tr in trials])):
            pts = np.array([tr[key][i] - tr[key][np.argmin(tr["s"])] - (tr["s"][i] - tr["s"].min()) * u
                            for tr in trials for i in np.flatnonzero(tr["target"] == target)])
            if len(pts) >= 2:
                rep_all.extend(np.linalg.norm(pts - pts.mean(0), axis=1).tolist())
        err_axis = np.asarray(err_axis)
        all_err.extend(err_axis.tolist())
        axes[axis] = {
            "trials": len(trials), "points": int(len(s_all)),
            "heldout_rmse_mm": float(np.sqrt(np.mean(err_axis ** 2))),
            "heldout_max_mm": float(err_axis.max()),
            "scale": scale, "scale_error_percent": 100.0 * (scale - 1.0),
            "cross_axis_rmse_mm": float(np.sqrt(np.mean(np.asarray(perp_axis) ** 2))),
            "direction": u.tolist(),
        }
    all_err, rep_all = np.asarray(all_err), np.asarray(rep_all)
    out = {
        "n_points": int(len(all_err)),
        "heldout_rmse_mm": float(np.sqrt(np.mean(all_err ** 2))),
        "heldout_max_mm": float(all_err.max()),
        "repeatability_p95_mm": float(np.quantile(rep_all, 0.95)) if len(rep_all) else None,
        "axes": axes,
    }
    th = CALIPER_THRESHOLDS
    out["passed"] = bool(
        len(axes) == 3 and
        out["heldout_rmse_mm"] <= th["heldout_rmse_mm_max"] and
        out["heldout_max_mm"] <= th["heldout_max_mm_max"] and
        max(abs(a["scale_error_percent"]) for a in axes.values()) <= th["axis_scale_error_percent_max_abs"] and
        max(a["cross_axis_rmse_mm"] for a in axes.values()) <= th["cross_axis_rmse_mm_max"] and
        out["repeatability_p95_mm"] is not None and out["repeatability_p95_mm"] <= th["repeatability_p95_mm_max"]
    )
    return out


CAMERA_STILL_DEG = 0.5


def _tilt_check(per_axis: dict) -> dict | None:
    """Tilt of the board between the holds of a trial, from PnP and from the
    accelerometer of the IMU on the sliding jaw. In the X mounting the board
    lies flat, so its normal there is the vertical in camera coordinates; it
    stays the vertical for every trial as long as the camera does not move,
    which the accelerometer of the camera IMU confirms."""
    x = per_axis.get("X")
    if not x:
        return None
    up = np.mean([tr["R"][int(np.argmin(tr["s"]))][2] for tr in x], axis=0)
    up /= np.linalg.norm(up)
    cams = [tr["c"][0] for trials in per_axis.values() for tr in trials if tr["c"][0] is not None]
    camera_moved = max((_angle_deg(c, cams[0]) for c in cams), default=0.0)
    if camera_moved > CAMERA_STILL_DEG:
        return {"skipped": f"camera orientation changed by {camera_moved:.2f} deg between trials"}
    per_trial, pairs = {}, []
    for axis, trials in sorted(per_axis.items()):
        for tr in trials:
            g = tr["g"]
            if any(v is None for v in g):
                return None
            i0 = int(np.argmin(tr["s"]))
            pnp = np.array([_angle_deg(Rk @ up, tr["R"][i0] @ up) for Rk in tr["R"]])
            imu = np.array([_angle_deg(gk, g[i0]) for gk in g])
            per_trial[f"{axis}{tr['trial']}"] = {
                "imu_tilt_max_deg": round(float(imu.max()), 3), "pnp_tilt_max_deg": round(float(pnp.max()), 3),
                "rms_diff_deg": round(float(np.sqrt(np.mean((pnp - imu) ** 2))), 3)}
            pairs.append(np.c_[pnp, imu])
    pairs = np.vstack(pairs)
    return {"assumption": "the board lies flat in the X mounting and the camera does not move",
            "camera_orientation_change_deg": round(camera_moved, 3),
            "imu_tilt_max_deg": round(float(pairs[:, 1].max()), 3),
            "pnp_tilt_max_deg": round(float(pairs[:, 0].max()), 3),
            "rms_diff_deg": round(float(np.sqrt(np.mean((pairs[:, 0] - pairs[:, 1]) ** 2))), 3),
            "corr": round(float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]), 3) if pairs[:, 1].std() > 0 else None,
            "per_trial": per_trial}


def analyze_caliper(plan: Path, output: Path, still_mm: float, min_s: float) -> dict:
    rows = list(csv.DictReader(plan.open(newline="", encoding="utf-8-sig")))
    missing = [f"{r['axis']}/trial{r['trial']}" for r in rows if not r["session"].strip()]
    if missing:
        raise RuntimeError(f"plan incomplete: {missing}")
    per_axis, details = {}, {}
    for r in rows:
        planned = np.array([float(v) for v in r["readings_mm"].split()])
        tr = caliper_trial(r["session"].strip(), planned, still_mm, min_s)
        if len(tr["segs"]) < 3:
            raise RuntimeError(f"{r['session']}: only {len(tr['segs'])} usable pauses")
        med = np.array([sg["median_mm"] for sg in tr["segs"]])
        per_axis.setdefault(r["axis"], []).append(
            {"trial": int(r["trial"]), "s": tr["readings"], "target": tr["targets"], "camera": med,
             "board": np.array([sg["board_mm"] for sg in tr["segs"]]),
             "R": [sg["R"] for sg in tr["segs"]], "g": [sg.get("board_imu_g") for sg in tr["segs"]],
             "c": [sg.get("camera_imu_g") for sg in tr["segs"]]})
        details[r["session"].strip()] = {
            "axis": r["axis"], "trial": int(r["trial"]), "pause_source": tr["source"],
            "readings_mm": tr["readings"].tolist(),
            "working_distance_mm": round(float(np.median(np.linalg.norm(med, axis=1))), 1),
            "static_p95_mm": [sg["p95_mm"] for sg in tr["segs"]],
            "frames": [sg["frames"] for sg in tr["segs"]],
            "rejected_holds": tr.get("rejected_holds"),
        }

    board, camera = _axis_metrics(per_axis, "board"), _axis_metrics(per_axis, "camera")
    distance = float(np.median([d["working_distance_mm"] for d in details.values()]))
    tilt = _tilt_check(per_axis)
    if tilt is not None:
        tilt["camera_centre_term_mm"] = round(distance * np.radians(tilt["imu_tilt_max_deg"]), 2)
    summary = {
        "n_recordings": len(rows), "quantity": "board centre in camera coordinates",
        "working_distance_mm": distance, **{k: v for k, v in board.items() if k != "passed"},
        "camera_centre": {k: v for k, v in camera.items() if k != "passed"},
        "tilt_check": tilt,
    }
    report = {
        "kind": "independent_caliper_validation_no_scale_fit",
        "plan": str(plan), "passed": board["passed"], "passed_camera_centre": camera["passed"],
        "thresholds": CALIPER_THRESHOLDS,
        "segmentation": {"source": sorted({d["pause_source"] for d in details.values()}),
                         "fallback_median_step_mm_max": still_mm, "fallback_min_duration_s": min_s},
        "summary": summary, "recordings": details,
        "limitations": [
            "The caliper is a hand-read vernier without a calibration certificate; its "
            "stated maximum permissible error and the operator's reading resolution "
            "bound the standard, which is therefore not SI-traceable.",
            "One-dimensional per axis; axes are separate mountings, so absolute "
            "orthogonality between axes is not tested.",
            "Readings were typed by the operator before each hold and never inferred from camera data.",
            "The caliper sets the translation of the sliding jaw, not its tilt. A tilt of the jaw "
            "moves the camera centre in board coordinates by the working distance times the tilt, "
            "so the primary comparison uses the board centre in camera coordinates and the "
            "camera-centre comparison is reported alongside; the IMU on the jaw measures the tilt.",
            "The jaw tilt also moves the board centre by its offset from the caliper's measuring "
            "line times the tilt (Abbe); this remains in the residuals.",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"],
                      "passed_camera_centre": report["passed_camera_centre"], "summary": summary},
                     ensure_ascii=False, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--mode", choices=("stage", "caliper"), default="stage")
    p_init.add_argument("--plan", type=Path)
    p_init.add_argument("--repeats", type=int, default=3)
    p_an = sub.add_parser("analyze")
    p_an.add_argument("--mode", choices=("stage", "caliper"), default="stage")
    p_an.add_argument("--plan", type=Path)
    p_an.add_argument("--output", type=Path)
    p_an.add_argument("--still-mm", type=float, default=0.2,
                      help="caliper mode: max median-filtered frame-to-frame step in a pause")
    p_an.add_argument("--min-s", type=float, default=1.5,
                      help="caliper mode: minimum pause duration")
    p_an.add_argument("--data-root", type=Path,
                      help="folder with pose_gt_raw/ and the session folders (rehearsals)")
    args = parser.parse_args()
    global DATA
    if getattr(args, "data_root", None):
        DATA = args.data_root
    default_plan = CALIPER_PLAN if args.mode == "caliper" else DEFAULT_PLAN
    plan = args.plan or DATA / default_plan.name
    if args.command == "init":
        (init_caliper_plan if args.mode == "caliper" else init_plan)(plan, args.repeats)
        return
    output = args.output or DATA / DEFAULT_REPORT.name
    if args.mode == "caliper":
        analyze_caliper(plan, output, args.still_mm, args.min_s)
    else:
        analyze(plan, output)


if __name__ == "__main__":
    main()
