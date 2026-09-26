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


def _static_segments(session: str, still_mm: float, min_s: float):
    """Median camera-in-board position of every stationary interval, in order."""
    raw = DATA / "pose_gt_raw" / f"{session}.npz"
    if not raw.is_file():
        raise FileNotFoundError(f"missing {raw}; run pose_gt export first")
    blob = np.load(raw)
    use = blob["usable"] == 1
    t = blob["t"][use]
    p = blob["p"][use].astype(np.float64) * 1000.0
    still = np.zeros(len(p), bool)
    for k in range(len(p)):
        w = np.abs(t - t[k]) <= 0.5
        if w.sum() >= 3:
            q = p[w]
            still[k] = np.linalg.norm(q - np.median(q, axis=0), axis=1).max() < still_mm
    segs, i = [], 0
    while i < len(p):
        if not still[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(p) and still[j + 1]:
            j += 1
        if t[j] - t[i] >= min_s:
            q = p[i:j + 1]
            med = np.median(q, axis=0)
            segs.append({
                "t0": float(t[i]), "t1": float(t[j]), "frames": int(j - i + 1),
                "median_mm": med,
                "p95_mm": float(np.quantile(np.linalg.norm(q - med, axis=1), 0.95)),
            })
        i = j + 1
    return segs


def _line_fit(s: np.ndarray, m: np.ndarray):
    """m ~= o + s * u with |u| = 1: direction and origin only, no scale."""
    sc = s - s.mean()
    g = (sc[:, None] * (m - m.mean(0))).sum(0) / max(float(sc @ sc), 1e-12)
    scale = float(np.linalg.norm(g))
    u = g / max(scale, 1e-12)
    o = m.mean(0) - s.mean() * u
    return u, o, scale


def analyze_caliper(plan: Path, output: Path, still_mm: float, min_s: float) -> dict:
    rows = list(csv.DictReader(plan.open(newline="", encoding="utf-8-sig")))
    missing = [f"{r['axis']}/trial{r['trial']}" for r in rows if not r["session"].strip()]
    if missing:
        raise RuntimeError(f"plan incomplete: {missing}")
    per_axis, details = {}, {}
    for r in rows:
        readings = np.array([float(v) for v in r["readings_mm"].split()])
        segs = _static_segments(r["session"].strip(), still_mm, min_s)
        if len(segs) != len(readings):
            raise RuntimeError(
                f"{r['session']}: found {len(segs)} stationary intervals, "
                f"plan lists {len(readings)} readings; re-record or adjust --still-mm")
        med = np.array([sg["median_mm"] for sg in segs])
        per_axis.setdefault(r["axis"], []).append((int(r["trial"]), readings, med))
        details[r["session"].strip()] = {
            "axis": r["axis"], "trial": int(r["trial"]),
            "readings_mm": readings.tolist(),
            "static_p95_mm": [sg["p95_mm"] for sg in segs],
            "frames": [sg["frames"] for sg in segs],
        }

    axes, all_err, rep_all = {}, [], []
    for axis, trials in sorted(per_axis.items()):
        s_all = np.concatenate([tr[1] for tr in trials])
        m_all = np.concatenate([tr[2] for tr in trials])
        err_axis, perp_axis = [], []
        # Each trial is a separate mounting, so its origin is refitted from its
        # own zero reading; the direction comes from the other trials only.
        for k, (_trial, s, m) in enumerate(trials):
            train_s = np.concatenate([tr[1] for j, tr in enumerate(trials) if j != k])
            train_m = np.concatenate([tr[2] - tr[2][np.argmin(tr[1])] for j, tr in enumerate(trials) if j != k])
            u, _o, _scale = _line_fit(train_s, train_m)
            origin = m[np.argmin(s)] - s.min() * u
            keep = s != s.min()
            e = (origin + s[keep, None] * u) - m[keep]
            err_axis.extend(np.linalg.norm(e, axis=1).tolist())
            along = e @ u
            perp_axis.extend(np.linalg.norm(e - along[:, None] * u, axis=1).tolist())
        rel = np.concatenate([tr[2] - tr[2][np.argmin(tr[1])] for tr in trials])
        u, _o, scale = _line_fit(s_all, rel)
        for reading in np.unique(s_all):
            pts = np.concatenate([tr[2][tr[1] == reading] - tr[2][np.argmin(tr[1])]
                                  for tr in trials])
            rep_all.extend(np.linalg.norm(pts - pts.mean(0), axis=1).tolist())
        err_axis = np.asarray(err_axis)
        all_err.extend(err_axis.tolist())
        axes[axis] = {
            "trials": len(trials), "points": int(len(s_all)),
            "heldout_rmse_mm": float(np.sqrt(np.mean(err_axis ** 2))),
            "heldout_max_mm": float(err_axis.max()),
            "scale": scale, "scale_error_percent": 100.0 * (scale - 1.0),
            "cross_axis_rmse_mm": float(np.sqrt(np.mean(np.asarray(perp_axis) ** 2))),
            "direction_in_board_frame": u.tolist(),
        }
    all_err, rep_all = np.asarray(all_err), np.asarray(rep_all)
    thresholds = {
        "heldout_rmse_mm_max": 1.0,
        "heldout_max_mm_max": 2.0,
        "axis_scale_error_percent_max_abs": 1.0,
        "cross_axis_rmse_mm_max": 0.5,
        "repeatability_p95_mm_max": 0.5,
    }
    summary = {
        "n_recordings": len(rows), "n_points": int(len(all_err)),
        "heldout_rmse_mm": float(np.sqrt(np.mean(all_err ** 2))),
        "heldout_max_mm": float(all_err.max()),
        "repeatability_p95_mm": float(np.quantile(rep_all, 0.95)),
        "axes": axes,
    }
    passed = (
        len(axes) == 3 and
        summary["heldout_rmse_mm"] <= thresholds["heldout_rmse_mm_max"] and
        summary["heldout_max_mm"] <= thresholds["heldout_max_mm_max"] and
        max(abs(a["scale_error_percent"]) for a in axes.values()) <=
        thresholds["axis_scale_error_percent_max_abs"] and
        max(a["cross_axis_rmse_mm"] for a in axes.values()) <=
        thresholds["cross_axis_rmse_mm_max"] and
        summary["repeatability_p95_mm"] <= thresholds["repeatability_p95_mm_max"]
    )
    report = {
        "kind": "independent_caliper_validation_no_scale_fit",
        "plan": str(plan), "passed": bool(passed), "thresholds": thresholds,
        "segmentation": {"still_mm_per_frame": still_mm, "min_duration_s": min_s},
        "summary": summary, "recordings": details,
        "limitations": [
            "Caliper calibration or gauge-block check must be archived separately.",
            "One-dimensional per axis; axes are separate mountings, so absolute "
            "orthogonality between axes is not tested.",
            "No result is valid if the readings were inferred from camera data.",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": passed, "summary": summary},
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
    p_an.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    p_an.add_argument("--still-mm", type=float, default=0.3,
                      help="caliper mode: max deviation from the 1-s window median in a pause")
    p_an.add_argument("--min-s", type=float, default=2.0,
                      help="caliper mode: minimum pause duration")
    args = parser.parse_args()
    plan = args.plan or (CALIPER_PLAN if args.mode == "caliper" else DEFAULT_PLAN)
    if args.command == "init":
        (init_caliper_plan if args.mode == "caliper" else init_plan)(plan, args.repeats)
    elif args.mode == "caliper":
        analyze_caliper(plan, args.output, args.still_mm, args.min_s)
    else:
        analyze(plan, args.output)


if __name__ == "__main__":
    main()
