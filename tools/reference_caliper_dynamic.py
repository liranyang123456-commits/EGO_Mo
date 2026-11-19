#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Label-level and in-motion scoring of the caliper recordings.

Complements ``reference_stage_validation.py analyze --mode caliper``, which
scores the median camera position of every pause. Here the reference is
scored the way it is used:

  label     every pair of single frames (a, b) that lie in two different
            pauses of one traversal gives a displacement that is compared
            with the caliper difference along the held-out axis direction.
            With ``--quantity board`` (default) it is the translation of the
            board centre in camera coordinates, which is what the caliper
            sets; with ``--quantity camera`` it is the label of the study,
            R_BC(t_a)^T [p_BC(t_b) - p_BC(t_a)], which also contains the tilt
            of the sliding jaw times the working distance. Endpoint noise and
            scale enter both. Both endpoints are at rest, so the time between
            them does not change the error; with a hand-read vernier
            consecutive holds are 8-15 s apart, and pairs are therefore not
            limited to the 1.8-4.5-s span of the study's windows
            (``--label-dt`` restores a limit).
  motion    the caliper constrains the board to a straight line, so the
            distance of every frame recorded while moving from that line is
            an in-motion error component (the along-axis position during a
            move is not known and is not scored).

The axis direction for a trial is fitted on the other trials of the same axis,
the origin on the trial's own first reading, and scale is never fitted.

    python tools/reference_caliper_dynamic.py --plan datasets/reference_caliper_plan.csv
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

import tools.reference_stage_validation as rsv
from tools.label_uncertainty import caliper_thresholds

SPEED_BINS = ((0.0, 5.0), (5.0, 15.0), (15.0, 1e9))
DISPLACEMENT_BINS = ((0.0, 10.0), (10.0, 20.0), (20.0, 45.0))
FRAME_STEP = 3   # every 3rd frame of a hold (5 Hz at 15 fps) keeps the pair count moderate


def _stats(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return {"n": 0}
    return {"n": int(len(x)), "median_mm": round(float(np.median(x)), 3),
            "mean_mm": round(float(x.mean()), 3), "p90_mm": round(float(np.percentile(x, 90)), 3),
            "p95_mm": round(float(np.percentile(x, 95)), 3), "rmse_mm": round(float(np.sqrt((x ** 2).mean())), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--data-root", type=Path,
                    help="folder with pose_gt_raw/ and the session folders (rehearsals)")
    ap.add_argument("--still-mm", type=float, default=0.2)
    ap.add_argument("--min-s", type=float, default=1.5)
    ap.add_argument("--label-dt", type=float, nargs=2, default=(0.0, 1e9),
                    help="limit the time between the two label endpoints (s)")
    ap.add_argument("--u-label-mm", type=float, nargs=2, default=None,
                    help="expanded label uncertainty (k=2) from Section V-B, nominal and perturbed")
    ap.add_argument("--quantity", choices=("board", "camera"), default="board",
                    help="board centre in camera coordinates, or camera centre in board coordinates")
    args = ap.parse_args()
    if args.u_label_mm is None:
        args.u_label_mm = caliper_thresholds()
    seg_key = "board_mm" if args.quantity == "board" else "median_mm"
    if args.data_root:
        rsv.DATA = args.data_root
    data = rsv.DATA
    plan = args.plan or data / rsv.CALIPER_PLAN.name
    output = args.output or data / ("reference_caliper_dynamic.json" if args.quantity == "board"
                                    else "reference_caliper_dynamic_camera.json")
    rows = list(csv.DictReader(plan.open(newline="", encoding="utf-8-sig")))
    missing = [f"{r['axis']}/trial{r['trial']}" for r in rows if not r["session"].strip()]
    if missing:
        raise SystemExit(f"plan incomplete: {missing}")

    trials = {}
    for r in rows:
        planned = np.array([float(v) for v in r["readings_mm"].split()])
        tr = rsv.caliper_trial(r["session"].strip(), planned, args.still_mm, args.min_s)
        trials.setdefault(r["axis"], []).append({
            "session": r["session"].strip(), "readings": tr["readings"], "segs": tr["segs"],
            "medians": np.array([s[seg_key] for s in tr["segs"]])})

    per_axis, all_label, all_true, all_static_perp, all_motion_perp, all_speed = {}, [], [], [], [], []
    all_hold_pair, all_hold_pair_step, all_adjacent, all_adjacent_step = [], [], [], []
    label_axis, motion_axis = [], []
    for axis, tr in sorted(trials.items()):
        label_err, label_true, label_meas = [], [], []
        hold_pair_err, hold_pair_step, adjacent_err, adjacent_step = [], [], [], []
        static_perp, motion_perp, motion_speed = [], [], []
        for k, trial in enumerate(tr):
            others = [x for j, x in enumerate(tr) if j != k]
            if not others:
                raise SystemExit(f"axis {axis}: at least two trials are needed for a held-out direction")
            s_tr = np.concatenate([x["readings"] for x in others])
            m_tr = np.concatenate([x["medians"] - x["medians"][np.argmin(x["readings"])] for x in others])
            u, _o, _scale = rsv._line_fit(s_tr, m_tr)
            s, m = trial["readings"], trial["medians"]
            origin = m[np.argmin(s)] - s.min() * u
            blob = np.load(data / "pose_gt_raw" / f"{trial['session']}.npz")
            use = np.flatnonzero(blob["usable"] == 1)
            t = blob["t"][use].astype(np.float64)
            R = blob["R"][use].astype(np.float64)
            p = rsv.frame_positions(blob, args.quantity)[use]
            U, _sv, Vt = np.linalg.svd(np.median(R, axis=0))
            r_true = np.eye(3) if args.quantity == "board" else U @ Vt
            hold_R = np.array([sg["R"] for sg in trial["segs"]])
            for i in range(len(s)):
                for j in range(i + 1, len(s)):
                    measured = m[j] - m[i] if args.quantity == "board" else \
                        hold_R[i].T @ (m[j] - m[i])
                    true = r_true.T @ (u * (s[j] - s[i]))
                    error = float(np.linalg.norm(measured - true))
                    hold_pair_err.append(error)
                    hold_pair_step.append(abs(float(s[j] - s[i])))
                    if j == i + 1:
                        adjacent_err.append(error)
                        adjacent_step.append(abs(float(s[j] - s[i])))
            pause_of = np.full(len(t), -1)
            for i, sg in enumerate(trial["segs"]):
                pause_of[(t >= sg["t0"]) & (t <= sg["t1"])] = i
            rel = p - origin
            along = rel @ u
            perp = np.linalg.norm(rel - along[:, None] * u, axis=1)
            static_perp.extend(perp[pause_of >= 0].tolist())
            first, last = trial["segs"][0]["t0"], trial["segs"][-1]["t1"]
            moving = (pause_of < 0) & (t > first) & (t < last)
            for i in np.flatnonzero(moving):
                near = np.abs(t - t[i]) <= 0.2
                if near.sum() >= 3:
                    speed = abs(np.polyfit(t[near], along[near], 1)[0])
                    motion_perp.append(perp[i])
                    motion_speed.append(speed)
            frames = [np.flatnonzero(pause_of == i)[::FRAME_STEP] for i in range(len(trial["segs"]))]
            for i in range(len(frames)):
                for j in range(len(frames)):
                    if i == j or len(frames[i]) == 0 or len(frames[j]) == 0:
                        continue
                    a, b = np.meshgrid(frames[i], frames[j], indexing="ij")
                    a, b = a.ravel(), b.ravel()
                    dt = t[b] - t[a]
                    ok = (dt >= args.label_dt[0]) & (dt <= args.label_dt[1])
                    a, b = a[ok], b[ok]
                    if len(a) == 0:
                        continue
                    meas = p[b] - p[a] if args.quantity == "board" else \
                        np.einsum("nji,nj->ni", R[a], p[b] - p[a])
                    true = r_true.T @ (u * (s[j] - s[i]))
                    label_err.extend(np.linalg.norm(meas - true, axis=1).tolist())
                    label_true.extend([float(np.linalg.norm(true))] * len(a))
                    label_meas.extend(np.linalg.norm(meas, axis=1).tolist())
        label_err, label_true, label_meas = map(np.asarray, (label_err, label_true, label_meas))
        speed = np.asarray(motion_speed)
        per_axis[axis] = {
            "trials": len(tr),
            "label_error": _stats(label_err),
            "label_error_by_displacement": {f"{lo:g}-{hi:g} mm": _stats(label_err[(label_true >= lo) & (label_true < hi)])
                                            for lo, hi in DISPLACEMENT_BINS},
            "label_scale": round(float(label_true @ label_meas / max(label_true @ label_true, 1e-9)), 4),
            "label_mean_true_mm": round(float(label_true.mean()), 2) if len(label_true) else None,
            "hold_pair_displacement": {
                **_stats(hold_pair_err),
                "by_step_mm": {
                    f"{step:g}": _stats(np.asarray(hold_pair_err)[np.isclose(hold_pair_step, step)])
                    for step in sorted(np.unique(hold_pair_step))
                },
            },
            "adjacent_moves": {
                **_stats(adjacent_err),
                "by_step_mm": {
                    f"{step:g}": _stats(np.asarray(adjacent_err)[np.isclose(adjacent_step, step)])
                    for step in sorted(np.unique(adjacent_step))
                },
            },
            "off_axis_static": _stats(static_perp),
            "off_axis_moving": _stats(motion_perp),
            "off_axis_moving_by_speed": {f"{lo:g}-{hi:g} mm/s" if hi < 1e8 else f">{lo:g} mm/s":
                                         _stats(np.asarray(motion_perp)[(speed >= lo) & (speed < hi)])
                                         for lo, hi in SPEED_BINS},
        }
        all_label.extend(label_err.tolist())
        all_true.extend(label_true.tolist())
        all_hold_pair.extend(hold_pair_err)
        all_hold_pair_step.extend(hold_pair_step)
        all_adjacent.extend(adjacent_err)
        all_adjacent_step.extend(adjacent_step)
        all_static_perp.extend(static_perp)
        all_motion_perp.extend(motion_perp)
        all_speed.extend(motion_speed)
        label_axis += [axis] * len(label_err)
        motion_axis += [axis] * len(motion_perp)

    all_label, all_true = np.asarray(all_label), np.asarray(all_true)
    all_hold_pair, all_hold_pair_step = np.asarray(all_hold_pair), np.asarray(all_hold_pair_step)
    all_adjacent, all_adjacent_step = np.asarray(all_adjacent), np.asarray(all_adjacent_step)
    all_motion_perp, all_speed = np.asarray(all_motion_perp), np.asarray(all_speed)
    report = {
        "kind": "caliper_label_and_motion_scoring",
        "plan": str(plan),
        "quantity": args.quantity,
        "U_thresholds_mm": list(args.u_label_mm),
        "label_dt_s": list(args.label_dt),
        "frame_step_in_holds": FRAME_STEP,
        "axes": per_axis,
        "overall": {
            "label_error": _stats(all_label),
            "label_error_by_displacement": {f"{lo:g}-{hi:g} mm": _stats(all_label[(all_true >= lo) & (all_true < hi)])
                                            for lo, hi in DISPLACEMENT_BINS},
            "label_error_within_U_label_nominal": round(float((all_label <= args.u_label_mm[0]).mean()), 3),
            "label_error_within_U_label_perturbed": round(float((all_label <= args.u_label_mm[1]).mean()), 3),
            "hold_pair_displacement": {
                **_stats(all_hold_pair),
                "by_step_mm": {
                    f"{step:g}": _stats(all_hold_pair[np.isclose(all_hold_pair_step, step)])
                    for step in sorted(np.unique(all_hold_pair_step))
                },
            },
            "adjacent_moves": {
                **_stats(all_adjacent),
                "by_step_mm": {
                    f"{step:g}": _stats(all_adjacent[np.isclose(all_adjacent_step, step)])
                    for step in sorted(np.unique(all_adjacent_step))
                },
            },
            "off_axis_static": _stats(all_static_perp),
            "off_axis_moving": _stats(all_motion_perp),
            "off_axis_moving_by_speed": {f"{lo:g}-{hi:g} mm/s" if hi < 1e8 else f">{lo:g} mm/s":
                                         _stats(all_motion_perp[(all_speed >= lo) & (all_speed < hi)])
                                         for lo, hi in SPEED_BINS},
        },
        "limitations": [
            "The caliper readings give independent start-to-end displacement truth for each actual move; "
            "the instantaneous along-axis jaw position during the transition is not recorded.",
            "Endpoints lie in holds, so blur at the endpoints is not tested; while moving, only the "
            "off-axis component is scored frame by frame.",
            "Board flatness and the caliper's own calibration are not part of this analysis.",
        ],
    }
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(output.with_suffix(".npz"), label_err_mm=all_label, label_true_mm=all_true,
                        hold_pair_error_mm=all_hold_pair, hold_pair_step_mm=all_hold_pair_step,
                        adjacent_move_error_mm=all_adjacent, adjacent_move_step_mm=all_adjacent_step,
                        label_axis=np.asarray(label_axis), motion_perp_mm=all_motion_perp,
                        motion_speed_mm_s=all_speed, motion_axis=np.asarray(motion_axis))
    print(json.dumps(report["overall"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
