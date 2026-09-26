#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical bound of zero-velocity-aided strapdown integration on this rig.

A window model cannot observe the mean camera velocity over its context, which
explains most of its residual. A detected zero-velocity phase removes exactly
that unknown: with the velocity pinned to zero at both ends of an interval, the
accelerometer alone determines the displacement and one constant bias per
interval is identifiable. This script measures the accuracy that becomes
available, separately for reference attitude (upper bound) and for the
integrated gyroscope (operational case), and counts how many such intervals the
present recordings actually contain.

The result is a measurement-design statement: it quantifies what a pause in the
acquisition protocol is worth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys

DATA = ROOT / "datasets"
G0 = phys.G0


def still_segments(t, ok, speed, thr, min_len, max_gap=0.35):
    """Maximal runs of reference frames slower than thr and at least min_len long."""
    idx = np.flatnonzero(ok & (speed < thr))
    if len(idx) == 0:
        return []
    segs, start = [], idx[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] > max_gap:
            if t[a] - t[start] >= min_len:
                segs.append((int(start), int(a)))
            start = b
    if t[idx[-1]] - t[start] >= min_len:
        segs.append((int(start), int(idx[-1])))
    return segs


def integrate(ses, t_nodes, R_nodes, ta, tb):
    """Displacement between ta and tb with v(ta)=v(tb)=0 and one constant bias.

    Returns (displacement, bias_mg) in the board frame, or None if too short.
    """
    m = (ses.t >= ta) & (ses.t <= tb)
    if int(m.sum()) < 50:
        return None
    tt = ses.t[m]
    k = np.searchsorted(t_nodes, tt).clip(0, len(t_nodes) - 1)
    a_b = np.einsum("nij,nj->ni", R_nodes[k] @ ses.R_ci, ses.acc[m])
    g_b = a_b.mean(0)
    g_b = g_b / np.linalg.norm(g_b) * G0
    lin = a_b - g_b
    dt = np.gradient(tt)
    # The bias is the constant acceleration that drives the end velocity to zero.
    bias = (lin * dt[:, None]).sum(0) / (tt[-1] - tt[0])
    v = np.cumsum((lin - bias) * dt[:, None], axis=0)
    d = np.cumsum(v * dt[:, None], axis=0)[-1]
    return d, float(np.linalg.norm(bias) / G0 * 1000)


def session_rows(name, thr, min_len, attitude):
    t, Rg, pg, usable = phys._load_gt(name)
    vg, vok = phys.gt_velocity(t, pg, usable)
    speed = np.linalg.norm(vg, axis=1) * 1000.0
    segs = still_segments(t, vok, speed, thr, min_len)
    ses = phys.Session(name)
    t_nodes = t[usable]
    if attitude == "gyro":
        from tools.fuse_trajectory import _gyro_orientation
        R_nodes = _gyro_orientation(name, t_nodes, Rg[usable[0]])
    else:
        R_nodes = Rg[usable]
    rows = []
    for (s0, s1), (n0, n1) in zip(segs[:-1], segs[1:]):
        # midpoints of the two still phases
        ia = s0 if speed[s0] <= speed[s1] else s1
        ib = n0 if speed[n0] <= speed[n1] else n1
        ta, tb = float(t[ia]), float(t[ib])
        if not (0.8 <= tb - ta <= 30.0):
            continue
        got = integrate(ses, t_nodes, R_nodes, ta, tb)
        if got is None:
            continue
        d, bias = got
        truth = pg[ib] - pg[ia]
        rows.append({
            "span_s": tb - ta,
            "true_mm": float(np.linalg.norm(truth) * 1000),
            "err_mm": float(np.linalg.norm(d - truth) * 1000),
            "bias_mg": bias,
        })
    return len(segs), rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr", type=float, default=3.0, help="stillness threshold, mm/s")
    ap.add_argument("--min-len", type=float, default=0.4, help="minimum still duration, s")
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    ap.add_argument("--output", type=Path, default=DATA / "zupt_bound.json")
    args = ap.parse_args()
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    groups = {"train": split["train"], "val": split["val"],
              "extra_train (rigid)": split.get("extra_train", []), "test": split["test"]}
    report = {"config": {"thr_mm_s": args.thr, "min_still_s": args.min_len}, "sessions": {}}
    for attitude in ("ref", "gyro"):
        pooled = {}
        for group, names in groups.items():
            rows, segs, seconds = [], 0, 0.0
            for name in names:
                n_seg, r = session_rows(name, args.thr, args.min_len, attitude)
                segs += n_seg
                t, _R, _p, us = phys._load_gt(name)
                seconds += float(t[us[-1]] - t[us[0]])
                rows += r
                report["sessions"].setdefault(name, {})[attitude] = {
                    "still_segments": n_seg, "intervals": len(r),
                    "err_mm": round(float(np.mean([x["err_mm"] for x in r])), 1) if r else None,
                }
            entry = {"sessions": len(names), "seconds": round(seconds, 1),
                     "still_segments": segs,
                     "still_segments_per_minute": round(segs / max(seconds / 60.0, 1e-6), 2),
                     "intervals": len(rows)}
            if rows:
                err = np.array([x["err_mm"] for x in rows])
                true = np.array([x["true_mm"] for x in rows])
                span = np.array([x["span_s"] for x in rows])
                bias = np.array([x["bias_mg"] for x in rows])
                entry |= {
                    "span_s_mean": round(float(span.mean()), 1),
                    "true_mm_mean": round(float(true.mean()), 1),
                    "err_mm_mean": round(float(err.mean()), 1),
                    "err_mm_median": round(float(np.median(err)), 1),
                    "err_mm_p90": round(float(np.percentile(err, 90)), 1),
                    "solved_bias_mg_mean": round(float(bias.mean()), 2),
                    "by_span": {},
                }
                for lo, hi in ((0.8, 2), (2, 4), (4, 8), (8, 30)):
                    s = (span >= lo) & (span < hi)
                    if s.any():
                        entry["by_span"][f"{lo}-{hi}s"] = {
                            "n": int(s.sum()),
                            "true_mm": round(float(true[s].mean()), 1),
                            "err_mm": round(float(err[s].mean()), 1),
                        }
            pooled[group] = entry
        report[attitude] = pooled
        print(json.dumps({attitude: pooled}, ensure_ascii=False, indent=2), flush=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
