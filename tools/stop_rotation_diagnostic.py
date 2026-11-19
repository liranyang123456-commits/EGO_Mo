"""Error of stop-anchored switching versus camera rotation within the window.

Reads the per-window output of eval_stop_protocol_experiment.py (paused recordings), takes the
rotation between the usable reference frames at the window ends, and reports the errors per
rotation bin.  The gated variant switches only when that rotation is below --gate-deg; it uses the
reference rotation as a stand-in for the gyroscope rotation and is exploratory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def window_rotation(sessions: np.ndarray, t_a: np.ndarray, horizon: float, tol: float) -> np.ndarray:
    rot = np.full(len(t_a), np.nan)
    for name in np.unique(sessions):
        raw = np.load(DATA / "pose_gt_raw" / f"{name}.npz")
        usable = raw["usable"].astype(bool)
        t, R = raw["t"][usable], raw["R"][usable].astype(float)
        for i in np.where(sessions == name)[0]:
            ia = int(np.argmin(np.abs(t - t_a[i])))
            ib = int(np.argmin(np.abs(t - t_a[i] - horizon)))
            if abs(t[ia] - t_a[i]) < tol and abs(t[ib] - t_a[i] - horizon) < tol:
                rot[i] = np.degrees(Rotation.from_matrix(R[ib] @ R[ia].T).magnitude())
    return rot


def reduction(err: np.ndarray, zero: np.ndarray) -> float:
    return round(float(1.0 - np.mean(err) / np.mean(zero)), 3)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--result", default=str(DATA / "stop_protocol_experiment.npz"))
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--bins", default="0,3,8,15,45")
    ap.add_argument("--gate-deg", type=float, default=8.0)
    args = ap.parse_args()

    z = np.load(args.result, allow_pickle=True)
    paused = z["condition"] == "paused"
    sessions, t_a = z["session"][paused], z["t_a"][paused]
    zero, net, switch = (z[k][paused] for k in ("err_zero_mm", "err_physnet_mm", "err_physnet_switch_mm"))
    switched = np.abs(switch - net) > 1e-3  # mm; float32/float64 rounding differs by up to ~1e-5
    rot = window_rotation(sessions, t_a, args.horizon, args.tol)
    known = switched & np.isfinite(rot)

    edges = [float(b) for b in args.bins.split(",")]
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = known & (rot >= lo) & (rot < hi)
        errs = {"stop_anchored": switch[b], "physnet": net[b], "zero": zero[b]}
        bins.append({"deg": [lo, hi], "n": int(b.sum()),
                     "mean_mm": {k: round(float(np.mean(v)), 1) for k, v in errs.items()},
                     "median_mm": {k: round(float(np.median(v)), 1) for k, v in errs.items()}})

    gated = np.where(known & (rot < args.gate_deg), switch, net)
    per_session = {str(s): {"physnet": reduction(net[sessions == s], zero[sessions == s]),
                            "switch": reduction(switch[sessions == s], zero[sessions == s]),
                            "gated": reduction(gated[sessions == s], zero[sessions == s])}
                   for s in np.unique(sessions)}
    out = {"result": str(args.result), "paused_windows": int(paused.sum()), "switched": int(switched.sum()),
           "switched_with_rotation": int(known.sum()),
           "rotation_deg_median": round(float(np.median(rot[known])), 1),
           "rotation_deg_p90": round(float(np.percentile(rot[known], 90)), 1),
           "share_switched_at_or_above_gate": round(float(np.mean(rot[known] >= args.gate_deg)), 3),
           "bins": bins,
           "reduction_vs_zero": {"physnet": reduction(net, zero), "switch": reduction(switch, zero),
                                 "gated": reduction(gated, zero), "per_session": per_session},
           "gate_deg": args.gate_deg, "gate_note": "reference rotation used in place of the gyroscope rotation"}
    path = Path(args.result).with_name(Path(args.result).stem + "_rotation.json")
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
