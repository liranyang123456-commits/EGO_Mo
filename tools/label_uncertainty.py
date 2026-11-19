"""Recompute the 3-s label budget from frozen frame pairs and raw PnP poses.

Run from any directory. The three pose exports and the frozen evaluation pairs
are sufficient. No IMU prediction or trained checkpoint enters the calculation.
"""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = Path(__file__).resolve().parents[1] / "datasets"
PAIRS = DATA / "physnet_v1/eval_wd1.npz"
SPLIT = DATA / "trajectory_split_paper.json"
MC = DATA / "sim_reference_validation_psf.json"
DELAYS_MS = (5, 10)


def budget_ranges():
    summary = json.loads((DATA / "label_uncertainty.json").read_text())
    return summary["ranges_of_medians_mm"]


def caliper_thresholds():
    ranges = budget_ranges()
    return ranges["U_nominal"][0], ranges["U_perturbed"][1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def session_rows(group, name, pairs, times, labels, scale, pose_file):
    with np.load(pose_file) as poses:
        t, position, rotation = (poses[k].astype(float) for k in ("t", "p", "R"))
        usable = np.flatnonzero(poses["usable"] == 1)
    a, b = pairs.T
    if not np.isin(pairs, usable).all() or not (b > a).all():
        raise ValueError(f"{name}: invalid endpoint index")
    # The raw camera clock and frozen evaluation clock differ by one session delay.
    offset = t[pairs] - times
    np.testing.assert_allclose(offset, np.median(offset), rtol=0, atol=5e-7)
    displacement = np.einsum("nji,nj->ni", rotation[a], position[b] - position[a])
    np.testing.assert_allclose(displacement, labels, rtol=1e-6, atol=1e-8)
    dp = displacement * 1000
    tu = t[usable]
    if not np.all(np.diff(tu) > 0):
        raise ValueError(f"{name}: non-increasing pose timestamps")
    # Central secant in the board frame, one-sided at the recording boundaries.
    index = np.arange(len(tu))
    lo, hi = np.maximum(index - 1, 0), np.minimum(index + 1, len(tu) - 1)
    span = tu[hi] - tu[lo]
    vel = (position[usable[hi]] - position[usable[lo]]) * 1000 / span[:, None]
    ia, ib = np.searchsorted(usable, pairs).T
    dv = np.linalg.norm(vel[ib] - vel[ia], axis=1)
    travel = np.linalg.norm(dp, axis=1)
    end = np.hypot(np.sqrt(2) * 0.75, travel * np.deg2rad(0.21))
    nominal = 0.018 * travel
    perturbed = np.linalg.norm(dp * scale, axis=1)
    for i, (start, stop) in enumerate(pairs):
        row = dict(split=group, session=name, frame_a=start, frame_b=stop,
                   duration_s=t[stop]-t[start], velocity_span_a_s=span[ia[i]],
                   velocity_span_b_s=span[ib[i]], dx_mm=dp[i, 0], dy_mm=dp[i, 1],
                   dz_mm=dp[i, 2], delta_v_mm_s=dv[i], end_mm=end[i],
                   cal_nominal_mm=nominal[i], cal_perturbed_mm=perturbed[i])
        for ms in DELAYS_MS:
            timing = dv[i] * ms / 1000
            row[f"time_{ms}ms_mm"] = timing
            for case, cal in (("nominal", nominal[i]), ("perturbed", perturbed[i])):
                row[f"U_{case}_{ms}ms_mm"] = 2 * np.linalg.norm([end[i], timing, cal])
        yield row


def main():
    split = json.loads(SPLIT.read_text())
    mc = json.loads(MC.read_text())["depths"]["250"]["monte_carlo"]
    scale = np.array([mc["scale_error_percent_std"][c] for c in "XYZ"]) / 100
    inputs = [PAIRS, SPLIT, MC]
    rows, medians = [], {}
    with np.load(PAIRS) as frozen:
        for group in ("val", "test"):
            sessions = frozen[group + "_session"]
            if set(sessions) != set(split[group]):
                raise ValueError(f"{group}: session split mismatch")
            group_rows = []
            for name in split[group]:
                pose = DATA / "pose_gt_raw" / f"{name}.npz"
                inputs.append(pose)
                mask = sessions == name
                args = [frozen[f"{group}_{key}"][mask] for key in ("pair", "t_pair", "y")]
                group_rows.extend(session_rows(group, name, *args, scale, pose))
            keys = [k for k in group_rows[0] if k.startswith(("end_", "cal_", "U_", "time_"))]
            medians[group] = {key: float(np.median([r[key] for r in group_rows])) for key in keys}
            rows.extend(group_rows)
    table = DATA / "label_uncertainty_windows.csv"
    with table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    ranges = {}
    for term, keys in {
        "end": ["end_mm"], "time": [f"time_{ms}ms_mm" for ms in DELAYS_MS],
        **{f"cal_{case}": [f"cal_{case}_mm"] for case in ("nominal", "perturbed")},
        **{f"U_{case}": [f"U_{case}_{ms}ms_mm" for ms in DELAYS_MS] for case in ("nominal", "perturbed")},
    }.items():
        values = [m[k] for m in medians.values() for k in keys]
        ranges[term] = [min(values), max(values)]
    report = dict(position_sigma_mm=0.75, rotation_sigma_deg=0.21, nominal_fraction=0.018,
                  axis_scale_fraction_xyz=scale.tolist(), delay_sigma_ms=DELAYS_MS, coverage_factor=2,
                  counts={s: sum(r["session"] == s for r in rows) for s in split["val"]+split["test"]},
                  medians_mm=medians, ranges_of_medians_mm=ranges,
                  inputs_sha256={p.relative_to(DATA).as_posix(): sha(p) for p in inputs},
                  windows_csv_sha256=sha(table))
    (DATA / "label_uncertainty.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
