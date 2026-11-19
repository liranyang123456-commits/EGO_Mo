#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Error by motion class on the real validation recordings (fixed split).

Every 3-s window is classified twice:

* reference class (analysis only, uses the chessboard reference): speed at the
  window start |v(t_a)| and the time since the last full stop (>= 0.3 s below
  10 mm/s) before t_a;
* IMU class (available at inference): the rig is still at t_a according to
  the gyroscope/accelerometer (|w| < 3 deg/s and ||a| - g| < 0.03 g over +-0.25 s).

Methods are seed ensembles of v3 PhysNet checkpoints or of external baselines
(frozen trainer checkpoints); all see the same windows.

    python tools/twin_motion_class_eval.py --physnet base pre_realistic \
        --external imunet:external_twin_realistic/imunet_real imunet_pre:external_twin_realistic/imunet_fine
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet_v3 as v3  # noqa: E402
import tools.train_seq as api  # noqa: E402
from tools.train_external_architectures import make_model, predict as predict_external  # noqa: E402

DATA = ROOT / "datasets"
SPLIT = "trajectory_split_paper.json"
REST_MM_S, REST_MIN_S = 10.0, 0.3
SPEED_BINS = (0.0, 15.0, 40.0, 80.0, np.inf)
SINCE_BINS = (0.0, 1.0, 3.0, 6.0, np.inf)


def window_features(name, data) -> dict:
    """Reference speed at t_a, time since the last full stop, IMU stillness at t_a."""
    t, Rg, pg, usable = v3._load_gt(name)
    vg, v_ok = v3.gt_velocity(t, pg, usable, half=0.4)
    speed = np.linalg.norm(vg, axis=1) * 1000
    rest = v_ok & (speed < REST_MM_S)
    # Full stops: runs of rest frames lasting at least REST_MIN_S.
    stop_end = np.full(len(t), -np.inf)
    k = 0
    while k < len(t):
        if rest[k]:
            j = k
            while j + 1 < len(t) and rest[j + 1]:
                j += 1
            if t[j] - t[k] >= REST_MIN_S:
                stop_end[k:j + 1] = t[k:j + 1]
            k = j + 1
        else:
            k += 1
    last_stop = np.maximum.accumulate(stop_end)
    a, b = data["pair"][:, 0], data["pair"][:, 1]
    stop_t = t[np.isfinite(stop_end)]
    # Where the rig is at a full stop relative to the window: inside (t_a, t_b]
    # (zero velocity somewhere in the window), only in the 2-s context before
    # t_a (initial velocity follows from the context), or nowhere (free).
    in_win = np.array([((stop_t > t[i]) & (stop_t <= t[j])).any() or (t[i] - last_stop[i] < 0.05)
                       for i, j in zip(a, b)])
    in_ctx = np.array([((stop_t >= t[i] - 2.0) & (stop_t <= t[i])).any() for i in a]) & ~in_win
    stop_class = np.where(in_win, "window", np.where(in_ctx, "context", "free"))
    ses = v3.Session(name)
    g = np.linalg.norm(ses.acc, axis=1) / v3.G0
    w = np.degrees(np.linalg.norm(ses.gyro, axis=1))
    still = np.zeros(len(a), bool)
    for m, ta in enumerate(t[a]):
        sel = (ses.t >= ta - 0.25) & (ses.t <= ta + 0.25)
        still[m] = sel.any() and w[sel].max() < 3.0 and np.abs(g[sel] - 1).max() < 0.03
    return {"speed_a": np.where(v_ok[a], speed[a], np.nan),
            "since_stop": t[a] - last_stop[a], "imu_still": still, "stop_class": stop_class}


def _metrics(p, y):
    e = np.linalg.norm(p - y, axis=1) * 1000
    return {"n": int(len(y)), "err_mm": round(float(e.mean()), 2),
            "zero_mm": round(float(np.linalg.norm(y, axis=1).mean() * 1000), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--physnet", nargs="*", default=["base"],
                    help="v3 config names (datasets/physnet_v3/physnet_<name>_s<seed>.pt)")
    ap.add_argument("--external", nargs="*", default=[],
                    help="label:dir/stem, checkpoints <stem>_s<seed>.pt under datasets/")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--output", type=Path, default=DATA / "twin_motion_class_eval.json")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = json.loads((DATA / SPLIT).read_text(encoding="utf-8"))

    phys = {n: [v3.load_model(DATA / "physnet_v3" / f"physnet_{n}_s{s}.pt", device)
                for s in args.seeds] for n in args.physnet}
    ext = {}
    for spec in args.external:
        label, stem = spec.split(":", 1)
        nets = []
        for s in args.seeds:
            path = DATA / f"{stem}_s{s}.pt"
            if not path.is_file():
                continue
            net = make_model("imunet").to(device)
            net.load_state_dict(torch.load(path, map_location=device))
            net.eval()
            nets.append(net)
        ext[label] = nets

    P = {k: [] for k in list(phys) + list(ext)}
    Y, F = [], {"speed_a": [], "since_stop": [], "imu_still": [], "stop_class": [], "session": []}
    for ses in split["val"]:
        data = v3.build_group([ses], context, horizon, length)
        X, y = api.build(ses, context, length)
        assert np.allclose(data["y"], y, atol=1e-6)
        dt = v3.to_device(data, device)
        for k, models in phys.items():
            P[k].append(np.mean([m.predict(dt) for m in models], axis=0))
        for k, nets in ext.items():
            P[k].append(np.mean([predict_external(n, X, device) for n in nets], axis=0))
        Y.append(y)
        for key, val in window_features(ses, data).items():
            F[key].append(val)
        F["session"].append(np.array([ses] * len(y)))
    y = np.concatenate(Y)
    P = {k: np.concatenate(v) for k, v in P.items()}
    F = {k: np.concatenate(v) for k, v in F.items()}

    def table(mask_by_class):
        out = {}
        for cls, mask in mask_by_class.items():
            if mask.sum() < 20:
                continue
            ym = y[mask]
            row = {"n": int(mask.sum()), "share": round(float(mask.mean()), 3),
                   "zero_mm": _metrics(ym, ym)["zero_mm"]}
            for k, p in P.items():
                row[k] = _metrics(p[mask], ym)["err_mm"]
                # Pooled slope of prediction on reference (1 = no shrinkage).
                row[f"{k}_slope"] = round(float((p[mask] * ym).sum() / (ym * ym).sum()), 3)
            out[cls] = row
        return out

    report = {"sessions": split["val"], "methods": {k: len(v) for k, v in {**phys, **ext}.items()},
              "all": table({"all": np.ones(len(y), bool)})["all"]}
    sp = F["speed_a"]
    report["by_start_speed_mm_s"] = table({
        f"{lo:g}-{hi:g}": (sp >= lo) & (sp < hi) for lo, hi in zip(SPEED_BINS[:-1], SPEED_BINS[1:])})
    ss = F["since_stop"]
    report["by_time_since_stop_s"] = table({
        f"{lo:g}-{hi:g}": (ss >= lo) & (ss < hi) for lo, hi in zip(SINCE_BINS[:-1], SINCE_BINS[1:])})
    report["by_imu_still_start"] = table({"still": F["imu_still"], "moving": ~F["imu_still"]})
    report["by_stop_position"] = table({c: F["stop_class"] == c for c in ("free", "context", "window")})
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for key in ("all",):
        print(key, report[key])
    for key in ("by_start_speed_mm_s", "by_time_since_stop_s", "by_imu_still_start",
                "by_stop_position"):
        print(key)
        for cls, row in report[key].items():
            print(f"  {cls:10s}", row)


if __name__ == "__main__":
    main()
