#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Motion-class switching: stop-anchored strapdown where a stop is detected.

A window is switched to the physics estimate when an IMU-detected full stop
(|w| < 1 deg/s and ||a| - g| < 0.01 g for 0.5 s) lies in [t_a - context, t_b]
and no stretch longer than `max_span` seconds has to be integrated without a
bracketing stop. Gravity is taken from the still samples (not from the
whole-window mean used by the PhysNet featurizer), the velocity drift is
removed linearly between still samples (two-sided zero-velocity update), and
the anchor-frame acceleration is integrated twice. All other windows keep the
learned prediction. `max_span` is chosen on the synthetic validation split.
Use models trained with --bias-mode none, whose windows carry no false bias.

    python tools/stop_anchor_check.py --target pause_4 pause_12.5 --model pause_4_nb pause_12.5_nb \
        --real --physnet bias_none --seeds 0 1
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
from tools.twin_motion_class_eval import window_features  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"
W_MAX_DPS, ACC_TOL_G, STILL_S = 1.0, 0.01, 0.5


def true_rest(data: dict, context: float, hz: float = v3.HZ) -> np.ndarray:
    """Synthetic only: samples where the true camera speed is below 1 mm/s."""
    out = np.zeros(data["valid"].shape, bool)
    T = out.shape[1]
    for name in np.unique(data["session"]):
        gt = np.load(Path(name) / "ground_truth.npz")
        speed = np.linalg.norm(np.gradient(gt["imu_p_W_C"].astype(np.float64), gt["imu_t"], axis=0),
                               axis=1)
        for k in np.flatnonzero(data["session"] == name):
            grid = data["t_pair"][k, 0] - context + np.arange(T) / hz
            out[k] = np.interp(grid, gt["imu_t"], speed, left=1.0, right=1.0) < 1e-3
    return out


def stop_anchor(data: dict, anchor: int, hz: float = v3.HZ, rest: np.ndarray | None = None):
    """Per-window stop-anchored displacement (NaN where no stop is detected).

    `rest` replaces the IMU still detector by a given per-sample rest mask.
    """
    from scipy.ndimage import minimum_filter1d
    acc, gyro, quat, valid = data["acc"], data["gyro"], data["quat"], data["valid"] > 0.5
    tau_b = data["tau_b"]
    R = v3.quat_to_mat(torch.from_numpy(quat)).numpy()
    a_rot = np.einsum("btij,btj->bti", R, acc)
    ok = ((np.degrees(np.linalg.norm(gyro, axis=-1)) < W_MAX_DPS)
          & (np.abs(np.linalg.norm(acc, axis=-1) / v3.G0 - 1.0) < ACC_TOL_G) & valid)
    if rest is not None:
        ok = rest & valid
    # A sample is still if the whole STILL_S run around it passes the test
    # (erosion followed by dilation keeps only runs of at least STILL_S).
    from scipy.ndimage import maximum_filter1d
    w = int(STILL_S * hz)
    still = maximum_filter1d(minimum_filter1d(ok.astype(np.uint8), w, axis=1), w, axis=1) > 0
    still &= ok
    out = np.full((len(acc), 3), np.nan)
    span = np.full(len(acc), np.inf)
    dt = 1.0 / hz
    for k in range(len(acc)):
        end = int(np.ceil(tau_b[k])) + 1
        s = still[k, :end]
        if s.sum() < w:
            continue
        g = a_rot[k, :end][s].mean(0)
        lin = (a_rot[k, :end] - g) * valid[k, :end, None]
        v = np.cumsum(lin, 0) * dt
        # Two-sided zero-velocity update: remove the velocity drift linearly
        # between still samples, hold it constant before the first/after the last.
        idx = np.flatnonzero(s)
        tk = np.arange(end)
        v = v - np.column_stack([np.interp(tk, idx, v[idx, c]) for c in range(3)])
        d = np.cumsum(v, 0) * dt
        b0 = int(np.floor(tau_b[k]))
        f = tau_b[k] - b0
        d_b = d[b0] * (1 - f) + d[min(b0 + 1, end - 1)] * f
        out[k] = d_b - d[anchor]
        # Longest stretch integrated without a bracketing stop: before the first
        # still sample (back to t_a) or after the last one (up to t_b).
        span[k] = max(idx[0] - anchor, tau_b[k] - idx[-1], 0.0) / hz
    return out, span


SPANS = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def predictions(names, models, context=2.0, horizon=3.0, cap=160, oracle=False) -> dict:
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))
    anchor = int(round(context * v3.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = v3.build_group(names, context, horizon, length, cap=cap,
                          rng=np.random.default_rng(123) if cap else None)
    if models:
        dt = v3.to_device(data, device)
        learned = np.mean([m.predict(dt) for m in models], axis=0)
    else:
        learned = np.zeros_like(data["y"])
    phys, span = stop_anchor(data, anchor, rest=true_rest(data, context) if oracle else None)
    cls = []
    for n in names:
        sel = data["session"] == str(n)
        cls.append(window_features(n, {"pair": data["pair"][sel]})["stop_class"])
    return {"y": data["y"], "learned": learned, "phys": phys, "span": span,
            "cls": np.concatenate(cls)}


def hybrid(P: dict, max_span: float) -> np.ndarray:
    """Stop-anchored estimate where a detected stop brackets the window closely enough."""
    use = P["span"] <= max_span
    return np.where(use[:, None], P["phys"], P["learned"])


def summarize(P: dict, max_span: float) -> dict:
    y = P["y"]
    h = hybrid(P, max_span)
    use = P["span"] <= max_span

    def err(p, m):
        return round(float(np.linalg.norm(p[m] - y[m], axis=1).mean() * 1000), 2) if m.any() else None

    every = np.ones(len(y), bool)
    rep = {"n": int(len(y)), "max_span_s": max_span, "physics_share": round(float(use.mean()), 3),
           "all": {"zero": err(np.zeros_like(y), every), "learned": err(P["learned"], every),
                   "hybrid": err(h, every),
                   "learned_on_switched": err(P["learned"], use),
                   "physics_on_switched": err(P["phys"], use)}}
    for c in ("free", "context", "window"):
        m = P["cls"] == c
        if m.sum() < 20:
            continue
        det = m & ~np.isnan(P["phys"][:, 0])
        rep[c] = {"n": int(m.sum()), "switched": int((m & use).sum()),
                  "zero": err(np.zeros_like(y), m), "learned": err(P["learned"], m),
                  "hybrid": err(h, m), "detected": int(det.sum()),
                  "physics_on_detected": err(P["phys"], det),
                  "learned_on_detected": err(P["learned"], det)}
    return rep


def pooled_err(Ps, max_span):
    e = np.concatenate([np.linalg.norm(hybrid(P, max_span) - P["y"], axis=1) for P in Ps])
    return float(e.mean() * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", nargs="*", default=[], help="protocol corpora (test split)")
    ap.add_argument("--model", nargs="*", default=[], help="protocol model tag per target")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--oracle-stop", action="store_true",
                    help="synthetic: anchor at true rest samples instead of IMU-detected ones")
    ap.add_argument("--physnet", default="base")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--output", type=Path, default=DATA / "stop_anchor_check.json")
    args = ap.parse_args()
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {"still_detector": {"gyro_max_dps": W_MAX_DPS, "acc_tol_g": ACC_TOL_G,
                                 "min_s": STILL_S},
              "oracle_stop": args.oracle_stop, "synthetic": {}}
    val, test = {}, {}
    for target, tag in zip(args.target, args.model):
        root = Path((PROTO / target / "latest.txt").read_text(encoding="utf-8").strip())
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        models = [v3.load_model(DATA / "physnet_proto" / f"physnet_{tag}_s{s}.pt", device)
                  for s in args.seeds]
        for split, store in (("val", val), ("test", test)):
            names = [v3._corpus_path(root, e["path"]) for e in manifest["imu"][split]]
            store[target] = predictions(names, models, oracle=args.oracle_stop)
    best = None
    if val:
        # Switching threshold chosen on the synthetic validation split only.
        curve = {s: round(pooled_err(list(val.values()), s), 2) for s in SPANS}
        best = min(curve, key=curve.get)
        report["val_pooled_err_by_max_span"] = curve
        report["chosen_max_span_s"] = best
        for target, P in test.items():
            report["synthetic"][target] = summarize(P, best)
            print(target, json.dumps(report["synthetic"][target]), flush=True)
    if args.real:
        split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
        models = [v3.load_model(DATA / "physnet_v3" / f"physnet_{args.physnet}_s{s}.pt", device)
                  for s in args.seeds]
        P = predictions(split["val"], models, cap=0)
        span = best if best is not None else 1.0
        report[f"real_val|{args.physnet}"] = summarize(P, span)
        print("real", json.dumps(report[f"real_val|{args.physnet}"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
