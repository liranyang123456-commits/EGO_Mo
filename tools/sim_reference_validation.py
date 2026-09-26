#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Image-level simulation of the known-displacement (caliper) protocol.

The digital twin renders the GP050 board, with the calibrated lens distortion,
sensor noise and illumination flicker, at displacements that follow
datasets/reference_caliper_plan.csv. The rendered left images go through the
same corner detector and PnP as the real reference (tools/build_pose_gt.py),
and the pose streams are scored by the same no-scale-fit analysis as a real
caliper run (tools/reference_stage_validation.py --mode caliper).

Conditions
----------
nominal       PnP uses the rendering intrinsics and the nominal 3.0-mm pitch.
monte_carlo   Detected corners are re-solved with intrinsics drawn from the
              calibration standard deviations and a board-pitch error, which
              gives the reference error budget that a physical run would see.

This verifies the reference pipeline, not the physical rig: board flatness,
rolling shutter, real defocus and the true intrinsic error remain untested.
The renderer's background image is cached per seed; ``render`` only reads it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.reference_stage_validation as rsv
from ego_sim.core import SimConfig, StereoRenderer
from tools.build_pose_gt import _find, _pose

DATA = ROOT / "datasets"
DEFAULT_OUT = DATA / "sim_reference_validation.json"

# 1-sigma values from cv2.calibrateCameraExtended on 80 frames of the intrinsic
# recording (cam0). They include motion-blurred frames and are conservative.
INTRINSIC_STD = {"fx": 10.74, "fy": 10.79, "cx": 8.39, "cy": 7.50,
                 "dist": (0.032, 0.493, 0.002, 0.003, 2.699)}
PITCH_STD_REL = 1e-3


def _board_pose(depth_mm: float, tilt_deg: float, cfg: SimConfig):
    R_W_B = Rotation.from_euler("xy", [tilt_deg, 0.4 * tilt_deg], degrees=True).as_matrix()
    sq = cfg.square_mm / 1000.0
    centre_B = np.array([(cfg.board_cols - 2) * sq / 2, (cfg.board_rows - 2) * sq / 2, 0.0])
    p0 = np.array([0.0, 0.0, depth_mm / 1000.0]) - R_W_B @ centre_B
    return R_W_B, p0


def _timeline(readings, fps, pause_s, move_s):
    """(time, reading, is_pause) samples for one caliper traversal."""
    rows, t = [], 0.0
    for k, s in enumerate(readings):
        for _ in range(int(round(pause_s * fps))):
            rows.append((t, s, True)); t += 1.0 / fps
        if k + 1 < len(readings):
            n = int(round(move_s * fps))
            for a in np.linspace(0, 1, n + 2)[1:-1]:
                rows.append((t, s + a * (readings[k + 1] - s), False)); t += 1.0 / fps
    return rows


def _checker(cfg: SimConfig, px: int) -> np.ndarray:
    h, w = cfg.board_rows * px, cfg.board_cols * px
    image = np.empty((h, w, 3), np.uint8)
    for row in range(cfg.board_rows):
        for col in range(cfg.board_cols):
            image[row * px:(row + 1) * px, col * px:(col + 1) * px] = 235 if (row + col) % 2 == 0 else 18
    return image


def _renderer(cfg: SimConfig, depth_mm: float) -> StereoRenderer:
    """Renderer with two corrections that matter for sub-pixel corners.

    The default 70-px texture is minified ~5x by warpPerspective without
    anti-aliasing, and cv2.undistortPoints' 5 iterations do not invert this
    lens's k3 = 6.6 model near the image edge. The texture is resized to about
    twice the projected square size and the distortion map is inverted to
    convergence.
    """
    r = StereoRenderer(cfg)
    fx = r.K0[0, 0]
    r.texture = _checker(cfg, max(4, int(round(2 * cfg.square_mm * fx / depth_mm))))
    yy, xx = np.mgrid[0:cfg.height, 0:cfg.width].astype(np.float32)
    pts = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
    und = cv2.undistortPointsIter(
        pts, r.K0, r.d0, None, r.K0,
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-10),
    ).reshape(cfg.height, cfg.width, 2)
    r._maps["cam0"] = (und[:, :, 0].copy(), und[:, :, 1].copy())
    cache: dict[int, np.ndarray] = {}
    base = r._background

    def background(seed: int) -> np.ndarray:
        if seed not in cache:
            cache[seed] = base(seed)
        return cache[seed]

    r._background = background
    return r


def render_trials(plan_rows, depth_mm, tilt_deg, fps, pause_s, move_s, seed):
    cfg = SimConfig(width=1280, height=720, seed=seed, image_noise_std=2.0,
                    light_flicker=0.08, motion_blur=0.0)
    renderer = _renderer(cfg, depth_mm)
    R_W_B, p0 = _board_pose(depth_mm, tilt_deg, cfg)
    axes = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, -1.0])}
    trials, index = [], seed * 100000
    for row in plan_rows:
        readings = [float(v) for v in row["readings_mm"].split()]
        t, corners, patterns, detected = [], [], [], []
        for ti, s, _pause in _timeline(readings, fps, pause_s, move_s):
            p_W_B = p0 + axes[row["axis"]] * s / 1000.0
            img = renderer.render(np.eye(3), np.zeros(3), R_W_B, p_W_B, index)
            index += 1
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            c, pattern = _find(gray)
            t.append(ti)
            detected.append(c is not None)
            corners.append(c if c is not None else None)
            patterns.append(pattern)
        trials.append({"axis": row["axis"], "trial": int(row["trial"]),
                       "readings_mm": row["readings_mm"], "t": np.array(t),
                       "corners": corners, "patterns": patterns,
                       "detected": np.array(detected)})
    return trials, renderer.K0.copy(), renderer.d0.reshape(-1, 1).copy()


def cached_render(plan_rows, depth_mm, tilt_deg, fps, pause_s, move_s, seed):
    """Rendering dominates the run time; detected corners are cached on disk."""
    import pickle
    key = f"{depth_mm:g}_{tilt_deg:g}_{fps:g}_{pause_s:g}_{move_s:g}_{seed}_{len(plan_rows)}"
    path = DATA / "sim_reference_cache" / f"{key}.pkl"
    if path.is_file():
        with path.open("rb") as f:
            return pickle.load(f)
    out = render_trials(plan_rows, depth_mm, tilt_deg, fps, pause_s, move_s, seed)
    path.parent.mkdir(exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(out, f)
    return out


def _solve(trials, K, dist, pitch_scale):
    out = []
    for tr in trials:
        p = np.zeros((len(tr["t"]), 3), np.float32)
        usable = np.zeros(len(tr["t"]), np.uint8)
        for i, (c, pattern) in enumerate(zip(tr["corners"], tr["patterns"])):
            if c is None:
                continue
            res = _pose(c, pattern, K, dist)
            if res is None:
                continue
            p[i] = res[1] * pitch_scale
            usable[i] = 1
        out.append(p)
        tr["usable_now"] = usable
    return out


def _score(trials, positions, still_mm, min_s):
    """Run the caliper analysis on simulated pose streams in a scratch folder."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "pose_gt_raw").mkdir()
        plan = tmp / "plan.csv"
        with plan.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=["axis", "trial", "readings_mm", "session"])
            w.writeheader()
            for k, (tr, p) in enumerate(zip(trials, positions)):
                name = f"sim_{tr['axis']}{tr['trial']}_{k}"
                np.savez(tmp / "pose_gt_raw" / f"{name}.npz", t=tr["t"], p=p,
                         usable=tr["usable_now"],
                         R=np.zeros((len(p), 3, 3), np.float32),
                         px=np.zeros(len(p), np.float32))
                w.writerow({"axis": tr["axis"], "trial": tr["trial"],
                            "readings_mm": tr["readings_mm"], "session": name})
        saved = rsv.DATA
        rsv.DATA = tmp
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                rep = rsv.analyze_caliper(plan, tmp / "report.json", still_mm, min_s)
        finally:
            rsv.DATA = saved
    s = rep["summary"]
    return {
        "passed": rep["passed"],
        "heldout_rmse_mm": s["heldout_rmse_mm"],
        "heldout_max_mm": s["heldout_max_mm"],
        "repeatability_p95_mm": s["repeatability_p95_mm"],
        "scale_error_percent": {a: v["scale_error_percent"] for a, v in s["axes"].items()},
        "cross_axis_rmse_mm": {a: v["cross_axis_rmse_mm"] for a, v in s["axes"].items()},
    }


def _perturb(K, dist, rng):
    Kp, dp = K.copy(), dist.copy().reshape(-1)
    Kp[0, 0] += rng.normal() * INTRINSIC_STD["fx"]
    Kp[1, 1] += rng.normal() * INTRINSIC_STD["fy"]
    Kp[0, 2] += rng.normal() * INTRINSIC_STD["cx"]
    Kp[1, 2] += rng.normal() * INTRINSIC_STD["cy"]
    dp[:5] += rng.normal(size=5) * np.asarray(INTRINSIC_STD["dist"])
    return Kp, dp.reshape(-1, 1), 1.0 + rng.normal() * PITCH_STD_REL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, default=rsv.CALIPER_PLAN)
    ap.add_argument("--depths-mm", type=float, nargs="+", default=[150.0, 250.0, 350.0])
    ap.add_argument("--tilt-deg", type=float, default=25.0)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--pause-s", type=float, default=5.0)
    ap.add_argument("--move-s", type=float, default=1.5)
    ap.add_argument("--mc", type=int, default=50)
    ap.add_argument("--still-mm", type=float, default=0.2)
    ap.add_argument("--min-s", type=float, default=1.5)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--render-only", action="store_true",
                    help="fill the render cache and exit (for parallel runs)")
    ap.add_argument("--seed-base", type=int, default=11)
    args = ap.parse_args()
    plan_rows = list(csv.DictReader(args.plan.open(newline="", encoding="utf-8-sig")))
    rng = np.random.default_rng(20260926)

    report = {
        "kind": "simulated_caliper_reference_validation",
        "scope": "reference pipeline (render -> corner detection -> PnP -> no-scale-fit "
                 "analysis); not a physical, SI-traceable validation",
        "plan": str(args.plan), "tilt_deg": args.tilt_deg,
        "timing": {"fps": args.fps, "pause_s": args.pause_s, "move_s": args.move_s},
        "intrinsic_std_1sigma": INTRINSIC_STD, "pitch_std_relative": PITCH_STD_REL,
        "thresholds": None, "depths": {},
    }
    rendered = [cached_render(plan_rows, depth, args.tilt_deg, args.fps,
                              args.pause_s, args.move_s, seed=args.seed_base + k)
                for k, depth in enumerate(args.depths_mm)]
    if args.render_only:
        return
    for depth, (trials, K, dist) in zip(args.depths_mm, rendered):
        det = float(np.mean(np.concatenate([tr["detected"] for tr in trials])))
        try:
            nominal = _score(trials, _solve(trials, K, dist, 1.0), args.still_mm, args.min_s)
        except RuntimeError as exc:
            nominal = {"passed": False, "error": str(exc)}
        mc = []
        for _ in range(args.mc):
            Kp, dp, pitch = _perturb(K, dist, rng)
            try:
                mc.append(_score(trials, _solve(trials, Kp, dp, pitch),
                                 args.still_mm, args.min_s))
            except RuntimeError as exc:
                mc.append({"passed": False, "error": str(exc)})
        ok = [m for m in mc if "error" not in m]
        summary_mc = {"draws": len(mc), "scored": len(ok),
                      "pass_rate": float(np.mean([m["passed"] for m in mc])) if mc else None}
        if ok:
            scale = np.array([[m["scale_error_percent"][a] for a in "XYZ"] for m in ok])
            cross = np.array([[m["cross_axis_rmse_mm"][a] for a in "XYZ"] for m in ok])
            rmse = np.array([m["heldout_rmse_mm"] for m in ok])
            summary_mc.update({
                "heldout_rmse_mm_p50_p95": np.quantile(rmse, [0.5, 0.95]).tolist(),
                "scale_error_percent_mean": dict(zip("XYZ", scale.mean(0).tolist())),
                "scale_error_percent_std": dict(zip("XYZ", scale.std(0).tolist())),
                "scale_error_percent_p95_abs": dict(zip("XYZ", np.quantile(np.abs(scale), 0.95, axis=0).tolist())),
                "cross_axis_rmse_mm_p95": dict(zip("XYZ", np.quantile(cross, 0.95, axis=0).tolist())),
            })
        else:
            summary_mc["error"] = mc[0].get("error") if mc else None
        report["depths"][f"{depth:g}"] = {
            "detection_rate": det, "nominal": nominal, "monte_carlo": summary_mc,
        }
        print(json.dumps({f"{depth:g} mm": report["depths"][f"{depth:g}"]},
                         ensure_ascii=False), flush=True)
    report["thresholds"] = {
        "heldout_rmse_mm_max": 1.0, "heldout_max_mm_max": 2.0,
        "axis_scale_error_percent_max_abs": 1.0, "cross_axis_rmse_mm_max": 0.5,
        "repeatability_p95_mm_max": 0.5,
    }
    report["passed_nominal_all_depths"] = all(
        d["nominal"]["passed"] for d in report["depths"].values())
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output),
                      "passed_nominal_all_depths": report["passed_nominal_all_depths"]}, indent=2))


if __name__ == "__main__":
    main()
