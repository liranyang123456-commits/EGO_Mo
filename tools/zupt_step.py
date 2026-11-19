#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.2 s steps with gravity removed in the current attitude.

The previous tilt feedback rotated away from the accelerometer, so every
window integrated about 2 g. Velocity was also zeroed during gentle
translation, which looks the same as rest. This run corrects the tilt sign,
and keeps a leaky velocity so a coast of one or two seconds still counts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import SPECIFIC_FORCE_UP, exp_so3
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split

GT = ROOT / "datasets" / "pose_gt"
G = np.array([0.0, 0.0, -9.81])
UP = SPECIFIC_FORCE_UP
TAUS = (1.0, 2.0)


def _propagate(usb_t, usb, ba, R0):
    n = len(usb_t)
    specific = usb[:, 0:3] * 9.81
    gyro_dps = usb[:, 3:6]
    R = np.zeros((n, 3, 3))
    R[0] = R0
    p = {tau: np.zeros((n, 3)) for tau in TAUS}
    v = {tau: np.zeros(3) for tau in TAUS}
    lp = {tau: np.zeros(3) for tau in TAUS}
    grav_err = []
    t0 = float(usb_t[0])
    for i in range(1, n):
        dt = float(usb_t[i] - usb_t[i - 1])
        if dt <= 0.0 or dt > 0.05:
            R[i] = R[i - 1]
            for tau in TAUS:
                p[tau][i] = p[tau][i - 1]
            continue
        w = gyro_dps[i] * (np.pi / 180.0)
        R[i] = R[i - 1] @ exp_so3(w * dt)
        g_body = R[i].T @ UP
        if np.linalg.norm(gyro_dps[i]) < 5.0 and abs(np.linalg.norm(specific[i]) - 9.81) < 0.8:
            # Rotate predicted gravity toward the measured specific force.
            delta = np.cross(specific[i], g_body) / 96.2
            R[i] = R[i] @ exp_so3(np.clip(delta, -0.02, 0.02))
            g_body = R[i].T @ UP
            grav_err.append(float(np.linalg.norm(g_body - specific[i])))
        a = R[i] @ (specific[i] - ba) + G
        warm = float(usb_t[i]) - t0 < 0.5
        for tau in TAUS:
            if warm:
                lp[tau] = a
                v[tau][:] = 0.0
                p[tau][i] = p[tau][i - 1]
                continue
            lp[tau] = lp[tau] + (dt / tau) * (a - lp[tau])
            hp = a - lp[tau]
            v[tau] = (v[tau] + hp * dt) * np.exp(-dt / tau)
            p[tau][i] = p[tau][i - 1] + v[tau] * dt
    return R, p, float(np.median(grav_err)) if grav_err else None


def _step(R, p, usb_t, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 0), len(usb_t) - 1)
    i1 = min(max(i1, 0), len(usb_t) - 1)
    if i1 <= i0:
        return None
    return R[i0].T @ (p[i1] - p[i0])


def _local(R, usb_t, usb, ba, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 1), len(usb_t) - 1)
    i1 = min(max(i1, 1), len(usb_t) - 1)
    if i1 <= i0:
        return None
    specific = usb[:, 0:3] * 9.81
    v = np.zeros(3)
    p = np.zeros(3)
    for i in range(i0 + 1, i1 + 1):
        dt = float(usb_t[i] - usb_t[i - 1])
        if dt <= 0.0 or dt > 0.05:
            continue
        a = R[i] @ (specific[i] - ba) + G
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
    return R[i0].T @ p


def _angles(pred, truth):
    out = []
    for a, b in zip(pred, truth):
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 0.002 or nb < 0.002:
            continue
        out.append(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))
    return out


def _fit_and_score(src_train, dst_train, src, dst):
    design = np.concatenate([src_train, np.ones((len(src_train), 1))], 1)
    coef, *_ = np.linalg.lstsq(design, dst_train, rcond=None)
    pred = np.concatenate([src, np.ones((len(src), 1))], 1) @ coef
    zero = float(np.linalg.norm(dst, axis=1).mean() * 1000.0)
    step = float(np.linalg.norm(pred - dst, axis=1).mean() * 1000.0)
    speed = np.linalg.norm(dst, axis=1)
    fast = speed >= np.quantile(speed, 0.67)
    resid = np.sum((pred - dst) ** 2)
    base = np.sum((dst - dst.mean(0)) ** 2)
    return {
        "zero_mm": round(zero, 2),
        "mapped_mm": round(step, 2),
        "fast_mm": round(float(np.linalg.norm(pred[fast] - dst[fast], axis=1).mean() * 1000.0), 2),
        "pred_mm": round(float(np.median(np.linalg.norm(pred, axis=1)) * 1000.0), 2),
        "truth_mm": round(float(np.median(speed) * 1000.0), 2),
        "r2": round(float(1.0 - resid / max(base, 1e-12)), 3),
        "singular": [round(float(v), 3) for v in np.linalg.svd(coef[:3], compute_uv=False)],
    }


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {name: build_pairs(raw[name]) for name in names}
    feat = {tau: {} for tau in TAUS}
    feat["local"] = {}
    report = {"gravity_mmps2": {}}
    for name in names:
        pack = raw[name]
        R, p, grav = _propagate(pack["usb_t"], pack["usb"], pack["ba"], pack["R0"])
        report["gravity_mmps2"][name] = None if grav is None else round(grav * 1000.0, 1)
        rows = {tau: [] for tau in TAUS}
        rows["local"] = []
        for t0, t1 in zip(built[name]["t_start"], built[name]["t_end"]):
            for tau in TAUS:
                d = _step(R, p[tau], pack["usb_t"], float(t0), float(t1))
                rows[tau].append(np.zeros(3) if d is None else d)
            d = _local(R, pack["usb_t"], pack["usb"], pack["ba"], float(t0), float(t1))
            rows["local"].append(np.zeros(3) if d is None else d)
        truth = built[name]["dp"]
        summary = {"gravity_mmps2": report["gravity_mmps2"][name]}
        for key in list(TAUS) + ["local"]:
            feat[key][name] = np.stack(rows[key])
            ang = _angles(feat[key][name], truth)
            summary[str(key)] = {
                "pred_mm": round(float(np.median(np.linalg.norm(feat[key][name], axis=1)) * 1000.0), 2),
                "deg": round(float(np.median(ang)), 1) if ang else None,
            }
        summary["truth_mm"] = round(float(np.median(np.linalg.norm(truth, axis=1)) * 1000.0), 2)
        print(json.dumps({name: summary}, ensure_ascii=False), flush=True)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    for key in list(TAUS) + ["local"]:
        src = np.concatenate([feat[key][n] for n in train])
        dst = np.concatenate([built[n]["dp"] for n in train])
        block = {}
        for name in (VAL_NAME, TEST_NAME):
            block[name] = _fit_and_score(src, dst, feat[key][name], built[name]["dp"])
            print(json.dumps({str(key): {name: block[name]}}, ensure_ascii=False), flush=True)
        report[str(key)] = block
    out = ROOT / "datasets" / "traj_run_v6"
    out.mkdir(parents=True, exist_ok=True)
    (out / "zupt_step.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
