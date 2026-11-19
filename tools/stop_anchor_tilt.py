#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop-anchored integration with the tilt pulled back at every stop.

The published anchor rotates specific force with the raw gyro attitude and
subtracts one gravity vector. A bias of a few tenths of a degree per second
turns that attitude by several degrees between 4 s stops, so gravity leaks
into the horizontal acceleration and the double integral runs away. Here the
gyro bias is the mean rate on the detected stops, the attitude is integrated
again from the anchor, and the remaining tilt — the angle between the
integrated down direction and the measured specific force — is taken at each
stop and interpolated across the gaps before the zero-velocity update.

    python tools/stop_anchor_tilt.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.stop_anchor_check as sac
import tools.train_physnet_v3 as v3

HZ = v3.HZ
DT = 1.0 / HZ
W = int(sac.STILL_S * HZ)


def _still_mask(acc, gyro, valid):
    ok = ((np.degrees(np.linalg.norm(gyro, axis=-1)) < sac.W_MAX_DPS)
          & (np.abs(np.linalg.norm(acc, axis=-1) / v3.G0 - 1.0) < sac.ACC_TOL_G)
          & valid)
    from scipy.ndimage import maximum_filter1d, minimum_filter1d
    still = maximum_filter1d(minimum_filter1d(ok.astype(np.uint8), W, axis=1), W, axis=1) > 0
    return still & ok


def _segments(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    bounds = np.concatenate([[0], cuts + 1, [len(idx)]])
    return [(int(idx[a]), int(idx[b - 1]) + 1) for a, b in zip(bounds[:-1], bounds[1:])]


def _integrate(acc, gyro, bias, anchor, end):
    """Specific force in the anchor frame. R maps the current camera frame to the anchor."""
    out = np.zeros((end, 3))
    out[anchor] = acc[anchor]
    R = np.eye(3)
    for t in range(anchor, end - 1):
        R = R @ Rotation.from_rotvec((gyro[t] - bias) * DT).as_matrix()
        out[t + 1] = R @ acc[t + 1]
    R = np.eye(3)
    for t in range(anchor, 0, -1):
        R = Rotation.from_rotvec(-(gyro[t - 1] - bias) * DT).as_matrix() @ R
        out[t - 1] = R @ acc[t - 1]
    return out


def _apply_tilt(a_anchor, segments):
    """Rotate the anchor-frame specific force so every stop agrees on one down direction."""
    if len(segments) < 2:
        return a_anchor
    centers, downs = [], []
    for a, b in segments:
        f = a_anchor[a:b].mean(0)
        n = np.linalg.norm(f)
        if n < 0.5 * v3.G0:
            continue
        centers.append(0.5 * (a + b - 1))
        downs.append(f / n)
    if len(centers) < 2:
        return a_anchor
    centers = np.asarray(centers)
    downs = np.asarray(downs)
    target = downs.mean(0)
    target /= np.linalg.norm(target)
    # Rotation taking each stop's down direction onto the common one.
    axes = np.cross(downs, target)
    sina = np.linalg.norm(axes, axis=1)
    cosa = np.clip(downs @ target, -1.0, 1.0)
    angle = np.arctan2(sina, cosa)
    axes = axes / np.maximum(sina, 1e-12)[:, None]
    rotvec = axes * angle[:, None]
    order = np.argsort(centers)
    centers, rotvec = centers[order], rotvec[order]
    t = np.arange(len(a_anchor))
    corr = np.column_stack([np.interp(t, centers, rotvec[:, c]) for c in range(3)])
    applied = Rotation.from_rotvec(corr).apply(a_anchor)
    return applied


def _displacement(field, valid_row, still_row, tau, anchor):
    end = len(field)
    g = field[still_row].mean(0)
    lin = (field - g) * valid_row[:end, None]
    v = np.cumsum(lin, 0) * DT
    idx = np.flatnonzero(still_row)
    tk = np.arange(end)
    v = v - np.column_stack([np.interp(tk, idx, v[idx, c]) for c in range(3)])
    d = np.cumsum(v, 0) * DT
    b0 = min(int(np.floor(tau)), end - 1)
    frac = float(tau - b0)
    d_b = d[b0] * (1.0 - frac) + d[min(b0 + 1, end - 1)] * frac
    return d_b - d[min(anchor, end - 1)]


def _fit_bias(acc, gyro, still, anchor, end, bias0):
    """Bias whose integrated attitude keeps gravity fixed on the detected stops."""
    bias = bias0.copy()
    step = np.radians(0.2)
    for _ in range(6):
        base = _integrate(acc, gyro, bias, anchor, end)
        ref = np.mean(np.sum((base[still] - base[still].mean(0)) ** 2))
        grad = np.zeros(3)
        for c in range(3):
            trial = bias.copy()
            trial[c] += step
            moved = _integrate(acc, gyro, trial, anchor, end)
            grad[c] = (np.mean(np.sum((moved[still] - moved[still].mean(0)) ** 2)) - ref) / step
        if np.linalg.norm(grad) < 1e-6:
            break
        bias = bias - grad * (np.radians(0.05) / max(np.linalg.norm(grad), 1e-9))
        bias = np.clip(bias, bias0 - np.radians(1.5), bias0 + np.radians(1.5))
    return bias


def tilt_on_published(data, anchor):
    """Same gyro attitude as the published anchor; only the stop-to-stop tilt is removed."""
    import torch
    acc, gyro, quat, valid = data["acc"], data["gyro"], data["quat"], data["valid"] > 0.5
    tau_b = data["tau_b"]
    still = _still_mask(acc, gyro, valid)
    rot = v3.quat_to_mat(torch.from_numpy(quat)).numpy()
    turned = np.einsum("btij,btj->bti", rot, acc)
    out = np.full((len(acc), 3), np.nan)
    span = np.full(len(acc), np.inf)
    for k in range(len(acc)):
        end = min(int(np.ceil(tau_b[k])) + 1, acc.shape[1])
        if anchor >= end:
            continue
        s = still[k, :end]
        if int(s.sum()) < W:
            continue
        field = _apply_tilt(turned[k, :end].copy(), _segments(s))
        out[k] = _displacement(field, valid[k], s, tau_b[k], anchor)
        idx = np.flatnonzero(s)
        span[k] = max(idx[0] - anchor, tau_b[k] - idx[-1], 0.0) / HZ
    return out, span


def corrected_anchor(data, anchor):
    acc, gyro, valid = data["acc"], data["gyro"], data["valid"] > 0.5
    tau_b = data["tau_b"]
    still = _still_mask(acc, gyro, valid)
    out_bias = np.full((len(acc), 3), np.nan)
    out_fit = np.full((len(acc), 3), np.nan)
    span = np.full(len(acc), np.inf)
    for k in range(len(acc)):
        end = min(int(np.ceil(tau_b[k])) + 1, acc.shape[1])
        if anchor >= end:
            continue
        s = still[k, :end]
        if int(s.sum()) < W:
            continue
        bias0 = gyro[k, :end][s].mean(0)
        field = _integrate(acc[k], gyro[k], bias0, anchor, end)
        out_bias[k] = _displacement(field, valid[k], s, tau_b[k], anchor)
        # Only spend the search on windows whose stops already disagree about down.
        downs = []
        for a, b in _segments(s):
            f = field[a:b].mean(0)
            n = np.linalg.norm(f)
            if n > 0.5 * v3.G0:
                downs.append(f / n)
        disagree = 0.0 if len(downs) < 2 else float(np.degrees(np.arccos(np.clip(
            min(d1 @ d2 for d1 in downs for d2 in downs), -1, 1))))
        use = bias0 if disagree < 1.0 else _fit_bias(acc[k], gyro[k], s, anchor, end, bias0)
        field = _integrate(acc[k], gyro[k], use, anchor, end)
        field = _apply_tilt(field, _segments(s))
        out_fit[k] = _displacement(field, valid[k], s, tau_b[k], anchor)
        idx = np.flatnonzero(s)
        span[k] = max(idx[0] - anchor, tau_b[k] - idx[-1], 0.0) / HZ
    return out_bias, out_fit, span


def _err(pred, y):
    return np.linalg.norm(pred - y, axis=1) * 1000.0


SESSIONS = ("traj_20261109_022758", "traj_20261109_023932")   # P1, P2 of the tripod batch
RIGID = "rigid_20261109_023120"
FROZEN = ROOT / "datasets" / "stop_eval_20261109_batch.npz"
OUTPUT = ROOT / "datasets" / "stop_anchor_tilt_20261109.json"


def _one(name: str, frozen: dict) -> dict:
    import tools.eval_stop_protocol_experiment as esp
    length = int(round((1.5 * 3.0 + 4.0) * v3.HZ))
    anchor = int(round(2.0 * v3.HZ))
    rig = json.loads((ROOT / "datasets" / RIGID / "rigidity.json").read_text(encoding="utf-8"))
    today = np.asarray(rig["R_camera_imu"], np.float64)
    v3.BIAS_MODE = "none"
    with esp.acc_at_rest(name), esp.imu_mount(today):
        data = v3.build_group([name], 2.0, 3.0, length)
    base, span = sac.stop_anchor(data, anchor)
    bias_only, _both, _span = corrected_anchor(data, anchor)
    tilt, _ = tilt_on_published(data, anchor)
    y = data["y"]
    # Rotation of the reference over the 3 s target, for the same bins as the
    # switching diagnostic.
    raw = np.load(ROOT / "datasets" / "pose_gt_raw" / f"{name}.npz")
    usable = raw["usable"].astype(bool)
    t, R = raw["t"][usable], raw["R"][usable].astype(float)
    rot = np.full(len(y), np.nan)
    for i, t_a in enumerate(data["t_pair"][:, 0]):
        ia = int(np.argmin(np.abs(t - t_a)))
        ib = int(np.argmin(np.abs(t - t_a - 3.0)))
        if abs(t[ia] - t_a) < 0.15 and abs(t[ib] - t_a - 3.0) < 0.15:
            rot[i] = np.degrees(Rotation.from_matrix(R[ib] @ R[ia].T).magnitude())
    # Frozen gated PhysNet on the same windows, from the scored comparison.
    sel = frozen["session"] == name
    f_t, f_err = frozen["t_a"][sel], frozen["err_physnet_mm"][sel]
    net = np.full(len(y), np.nan)
    order = np.argsort(f_t)
    pos = np.clip(np.searchsorted(f_t[order], data["t_pair"][:, 0]), 0, len(order) - 1)
    hit = np.abs(f_t[order][pos] - data["t_pair"][:, 0]) < 1e-6
    net[hit] = f_err[order][pos][hit]
    # Windows the published rule would switch, and that the comparison scored.
    use = (np.isfinite(base).all(1) & np.isfinite(tilt).all(1) & (span <= 2.0)
           & np.isfinite(rot) & np.isfinite(net))
    zero = _err(np.zeros_like(y), y)
    columns = {"zero": zero, "published_anchor": _err(base, y),
               "stop_gyro_bias": _err(bias_only, y), "tilt_corrected": _err(tilt, y),
               "frozen_physnet": net}

    def stats(mask):
        return {"n": int(mask.sum()),
                **{k: round(float(np.nanmean(v[mask])), 2) for k, v in columns.items()}}

    bins = [{"deg": [lo, hi], **stats(use & (rot >= lo) & (rot < hi))}
            for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 45))]
    return {"session": name, **stats(use), "bins": bins,
            "_arrays": {k: v[use] for k, v in columns.items()} | {"rot": rot[use]}}


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=OUTPUT)
    args = ap.parse_args()
    frozen = dict(np.load(FROZEN, allow_pickle=True))
    rows = [_one(name, frozen) for name in SESSIONS]
    arrays = {k: np.concatenate([r["_arrays"][k] for r in rows]) for k in rows[0]["_arrays"]}
    rot = arrays.pop("rot")

    def stats(mask):
        return {"n": int(mask.sum()),
                **{k: round(float(np.nanmean(v[mask])), 2) for k, v in arrays.items()}}

    pooled = {**stats(np.ones(len(rot), bool)),
              "bins": [{"deg": [lo, hi], **stats((rot >= lo) & (rot < hi))}
                       for lo, hi in ((0, 3), (3, 8), (8, 15), (15, 45))]}
    for r in rows:
        r.pop("_arrays")
    report = {
        "note": "Exploratory and not pre-specified. Mean 3-s error in mm on the windows of the paused "
                "tripod recordings that the published rule switches to stop anchoring (S <= 2 s). "
                "tilt_corrected rotates the anchor-frame specific force so that every detected stop "
                "agrees on one gravity direction, interpolating the correction between stops. "
                "stop_gyro_bias instead removes the mean gyroscope rate of the detected stops.",
        "rigid": RIGID, "pooled": pooled, "per_session": rows,
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
