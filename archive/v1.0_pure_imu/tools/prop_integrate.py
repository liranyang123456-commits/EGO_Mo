#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrate with attitude propagated from the start of the session."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import exp_so3
from tools.fuse_path import _nearest, _nodes, _solve
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split

GT = ROOT / "datasets" / "pose_gt"
G = np.array([0.0, 0.0, -9.81])


def _propagate(usb_t, usb, bg, ba, R0):
    n = len(usb_t)
    R = np.zeros((n, 3, 3))
    p = np.zeros((n, 3))
    R[0] = R0
    v = np.zeros(3)
    for i in range(1, n):
        dt = float(usb_t[i] - usb_t[i - 1])
        if dt <= 0.0 or dt > 0.05:
            R[i] = R[i - 1]
            p[i] = p[i - 1]
            continue
        w = (usb[i, 3:6] * (np.pi / 180.0) - bg) * dt
        R[i] = R[i - 1] @ exp_so3(w)
        f = usb[i, 0:3] * 9.81 - ba
        a = R[i] @ f + G
        p[i] = p[i - 1] + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
    return R, p


def _body_dp(R, p, usb_t, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 0), len(usb_t) - 1)
    i1 = min(max(i1, 0), len(usb_t) - 1)
    if i1 <= i0:
        return None
    return R[i0].T @ (p[i1] - p[i0])


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {name: build_pairs(raw[name]) for name in names}
    traj = {}
    for name in names:
        traj[name] = _propagate(
            raw[name]["usb_t"], raw[name]["usb"],
            raw[name]["bg"], raw[name]["ba"], raw[name]["R0"],
        )
    feat = {}
    for name in names:
        R, p = traj[name]
        usb_t = raw[name]["usb_t"]
        cols = []
        for t0, t1 in zip(built[name]["t_start"], built[name]["t_end"]):
            dp = _body_dp(R, p, usb_t, float(t0), float(t1))
            cols.append(np.zeros(3) if dp is None else dp)
        feat[name] = np.stack(cols)
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([feat[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)
    report = {"singular": [round(float(v), 3) for v in np.linalg.svd(coef[:3], compute_uv=False)]}
    for name in (VAL_NAME, TEST_NAME):
        pred = np.concatenate([feat[name], np.ones((len(feat[name]), 1))], 1) @ coef
        step = float(np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0)
        times, rotations, positions = _nodes_ok(name)
        a = _nearest(times, built[name]["t_start"])
        b = _nearest(times, built[name]["t_end"])
        edges = []
        for k in range(len(pred)):
            if abs(times[a[k]] - built[name]["t_start"][k]) > 0.03 or abs(times[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            edges.append((int(a[k]), int(b[k]), rotations[a[k]] @ pred[k]))
        # Gaps use the same propagated trajectory.
        R, p = traj[name]
        usb_t = raw[name]["usb_t"]
        angles = []
        for i in range(len(times) - 1):
            if float(times[i + 1] - times[i]) <= 0.30:
                continue
            raw_dp = _body_dp(R, p, usb_t, float(times[i]), float(times[i + 1]))
            if raw_dp is None:
                continue
            mapped = np.concatenate([raw_dp, [1.0]]) @ coef
            truth = rotations[i].T @ (positions[i + 1] - positions[i])
            ang = _angle(mapped, truth)
            if ang is not None:
                angles.append(ang)
            edges.append((i, i + 1, rotations[i] @ mapped))
        path = _solve(len(times), edges)
        path = path - path[0] + positions[0]
        report[name] = {
            "step_mm": round(step, 2),
            "gap_angle_deg": round(float(np.median(angles)), 1) if angles else None,
            "fused_mm": round(float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0), 1),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    print(json.dumps({"singular": report["singular"]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "propagated.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _nodes_ok(name):
    data = np.load(GT / f"{name}.npz")
    delay = float(data["delay_usb"][0])
    ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
    return data["t"][ok] - delay, data["R"][ok], data["p"][ok]


def _nearest(times, query):
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = idx - 1
    choose_left = np.abs(times[left] - query) <= np.abs(times[idx] - query)
    return np.where(choose_left, left, idx)


if __name__ == "__main__":
    main()
