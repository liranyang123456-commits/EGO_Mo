#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rotate the horizontal gap displacement around gravity by a train-only angle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _solve
from tools.gap_map import _gaps, _nearest_direct
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"


def _horiz(v, up):
    up = up / max(np.linalg.norm(up), 1e-8)
    return v - up * np.dot(v, up)


def _signed(a, b, up):
    up = up / max(np.linalg.norm(up), 1e-8)
    ha, hb = _horiz(a, up), _horiz(b, up)
    if np.linalg.norm(ha) < 1e-8 or np.linalg.norm(hb) < 1e-8:
        return None
    ha, hb = ha / np.linalg.norm(ha), hb / np.linalg.norm(hb)
    return float(np.degrees(np.arctan2(np.dot(up, np.cross(ha, hb)), np.dot(ha, hb))))


def _rot_about(up, deg):
    up = up / max(np.linalg.norm(up), 1e-8)
    th = np.radians(deg)
    k = np.array([[0, -up[2], up[1]], [up[2], 0, -up[0]], [-up[1], up[0], 0]])
    return np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built = {name: build_pairs(raw[name]) for name in names}
    windows = {}
    for name in names:
        acc = torch.from_numpy(built[name]["acc"]).to(device)
        gyro = torch.from_numpy(built[name]["gyro"]).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(built[name]["bg"], device=device),
                torch.tensor(built[name]["ba"], device=device),
                torch.tensor(built[name]["R0"], device=device),
            )
        windows[name] = dp.cpu().numpy()
    up = {name: raw[name]["R0"].T @ np.array([0.0, 0.0, 1.0]) for name in names}
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    cached = {name: _gaps(name, raw[name], device) for name in names}
    angles = []
    for name in train:
        src, dst, *_r = cached[name]
        for dp, truth in zip(src[:, :3], dst):
            ang = _signed(dp, truth, up[name])
            if ang is not None:
                angles.append(ang)
    yaw = float(np.median(angles))
    report = {"train_yaw_deg": round(yaw, 1)}
    src_w = np.concatenate([windows[n] for n in train])
    dst_w = np.concatenate([built[n]["dp"] for n in train])
    step_coef, *_ = np.linalg.lstsq(np.concatenate([src_w, np.ones((len(src_w), 1))], 1), dst_w, rcond=None)
    for name in (VAL_NAME, TEST_NAME):
        src, dst, gap_i, times, rotations, positions = cached[name]
        R = _rot_about(up[name], yaw)
        pred = (src[:, :3] @ R.T)
        pred = np.concatenate([pred, src[:, 3:4], np.ones((len(pred), 1))], 1)
        # Length still comes from the short-step shrink, applied after the yaw.
        shrink = np.concatenate([src[:, :3], np.ones((len(src), 1))], 1) @ step_coef
        turned = np.stack([
            R @ s if np.linalg.norm(s) > 1e-8 else s
            for s in shrink
        ])
        err = np.linalg.norm(turned - dst, axis=1)
        step = np.concatenate([windows[name], np.ones((len(windows[name]), 1))], 1) @ step_coef
        edges = []
        a = _nearest_direct(times, built[name]["t_start"])
        b = _nearest_direct(times, built[name]["t_end"])
        for k in range(len(step)):
            if abs(times[a[k]] - built[name]["t_start"][k]) > 0.03 or abs(times[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            edges.append((int(a[k]), int(b[k]), rotations[a[k]] @ step[k]))
        for i, body in zip(gap_i, turned):
            edges.append((int(i), int(i) + 1, rotations[i] @ body))
        path = _solve(len(times), edges)
        path = path - path[0] + positions[0]
        report[name] = {
            "gap_err_mm_p50": round(float(np.median(err) * 1000.0), 1),
            "fused_mm": round(float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0), 1),
        }
        print(json.dumps({name: report[name], "yaw": report["train_yaw_deg"]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "yaw_gap.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _nearest_direct(times, query):
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = idx - 1
    choose_left = np.abs(times[left] - query) <= np.abs(times[idx] - query)
    return np.where(choose_left, left, idx)


if __name__ == "__main__":
    main()
