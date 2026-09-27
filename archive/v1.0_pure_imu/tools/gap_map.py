#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fit a displacement map on board-loss gaps from the training sessions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _nearest, _nodes, _solve
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"


def _poses(name):
    data = np.load(GT / f"{name}.npz")
    delay = float(data["delay_usb"][0])
    ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
    return data["t"][ok] - delay, data["R"][ok], data["p"][ok]


def _integrate(usb_t, usb, t0, t1, bg, ba, R0, device):
    sel = np.flatnonzero((usb_t >= t0) & (usb_t <= t1))
    if len(sel) < 8:
        return None
    acc = torch.from_numpy(usb[sel, 0:3] * 9.81).to(device).unsqueeze(0)
    gyro = torch.from_numpy(usb[sel, 3:6] * (np.pi / 180.0)).to(device).unsqueeze(0)
    with torch.no_grad():
        _dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
    return dp.cpu().numpy()[0]


def _gaps(name, raw, device):
    times, rotations, positions = _poses(name)
    bg = torch.tensor(raw["bg"], device=device)
    ba = torch.tensor(raw["ba"], device=device)
    R0 = torch.tensor(raw["R0"], device=device)
    src, dst, index = [], [], []
    for i in range(len(times) - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0.30 or dt > 2.5:
            continue
        raw_dp = _integrate(raw["usb_t"], raw["usb"], float(times[i]), float(times[i + 1]), bg, ba, R0, device)
        if raw_dp is None:
            continue
        truth = rotations[i].T @ (positions[i + 1] - positions[i])
        src.append(np.concatenate([raw_dp, [dt, 1.0]]))
        dst.append(truth)
        index.append(i)
    if not src:
        return None
    return np.stack(src), np.stack(dst), np.asarray(index), times, rotations, positions


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
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src_w = np.concatenate([windows[n] for n in train])
    dst_w = np.concatenate([built[n]["dp"] for n in train])
    step_coef, *_ = np.linalg.lstsq(np.concatenate([src_w, np.ones((len(src_w), 1))], 1), dst_w, rcond=None)
    gap_src, gap_dst = [], []
    cached = {}
    for name in names:
        got = _gaps(name, raw[name], device)
        cached[name] = got
        if name in train and got is not None:
            gap_src.append(got[0])
            gap_dst.append(got[1])
    gap_x = np.concatenate(gap_src)
    gap_y = np.concatenate(gap_dst)
    gap_coef, *_ = np.linalg.lstsq(gap_x, gap_y, rcond=None)
    report = {"train_gaps": int(len(gap_y))}
    for name in (VAL_NAME, TEST_NAME):
        src, dst, gap_i, times, rotations, positions = cached[name]
        pred_gap = src @ gap_coef
        old = src[:, :3]
        old_pred = np.concatenate([old, np.ones((len(old), 1))], 1) @ step_coef
        scale_new, scale_old, err_new = [], [], []
        for pred, prev, truth in zip(pred_gap, old_pred, dst):
            if np.linalg.norm(truth) < 0.01:
                continue
            scale_new.append(np.linalg.norm(pred) / np.linalg.norm(truth))
            scale_old.append(np.linalg.norm(prev) / np.linalg.norm(truth))
            err_new.append(np.linalg.norm(pred - truth) * 1000.0)
        pred_step = np.concatenate([windows[name], np.ones((len(windows[name]), 1))], 1) @ step_coef
        step = float(np.linalg.norm(pred_step - built[name]["dp"], axis=1).mean() * 1000.0)
        node_t, node_R, node_p = _nodes_direct(times, rotations, positions)
        a = _nearest_direct(node_t, built[name]["t_start"])
        b = _nearest_direct(node_t, built[name]["t_end"])
        edges = []
        for k in range(len(pred_step)):
            if abs(node_t[a[k]] - built[name]["t_start"][k]) > 0.03 or abs(node_t[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            edges.append((int(a[k]), int(b[k]), node_R[a[k]] @ pred_step[k]))
        for i, body in zip(gap_i, pred_gap):
            edges.append((int(i), int(i) + 1, node_R[i] @ body))
        path = _solve(len(node_t), edges)
        path = path - path[0] + node_p[0]
        report[name] = {
            "step_mm": round(step, 2),
            "gaps": int(len(dst)),
            "old_scale_p50": round(float(np.median(scale_old)), 2) if scale_old else None,
            "new_scale_p50": round(float(np.median(scale_new)), 2) if scale_new else None,
            "gap_err_mm_p50": round(float(np.median(err_new)), 1) if err_new else None,
            "fused_mm": round(float(np.sqrt(np.mean(np.sum((path - node_p) ** 2, axis=1))) * 1000.0), 1),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "gap_map.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _nodes_direct(times, rotations, positions):
    return times, rotations, positions


def _nearest_direct(times, query):
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = idx - 1
    choose_left = np.abs(times[left] - query) <= np.abs(times[idx] - query)
    return np.where(choose_left, left, idx)


if __name__ == "__main__":
    main()
