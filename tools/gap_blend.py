#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keep the better gap direction and the gap-map length."""

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


def _angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-6 or nb < 1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


def _blend(direction, length):
    n = np.linalg.norm(direction)
    if n < 1e-8:
        return length
    return direction / n * np.linalg.norm(length)


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
    cached = {name: _gaps(name, raw[name], device) for name in names}
    gap_x = np.concatenate([cached[n][0] for n in train if cached[n] is not None])
    gap_y = np.concatenate([cached[n][1] for n in train if cached[n] is not None])
    gap_coef, *_ = np.linalg.lstsq(gap_x, gap_y, rcond=None)
    for name in (VAL_NAME, TEST_NAME):
        src, dst, gap_i, times, rotations, positions = cached[name]
        new = src @ gap_coef
        old = np.concatenate([src[:, :3], np.ones((len(src), 1))], 1) @ step_coef
        blended = np.stack([_blend(o, n) for o, n in zip(old, new)])
        def stats(pred):
            ang, err = [], []
            for p, t in zip(pred, dst):
                a = _angle(p, t)
                if a is None:
                    continue
                ang.append(a)
                err.append(np.linalg.norm(p - t) * 1000.0)
            return round(float(np.median(ang)), 1), round(float(np.median(err)), 1)
        edges = []
        pred_step = np.concatenate([windows[name], np.ones((len(windows[name]), 1))], 1) @ step_coef
        a = _nearest_direct(times, built[name]["t_start"])
        b = _nearest_direct(times, built[name]["t_end"])
        for k in range(len(pred_step)):
            if abs(times[a[k]] - built[name]["t_start"][k]) > 0.03 or abs(times[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            edges.append((int(a[k]), int(b[k]), rotations[a[k]] @ pred_step[k]))
        for i, body in zip(gap_i, blended):
            edges.append((int(i), int(i) + 1, rotations[i] @ body))
        path = _solve(len(times), edges)
        path = path - path[0] + positions[0]
        old_ang, old_err = stats(old)
        new_ang, new_err = stats(new)
        blend_ang, blend_err = stats(blended)
        report = {
            "old_angle_deg": old_ang,
            "new_angle_deg": new_ang,
            "blend_angle_deg": blend_ang,
            "old_err_mm": old_err,
            "new_err_mm": new_err,
            "blend_err_mm": blend_err,
            "fused_mm": round(float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0), 1),
        }
        print(json.dumps({name: report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
