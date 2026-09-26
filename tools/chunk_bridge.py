#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bridge board-loss gaps with chained 0.2 s IMU steps."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.fuse_path import _solve
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

GT = ROOT / "datasets" / "pose_gt"


def _chunk(usb_t, usb, t0, t1, bg, ba, R0, device):
    sel = np.flatnonzero((usb_t >= t0) & (usb_t <= t1))
    if len(sel) < 4:
        return None, None
    acc = torch.from_numpy(usb[sel, 0:3] * 9.81).to(device).unsqueeze(0)
    gyro = torch.from_numpy(usb[sel, 3:6] * (np.pi / 180.0)).to(device).unsqueeze(0)
    with torch.no_grad():
        dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
    return dR.cpu().numpy()[0], dp.cpu().numpy()[0]


def _reachable(n, edges):
    seen = {0}
    changed = True
    while changed:
        changed = False
        for a, b, _m in edges:
            if a in seen and b not in seen:
                seen.add(b)
                changed = True
            elif b in seen and a not in seen:
                seen.add(a)
                changed = True
    return seen


def _rmse(path, positions):
    return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    raw = load_split(names)
    built, imu_dp = {}, {}
    for name in names:
        built[name] = build_pairs(raw[name])
        acc = torch.from_numpy(built[name]["acc"]).to(device)
        gyro = torch.from_numpy(built[name]["gyro"]).to(device)
        with torch.no_grad():
            _dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(built[name]["bg"], device=device),
                torch.tensor(built[name]["ba"], device=device),
                torch.tensor(built[name]["R0"], device=device),
            )
        imu_dp[name] = dp.cpu().numpy()
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    src = np.concatenate([imu_dp[n] for n in train])
    dst = np.concatenate([built[n]["dp"] for n in train])
    coef, *_ = np.linalg.lstsq(np.concatenate([src, np.ones((len(src), 1))], 1), dst, rcond=None)

    report = {}
    for name in (VAL_NAME, TEST_NAME):
        data = np.load(GT / f"{name}.npz")
        delay = float(data["delay_usb"][0])
        ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
        times = data["t"][ok] - delay
        rotations = data["R"][ok]
        positions = data["p"][ok]
        pred = np.concatenate([imu_dp[name], np.ones((len(imu_dp[name]), 1))], 1) @ coef
        step = float(np.linalg.norm(pred - built[name]["dp"], axis=1).mean() * 1000.0)
        edges = []
        pack = built[name]
        order_t = times
        a_idx = np.searchsorted(order_t, pack["t_start"])
        for k in range(len(pred)):
            a = int(np.clip(a_idx[k], 1, len(times) - 1))
            if abs(times[a - 1] - pack["t_start"][k]) < abs(times[a] - pack["t_start"][k]):
                a -= 1
            b = int(np.clip(np.searchsorted(times, pack["t_end"][k]), 1, len(times) - 1))
            if abs(times[b - 1] - pack["t_end"][k]) < abs(times[b] - pack["t_end"][k]):
                b -= 1
            if abs(times[a] - pack["t_start"][k]) > 0.03 or abs(times[b] - pack["t_end"][k]) > 0.03 or a >= b:
                continue
            edges.append((a, b, rotations[a] @ pred[k]))
        seen = _reachable(len(times), edges)
        plain = _solve(len(times), edges)
        plain = plain - plain[0] + positions[0]
        bg = torch.tensor(built[name]["bg"], device=device)
        ba = torch.tensor(built[name]["ba"], device=device)
        R0 = torch.tensor(built[name]["R0"], device=device)
        usb_t = raw[name]["usb_t"]
        usb = raw[name]["usb"]
        bridges = 0
        for i in range(len(times) - 1):
            if float(times[i + 1] - times[i]) <= 0.30:
                continue
            t_cursor = float(times[i])
            t_stop = float(times[i + 1])
            R_rel = np.eye(3)
            p_delta = np.zeros(3)
            ok_gap = False
            while t_cursor < t_stop - 0.02:
                t_next = min(t_cursor + 0.2, t_stop)
                sel = np.flatnonzero((usb_t >= t_cursor) & (usb_t <= t_next))
                if len(sel) < 4:
                    t_cursor = t_next
                    continue
                acc = torch.from_numpy(usb[sel, 0:3] * 9.81).to(device).unsqueeze(0)
                gyro = torch.from_numpy(usb[sel, 3:6] * (np.pi / 180.0)).to(device).unsqueeze(0)
                with torch.no_grad():
                    dR, dp = integrate_aligned(acc, gyro, bg, ba, R0)
                dp = dp.cpu().numpy()[0]
                dR = dR.cpu().numpy()[0]
                mapped = np.concatenate([dp, [1.0]]) @ coef
                p_delta = p_delta + R_rel @ mapped
                R_rel = R_rel @ dR
                ok_gap = True
                t_cursor = t_next
            if ok_gap:
                edges.append((i, i + 1, rotations[i] @ p_delta))
                bridges += 1
        bridged = _solve(len(times), edges)
        bridged = bridged - bridged[0] + positions[0]
        lost = ~np.isin(np.arange(len(times)), list(seen))
        report = {
            "step_mm": round(step, 2),
            "disconnected": int(lost.sum()),
            "fused_mm": round(float(np.sqrt(np.mean(np.sum((plain - positions) ** 2, axis=1))) * 1000.0), 1),
            "bridged_mm": round(float(np.sqrt(np.mean(np.sum((bridged - positions) ** 2, axis=1))) * 1000.0), 1),
            "bridges": bridges,
        }
        if lost.any():
            report["disconnected_rmse_mm"] = round(float(np.sqrt(np.mean(np.sum((plain[lost] - positions[lost]) ** 2, axis=1))) * 1000.0), 1)
        print(json.dumps({name: report}, ensure_ascii=False), flush=True)
    # step was computed before the loop variable reuse; recompute is already inside. 
    out = ROOT / "datasets" / "traj_run_v3" / "chunk_bridge.json"
    # The prints are the record. Also store the last name only if we collect.
    _ = out


if __name__ == "__main__":
    main()
