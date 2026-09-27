#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pose stream by chaining the long-context 0.2 s IMU model.

Between two gyro-locked poses the interval is cut into ~0.2 s pieces. Each
piece is predicted by the five-seed ensemble from the IMU around it, then
rotated into the start camera frame with the USB gyro and summed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
from tools.train_seq import HZ, SeqNet, _grid
from tools.train_stream import BUCKETS, load_session
from tools.train_trajectory import TEST_NAME, VAL_NAME

DATA = ROOT / "datasets"
CONTEXT = 0.5
SEEDS = range(5)
PER_BUCKET = 400


def _models(device):
    nets = []
    for s in SEEDS:
        net = SeqNet().to(device)
        net.load_state_dict(torch.load(DATA / "traj_run_v7" / f"seq_ctx{CONTEXT:g}_s{s}.pt", map_location=device))
        net.eval()
        nets.append(net)
    return nets


def _predict(nets, X, device):
    with torch.no_grad():
        x = torch.from_numpy(X).to(device)
        return np.mean([net(x).cpu().numpy() for net in nets], 0) * 0.01


def run(name: str, nets, device, rng) -> dict:
    sess = load_session(name)
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    grid0 = usb_t[0]
    t_lab = grid0 + sess["k"] / HZ
    length = int(round((0.30 + 2 * CONTEXT) * HZ))
    report = {}
    for lo, hi in BUCKETS:
        a_idx, b_idx = np.meshgrid(np.arange(len(t_lab)), np.arange(len(t_lab)), indexing="ij")
        gap = t_lab[b_idx] - t_lab[a_idx]
        m = (gap >= lo) & (gap <= hi)
        pairs = np.stack([a_idx[m], b_idx[m]], 1)
        pairs = pairs[(t_lab[pairs[:, 0]] - CONTEXT > usb_t[0]) & (t_lab[pairs[:, 1]] + CONTEXT < usb_t[-1])]
        if len(pairs) > PER_BUCKET:
            pairs = pairs[rng.choice(len(pairs), PER_BUCKET, replace=False)]
        if len(pairs) < 5:
            continue
        pred_all, truth_all = [], []
        for a, b in pairs:
            ta, tb = float(t_lab[a]), float(t_lab[b])
            n = max(1, int(round((tb - ta) / 0.2)))
            edges = np.linspace(ta, tb, n + 1)
            X = []
            for t0, t1 in zip(edges[:-1], edges[1:]):
                feat = _grid(usb_t, usb, t0, t1, CONTEXT)
                if len(feat) < length:
                    feat = np.vstack([feat, np.zeros((length - len(feat), 8), np.float32)])
                X.append(feat[:length])
            steps = _predict(nets, np.stack(X), device)
            ka = int(sess["k"][a])
            Ca = sess["C"][ka].astype(np.float64)
            total = np.zeros(3)
            for t0, dp in zip(edges[:-1], steps):
                k0 = int(round((t0 - grid0) * HZ))
                k0 = min(max(k0, 0), len(sess["C"]) - 1)
                total += Ca.T @ sess["C"][k0].astype(np.float64) @ dp
            truth = sess["R"][a].T @ (sess["p"][b] - sess["p"][a])
            pred_all.append(total)
            truth_all.append(truth)
        pred = np.stack(pred_all)
        y = np.stack(truth_all)
        err = np.linalg.norm(pred - y, axis=1)
        mag = np.linalg.norm(y, axis=1)
        fast = mag >= np.quantile(mag, 0.67)
        resid = np.sum((pred - y) ** 2)
        base = np.sum((y - y.mean(0)) ** 2)
        report[f"{lo:.2f}-{hi:.2f}s"] = {
            "n": int(len(y)),
            "zero_mm": round(float(mag.mean() * 1000), 2),
            "err_mm": round(float(err.mean() * 1000), 2),
            "fast_zero_mm": round(float(mag[fast].mean() * 1000), 2),
            "fast_mm": round(float(err[fast].mean() * 1000), 2),
            "r2": round(float(1 - resid / max(base, 1e-12)), 3),
        }
        print(json.dumps({name: {f"{lo:.2f}-{hi:.2f}s": report[f"{lo:.2f}-{hi:.2f}s"]}}), flush=True)
    return report


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    nets = _models(device)
    report = {n: run(n, nets, device, rng) for n in (VAL_NAME, TEST_NAME)}
    out = DATA / "traj_run_v8" / "chain_steps.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
