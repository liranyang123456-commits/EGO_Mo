#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera translation as rotation about a fixed point in the camera frame.

dp = (I - dR) @ r. If the scope swings about the IMU, r is the IMU's
position in the camera frame and the accelerometer never has to see the step.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_trajectory import TEST_NAME, VAL_NAME

GT = ROOT / "datasets" / "pose_gt"


def _pairs(name: str):
    gt = np.load(GT / f"{name}.npz")
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    t = gt["t"] - float(gt["delay_usb"][0])
    R = gt["R"]
    p = gt["p"]
    dR, dp, ang = [], [], []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))]
        dt = float(t[i] - t[j])
        if dt < 0.12 or dt > 0.30:
            continue
        rel = R[j].T @ R[i]
        dR.append(rel)
        dp.append(R[j].T @ (p[i] - p[j]))
        c = float(np.clip((np.trace(rel) - 1.0) * 0.5, -1.0, 1.0))
        ang.append(np.degrees(np.arccos(c)))
    return np.stack(dR), np.stack(dp), np.asarray(ang)


def _design(dR: np.ndarray) -> np.ndarray:
    eye = np.eye(3)
    return np.stack([eye - rel for rel in dR])


def _fit(dR, dp):
    A = _design(dR).reshape(-1, 3)
    r, *_ = np.linalg.lstsq(A, dp.reshape(-1), rcond=None)
    return r


def _apply(dR, r):
    return np.einsum("nij,j->ni", _design(dR), r)


def _score(pred, dp, ang) -> dict:
    err = np.linalg.norm(pred - dp, axis=1)
    truth = np.linalg.norm(dp, axis=1)
    fast = truth >= np.quantile(truth, 0.67)
    resid = np.sum((pred - dp) ** 2)
    base = np.sum((dp - dp.mean(0)) ** 2)
    return {
        "n": int(len(err)),
        "zero_mm": round(float(truth.mean() * 1000.0), 2),
        "err_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "pred_mm": round(float(np.median(np.linalg.norm(pred, axis=1)) * 1000.0), 2),
        "r2": round(float(1.0 - resid / max(base, 1e-12)), 3),
        "rot_corr": round(float(np.corrcoef(ang, truth)[0, 1]), 3) if len(ang) > 5 else None,
    }


def _half(dR, dp):
    mid = len(dp) // 2
    r = _fit(dR[:mid], dp[:mid])
    return _score(_apply(dR[mid:], r), dp[mid:], np.zeros(len(dp) - mid)), r


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = {name: _pairs(name) for name in names}
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    dR = np.concatenate([packs[n][0] for n in train])
    dp = np.concatenate([packs[n][1] for n in train])
    r = _fit(dR, dp)
    report = {"r_mm": [round(float(v) * 1000.0, 1) for v in r], "sessions": {}}
    print(json.dumps({"r_mm": report["r_mm"]}, ensure_ascii=False), flush=True)
    for name in names:
        pred = _apply(packs[name][0], r)
        block = _score(pred, packs[name][1], packs[name][2])
        half, r_half = _half(packs[name][0], packs[name][1])
        block["half_mm"] = half["err_mm"]
        block["half_zero_mm"] = half["zero_mm"]
        block["half_r_mm"] = [round(float(v) * 1000.0, 1) for v in r_half]
        report["sessions"][name] = block
        print(json.dumps({name: block}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v6" / "lever_arm.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
