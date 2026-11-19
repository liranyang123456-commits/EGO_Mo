#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Does the USB gyro track the camera rotation, at the right time?

The 0.2 s chessboard turn is about 1 deg. If the gyro sees it, a linear map
from that rotation to the camera step is the lever arm. The shift is chosen
on the training sessions only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
from tools.train_trajectory import TEST_NAME, VAL_NAME

GT = ROOT / "datasets" / "pose_gt"
DATA = ROOT / "datasets"
SHIFTS = np.linspace(-0.20, 0.20, 21)


def _so3_log(R: np.ndarray) -> np.ndarray:
    cos = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(cos))
    if th < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return w * (th / (2.0 * np.sin(th)))


def _load(name: str):
    gt = np.load(GT / f"{name}.npz")
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    delay = float(gt["delay_usb"][0])
    t = gt["t"] - delay
    pairs = []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))]
        dt = float(t[i] - t[j])
        if dt < 0.12 or dt > 0.30:
            continue
        rel = gt["R"][j].T @ gt["R"][i]
        dp = gt["R"][j].T @ (gt["p"][i] - gt["p"][j])
        pairs.append((float(t[j]), float(t[i]), _so3_log(rel), dp))
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * (np.pi / 180.0)
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else gyro.mean(0)
    dt = np.diff(usb_t, prepend=usb_t[0])
    dt[0] = 0.0
    dt[(dt <= 0.0) | (dt > 0.05)] = 0.0
    inc = np.cumsum((gyro - bg) * dt[:, None], axis=0)
    euler = np.unwrap(usb[:, 6:9] * (np.pi / 180.0), axis=0)
    return pairs, usb_t, inc, euler


def _at(cum, usb_t, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 0), len(usb_t) - 1)
    i1 = min(max(i1, 0), len(usb_t) - 1)
    if i1 <= i0:
        return None
    return cum[i1] - cum[i0]


def _corr(a, b) -> float:
    if len(a) < 8 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _collect(pack, shift: float, source: str):
    pairs, usb_t, inc, euler = pack
    cum = inc if source == "gyro" else euler
    w_imu, w_cam, dp = [], [], []
    for t0, t1, wc, step in pairs:
        got = _at(cum, usb_t, t0 + shift, t1 + shift)
        if got is None:
            continue
        w_imu.append(got)
        w_cam.append(wc)
        dp.append(step)
    return np.stack(w_imu), np.stack(w_cam), np.stack(dp)


def _score(pred, dp) -> dict:
    err = np.linalg.norm(pred - dp, axis=1)
    truth = np.linalg.norm(dp, axis=1)
    fast = truth >= np.quantile(truth, 0.67)
    resid = np.sum((pred - dp) ** 2)
    base = np.sum((dp - dp.mean(0)) ** 2)
    return {
        "zero_mm": round(float(truth.mean() * 1000.0), 2),
        "err_mm": round(float(err.mean() * 1000.0), 2),
        "fast_mm": round(float(err[fast].mean() * 1000.0), 2),
        "r2": round(float(1.0 - resid / max(base, 1e-12)), 3),
    }


def _fit(src, dst):
    design = np.concatenate([src, np.ones((len(src), 1))], 1)
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return coef


def main() -> None:
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = {name: _load(name) for name in names}
    train = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    report = {"shifts": []}
    best = None
    for source in ("gyro", "euler"):
        for shift in SHIFTS:
            corrs = []
            for name in train:
                w_imu, w_cam, dp = _collect(packs[name], float(shift), source)
                corrs.append(_corr(np.linalg.norm(w_imu, axis=1), np.linalg.norm(w_cam, axis=1)))
            mean_corr = float(np.mean(corrs))
            row = {"source": source, "shift_ms": round(float(shift) * 1000.0, 1), "train_corr": round(mean_corr, 3)}
            report["shifts"].append(row)
            if best is None or mean_corr > best[0]:
                best = (mean_corr, source, float(shift))
    print(json.dumps({"best": {"corr": round(best[0], 3), "source": best[1], "shift_ms": round(best[2] * 1000, 1)}}, ensure_ascii=False), flush=True)
    source, shift = best[1], best[2]
    built = {name: _collect(packs[name], shift, source) for name in names}
    src = np.concatenate([built[n][0] for n in train])
    dst = np.concatenate([built[n][2] for n in train])
    coef = _fit(src, dst)
    # Also the map from the camera's own rotation, as the ceiling at this pair set.
    src_c = np.concatenate([built[n][1] for n in train])
    coef_c = _fit(src_c, dst)
    for name in (VAL_NAME, TEST_NAME):
        w_imu, w_cam, dp = built[name]
        pred = np.concatenate([w_imu, np.ones((len(w_imu), 1))], 1) @ coef
        pred_c = np.concatenate([w_cam, np.ones((len(w_cam), 1))], 1) @ coef_c
        block = {
            "imu": _score(pred, dp),
            "chessboard_rot": _score(pred_c, dp),
            "mag_corr": round(_corr(np.linalg.norm(w_imu, axis=1), np.linalg.norm(w_cam, axis=1)), 3),
            "rot_med_deg": round(float(np.median(np.linalg.norm(w_cam, axis=1)) * 180.0 / np.pi), 2),
            "imu_med_deg": round(float(np.median(np.linalg.norm(w_imu, axis=1)) * 180.0 / np.pi), 2),
        }
        print(json.dumps({name: block}, ensure_ascii=False), flush=True)
        report[name] = block
    out = ROOT / "datasets" / "traj_run_v6" / "gyro_sync.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
