#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""See whether the translation left after gyro lock lives in one axis or in the accelerometer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import exp_so3
from ego_capture.sync import load_imu
from tools.disambiguate_steps import _fit_t, _gyro_w, _k, _obj, _score
from tools.train_trajectory import TEST_NAME, VAL_NAME

DATA = ROOT / "datasets"
# Body gyro [x,y,z] -> camera [x,y,z], fit on the moving training sessions.
BODY_TO_CAM = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _window(usb_t, acc, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 1), len(usb_t) - 1)
    i1 = min(max(i1, 1), len(usb_t) - 1)
    if i1 <= i0:
        return None
    v = np.zeros(3)
    p = np.zeros(3)
    for i in range(i0 + 1, i1 + 1):
        dt = float(usb_t[i] - usb_t[i - 1])
        if dt <= 0.0 or dt > 0.05:
            continue
        a = acc[i]
        p = p + v * dt + 0.5 * a * dt * dt
        v = v + a * dt
    return acc[i0:i1 + 1].mean(0), p


def session(name: str):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    obj = _obj()
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    delay = float(gt["delay_usb"][0])
    t = gt["t"] - delay
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * (np.pi / 180.0)
    acc = (usb[:, 0:3] * 9.81) @ BODY_TO_CAM.T
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    gyro = gyro - bg
    gravity = acc[still].mean(0) if int(still.sum()) >= 50 else acc.mean(0)
    acc = acc - gravity
    w_list, dp_list, mean_list, integ_list, px_list = [], [], [], [], []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))])
        if not (0.12 <= float(t[i] - t[j]) <= 0.30):
            continue
        w = _gyro_w(usb_t, gyro, float(t[j]) - 0.04, float(t[i]) - 0.04)
        got = _window(usb_t, acc, float(t[j]) - 0.04, float(t[i]) - 0.04)
        if w is None or got is None:
            continue
        R_j = gt["R"][j]
        R_i = gt["R"][i]
        t_stored = (-R_i.T @ gt["p"][i]) * 1000.0
        rvec, _ = cv2.Rodrigues(R_i.T)
        corners, _ = cv2.projectPoints(obj, rvec, t_stored, K, dist)
        R_lock = R_j @ exp_so3(w)
        t_new, px = _fit_t(obj, corners, R_lock.T, t_stored, K, dist)
        p_new = -R_lock @ (t_new / 1000.0)
        w_list.append(w)
        dp_list.append(R_j.T @ (p_new - gt["p"][j]))
        mean_list.append(got[0])
        integ_list.append(got[1])
        px_list.append(px)
    return {
        "w": np.stack(w_list),
        "dp": np.stack(dp_list),
        "mean": np.stack(mean_list),
        "integ": np.stack(integ_list),
        "px": np.asarray(px_list),
    }


def _fit(src, dst):
    weight = np.clip(np.linalg.norm(dst, axis=1), 1e-6, np.quantile(np.linalg.norm(dst, axis=1), 0.95))
    sw = np.sqrt(weight)[:, None]
    design = np.concatenate([src, np.ones((len(src), 1))], 1) * sw
    coef, *_ = np.linalg.lstsq(design, dst * sw, rcond=None)
    return coef


def _apply(src, coef):
    return np.concatenate([src, np.ones((len(src), 1))], 1) @ coef


def main() -> None:
    index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    built = {}
    for name in names:
        built[name] = session(name)
        pack = built[name]
        m = pack["px"] < 1.5
        dp = pack["dp"][m]
        rms = np.sqrt(np.mean(dp ** 2, axis=0)) * 1000.0
        print(json.dumps({
            name: {
                "n": int(m.sum()),
                "mean_mm": round(float(np.linalg.norm(dp, axis=1).mean() * 1000), 2),
                "axis_rms_mm": [round(float(v), 2) for v in rms],
            }
        }, ensure_ascii=False), flush=True)
    train = [n for n in ("traj_20260923_020132", "traj_20260923_020407")]
    report = {"sessions": {}}
    for kind in ("mean", "integ", "w"):
        src = np.concatenate([built[n][kind][built[n]["px"] < 1.5] for n in train])
        dst = np.concatenate([built[n]["dp"][built[n]["px"] < 1.5] for n in train])
        coef = _fit(src, dst)
        block = {}
        for name in (VAL_NAME, TEST_NAME):
            m = built[name]["px"] < 1.5
            pred = _apply(built[name][kind][m], coef)
            block[name] = _score(pred, built[name]["dp"][m])
        report[kind] = block
        print(json.dumps({kind: block}, ensure_ascii=False), flush=True)
    out = DATA / "traj_run_v6" / "residual_acc.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
