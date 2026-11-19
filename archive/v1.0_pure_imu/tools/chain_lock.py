#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Gyro-locked chessboard trajectory, then predict each step from the previous one.

Rotation is chained from the USB gyro. Translation is whatever still hits the
stored board projection. A step whose reprojection exceeds 1.5 px is put back
on the original pose so a drift does not run away.
"""

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
from tools.disambiguate_steps import _fit_t, _k, _obj, _score
from tools.train_trajectory import TEST_NAME, VAL_NAME

DATA = ROOT / "datasets"
BODY_TO_CAM = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])


def _integrate(usb_t, gyro_cam, t0, t1):
    i0 = int(np.searchsorted(usb_t, t0))
    i1 = int(np.searchsorted(usb_t, t1))
    i0 = min(max(i0, 0), len(usb_t) - 1)
    i1 = min(max(i1, 0), len(usb_t) - 1)
    if i1 <= i0:
        return np.zeros(3)
    acc = np.zeros(3)
    for k in range(i0 + 1, i1 + 1):
        dt = float(usb_t[k] - usb_t[k - 1])
        if dt <= 0.0 or dt > 0.05:
            continue
        acc += gyro_cam[k] * dt
    return acc


def lock_session(name: str):
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    obj = _obj()
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * (np.pi / 180.0)
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    gyro_cam = (gyro - bg) @ BODY_TO_CAM.T
    t = gt["t"]
    R = gt["R"][idx[0]].copy()
    p = gt["p"][idx[0]].copy()
    poses_R = {int(idx[0]): R.copy()}
    poses_p = {int(idx[0]): p.copy()}
    anchored = 0
    px_ok = []
    for n in range(1, len(idx)):
        j = int(idx[n - 1])
        i = int(idx[n])
        R = R @ exp_so3(_integrate(usb_t, gyro_cam, float(t[j]), float(t[i])))
        t_stored = (-gt["R"][i].T @ gt["p"][i]) * 1000.0
        rvec, _ = cv2.Rodrigues(gt["R"][i].T)
        corners, _ = cv2.projectPoints(obj, rvec, t_stored, K, dist)
        t_new, px = _fit_t(obj, corners, R.T, t_stored, K, dist)
        if px > 1.5:
            R = gt["R"][i].copy()
            p = gt["p"][i].copy()
            anchored += 1
        else:
            p = -R @ (t_new / 1000.0)
            px_ok.append(px)
        poses_R[i] = R.copy()
        poses_p[i] = p.copy()
    return {
        "t": t,
        "idx": idx,
        "R": poses_R,
        "p": poses_p,
        "anchored": anchored,
        "px_med": float(np.median(px_ok)) if px_ok else None,
    }


def _pairs(pack):
    t = pack["t"]
    idx = pack["idx"]
    start = {}
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - 0.2)))])
        dt = float(t[i] - t[j])
        if dt < 0.12 or dt > 0.30:
            continue
        if j not in pack["R"] or int(i) not in pack["R"]:
            continue
        start[int(i)] = (j, dt)
    pred, truth = [], []
    for i, (j, dt) in start.items():
        if j not in start:
            continue
        k, dt_prev = start[j]
        R_j = pack["R"][j]
        R_k = pack["R"][k]
        dp_prev = R_k.T @ (pack["p"][j] - pack["p"][k])
        scale = dt / max(dt_prev, 1e-3)
        pred.append(R_j.T @ R_k @ (dp_prev * scale))
        truth.append(R_j.T @ (pack["p"][i] - pack["p"][j]))
    if not pred:
        return None
    return np.stack(pred), np.stack(truth)


def main() -> None:
    index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    held = {}
    train_pred, train_truth = [], []
    moving = {"traj_20260923_020132", "traj_20260923_020407"}
    report = {}
    for name in names:
        pack = lock_session(name)
        got = _pairs(pack)
        if got is None:
            print(json.dumps({name: {"anchored": pack["anchored"]}}), flush=True)
            continue
        pred, truth = got
        held[name] = (pred, truth, pack)
        if name in moving:
            train_pred.append(pred)
            train_truth.append(truth)
        block = _score(pred, truth)
        block["anchored"] = pack["anchored"]
        block["frames"] = int(len(pack["idx"]))
        block["px_med"] = None if pack["px_med"] is None else round(pack["px_med"], 3)
        report[name] = block
        print(json.dumps({name: block}, ensure_ascii=False), flush=True)
    src = np.concatenate(train_pred)
    dst = np.concatenate(train_truth)
    alpha = float(np.sum(src * dst) / max(np.sum(src * src), 1e-12))
    report["alpha"] = round(alpha, 3)
    print(json.dumps({"alpha": report["alpha"]}), flush=True)
    for name in (VAL_NAME, TEST_NAME):
        pred, truth, _pack = held[name]
        report[name]["shrunk"] = _score(pred * alpha, truth)
        print(json.dumps({name + "_shrunk": report[name]["shrunk"]}), flush=True)
    out = DATA / "traj_run_v6" / "chain_coast.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
