#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export gyro-locked chessboard poses for later egomotion ground truth.

A frame is usable when the board is still, the stored solve is already under
1.5 px, and locking rotation to the USB gyro still hits that projection
within 1 px. Frames that had to fall back to the raw solve are left out.
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
from tools.chain_lock import BODY_TO_CAM, _integrate
from tools.disambiguate_steps import _fit_t, _k, _obj

DATA = ROOT / "datasets"
OUT = DATA / "pose_gt_locked"
PX_MAX = 1.0


def export_session(name: str, shift: float = 0.0, out_dir: Path = OUT) -> dict:
    """`shift` is added to the per-session camera delay, for sensitivity runs."""
    gt = np.load(DATA / "pose_gt" / f"{name}.npz")
    n = len(gt["t"])
    t_imu = gt["t"] - float(gt["delay_usb"][0]) - shift
    K, dist = _k(int(gt["image_size"][0]), int(gt["image_size"][1]))
    obj = _obj()
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    gyro = usb[:, 3:6] * (np.pi / 180.0)
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    gyro_cam = (gyro - bg) @ BODY_TO_CAM.T
    R_out = np.tile(np.eye(3), (n, 1, 1))
    p_out = np.zeros((n, 3))
    px = np.full(n, np.nan)
    usable = np.zeros(n, dtype=np.uint8)
    if len(idx) == 0:
        return {"name": name, "frames": int(n), "usable": 0}
    i0 = int(idx[0])
    R = gt["R"][i0].copy()
    p = gt["p"][i0].copy()
    R_out[i0] = R
    p_out[i0] = p
    px[i0] = float(gt["reproj"][i0])
    usable[i0] = int(px[i0] <= PX_MAX)
    for n_i in range(1, len(idx)):
        j = int(idx[n_i - 1])
        i = int(idx[n_i])
        R_try = R @ exp_so3(_integrate(usb_t, gyro_cam, float(t_imu[j]), float(t_imu[i])))
        t_stored = (-gt["R"][i].T @ gt["p"][i]) * 1000.0
        rvec, _ = cv2.Rodrigues(gt["R"][i].T)
        corners, _ = cv2.projectPoints(obj, rvec, t_stored, K, dist)
        t_new, err = _fit_t(obj, corners, R_try.T, t_stored, K, dist)
        px[i] = err
        if err <= PX_MAX:
            R = R_try
            p = -R @ (t_new / 1000.0)
            usable[i] = 1
        elif err <= 1.5:
            R = R_try
            p = -R @ (t_new / 1000.0)
        else:
            R = gt["R"][i].copy()
            p = gt["p"][i].copy()
        R_out[i] = R
        p_out[i] = p
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{name}.npz",
        t=gt["t"],
        R=R_out.astype(np.float32),
        p=p_out.astype(np.float32),
        px=px.astype(np.float32),
        usable=usable,
        raw_reproj=gt["reproj"].astype(np.float32),
    )
    return {
        "name": name,
        "frames": int(n),
        "board_ok": int(ok.sum()),
        "usable": int(usable.sum()),
        "px_med": round(float(np.nanmedian(px[usable == 1])), 3) if int(usable.sum()) else None,
    }


def main() -> None:
    index = json.loads((DATA / "pose_gt" / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    rows = [export_session(name) for name in names]
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (OUT / "index.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
