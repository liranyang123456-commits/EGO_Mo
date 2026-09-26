#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Is the IMU acceleration consistent with the camera motion it should measure?

For every recording, the low-frequency camera acceleration from the chessboard
reference (cubic spline through 10-fps PnP positions, 0.3-2 Hz band) is
compared with the IMU specific force rotated into the board frame by the
reference attitude and R_CI, after removing gravity (the band-pass removes
it). Per-axis correlation is scanned over a time shift of the IMU stream; a
rigid, correctly synchronized recording peaks near zero shift with high
correlation, a loose mount or a timing error does not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation, Slerp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.train_seq as api
from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
HZ = 100.0


def analyse(name, R_ci, shifts=np.arange(-0.20, 0.201, 0.01)):
    raw = np.load(DATA / "pose_gt_raw" / f"{name}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{name}.npz")["delay_usb"][0])
    use = raw["usable"] == 1
    tc = raw["t"][use] - delay
    p = raw["p"][use].astype(np.float64)
    R = raw["R"][use].astype(np.float64)
    ti, usb = load_imu(DATA / name / "imu_stream.csv")
    f_i = usb[:, 0:3] * 9.80665
    sos = butter(3, [0.3, 2.0], btype="band", fs=HZ, output="sos")
    gaps = np.flatnonzero(np.diff(tc) > 0.35)
    spans = np.split(np.arange(len(tc)), gaps + 1)
    best = None
    table = []
    for sh in shifts:
        A, B = [], []
        for idx in spans:
            if len(idx) < 40:
                continue
            g = np.arange(tc[idx[0]] + 1.0, tc[idx[-1]] - 1.0, 1.0 / HZ)
            if len(g) < 300:
                continue
            acc_ref = CubicSpline(tc[idx], p[idx], axis=0)(g, 2)
            Rg = Slerp(tc[idx], Rotation.from_matrix(R[idx]))(g).as_matrix()
            f = np.column_stack([np.interp(g + sh, ti, f_i[:, k]) for k in range(3)])
            acc_imu = np.einsum("nij,nj->ni", Rg, f @ R_ci.T)   # board-frame specific force
            A.append(sosfiltfilt(sos, acc_ref, axis=0)); B.append(sosfiltfilt(sos, acc_imu, axis=0))
        if not A:
            return None
        A, B = np.concatenate(A), np.concatenate(B)
        c = [float(np.corrcoef(A[:, k], B[:, k])[0, 1]) for k in range(3)]
        gain = [float(np.dot(A[:, k], B[:, k]) / np.dot(B[:, k], B[:, k])) for k in range(3)]
        table.append((sh, c, gain))
        if best is None or np.mean(c) > np.mean(best[1]):
            best = (sh, c, gain)
    zero = min(table, key=lambda r: abs(r[0]))
    return {"best_shift_s": best[0], "best_corr": best[1], "gain_at_best": best[2],
            "corr_at_zero_shift": zero[1], "stored_delay_s": delay}


def main():
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    names = split["train"] + split["val"] + split["test"] + split["extra_train"]
    new = api.R_CAMERA_IMU_NEW.astype(np.float64)
    out = {}
    for n in names:
        r = analyse(n, new)
        if r is None:
            print(n, "no long spans"); continue
        out[n] = r
        print(f"{n:24s} corr@0 {np.round(r['corr_at_zero_shift'], 2)}  best shift {r['best_shift_s']:+.2f} s "
              f"corr {np.round(r['best_corr'], 2)}  gain {np.round(r['gain_at_best'], 2)}", flush=True)
    (DATA / "accel_consistency.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
