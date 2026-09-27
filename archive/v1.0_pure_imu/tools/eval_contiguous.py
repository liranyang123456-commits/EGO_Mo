#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Contiguous trajectory error on the held-out session.

Overlapping 0.2 s windows are not added on top of each other. A step is kept
only when it starts after the previous step ends.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import MotionTrajectoryCorrector
from tools.train_trajectory import (
    TEST_NAME,
    VAL_NAME,
    _correct,
    _loader,
    _rotate_labels,
    build_pairs,
    integrate_aligned,
    load_split,
)

GT = ROOT / "datasets" / "pose_gt"
CKPT = ROOT / "datasets" / "traj_run_v3" / "corrector.pt"


def _fit(train_packs, device):
    positions = []
    rotations = []
    cam_dp = []
    cam_dR = []
    with torch.no_grad():
        for pack in train_packs:
            acc = torch.from_numpy(pack["acc"]).to(device)
            gyro = torch.from_numpy(pack["gyro"]).to(device)
            dR, dp = integrate_aligned(
                acc, gyro,
                torch.tensor(pack["bg"], device=device),
                torch.tensor(pack["ba"], device=device),
                torch.tensor(pack["R0"], device=device),
            )
            rotations.append(dR.cpu().numpy())
            positions.append(dp.cpu().numpy())
            cam_dp.append(pack["dp"])
            cam_dR.append(pack["dR"])
    imu_dp = np.concatenate(positions)
    design = np.concatenate([imu_dp, np.ones((len(imu_dp), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, np.concatenate(cam_dp), rcond=None)
    return torch.tensor(coef, dtype=torch.float32, device=device)


def _predict(net, pack, coef, device):
    loader = _loader(pack, False)
    bg = torch.tensor(pack["bg"], device=device)
    ba = torch.tensor(pack["ba"], device=device)
    R0 = torch.tensor(pack["R0"], device=device)
    pred_p, base_p, pred_R = [], [], []
    net.eval()
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]
            usb, bt, usb_t, bt_t, cam, cam_m, cam_t, _dR, _dp, _state, acc, gyro = batch
            dRs, dps = integrate_aligned(acc, gyro, bg, ba, R0)
            _logits, xi, _log_var, _h = net(usb, bt, cam, None, cam_m, None, usb_t, bt_t, cam_t, None)
            dRc, dpc, dplin = _correct(dRs, dps, xi, coef)
            pred_p.append(dpc.cpu().numpy())
            base_p.append(dplin.cpu().numpy())
            pred_R.append(dRc.cpu().numpy())
    return np.concatenate(pred_p), np.concatenate(base_p), np.concatenate(pred_R)


def _keep(t_start, t_end) -> np.ndarray:
    chosen = []
    last = -1e9
    for i in np.argsort(t_end):
        if t_start[i] < last - 0.02:
            continue
        chosen.append(int(i))
        last = float(t_end[i])
    return np.asarray(chosen, dtype=np.int64)


def _chain(dp, dR, index) -> np.ndarray:
    R = np.eye(3)
    p = np.zeros(3)
    out = []
    for i in index:
        p = p + R @ dp[i]
        R = R @ dR[i]
        out.append(p.copy())
    return np.stack(out)


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))) * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = load_split(names)
    built = {name: build_pairs(packs[name]) for name in names}
    train_names = [name for name in names if name not in (TEST_NAME, VAL_NAME)]
    train_packs = [built[name] for name in train_names]
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    for pack in built.values():
        _rotate_labels(pack, ckpt["R_ci"])
    coef = _fit(train_packs, device)
    net = MotionTrajectoryCorrector().to(device)
    net.load_state_dict(ckpt["state_dict"])
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        pack = built[name]
        pred, base, dR = _predict(net, pack, coef, device)
        gt = pack["dp"]
        step_net = float(np.linalg.norm(pred - gt, axis=1).mean() * 1000.0)
        step_lin = float(np.linalg.norm(base - gt, axis=1).mean() * 1000.0)
        keep = _keep(pack["t_start"], pack["t_end"])
        # Gyro integration rotation, expressed like the label, is pack["dR"] after the extrinsic fit.
        path_g = _chain(gt, pack["dR"], keep)
        path_n = _chain(pred, pack["dR"], keep)
        path_b = _chain(base, pack["dR"], keep)
        report[name] = {
            "steps": int(len(gt)),
            "contiguous": int(len(keep)),
            "step_net_mm": round(step_net, 2),
            "step_linear_mm": round(step_lin, 2),
            "path_net_mm": round(_rmse(path_n, path_g), 1),
            "path_linear_mm": round(_rmse(path_b, path_g), 1),
            "duration_s": round(float(pack["t_end"][keep[-1]] - pack["t_start"][keep[0]]), 1),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "contiguous_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
