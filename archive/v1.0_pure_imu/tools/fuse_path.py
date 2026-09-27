#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse overlapping 0.2 s displacements into one path. Step error is unchanged."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.eval_contiguous import _fit, _predict
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split
from ego_capture.mapping.inertial import MotionTrajectoryCorrector

GT = ROOT / "datasets" / "pose_gt"


def _nodes(session: str):
    data = np.load(GT / f"{session}.npz")
    delay = float(data["delay_usb"][0])
    ok = (data["ok"] == 1) & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
    t = data["t"][ok] - delay
    return t, data["R"][ok], data["p"][ok]


def _nearest(times: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(times, query)
    idx = np.clip(idx, 1, len(times) - 1)
    left = idx - 1
    choose_left = np.abs(times[left] - query) <= np.abs(times[idx] - query)
    return np.where(choose_left, left, idx)


def _solve(n: int, edges: list[tuple[int, int, np.ndarray]]) -> np.ndarray:
    dim = (n - 1) * 3
    ata = np.zeros((dim, dim))
    atb = np.zeros(dim)
    for a, b, meas in edges:
        # p_b - p_a = meas, p_0 fixed at 0
        if a == 0 and b == 0:
            continue
        row_b = None if b == 0 else (b - 1) * 3
        row_a = None if a == 0 else (a - 1) * 3
        if row_b is not None:
            ata[row_b:row_b + 3, row_b:row_b + 3] += np.eye(3)
            atb[row_b:row_b + 3] += meas
        if row_a is not None:
            ata[row_a:row_a + 3, row_a:row_a + 3] += np.eye(3)
            atb[row_a:row_a + 3] -= meas
        if row_a is not None and row_b is not None:
            ata[row_a:row_a + 3, row_b:row_b + 3] -= np.eye(3)
            ata[row_b:row_b + 3, row_a:row_a + 3] -= np.eye(3)
    delta = np.linalg.solve(ata + np.eye(dim) * 1e-8, atb)
    path = np.zeros((n, 3))
    path[1:] = delta.reshape(n - 1, 3)
    return path


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = load_split(names)
    built = {name: build_pairs(packs[name]) for name in names}
    train_names = [name for name in names if name not in (TEST_NAME, VAL_NAME)]
    coef = _fit([built[name] for name in train_names], device)
    net = MotionTrajectoryCorrector().to(device)
    ckpt = torch.load(ROOT / "datasets" / "traj_run_v3" / "corrector.pt", map_location=device, weights_only=False)
    net.load_state_dict(ckpt["state_dict"])
    report = {}
    for name in (VAL_NAME, TEST_NAME):
        _pred, base, _dR = _predict(net, built[name], coef, device)
        times, rotations, positions = _nodes(name)
        a = _nearest(times, built[name]["t_start"])
        b = _nearest(times, built[name]["t_end"])
        edges_net, edges_lin = [], []
        step_net, step_lin = [], []
        for k in range(len(base)):
            if abs(times[a[k]] - built[name]["t_start"][k]) > 0.03:
                continue
            if abs(times[b[k]] - built[name]["t_end"][k]) > 0.03:
                continue
            if a[k] >= b[k]:
                continue
            meas_n = rotations[a[k]] @ _pred[k]
            meas_l = rotations[a[k]] @ base[k]
            truth = positions[b[k]] - positions[a[k]]
            step_net.append(np.linalg.norm(meas_n - truth))
            step_lin.append(np.linalg.norm(meas_l - truth))
            edges_net.append((int(a[k]), int(b[k]), meas_n))
            edges_lin.append((int(a[k]), int(b[k]), meas_l))
        path_n = _solve(len(times), edges_net)
        path_l = _solve(len(times), edges_lin)
        # Put both paths in the same board frame as the first pose.
        path_n = path_n - path_n[0] + positions[0]
        path_l = path_l - path_l[0] + positions[0]
        def rmse(path):
            return float(np.sqrt(np.mean(np.sum((path - positions) ** 2, axis=1))) * 1000.0)
        report[name] = {
            "edges": len(edges_lin),
            "step_net_mm": round(float(np.mean(step_net) * 1000.0), 2),
            "step_linear_mm": round(float(np.mean(step_lin) * 1000.0), 2),
            "fused_net_mm": round(rmse(path_n), 1),
            "fused_linear_mm": round(rmse(path_l), 1),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "fused_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
