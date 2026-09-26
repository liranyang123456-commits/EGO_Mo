#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Anchor a body-frame displacement bias on the first 8 s, score the rest."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import MotionTrajectoryCorrector
from tools.eval_contiguous import _fit, _predict
from tools.fuse_path import _nearest, _nodes, _solve
from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, load_split

GT = ROOT / "datasets" / "pose_gt"
ANCHOR_S = 8.0


def _edges(pack, pred, times, rotations, positions):
    a = _nearest(times, pack["t_start"])
    b = _nearest(times, pack["t_end"])
    rows = []
    for k in range(len(pred)):
        if abs(times[a[k]] - pack["t_start"][k]) > 0.03:
            continue
        if abs(times[b[k]] - pack["t_end"][k]) > 0.03:
            continue
        if a[k] >= b[k]:
            continue
        truth = rotations[a[k]].T @ (positions[b[k]] - positions[a[k]])
        rows.append((int(a[k]), int(b[k]), pred[k], truth, float(pack["t_end"][k])))
    return rows


def _score(rows, bias, times, rotations, positions):
    if not rows:
        return None
    step = []
    edges = []
    for a, b, pred, truth, _t in rows:
        corrected = pred - bias
        step.append(np.linalg.norm(corrected - truth))
        edges.append((a, b, rotations[a] @ corrected))
    path = _solve(len(times), edges)
    path = path - path[0] + positions[0]
    used = np.zeros(len(times), dtype=bool)
    for a, b, _meas in edges:
        used[a] = True
        used[b] = True
    err = path[used] - positions[used]
    rmse = float(np.sqrt(np.mean(np.sum(err ** 2, axis=1))) * 1000.0)
    return {
        "edges": len(rows),
        "step_mm": round(float(np.mean(step) * 1000.0), 2),
        "fused_mm": round(rmse, 1),
    }


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
        rows = _edges(built[name], base, times, rotations, positions)
        t0 = float(times[0])
        anchor = [row for row in rows if row[4] <= t0 + ANCHOR_S]
        rest = [row for row in rows if row[4] > t0 + ANCHOR_S]
        if anchor:
            bias = np.mean([pred - truth for _a, _b, pred, truth, _t in anchor], axis=0)
        else:
            bias = np.zeros(3)
        report[name] = {
            "bias_mm": [round(float(v) * 1000.0, 2) for v in bias],
            "anchor_edges": len(anchor),
            "rest_raw": _score(rest, np.zeros(3), times, rotations, positions),
            "rest_anchored": _score(rest, bias, times, rotations, positions),
        }
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)
    out = ROOT / "datasets" / "traj_run_v3" / "anchor_path.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
