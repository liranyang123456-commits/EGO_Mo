#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How much of the 0.2 s position error is a constant frame map."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.train_trajectory import TEST_NAME, VAL_NAME, build_pairs, integrate_aligned, load_split

DATA = ROOT / "datasets" / "pose_gt"


def displacements(pack: dict, device) -> tuple[np.ndarray, np.ndarray]:
    built = build_pairs(pack)
    acc = torch.from_numpy(built["acc"]).to(device)
    gyro = torch.from_numpy(built["gyro"]).to(device)
    with torch.no_grad():
        _dR, dp = integrate_aligned(
            acc, gyro,
            torch.tensor(built["bg"], device=device),
            torch.tensor(built["ba"], device=device),
            torch.tensor(built["R0"], device=device),
        )
    return dp.cpu().numpy(), built["dp"]


def fit_map(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    design = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return coef


def apply_map(coef: np.ndarray, src: np.ndarray) -> np.ndarray:
    design = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    return design @ coef


def mm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean() * 1000.0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    index = json.loads((DATA / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    packs = load_split(names)
    train_names = [name for name in names if name not in (TEST_NAME, VAL_NAME)]
    srcs, dsts = [], []
    for name in train_names:
        src, dst = displacements(packs[name], device)
        srcs.append(src)
        dsts.append(dst)
        print(name, "strap", round(mm(src, dst), 1), "n", len(src), flush=True)
    coef = fit_map(np.concatenate(srcs), np.concatenate(dsts))
    for name in (VAL_NAME, TEST_NAME):
        src, dst = displacements(packs[name], device)
        pred = apply_map(coef, src)
        print(json.dumps({
            "name": name,
            "strap_mm": round(mm(src, dst), 2),
            "linear_mm": round(mm(pred, dst), 2),
        }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
