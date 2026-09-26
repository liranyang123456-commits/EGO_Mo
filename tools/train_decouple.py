#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train absolute camera/board decomposition on independent-board simulation.

Sequences are generated in memory (no images). Each window is anchored on the
camera pose at its first sample. The visual token is the relative pose at
that instant, dropped out so the two IMUs have to assign the later motion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.mapping.decouple import DecoupledEgomotion, decomposition_loss
from ego_capture.mapping.so3 import R_to_rot6d
from ego_sim import SimConfig, build_sequence

WINDOW = 400
KNOTS = 5
STRIDE = 200


def _windows(seed: int, duration: float, board_mm: float, board_deg: float) -> list[dict]:
    config = SimConfig(
        duration_s=duration,
        camera_fps=10.0,
        imu_hz=200.0,
        seed=seed,
        motion="random_spline",
        translation_mm=40.0,
        depth_mm=180.0,
        depth_change_mm=30.0,
        rotation_deg=12.0,
        tracking_gain=0.05,
        speed=1.4,
        noise_scale=1.0,
        board_motion=True,
        board_motion_mode="independent",
        board_motion_mm=board_mm,
        board_rotation_deg=board_deg,
        render_images=False,
        gravity_board=(0.0, 0.87, 0.49),
    )
    sequence = build_sequence(config)
    poses = sequence.poses
    usb = sequence.imu_usb[:, :19].astype(np.float32)
    ble = sequence.imu_bt[:, :19].astype(np.float32)
    knots = np.linspace(0, WINDOW - 1, KNOTS).astype(np.int64)
    rows = []
    for start in range(0, len(poses.t) - WINDOW, STRIDE):
        idx = start + knots
        visual_R = poses.R_W_C[start].T @ poses.R_W_B[start]
        visual_p = poses.R_W_C[start].T @ (poses.p_W_B[start] - poses.p_W_C[start])
        visual = np.concatenate([
            R_to_rot6d(torch.from_numpy(visual_R[None].astype(np.float32)))[0].numpy(),
            visual_p.astype(np.float32),
        ])
        rows.append({
            "usb": usb[start:start + WINDOW],
            "ble": ble[start:start + WINDOW],
            "visual": visual.astype(np.float32),
            "R_c": poses.R_W_C[idx].astype(np.float32),
            "p_c": poses.p_W_C[idx].astype(np.float32),
            "R_b": poses.R_W_B[idx].astype(np.float32),
            "p_b": poses.p_W_B[idx].astype(np.float32),
        })
    return rows


class WindowSet(Dataset):
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        return {key: torch.from_numpy(value) for key, value in row.items()}


def _batch_to(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) for key, value in batch.items()}


def _run_epoch(model, loader, device, optimizer, supervise: str, dropout_visual: float) -> dict:
    train = optimizer is not None
    model.train(train)
    totals = []
    for batch in loader:
        batch = _batch_to(batch, device)
        mask = torch.ones(batch["usb"].shape[0], dtype=torch.bool, device=device)
        if train and dropout_visual > 0:
            mask = torch.rand(mask.shape, device=device) > dropout_visual
        pred = model(batch["usb"], batch["ble"], batch["visual"], mask)
        losses = decomposition_loss(
            pred, batch["R_c"], batch["p_c"], batch["R_b"], batch["p_b"], supervise=supervise,
        )
        if train:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        totals.append({key: float(value.detach()) for key, value in losses.items()})
    keys = totals[0].keys()
    return {key: float(np.mean([row[key] for row in totals])) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[101, 102, 103, 104])
    parser.add_argument("--val-seeds", type=int, nargs="+", default=[201, 202])
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--board-mm", type=float, default=40.0)
    parser.add_argument("--board-deg", type=float, default=20.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--supervise", choices=("absolute", "relative"), default="absolute")
    parser.add_argument("--visual-dropout", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=ROOT / "datasets" / "decouple_smoke")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_rows = []
    for seed in args.train_seeds:
        train_rows.extend(_windows(seed, args.duration, args.board_mm, args.board_deg))
    val_rows = []
    for seed in args.val_seeds:
        val_rows.extend(_windows(seed, args.duration, args.board_mm, args.board_deg))
    train_loader = DataLoader(WindowSet(train_rows), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(WindowSet(val_rows), batch_size=args.batch)
    model = DecoupledEgomotion(knots=KNOTS).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        train_metrics = _run_epoch(model, train_loader, device, optimizer, args.supervise, args.visual_dropout)
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, device, None, args.supervise, 0.0)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            f"epoch {epoch} train_loss {train_metrics['loss']:.4f} "
            f"val_pos_c_mm {val_metrics['pos_c'] * 1000:.1f} "
            f"val_pos_b_mm {val_metrics['pos_b'] * 1000:.1f} "
            f"val_pos_rel_mm {val_metrics['pos_rel'] * 1000:.1f}",
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "supervise": args.supervise, "knots": KNOTS}, args.out / "model.pt")
    (args.out / "history.json").write_text(json.dumps({
        "windows_train": len(train_rows),
        "windows_val": len(val_rows),
        "board_mm": args.board_mm,
        "board_deg": args.board_deg,
        "history": history,
    }, indent=2), encoding="utf-8")
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
