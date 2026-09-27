#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the image+IMU 6-DoF pose net on datasets/vi_pose."""

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

from ego_capture.mapping.visual_pose import VisualInertialPose, pose_loss

DATA = ROOT / "datasets" / "vi_pose"


class PoseWindows(Dataset):
    def __init__(self, group: str):
        images, imus, rotations, positions = [], [], [], []
        groups = ("train", "extra_train") if group == "train" else (group,)
        for folder in groups:
            for path in sorted((DATA / folder).glob("*.npz")):
                blob = np.load(path)
                images.append(blob["image"])
                imus.append(blob["imu"])
                rotations.append(blob["R"])
                positions.append(blob["p"])
        self.image = np.concatenate(images)
        self.imu = np.concatenate(imus)
        self.R = np.concatenate(rotations)
        self.p = np.concatenate(positions)

    def __len__(self) -> int:
        return len(self.image)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = torch.from_numpy(self.image[index]).float().unsqueeze(0) / 255.0
        return {
            "image": image,
            "imu": torch.from_numpy(self.imu[index]),
            "R": torch.from_numpy(self.R[index]),
            "p": torch.from_numpy(self.p[index]),
        }


def _epoch(model, loader, device, optimizer) -> dict[str, float]:
    model.train(optimizer is not None)
    rows = []
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        pred = model(batch["image"], batch["imu"])
        losses = pose_loss(pred, batch["R"], batch["p"])
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            optimizer.step()
        rows.append({key: float(value.detach()) for key, value in losses.items()})
    return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--out", type=Path, default=DATA / "model")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(PoseWindows("train"), batch_size=args.batch, shuffle=True)
    val_set = PoseWindows("val")
    val_loader = DataLoader(val_set, batch_size=args.batch)
    model = VisualInertialPose().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        train_metrics = _epoch(model, train_loader, device, optimizer)
        with torch.no_grad():
            val_metrics = _epoch(model, val_loader, device, None)
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        print(
            f"epoch {epoch} train {train_metrics['loss']:.3f} "
            f"val_rot_deg {np.degrees(val_metrics['rot']):.2f} "
            f"val_pos_mm {val_metrics['pos'] * 1000:.1f}",
            flush=True,
        )
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.out / "visual_inertial_pose.pt")
    (args.out / "history.json").write_text(json.dumps({
        "train": len(train_loader.dataset),
        "val": len(val_set),
        "history": history,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
