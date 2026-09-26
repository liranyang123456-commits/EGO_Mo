#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera image plus USB IMU to a 6-DoF pose in the chessboard frame.

The board is the world while it stays fixed, so this pose is the camera
egomotion. The image is an input, not only a label. PnP remains the label.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .inertial import geodesic_per
from .so3 import rot6d_to_R


class VisualInertialPose(nn.Module):
    """image (B,1,H,W) in [0,1], imu (B,T,6) acc m/s^2 and gyro rad/s."""

    def __init__(self, imu_len: int = 200):
        super().__init__()
        self.imu_len = imu_len
        self.vision = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.imu = nn.Sequential(
            nn.Conv1d(6, 32, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 9),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, image: torch.Tensor, imu: torch.Tensor) -> dict[str, torch.Tensor]:
        visual = self.vision(image).flatten(1)
        inertial = self.imu(imu.transpose(1, 2)).flatten(1)
        out = self.head(torch.cat((visual, inertial), dim=1))
        translation = out[:, 6:] * 0.05
        return {"R": rot6d_to_R(out[:, :6]), "p": translation}


def pose_loss(pred: dict[str, torch.Tensor], R: torch.Tensor, p: torch.Tensor) -> dict[str, torch.Tensor]:
    rot = geodesic_per(pred["R"], R).mean()
    pos = (pred["p"] - p).norm(dim=-1).mean()
    return {"loss": rot + 40.0 * pos, "rot": rot.detach(), "pos": pos.detach()}
