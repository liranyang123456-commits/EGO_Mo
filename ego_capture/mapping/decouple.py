#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Absolute two-body egomotion.

Images observe only the relative pose between the camera and the board.
The gauge is fixed by the camera pose at the start of the window: both
trajectories are written in that frame, so each is an absolute pose inside
the window, not a frame-to-frame increment. USB IMU integrates the camera.
Bluetooth IMU integrates the board. A sparse visual token supplies the
relative pose at the anchor; dropout forces the IMUs to explain the rest.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .inertial import exp_so3_b, geodesic_per
from .so3 import rot6d_to_R

IMU_SIM_DIM = 19
IMU_SIM_SCALE = (
    16, 16, 16,
    2000, 2000, 2000,
    180, 180, 180,
    8000, 8000, 8000,
    1100, 200, 80,
    1, 1, 1, 1,
)


def _integrate(xi: torch.Tensor, R0: torch.Tensor, p0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-increment xi (B, K-1, 6) onto the anchor pose."""
    rotations = [R0]
    positions = [p0]
    for k in range(xi.shape[1]):
        step_R = exp_so3_b(xi[:, k, :3])
        rotations.append(torch.matmul(rotations[-1], step_R))
        positions.append(positions[-1] + torch.matmul(rotations[-1], xi[:, k, 3:, None]).squeeze(-1))
    return torch.stack(rotations, dim=1), torch.stack(positions, dim=1)


def anchor_to_camera_start(
    R_c: torch.Tensor, p_c: torch.Tensor, R_b: torch.Tensor, p_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Express both bodies in the camera frame at index 0. Shapes (B, K, ...)."""
    R0 = R_c[:, 0].transpose(-1, -2)
    p0 = p_c[:, 0]
    R_c_a = torch.matmul(R0.unsqueeze(1), R_c)
    p_c_a = torch.matmul(R0.unsqueeze(1), (p_c - p0[:, None, :]).unsqueeze(-1)).squeeze(-1)
    R_b_a = torch.matmul(R0.unsqueeze(1), R_b)
    p_b_a = torch.matmul(R0.unsqueeze(1), (p_b - p0[:, None, :]).unsqueeze(-1)).squeeze(-1)
    return R_c_a, p_c_a, R_b_a, p_b_a


def relative_pose(R_c: torch.Tensor, p_c: torch.Tensor, R_b: torch.Tensor, p_b: torch.Tensor):
    """T_C_B = inv(T_W_C) @ T_W_B, batched over time."""
    R_cb = torch.matmul(R_c.transpose(-1, -2), R_b)
    p_cb = torch.matmul(R_c.transpose(-1, -2), (p_b - p_c).unsqueeze(-1)).squeeze(-1)
    return R_cb, p_cb


class DecoupledEgomotion(nn.Module):
    """Predict absolute camera and board trajectories inside one window.

    usb, ble: (B, T, 19) synthetic or real packets without age channels.
    visual: (B, 9) rot6d + translation of the board in the camera at the
    anchor. visual_mask: (B,) True when that relative pose was observed.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 2, dropout: float = 0.1, knots: int = 5):
        super().__init__()
        if knots < 2:
            raise ValueError("knots includes the anchor, so it must be at least 2")
        self.knots = knots
        self.register_buffer("scale", torch.tensor(IMU_SIM_SCALE, dtype=torch.float32), persistent=False)
        self.imu_proj = nn.Linear(IMU_SIM_DIM, d_model)
        self.stream = nn.Embedding(2, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            ),
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.visual_proj = nn.Linear(9, d_model)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cam_delta = nn.Linear(d_model, (knots - 1) * 6)
        self.board_delta = nn.Linear(d_model, (knots - 1) * 6)
        self.board_residual = nn.Linear(d_model, 6)
        nn.init.zeros_(self.cam_delta.weight)
        nn.init.zeros_(self.cam_delta.bias)
        nn.init.zeros_(self.board_delta.weight)
        nn.init.zeros_(self.board_delta.bias)
        nn.init.zeros_(self.board_residual.weight)
        nn.init.zeros_(self.board_residual.bias)

    def _encode_pair(self, usb: torch.Tensor, ble: torch.Tensor) -> torch.Tensor:
        step = max(1, usb.shape[1] // 64)
        usb_s = usb[:, ::step]
        ble_s = ble[:, ::step]
        tokens = torch.cat((
            self.imu_proj(usb_s / self.scale) + self.stream.weight[0],
            self.imu_proj(ble_s / self.scale) + self.stream.weight[1],
        ), dim=1)
        return self.encoder(tokens)

    def forward(
        self,
        usb: torch.Tensor,
        ble: torch.Tensor,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        memory = self._encode_pair(usb, ble)
        pooled = memory.mean(dim=1)
        seen = visual_mask.float().unsqueeze(-1)
        visual_h = self.visual_proj(visual) * seen
        query = self.query.expand(usb.shape[0], -1, -1) + visual_h.unsqueeze(1)
        mixed, _ = self.cross(query, memory, memory)
        hidden = mixed[:, 0] + pooled

        eye = torch.eye(3, device=usb.device, dtype=usb.dtype).expand(usb.shape[0], 3, 3)
        zeros = usb.new_zeros(usb.shape[0], 3)
        cam_xi = self.cam_delta(hidden).view(usb.shape[0], self.knots - 1, 6) * 0.05
        R_c, p_c = _integrate(cam_xi, eye, zeros)

        observed_R = rot6d_to_R(visual[:, :6])
        observed_p = visual[:, 6:]
        residual = self.board_residual(hidden)
        R_b0 = torch.matmul(observed_R, exp_so3_b(residual[:, :3] * seen))
        p_b0 = observed_p * seen + residual[:, 3:] * 0.01
        board_xi = self.board_delta(hidden).view(usb.shape[0], self.knots - 1, 6) * 0.05
        R_b, p_b = _integrate(board_xi, R_b0, p_b0)
        R_rel, p_rel = relative_pose(R_c, p_c, R_b, p_b)
        return {
            "R_c": R_c, "p_c": p_c, "R_b": R_b, "p_b": p_b,
            "R_rel": R_rel, "p_rel": p_rel,
        }


def decomposition_loss(
    pred: dict[str, torch.Tensor],
    R_c: torch.Tensor,
    p_c: torch.Tensor,
    R_b: torch.Tensor,
    p_b: torch.Tensor,
    supervise: str = "absolute",
) -> dict[str, torch.Tensor]:
    """supervise='absolute' scores both world trajectories and the relative pose.

    supervise='relative' scores only T_C_B, which is the gauge-free quantity
    most pose networks are trained on.
    """
    R_c, p_c, R_b, p_b = anchor_to_camera_start(R_c, p_c, R_b, p_b)
    R_rel, p_rel = relative_pose(R_c, p_c, R_b, p_b)
    rot_c = geodesic_per(pred["R_c"], R_c).mean()
    rot_b = geodesic_per(pred["R_b"], R_b).mean()
    pos_c = (pred["p_c"] - p_c).norm(dim=-1).mean()
    pos_b = (pred["p_b"] - p_b).norm(dim=-1).mean()
    rot_rel = geodesic_per(pred["R_rel"], R_rel).mean()
    pos_rel = (pred["p_rel"] - p_rel).norm(dim=-1).mean()
    # Translation is in metres. Weight it so a centimetre competes with a small rotation.
    pos_w = 40.0
    if supervise == "relative":
        total = rot_rel + pos_w * pos_rel
    elif supervise == "absolute":
        total = rot_c + rot_b + pos_w * (pos_c + pos_b) + 0.3 * (rot_rel + pos_w * pos_rel)
    else:
        raise ValueError(supervise)
    return {
        "loss": total,
        "rot_c": rot_c.detach(),
        "rot_b": rot_b.detach(),
        "pos_c": pos_c.detach(),
        "pos_b": pos_b.detach(),
        "rot_rel": rot_rel.detach(),
        "pos_rel": pos_rel.detach(),
    }
