#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply a constant SE(3) prior, then a learned residual on SO(3)×R^3."""

from __future__ import annotations

import torch


def apply_residual(R_geo: torch.Tensor, t_geo: torch.Tensor, R_res: torch.Tensor, t_res: torch.Tensor):
    """R_cam = R_geo @ R_res,  t_cam = t_geo + R_geo @ t_res  (metres)."""
    R = torch.matmul(R_geo, R_res)
    t = t_geo + torch.matmul(R_geo, t_res.unsqueeze(-1)).squeeze(-1)
    return R, t


def compose_camera_from_imu(
    R_I: torch.Tensor,
    t_I: torch.Tensor,
    R_C_I: torch.Tensor,
    t_C_I: torch.Tensor,
):
    """p_C = R_C_I p_I + t_C_I  →  pose of IMU expressed in camera: T_C = T_C_I @ T_I."""
    R = torch.matmul(R_C_I, R_I)
    t = torch.matmul(R_C_I, t_I.unsqueeze(-1)).squeeze(-1) + t_C_I
    return R, t
