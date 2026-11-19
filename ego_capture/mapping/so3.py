#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Continuous 6D rotation (Zhou et al., CVPR 2019) and geodesic loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rot6d_to_R(x6: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if x6.ndim != 2 or x6.shape[-1] != 6:
        raise ValueError(f"expected (B,6), got {tuple(x6.shape)}")
    a1, a2 = x6[:, :3], x6[:, 3:]
    b1 = F.normalize(a1, dim=1, eps=eps)
    a2o = a2 - (b1 * a2).sum(1, keepdim=True) * b1
    b2 = F.normalize(a2o, dim=1, eps=eps)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def R_to_rot6d(R: torch.Tensor) -> torch.Tensor:
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=1)


def geodesic_loss(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    return torch.acos(cos.clamp(-1.0 + 1e-7, 1.0)).mean()
