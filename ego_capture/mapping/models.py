#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dual-IMU residual on a constant camera–IMU extrinsic.

Kept as the extrinsic experiment. On calib_board_bt_20260922_124351 the
geometric prior beat mlp / gru / gru_xattn, so this is not the trajectory model.

Camera pose for ego-motion lives in inertial.py: strapdown, then a causal
Transformer with four motion experts corrects that integral. PnP overwrites
the pose whenever the board is fully seen. Do not train this residual mapper
to emit a trajectory.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

from .so3 import rot6d_to_R

IMU_DIM = 13  # acc3 + gyro3 + mag3 + quat4
DEFAULT_IN_DIM = IMU_DIM * 2  # usb || bt


class _ResMLP(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class _Heads(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.rot = nn.Sequential(_ResMLP(dim, dropout), nn.Linear(dim, 6))
        self.trans = nn.Sequential(_ResMLP(dim, dropout), nn.Linear(dim, 3))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r6 = self.rot(z)
        t = self.trans(z)
        return rot6d_to_R(r6), t, r6


class FrameMapMLP(nn.Module):
    """Ablation only: flatten (B,T,D) → residual SE(3)."""

    def __init__(self, seq_len: int = 100, input_dim: int = DEFAULT_IN_DIM, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.net = nn.Sequential(
            nn.Linear(seq_len * input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.heads = _Heads(hidden, dropout)

    def forward(self, x: torch.Tensor):
        z = self.net(x.reshape(x.shape[0], -1))
        return self.heads(z)


class _IMUStreamGRU(nn.Module):
    def __init__(self, input_dim: int = IMU_DIM, hidden: int = 128, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.gru = nn.GRU(
            hidden, hidden, num_layers=layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(self.proj(x))
        return h  # (B,T,H)


class DualStreamFrameMapper(nn.Module):
    """
    USB window (B,T,13) and BT window (B,T,13) → residual R, t.

    If xattn=False, last GRU states are concatenated (pure GRU).
    If xattn=True, each stream attends to the other (2-layer Transformer decoder).
    """

    def __init__(
        self,
        hidden: int = 128,
        layers: int = 2,
        dropout: float = 0.1,
        xattn: bool = True,
        nhead: int = 4,
    ):
        super().__init__()
        self.usb = _IMUStreamGRU(IMU_DIM, hidden, layers, dropout)
        self.bt = _IMUStreamGRU(IMU_DIM, hidden, layers, dropout)
        self.xattn = xattn
        if xattn:
            layer = nn.TransformerDecoderLayer(
                d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
                dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
            )
            self.fuse_usb = nn.TransformerDecoder(layer, num_layers=2)
            self.fuse_bt = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=hidden, nhead=nhead, dim_feedforward=hidden * 4,
                    dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
                ),
                num_layers=2,
            )
            out_dim = hidden * 2
        else:
            self.fuse_usb = None
            self.fuse_bt = None
            out_dim = hidden * 2
        self.bottleneck = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = _Heads(hidden, dropout)

    def forward(self, usb: torch.Tensor, bt: torch.Tensor):
        hu = self.usb(usb)
        hb = self.bt(bt)
        if self.xattn and self.fuse_usb is not None and self.fuse_bt is not None:
            fu = self.fuse_usb(hu, hb)
            fb = self.fuse_bt(hb, hu)
            z = torch.cat([fu.mean(1), fb.mean(1)], dim=-1)
        else:
            z = torch.cat([hu[:, -1], hb[:, -1]], dim=-1)
        return self.heads(self.bottleneck(z))


class FrameMappingSystem(nn.Module):
    """Geometric T_C_I plus learned residual. usb,bt: (B,T,13)."""

    def __init__(self, mapper: nn.Module, R_geo: torch.Tensor, t_geo: torch.Tensor):
        super().__init__()
        self.mapper = mapper
        self.register_buffer("R_geo", R_geo.reshape(1, 3, 3))
        self.register_buffer("t_geo", t_geo.reshape(1, 3))

    def forward(self, usb: torch.Tensor, bt: torch.Tensor):
        from .prior import apply_residual

        if isinstance(self.mapper, FrameMapMLP):
            x = torch.cat([usb, bt], dim=-1)
            R_res, t_res, r6 = self.mapper(x)
        else:
            R_res, t_res, r6 = self.mapper(usb, bt)
        R, t = apply_residual(self.R_geo.expand(usb.size(0), -1, -1), self.t_geo.expand(usb.size(0), -1), R_res, t_res)
        return R, t, r6


def build_mapper(
    kind: Literal["mlp", "gru", "gru_xattn"] = "gru_xattn",
    seq_len: int = 100,
    hidden: int = 128,
    dropout: float = 0.1,
) -> nn.Module:
    if kind == "mlp":
        return FrameMapMLP(seq_len=seq_len, hidden=max(hidden * 2, 256), dropout=dropout)
    return DualStreamFrameMapper(hidden=hidden, dropout=dropout, xattn=(kind == "gru_xattn"))
