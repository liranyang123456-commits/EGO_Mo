#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Corrected camera pose from the USB IMU.

The quantity to estimate is the camera pose in the chessboard frame, because
that is the ego-motion ground truth. PnP is that pose whenever the board is
fully seen. Between those frames the pose is strapdown integration plus a
residual. The network does not invent a second trajectory.

Each sensor is its own stream, and each token is that sensor's full packet.
USB and Bluetooth carry acc, gyro, angle, mag, pressure, altitude, temperature,
quaternion, plus the age of mag and quaternion so a held sample is not read
as a new one. Left and right cameras are two more streams: a past PnP pose
when the board was complete, and no token when it was not. The query sits at
the newest USB time and cannot read anything later, including the current
PnP label. Four motion experts correct the USB strapdown increment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# One IMU token, CSV order plus how long mag and quat have been held (seconds).
# acc(g), gyro(deg/s), angle(deg), mag, pressure(hPa), altitude(m), temp(C), quat, mag_age, quat_age
IMU_DIM = 21
IMU_SCALE = (
    16, 16, 16,
    2000, 2000, 2000,
    180, 180, 180,
    8000, 8000, 8000,
    1100, 200, 80,
    1, 1, 1, 1,
    1, 1,
)

HZ = 200
WINDOW_S = 1.0
DISP_S = 0.2
WINDOW_LEN = int(HZ * WINDOW_S)
DISP_LEN = int(HZ * DISP_S)

STATIC, SLOW, SMOOTH, FAST = 0, 1, 2, 3
N_MOTION = 4
MOTION_NAMES = ("static", "slow", "smooth", "fast")

G_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)
SPECIFIC_FORCE_UP = np.array([0.0, 0.0, 9.81], dtype=np.float64)
G_WORLD_T = (0.0, 0.0, -9.81)

STATIC_GYRO = np.deg2rad(2.0)
STATIC_SPEED = 0.002
FAST_GYRO = np.deg2rad(60.0)
FAST_SPEED = 0.10


def wit_to_si(acc_g: np.ndarray, gyro_dps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(acc_g, dtype=np.float64) * 9.81, np.asarray(gyro_dps, dtype=np.float64) * (np.pi / 180.0)


def label_motion(speed_m_s: float, gyro_rad_s: float) -> int:
    if gyro_rad_s < STATIC_GYRO and speed_m_s < STATIC_SPEED:
        return STATIC
    if gyro_rad_s > FAST_GYRO or speed_m_s > FAST_SPEED:
        return FAST
    if speed_m_s < 0.02 and gyro_rad_s < np.deg2rad(15.0):
        return SLOW
    return SMOOTH


def _skew(w: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(w, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def exp_so3(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(w))
    K = _skew(w)
    if th < 1e-8:
        return np.eye(3) + K
    s, c = np.sin(th), np.cos(th)
    return np.eye(3) + (s / th) * K + ((1.0 - c) / (th * th)) * (K @ K)


def rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    s = float(np.linalg.norm(v))
    d = float(np.dot(a, b))
    if s < 1e-8:
        if d > 0.0:
            return np.eye(3)
        axis = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, axis)
        return exp_so3(axis / np.linalg.norm(axis) * np.pi)
    K = _skew(v)
    return np.eye(3) + K + K @ K * ((1.0 - d) / (s * s))


def align_R_from_acc(acc_mps2: np.ndarray) -> np.ndarray:
    return rot_a_to_b(acc_mps2, np.array([0.0, 0.0, 1.0]))


def _skew_b(w: torch.Tensor) -> torch.Tensor:
    x, y, z = w[:, 0], w[:, 1], w[:, 2]
    o = torch.zeros_like(x)
    return torch.stack(
        (
            torch.stack((o, -z, y), dim=-1),
            torch.stack((z, o, -x), dim=-1),
            torch.stack((-y, x, o), dim=-1),
        ),
        dim=-2,
    )


def exp_so3_b(w: torch.Tensor) -> torch.Tensor:
    th = torch.linalg.norm(w, dim=-1, keepdim=True).clamp(min=1e-8)
    K = _skew_b(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype).expand(w.shape[0], 3, 3)
    th_ = th.unsqueeze(-1)
    s = torch.sin(th_)
    c = torch.cos(th_)
    return eye + (s / th_) * K + ((1.0 - c) / (th_ * th_)) * (K @ K)


def geodesic_per(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    rel = torch.matmul(R_pred.transpose(-1, -2), R_gt)
    tr = rel[..., 0, 0] + rel[..., 1, 1] + rel[..., 2, 2]
    cos = ((tr - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos)


def mix_residual(expert_out: torch.Tensor, prob: torch.Tensor) -> torch.Tensor:
    """expert_out (B,4,6), prob (B,4) → rotation and position residual.

    Static contributes nothing. Fast contributes rotation only.
    """
    xi = (prob.unsqueeze(-1) * expert_out).sum(dim=1)
    moving = 1.0 - prob[:, STATIC:STATIC + 1]
    xi = xi * moving
    pos = xi[:, 3:] * (1.0 - prob[:, FAST:FAST + 1])
    return torch.cat((xi[:, :3], pos), dim=-1)


def apply_correction(dR: torch.Tensor, dp: torch.Tensor, xi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Right-multiply the strapdown rotation and add a position residual."""
    dR_c = torch.matmul(dR, exp_so3_b(xi[:, :3]))
    dp_c = dp + xi[:, 3:]
    return dR_c, dp_c


class _TimeEmbed(nn.Module):
    """Sinusoids of the timestamp in seconds, not the index inside one tensor."""

    def __init__(self, d_model: int):
        super().__init__()
        half = d_model // 2
        freq = torch.exp(torch.linspace(np.log(0.5), np.log(200.0), half))
        self.register_buffer("freq", freq, persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        arg = t.unsqueeze(-1) * self.freq
        return torch.cat((torch.sin(arg), torch.cos(arg)), dim=-1)


class MotionTrajectoryCorrector(nn.Module):
    """
    Four sensor streams and one query.

    usb, ble: (B, T, 21) full packets on their own timestamps.
    cam0, cam1: (B, Tc, 9) past rot6d+t. Mask False where that camera lost the
    board. A pose at the query time is the label and is not readable.
    """

    def __init__(self, d_model: int = 128, nhead: int = 4, layers: int = 4, dropout: float = 0.1):
        super().__init__()
        if d_model % 2 or d_model % nhead:
            raise ValueError("d_model must be divisible by 2 and by nhead")
        self.d_model = d_model
        self.register_buffer("imu_scale", torch.tensor(IMU_SCALE, dtype=torch.float32), persistent=False)
        self.imu_proj = nn.Linear(IMU_DIM, d_model)
        self.pnp_proj = nn.Linear(9, d_model)
        self.stream = nn.Embedding(5, d_model)
        self.time = _TimeEmbed(d_model)
        self.imu_encoder = self._make_encoder(d_model, nhead, layers, dropout)
        self.pnp_encoder = self._make_encoder(d_model, nhead, max(1, layers // 2), dropout)
        self.cross = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.gate = nn.Linear(d_model, N_MOTION)
        self.experts = nn.ModuleList(nn.Linear(d_model, 6) for _ in range(N_MOTION))
        self.log_var = nn.Linear(d_model, 6)
        for expert in self.experts:
            nn.init.zeros_(expert.weight)
            nn.init.zeros_(expert.bias)
        nn.init.zeros_(self.log_var.weight)
        nn.init.constant_(self.log_var.bias, -4.0)

    @staticmethod
    def _make_encoder(d_model: int, nhead: int, layers: int, dropout: float) -> nn.TransformerEncoder:
        block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)

    @staticmethod
    def _causal_mask(n: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    def _uniform_time(self, n: int, batch: int, ref: torch.Tensor) -> torch.Tensor:
        t = torch.arange(n, device=ref.device, dtype=ref.dtype) / HZ
        return t.expand(batch, n)

    def encode_imu(self, x: torch.Tensor, t: torch.Tensor, stream_id: int) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != IMU_DIM:
            raise ValueError(f"expected (B,T,{IMU_DIM}) full IMU packet, got {tuple(x.shape)}")
        h = self.imu_proj(x / self.imu_scale) + self.stream.weight[stream_id] + self.time(t)
        return self.imu_encoder(h, mask=self._causal_mask(x.shape[1], x.device))

    def _encode_cam(self, pose: torch.Tensor, pose_t: torch.Tensor, pose_mask: torch.Tensor, stream_id: int) -> torch.Tensor:
        b, tp, _ = pose.shape
        h = self.pnp_proj(pose) + self.stream.weight[stream_id] + self.time(pose_t)
        present = pose_mask.any(dim=1)
        if not bool(present.any()):
            return h.new_zeros(b, tp, self.d_model)
        pad = ~pose_mask
        pad = pad.clone()
        pad[~present, 0] = False
        h = self.pnp_encoder(h, mask=self._causal_mask(tp, pose.device), src_key_padding_mask=pad)
        return h.masked_fill(~present.view(b, 1, 1), 0.0)

    def _cam_or_empty(self, cam, cam_mask, cam_t, t_now, ref):
        b = t_now.shape[0]
        if cam is None:
            cam = ref.new_zeros(b, 1, 9)
            cam_mask = torch.zeros(b, 1, dtype=torch.bool, device=ref.device)
            cam_t = t_now.unsqueeze(1) - 1.0
            return cam, cam_mask, cam_t
        if cam_t is None:
            raise ValueError("传入相机位姿流时必须带时间戳")
        if cam_mask is None:
            cam_mask = torch.ones(cam.shape[:2], dtype=torch.bool, device=ref.device)
        return cam, cam_mask.bool(), cam_t

    def forward(
        self,
        usb: torch.Tensor,
        ble: torch.Tensor,
        cam0: torch.Tensor | None = None,
        cam1: torch.Tensor | None = None,
        cam0_mask: torch.Tensor | None = None,
        cam1_mask: torch.Tensor | None = None,
        usb_t: torch.Tensor | None = None,
        ble_t: torch.Tensor | None = None,
        cam0_t: torch.Tensor | None = None,
        cam1_t: torch.Tensor | None = None,
    ):
        b, tu, _ = usb.shape
        if usb_t is None:
            usb_t = self._uniform_time(tu, b, usb)
        if ble_t is None:
            ble_t = self._uniform_time(ble.shape[1], b, ble)
        h_u = self.encode_imu(usb, usb_t, 0)
        h_b = self.encode_imu(ble, ble_t, 1)
        t_now = usb_t[:, -1]
        cam0, cam0_mask, cam0_t = self._cam_or_empty(cam0, cam0_mask, cam0_t, t_now, usb)
        cam1, cam1_mask, cam1_t = self._cam_or_empty(cam1, cam1_mask, cam1_t, t_now, usb)
        h0 = self._encode_cam(cam0, cam0_t, cam0_mask, 2)
        h1 = self._encode_cam(cam1, cam1_t, cam1_mask, 3)
        mem = torch.cat((h_u, h_b, h0, h1), dim=1)
        mem_t = torch.cat((usb_t, ble_t, cam0_t, cam1_t), dim=1)
        late = mem_t > (t_now.unsqueeze(1) + 1e-4)
        missing = torch.zeros(b, mem.shape[1], dtype=torch.bool, device=usb.device)
        n0 = tu + ble.shape[1]
        missing[:, n0:n0 + cam0.shape[1]] = ~cam0_mask
        missing[:, n0 + cam0.shape[1]:] = ~cam1_mask
        q = self.query.expand(b, 1, -1) + self.stream.weight[4] + self.time(t_now.unsqueeze(1))
        z, _ = self.cross(q, mem, mem, key_padding_mask=late | missing, need_weights=False)
        z = z[:, 0]
        logits = self.gate(z)
        prob = torch.softmax(logits, dim=-1)
        expert_out = torch.stack([expert(z) for expert in self.experts], dim=1)
        xi = mix_residual(expert_out, prob)
        log_var = self.log_var(z).clamp(-8.0, 4.0)
        return logits, xi, log_var, h_u


def integrate_increment(
    acc: torch.Tensor,
    gyro: torch.Tensor,
    bg: torch.Tensor,
    ba: torch.Tensor,
    dt: float,
    horizon: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Strapdown over the last `horizon` samples. Returns ΔR, Δp from identity."""
    acc_h = acc[:, -horizon:]
    gyro_h = gyro[:, -horizon:]
    b, n, _ = acc_h.shape
    eye = torch.eye(3, device=acc.device, dtype=acc.dtype)
    R = eye.expand(b, 3, 3).clone()
    v = acc.new_zeros(b, 3)
    p = acc.new_zeros(b, 3)
    g = acc.new_tensor(G_WORLD_T)
    bg = bg.reshape(1, 3).to(acc.dtype)
    ba = ba.reshape(1, 3).to(acc.dtype)
    for i in range(n):
        w = (gyro_h[:, i] - bg) * dt
        f = acc_h[:, i] - ba
        R = torch.matmul(R, exp_so3_b(w))
        a = torch.matmul(R, f.unsqueeze(-1)).squeeze(-1) + g
        p = p + v * dt + 0.5 * a * (dt * dt)
        v = v + a * dt
    return R, p


def imu_packet(
    acc_g: torch.Tensor,
    gyro_dps: torch.Tensor,
    angle_deg: torch.Tensor | None = None,
    mag: torch.Tensor | None = None,
    pressure: torch.Tensor | None = None,
    altitude: torch.Tensor | None = None,
    temp: torch.Tensor | None = None,
    quat: torch.Tensor | None = None,
    mag_age_s: torch.Tensor | None = None,
    quat_age_s: torch.Tensor | None = None,
) -> torch.Tensor:
    """Stack one IMU's CSV fields. Omitted slow fields get a 10 s age, not a fake fresh 0."""
    b, t, _ = acc_g.shape
    feat = acc_g.new_zeros(b, t, IMU_DIM)
    feat[..., 0:3] = acc_g
    feat[..., 3:6] = gyro_dps
    if angle_deg is not None:
        feat[..., 6:9] = angle_deg
    if mag is not None:
        feat[..., 9:12] = mag
    if pressure is not None:
        feat[..., 12] = pressure
    if altitude is not None:
        feat[..., 13] = altitude
    if temp is not None:
        feat[..., 14] = temp
    if quat is not None:
        feat[..., 15:19] = quat
    feat[..., 19] = 10.0 if mag_age_s is None else mag_age_s
    feat[..., 20] = 10.0 if quat_age_s is None else quat_age_s
    if mag is not None and mag_age_s is None:
        feat[..., 19] = 0.0
    if quat is not None and quat_age_s is None:
        feat[..., 20] = 0.0
    return feat


def correct_increment(
    net: MotionTrajectoryCorrector,
    acc: torch.Tensor,
    gyro: torch.Tensor,
    bg: torch.Tensor,
    ba: torch.Tensor,
    dt: float = 1.0 / HZ,
    horizon: int = DISP_LEN,
    usb: torch.Tensor | None = None,
    ble: torch.Tensor | None = None,
    cam0: torch.Tensor | None = None,
    cam1: torch.Tensor | None = None,
    cam0_mask: torch.Tensor | None = None,
    cam1_mask: torch.Tensor | None = None,
    usb_t: torch.Tensor | None = None,
    ble_t: torch.Tensor | None = None,
    cam0_t: torch.Tensor | None = None,
    cam1_t: torch.Tensor | None = None,
):
    """USB strapdown, corrected after the query reads every sensor stream."""
    if usb is None:
        usb = imu_packet(acc / 9.81, gyro * (180.0 / np.pi))
    if ble is None:
        ble = imu_packet(torch.zeros_like(acc) / 9.81, torch.zeros_like(gyro))
    logits, xi, log_var, h = net(
        usb, ble, cam0, cam1, cam0_mask, cam1_mask, usb_t, ble_t, cam0_t, cam1_t,
    )
    dR, dp = integrate_increment(acc, gyro, bg, ba, dt, horizon)
    dR_c, dp_c = apply_correction(dR, dp, xi)
    return dR_c, dp_c, logits, xi, log_var, h


def correction_loss(
    dR_c: torch.Tensor,
    dp_c: torch.Tensor,
    logits: torch.Tensor,
    xi: torch.Tensor,
    log_var: torch.Tensor,
    dR_gt: torch.Tensor,
    dp_gt: torch.Tensor,
    state_gt: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pose loss only where the board PnP is valid. Static residual is driven to 0."""
    mask_f = mask.reshape(-1).float()
    denom = mask_f.sum().clamp(min=1.0)
    rot = geodesic_per(dR_c, dR_gt)
    pos_var = torch.exp(-log_var[:, 3:]).clamp(max=1e3)
    pos = 0.5 * (pos_var * (dp_c - dp_gt) ** 2 + log_var[:, 3:]).sum(-1)
    pose = ((rot + pos) * mask_f).sum() / denom
    labeled = state_gt >= 0
    if bool(labeled.any()):
        ce = F.cross_entropy(logits[labeled], state_gt[labeled])
    else:
        ce = pose.new_zeros(())
    static = (state_gt == STATIC) & (mask_f > 0)
    if bool(static.any()):
        reg = xi[static].pow(2).mean()
    else:
        reg = pose.new_zeros(())
    loss = pose + ce + reg
    return loss, {"pose": float(pose.detach()), "ce": float(ce.detach()), "static": float(reg.detach())}


@dataclass
class NavState:
    R: np.ndarray
    v: np.ndarray
    p: np.ndarray
    bg: np.ndarray
    ba: np.ndarray

    @staticmethod
    def identity() -> "NavState":
        return NavState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))


def initialize_from_static(acc_mps2: np.ndarray, gyro_rad_s: np.ndarray) -> NavState:
    acc = np.asarray(acc_mps2, dtype=np.float64)
    gyro = np.asarray(gyro_rad_s, dtype=np.float64)
    a_mean = acc.mean(axis=0)
    R = align_R_from_acc(a_mean)
    ba = a_mean - R.T @ SPECIFIC_FORCE_UP
    return NavState(R=R, v=np.zeros(3), p=np.zeros(3), bg=gyro.mean(axis=0).copy(), ba=ba)


def propagate(state: NavState, gyro_rad_s: np.ndarray, acc_mps2: np.ndarray, dt: float) -> NavState:
    gyro = np.atleast_2d(np.asarray(gyro_rad_s, dtype=np.float64))
    acc = np.atleast_2d(np.asarray(acc_mps2, dtype=np.float64))
    for w_meas, f_meas in zip(gyro, acc):
        state.R = state.R @ exp_so3((w_meas - state.bg) * dt)
        a = state.R @ (f_meas - state.ba) + G_WORLD
        state.p = state.p + state.v * dt + 0.5 * a * (dt * dt)
        state.v = state.v + a * dt
    return state


def update_static(state: NavState, gyro_mean: np.ndarray, acc_mean: np.ndarray, bg_alpha: float = 0.2, ba_alpha: float = 0.05) -> NavState:
    state.v[:] = 0.0
    gyro_mean = np.asarray(gyro_mean, dtype=np.float64).reshape(3)
    acc_mean = np.asarray(acc_mean, dtype=np.float64).reshape(3)
    state.bg = (1.0 - bg_alpha) * state.bg + bg_alpha * gyro_mean
    if abs(float(np.linalg.norm(acc_mean)) - 9.81) < 1.5:
        target = acc_mean - state.R.T @ SPECIFIC_FORCE_UP
        state.ba = (1.0 - ba_alpha) * state.ba + ba_alpha * target
    return state


def snap_to_pnp(state: NavState, R_pnp: np.ndarray, p_pnp: np.ndarray) -> NavState:
    """Board is fully seen: the stored camera pose is PnP, not the integral."""
    state.R = np.asarray(R_pnp, dtype=np.float64).reshape(3, 3).copy()
    state.p = np.asarray(p_pnp, dtype=np.float64).reshape(3).copy()
    return state
