#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics-structured inertial displacement network (PhysNet).

Differences from train_seq.py and the adapted public baselines:

* Inputs are expressed in the camera frame at the pair start t_a through gyro
  pre-integration, so input and output share one frame.
* Gravity is removed in that frame with the window mean, and the strapdown
  velocity/position pre-integrals are supplied as extra channels.
* The head predicts a 200 Hz velocity stream that is integrated into a
  displacement stream anchored at t_a, so d(t_a) == 0 by construction.
* Every usable chessboard frame inside the window supervises the displacement
  stream (dense supervision), not only the pair end point.
* A learnable lever arm adds (R_{a<-k} - I) l for the camera/IMU offset.
* Optional coarse branch: a long, 20 Hz window around the pair estimates the
  low-frequency velocity offset that a 7 s window cannot observe.
* Optional exact time-reversal augmentation (acceleration even, rate odd).

`train` never loads the sealed test split; `eval` loads it once for a
validation-selected set of checkpoints.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
import tools.train_seq as api

DATA = ROOT / "datasets"
HZ = 200.0
COARSE_HZ = 20.0
G0 = 9.80665
FEATURES = 24
BINARY_CHANNELS = (21, 22, 23)  # interval mask, valid mask, time coordinate


# --------------------------------------------------------------------------
# quaternion helpers (x, y, z, w), numpy
# --------------------------------------------------------------------------
def _qmul(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    px, py, pz, pw = np.moveaxis(p, -1, 0)
    qx, qy, qz, qw = np.moveaxis(q, -1, 0)
    return np.stack((
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * qy - pz * qz,
    ), axis=-1)


def _qconj(q: np.ndarray) -> np.ndarray:
    out = -q.copy()
    out[..., 3] = q[..., 3]
    return out


def _qexp(v: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(v, axis=-1, keepdims=True)
    half = 0.5 * theta
    k = np.where(theta > 1e-12, np.sin(half) / np.maximum(theta, 1e-12), 0.5)
    return np.concatenate((v * k, np.cos(half)), axis=-1)


def _qfrom_mat(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(R).as_quat()


def _session_orientation(usb_t: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    """World<-body quaternion at every raw sample by trapezoidal integration."""
    dtheta = 0.5 * (gyro[1:] + gyro[:-1]) * np.diff(usb_t)[:, None]
    dq = _qexp(dtheta)
    Q = np.zeros((len(usb_t), 4))
    Q[0] = (0.0, 0.0, 0.0, 1.0)
    for i in range(len(dq)):
        Q[i + 1] = _qmul(Q[i], dq[i])
        Q[i + 1] /= np.linalg.norm(Q[i + 1])
    return Q


def _orientation_at(usb_t, gyro, Q, times):
    idx = np.clip(np.searchsorted(usb_t, times, side="right") - 1, 0, len(usb_t) - 1)
    dt = (times - usb_t[idx])[..., None]
    return _qmul(Q[idx], _qexp(gyro[idx] * dt))


# --------------------------------------------------------------------------
# window construction
# --------------------------------------------------------------------------
class Session:
    """Raw IMU stream of one recording with gyro-integrated orientation."""

    def __init__(self, name: str | Path, R_ci: np.ndarray | None = None):
        self.name = str(name)
        folder = Path(name) if Path(name).is_absolute() else DATA / name
        usb_t, usb = load_imu(folder / "imu_stream.csv")
        self.t = usb_t
        self.acc = usb[:, 0:3] * G0
        gyro = np.radians(usb[:, 3:6])
        still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
        if int(still.sum()) >= 50:
            gyro = gyro - gyro[still].mean(0)
        self.gyro = gyro
        self.Q = _session_orientation(usb_t, gyro)
        self.R_ci = (api.R_CAMERA_IMU_NEW if R_ci is None else R_ci).astype(np.float64)
        self.q_ci = _qfrom_mat(self.R_ci)
        # 20 Hz anti-aliased copies for the coarse branch.
        from scipy.ndimage import uniform_filter1d
        self.acc_lp = uniform_filter1d(self.acc, 10, axis=0, mode="nearest")
        self.gyro_lp = uniform_filter1d(self.gyro, 10, axis=0, mode="nearest")

    def window(self, t_a: float, t_b: float, context: float, length: int, sign: float,
               hz: float = HZ, lowpass: bool = False):
        """Camera-frame IMU window anchored at t_a with padding to `length`.

        Samples outside the recording are marked invalid instead of dropped so
        long coarse windows can extend beyond the session start/end.
        """
        n = min(int(round((abs(t_b - t_a) + 2 * context) * hz)), length)
        grid = t_a + sign * (np.arange(n) / hz - context)
        inside = (grid >= self.t[0]) & (grid <= self.t[-1])
        acc_src, gyro_src = (self.acc_lp, self.gyro_lp) if lowpass else (self.acc, self.gyro)
        acc_w = np.column_stack([np.interp(grid, self.t, acc_src[:, c]) for c in range(3)])
        gyro_w = sign * np.column_stack([np.interp(grid, self.t, gyro_src[:, c]) for c in range(3)])
        q_k = _orientation_at(self.t, self.gyro, self.Q, np.clip(grid, self.t[0], self.t[-1]))
        q_a = _orientation_at(self.t, self.gyro, self.Q, np.array([t_a]))
        q_rel = _qmul(_qconj(q_a), q_k)                               # IMU_a <- IMU_k
        q_rel = _qmul(_qmul(self.q_ci, q_rel), _qconj(self.q_ci))     # C_a <- C_k
        acc_p = np.zeros((length, 3), np.float32)
        gyro_p = np.zeros((length, 3), np.float32)
        quat_p = np.zeros((length, 4), np.float32)
        quat_p[:, 3] = 1.0
        valid = np.zeros(length, np.float32)
        acc_p[:n] = (acc_w @ self.R_ci.T) * inside[:, None]
        gyro_p[:n] = (gyro_w @ self.R_ci.T) * inside[:, None]
        quat_p[:n] = q_rel
        valid[:n] = inside
        tau_b = float((t_b - grid[0]) * hz * sign)
        anchor = int(round(context * hz))
        interval = ((np.arange(length) >= anchor) & (np.arange(length) <= tau_b)).astype(np.float32)
        return acc_p, gyro_p, quat_p, valid, interval, tau_b, grid[0]


def _corpus_path(corpus: str | Path, entry: str) -> Path:
    """Manifest entries are absolute (curriculum) or corpus-relative (generator)."""
    p = Path(entry)
    return p if p.is_absolute() else (Path(corpus) / p).resolve()


def _load_gt(name: str | Path):
    if Path(name).is_absolute():
        # Synthetic digital-twin session: exact poses at every frame, no delay.
        gt = np.load(Path(name) / "ground_truth.npz")
        return (gt["t"].astype(np.float64), gt["R"].astype(np.float64),
                gt["p"].astype(np.float64), np.arange(len(gt["t"])))
    gt = np.load(DATA / "pose_gt_raw" / f"{name}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{name}.npz")["delay_usb"][0])
    return (gt["t"].astype(np.float64) - delay, gt["R"].astype(np.float64),
            gt["p"].astype(np.float64), np.flatnonzero(gt["usable"] == 1))


def gt_velocity(t: np.ndarray, pg: np.ndarray, usable: np.ndarray, half: float = 0.2):
    """Board-frame camera velocity at every usable frame by a local linear fit.

    Frames with fewer than three neighbours inside +-half seconds get no
    velocity label. Returns (velocity[n,3], valid[n]) indexed like `t`.
    """
    vel = np.zeros((len(t), 3))
    ok = np.zeros(len(t), bool)
    tu = t[usable]
    for m in usable:
        lo = np.searchsorted(tu, t[m] - half)
        hi = np.searchsorted(tu, t[m] + half, side="right")
        near = usable[lo:hi]
        if len(near) < 3 or t[near[-1]] - t[near[0]] < half:
            continue
        A = np.column_stack((t[near] - t[m], np.ones(len(near))))
        vel[m] = np.linalg.lstsq(A, pg[near], rcond=None)[0][0]
        ok[m] = True
    return vel, ok


def enumerate_pairs(t, usable, horizon, context, usb_t):
    pairs = []
    for i in usable:
        prev = usable[usable < i]
        if len(prev) == 0:
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - horizon)))])
        dt = float(t[i] - t[j])
        if dt < 0.60 * horizon or dt > 1.50 * horizon:
            continue
        if t[j] - context < usb_t[0] or t[i] + context > usb_t[-1]:
            continue
        pairs.append((j, int(i)))
    return pairs


def build_session(name: str | Path, context: float, horizon: float, length: int,
                  reverse: bool = False, coarse: float = 0.0, cap: int = 0,
                  rng: np.random.Generator | None = None) -> dict | None:
    t, Rg, pg, usable = _load_gt(name)
    ses = Session(name)
    pairs = enumerate_pairs(t, usable, horizon, context, ses.t)
    if not pairs:
        return None
    if cap and len(pairs) > cap:
        rng = rng or np.random.default_rng(0)
        pairs = [pairs[k] for k in np.sort(rng.choice(len(pairs), cap, replace=False))]
    coarse_len = int(round((1.5 * horizon + 2 * coarse) * COARSE_HZ)) if coarse > 0 else 0
    vg, v_ok = gt_velocity(t, pg, usable)
    keys = ("acc", "gyro", "quat", "valid", "interval", "tau_b", "dense_tau", "dense_y", "dense_v",
            "dense_v_valid", "y", "pair", "t_pair", "cacc", "cgyro", "cquat", "cvalid", "cinterval")
    out = {k: [] for k in keys}
    for j, i in pairs:
        a, b, sign = (i, j, -1.0) if reverse else (j, i, 1.0)
        acc_p, gyro_p, quat_p, valid, interval, tau_b, g0 = ses.window(
            t[a], t[b], context, length, sign)
        lo, hi = sorted((t[a] - sign * context, t[b] + sign * context))
        dense = usable[(t[usable] >= lo) & (t[usable] <= hi)]
        out["dense_tau"].append(((t[dense] - g0) * HZ * sign).astype(np.float32))
        out["dense_y"].append(((pg[dense] - pg[a]) @ Rg[a]).astype(np.float32))
        out["dense_v"].append((sign * (vg[dense] @ Rg[a])).astype(np.float32))
        out["dense_v_valid"].append(v_ok[dense].astype(np.float32))
        out.setdefault("dense_frame", []).append(dense.astype(np.int64))
        out["y"].append((Rg[a].T @ (pg[b] - pg[a])).astype(np.float32))
        for key, value in zip(("acc", "gyro", "quat", "valid", "interval", "tau_b"),
                              (acc_p, gyro_p, quat_p, valid, interval, tau_b)):
            out[key].append(value)
        out["pair"].append((a, b))
        out["t_pair"].append((t[a], t[b]))
        if coarse > 0:
            cacc, cgyro, cquat, cvalid, cinterval, _, _ = ses.window(
                t[a], t[b], coarse, coarse_len, sign, hz=COARSE_HZ, lowpass=True)
            for key, value in zip(("cacc", "cgyro", "cquat", "cvalid", "cinterval"),
                                  (cacc, cgyro, cquat, cvalid, cinterval)):
                out[key].append(value)
    result = {k: np.stack(out[k]) for k in ("acc", "gyro", "quat", "valid", "interval", "y")}
    result["tau_b"] = np.asarray(out["tau_b"], np.float32)
    result["pair"] = np.asarray(out["pair"], np.int64)
    result["t_pair"] = np.asarray(out["t_pair"], np.float64)
    result["dense_tau"] = out["dense_tau"]
    result["dense_y"] = out["dense_y"]
    result["dense_v"] = out["dense_v"]
    result["dense_v_valid"] = out["dense_v_valid"]
    result["dense_frame"] = out["dense_frame"]
    result["session"] = np.array([str(name)] * len(pairs))
    result["reversed"] = np.full(len(pairs), reverse)
    if coarse > 0:
        for k in ("cacc", "cgyro", "cquat", "cvalid", "cinterval"):
            result[k] = np.stack(out[k])
    return result


def _pad_dense(groups: list[dict]) -> dict:
    """Concatenate session dictionaries, padding dense supervision to one width."""
    m = max(max(len(x) for x in g["dense_tau"]) for g in groups)
    merged = {}
    for key in groups[0]:
        if key in ("dense_tau", "dense_y", "dense_v", "dense_v_valid", "dense_frame"):
            continue
        merged[key] = np.concatenate([g[key] for g in groups])
    n = len(merged["y"])
    dense_tau = np.zeros((n, m), np.float32)
    dense_y = np.zeros((n, m, 3), np.float32)
    dense_v = np.zeros((n, m, 3), np.float32)
    dense_valid = np.zeros((n, m), np.float32)
    dense_v_valid = np.zeros((n, m), np.float32)
    dense_frame = np.full((n, m), -1, np.int64)
    k = 0
    for g in groups:
        for tau, yy, vv, vok, fr in zip(g["dense_tau"], g["dense_y"], g["dense_v"], g["dense_v_valid"],
                                        g["dense_frame"]):
            dense_tau[k, :len(tau)] = tau
            dense_y[k, :len(tau)] = yy
            dense_v[k, :len(tau)] = vv
            dense_valid[k, :len(tau)] = 1.0
            dense_v_valid[k, :len(tau)] = vok
            dense_frame[k, :len(tau)] = fr
            k += 1
    merged["dense_tau"] = dense_tau
    merged["dense_y"] = dense_y
    merged["dense_v"] = dense_v
    merged["dense_valid"] = dense_valid
    merged["dense_v_valid"] = dense_v_valid
    merged["dense_frame"] = dense_frame
    return merged


def build_group(names, context, horizon, length, reverse=False, coarse=0.0, cap=0,
                rng=None) -> dict:
    groups = []
    for name in names:
        got = build_session(name, context, horizon, length, coarse=coarse, cap=cap, rng=rng)
        if got is not None:
            groups.append(got)
            print(json.dumps({"built": Path(str(name)).name, "n": int(len(got["y"]))}), flush=True)
        if reverse:
            rev = build_session(name, context, horizon, length, reverse=True, coarse=coarse)
            if rev is not None:
                groups.append(rev)
    return _pad_dense(groups)


# --------------------------------------------------------------------------
# torch featurizer and network
# --------------------------------------------------------------------------
def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    return torch.stack((
        torch.stack((1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)), -1),
        torch.stack((2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)), -1),
        torch.stack((2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)), -1),
    ), -2)


def rodrigues(v: torch.Tensor) -> torch.Tensor:
    theta = v.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    k = v / theta
    K = torch.zeros(v.shape[:-1] + (3, 3), device=v.device, dtype=v.dtype)
    K[..., 0, 1], K[..., 0, 2] = -k[..., 2], k[..., 1]
    K[..., 1, 0], K[..., 1, 2] = k[..., 2], -k[..., 0]
    K[..., 2, 0], K[..., 2, 1] = -k[..., 1], k[..., 0]
    s = torch.sin(theta)[..., None]
    c = (1 - torch.cos(theta))[..., None]
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    return eye + s * K + c * (K @ K)


class Featurizer:
    """Turns raw camera-frame IMU windows into anchor-frame physics channels."""

    def __init__(self, anchor: int, horizon: float, hz: float = HZ, use_phys: bool = True,
                 prefix: str = ""):
        self.anchor = anchor
        self.horizon = horizon
        self.hz = hz
        self.use_phys = use_phys
        self.prefix = prefix
        self.mean = None
        self.std = None

    def raw(self, batch: dict, aug: dict | None = None):
        p = self.prefix
        acc, gyro, quat = batch[p + "acc"], batch[p + "gyro"], batch[p + "quat"]
        valid, interval = batch[p + "valid"], batch[p + "interval"]
        B, T, _ = acc.shape
        dev = acc.device
        R = quat_to_mat(quat)
        if aug:
            acc = acc + aug["acc_bias"] * torch.randn(B, 1, 3, device=dev)
            acc = acc + aug["acc_noise"] * torch.randn_like(acc)
            bg = aug["gyro_bias"] * torch.randn(B, 1, 3, device=dev)
            gyro = gyro + bg + aug["gyro_noise"] * torch.randn_like(gyro)
            tk = (torch.arange(T, device=dev, dtype=acc.dtype) - self.anchor) / self.hz
            R = R @ rodrigues(bg * tk[None, :, None])
            if aug.get("ext_rad", 0.0) > 0:
                eps = rodrigues(aug["ext_rad"] * torch.randn(B, 3, device=dev))[:, None]
                R = eps @ R @ eps.transpose(-1, -2)
                acc = torch.einsum("bij,btj->bti", eps[:, 0], acc)
                gyro = torch.einsum("bij,btj->bti", eps[:, 0], gyro)
            if aug.get("jitter", 0) > 0:
                # camera/IMU delay uncertainty: roll IMU samples against the anchor
                shift = int(torch.randint(-aug["jitter"], aug["jitter"] + 1, (1,)))
                if shift:
                    acc = torch.roll(acc, shift, 1)
                    gyro = torch.roll(gyro, shift, 1)
                    R = torch.roll(R, shift, 1)
        a_rot = torch.einsum("btij,btj->bti", R, acc)
        vmask = valid[..., None]
        g_hat = (a_rot * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
        lin = (a_rot - g_hat[:, None]) * vmask
        # Trajectory-level augmentation factors set by augment_batch(): a time
        # warp s (a -> s^2 a, w -> s w, gravity unchanged) and a translation
        # amplitude alpha applied to the gravity-free acceleration only.
        if "_warp" in batch:
            s = batch["_warp"][:, None, None]
            lin = lin * s * s
            gyro = gyro * s
        if "_amp" in batch:
            lin = lin * batch["_amp"][:, None, None]
        dt = 1.0 / self.hz
        v = torch.cumsum(lin, 1) * dt
        v = v - v[:, self.anchor:self.anchor + 1]
        d = torch.cumsum(v, 1) * dt
        d = d - d[:, self.anchor:self.anchor + 1]
        gdir = (g_hat / g_hat.norm(dim=-1, keepdim=True).clamp(min=1e-6))[:, None].expand(B, T, 3)
        rot6 = R[..., :, :2].reshape(B, T, 6)
        tcoord = ((torch.arange(T, device=dev, dtype=acc.dtype) - self.anchor)
                  / (self.hz * self.horizon))[None, :, None].expand(B, T, 1)
        if not self.use_phys:
            v = torch.zeros_like(v)
            d = torch.zeros_like(d)
        feats = torch.cat((lin, gyro, rot6, v, d, gdir, interval[..., None],
                           valid[..., None], tcoord), dim=-1)
        # `v` is also returned unnormalized: the ZUPT stage re-anchors this
        # strapdown velocity at detected zero-velocity samples.
        return feats, R, v

    def fit(self, batch: dict):
        feats, _, _ = self.raw(batch)
        valid = batch[self.prefix + "valid"] > 0.5
        sel = feats[valid]
        self.mean = sel.mean(0)
        self.std = sel.std(0).clamp(min=1e-6)
        for c in BINARY_CHANNELS:
            self.mean[c] = 0.0
            self.std[c] = 1.0

    def __call__(self, batch: dict, aug: dict | None = None):
        feats, R, v = self.raw(batch, aug)
        feats = ((feats - self.mean) / self.std).clamp(-8.0, 8.0)
        return feats, R, v

    def state(self):
        return {"mean": self.mean.cpu(), "std": self.std.cpu(), "anchor": self.anchor,
                "horizon": self.horizon, "hz": self.hz, "use_phys": self.use_phys,
                "prefix": self.prefix}

    @classmethod
    def from_state(cls, state, device):
        obj = cls(state["anchor"], state["horizon"], state.get("hz", HZ), state["use_phys"],
                  state.get("prefix", ""))
        obj.mean = state["mean"].to(device)
        obj.std = state["std"].to(device)
        return obj


ZUPT_CH = 4          # re-anchored velocity (3) + still-mass confidence (1)
ZUPT_V_SCALE = 0.05  # m/s, normalizes the re-anchored velocity channel
ZUPT_MASS = 100.0    # gate mass (samples) for full confidence: 0.5 s at 200 Hz


class PhysNet(nn.Module):
    """Velocity-stream displacement network with a zero-velocity stage.

    Stage 1 runs the trunk with the ZUPT channels zeroed and predicts a
    stillness gate. Stage 2 reuses the same weights after replacing those
    channels by the strapdown velocity re-anchored at the detected still
    samples, which is an estimate of the absolute velocity that an
    accelerometer cannot otherwise observe inside one window.
    """

    def __init__(self, width: int = 96, in_ch: int = FEATURES, lever: bool = True,
                 dilations=(1, 2, 4, 8, 16), vel_scale: float = 0.1, dropout: float = 0.0,
                 coarse: bool = False, coarse_width: int = 48,
                 still: bool = False, gate: bool = False, zupt: bool = False,
                 logvar: bool = False):
        super().__init__()
        self.use_still = still or gate or zupt
        self.use_gate = gate
        self.use_zupt = zupt
        self.use_logvar = logvar
        self.base_ch = in_ch
        in_ch = in_ch + (ZUPT_CH if zupt else 0)
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(width, width, 5, stride=2, padding=2), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(width, width, 5, padding=2 * d, dilation=d), nn.GELU(),
                          nn.Dropout(dropout))
            for d in dilations
        ])
        self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * width, width), nn.GELU(),
                                  nn.Linear(width, 3))
        self.still_head = nn.Linear(2 * width, 1) if self.use_still else None
        self.logvar_head = nn.Linear(2 * width, 3) if logvar else None
        if logvar:
            nn.init.zeros_(self.logvar_head.weight)
            nn.init.zeros_(self.logvar_head.bias)
        self.use_lever = lever
        self.lever = nn.Parameter(torch.zeros(3))
        self.vel_scale = vel_scale
        self.in_drop = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()
        self.use_coarse = coarse
        if coarse:
            self.coarse_stem = nn.Sequential(
                nn.Conv1d(in_ch, coarse_width, 5, padding=2), nn.GELU(),
                nn.Conv1d(coarse_width, coarse_width, 5, padding=4, dilation=2), nn.GELU(),
            )
            self.coarse_gru = nn.GRU(coarse_width, coarse_width, batch_first=True, bidirectional=True)
            self.coarse_offset = nn.Linear(2 * coarse_width, 3)
            self.coarse_cond = nn.Linear(2 * coarse_width, 2 * width)
            nn.init.zeros_(self.coarse_offset.weight)
            nn.init.zeros_(self.coarse_offset.bias)

    def _trunk(self, feats, cfeats=None, cvalid=None):
        h = self.stem(self.in_drop(feats.transpose(1, 2)))
        for block in self.blocks:
            h = h + block(h)
        h, _ = self.gru(h.transpose(1, 2))
        offset = None
        if self.use_coarse and cfeats is not None:
            c = self.coarse_stem(cfeats.transpose(1, 2)).transpose(1, 2)
            c, _ = self.coarse_gru(c)
            w = cvalid[..., None]
            c = (c * w).sum(1) / w.sum(1).clamp(min=1.0)
            offset = self.coarse_offset(c) * 0.02
            h = h + self.coarse_cond(c)[:, None, :]
        return h, offset

    def _up(self, x, T):
        return F.interpolate(x.transpose(1, 2), size=T, mode="linear",
                             align_corners=False).transpose(1, 2)

    def forward(self, feats: torch.Tensor, R: torch.Tensor, anchor: int,
                v_pre: torch.Tensor | None = None, valid: torch.Tensor | None = None,
                cfeats: torch.Tensor | None = None, cvalid: torch.Tensor | None = None):
        B, T, _ = feats.shape
        if self.use_zupt:
            zeros = feats.new_zeros(B, T, ZUPT_CH)
            h, _offset = self._trunk(torch.cat((feats, zeros), -1), cfeats, cvalid)
            still = self._up(self.still_head(h), T)
            gate = torch.sigmoid(still) * valid[..., None]
            mass = gate.sum(1)
            conf = (mass / ZUPT_MASS).clamp(max=1.0)
            v_still = (gate * v_pre).sum(1) / mass.clamp(min=1e-3)
            v_abs = (v_pre - v_still[:, None]) * conf[:, None] / ZUPT_V_SCALE
            extra = torch.cat((v_abs.clamp(-8.0, 8.0),
                               conf[:, None].expand(B, T, 1)), -1)
            h, offset = self._trunk(torch.cat((feats, extra), -1), cfeats, cvalid)
        else:
            h, offset = self._trunk(feats, cfeats, cvalid)
        still = self._up(self.still_head(h), T) if self.use_still else None
        vel = self._up(self.head(h), T) * self.vel_scale
        if offset is not None:
            vel = vel + offset[:, None, :]
        if self.use_gate:
            # Structural zero-velocity update: no displacement accumulates
            # while the camera is detected as still.
            vel = vel * (1.0 - torch.sigmoid(still))
        d = torch.cumsum(vel, 1) / HZ
        d = d - d[:, anchor:anchor + 1]
        if self.use_lever:
            eye = torch.eye(3, device=R.device, dtype=R.dtype)
            lever_d = torch.einsum("btij,j->bti", R - eye, self.lever)
            d = d + lever_d
            # camera velocity = IMU-point velocity + d/dt of the lever displacement
            lever_v = torch.diff(lever_d, dim=1, append=lever_d[:, -1:]) * HZ
            vel = vel + lever_v
        logvar = self._up(self.logvar_head(h), T) if self.use_logvar else None
        return d, vel, still, logvar


def _resample(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Linear resampling of a (B,T,C) tensor at fractional source indices (B,T)."""
    T = x.shape[1]
    lo = idx.floor().clamp(0, T - 2).long()
    frac = (idx - lo.to(idx.dtype)).clamp(0.0, 1.0)[..., None]
    g = lo[..., None].expand(-1, -1, x.shape[2])
    return torch.gather(x, 1, g) * (1 - frac) + torch.gather(x, 1, g + 1) * frac


def augment_batch(batch: dict, aug: dict, anchor: int) -> dict:
    """Physically exact trajectory-level augmentation of one training batch.

    Time warp s: the same path is traversed s times faster. Source index
    k' = anchor + (k - anchor) s; rotations are resampled, gravity-free
    acceleration is later scaled by s^2 and rate by s (Featurizer.raw), and
    every supervision time moves to anchor + (tau - anchor) / s. Amplitude
    alpha scales the translation path (and its velocity) while the rotation
    path is kept, so gravity-free acceleration and all position/velocity
    labels are multiplied by alpha.
    """
    warp_max, amp_range = aug.get("warp", 1.0), aug.get("amp", (1.0, 1.0))
    if warp_max <= 1.0 and amp_range == (1.0, 1.0):
        return batch
    acc = batch["acc"]
    B, T, _ = acc.shape
    dev = acc.device
    out = dict(batch)
    if warp_max > 1.0:
        s = torch.exp((torch.rand(B, device=dev) * 2 - 1) * np.log(warp_max))
        k = torch.arange(T, device=dev, dtype=acc.dtype)
        src = anchor + (k[None, :] - anchor) * s[:, None]
        inside = (src >= 0) & (src <= T - 1)
        n_valid = batch["valid"].sum(1, keepdim=True)
        inside = inside & (src <= n_valid - 1)
        out["acc"] = _resample(acc, src) * inside[..., None]
        out["gyro"] = _resample(batch["gyro"], src) * inside[..., None]
        q = _resample(batch["quat"], src)
        q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        ident = torch.zeros_like(q)
        ident[..., 3] = 1.0
        out["quat"] = torch.where(inside[..., None], q, ident)
        out["valid"] = inside.to(acc.dtype)
        out["tau_b"] = anchor + (batch["tau_b"] - anchor) / s
        out["interval"] = ((k[None, :] >= anchor) & (k[None, :] <= out["tau_b"][:, None])).to(acc.dtype)
        out["dense_tau"] = anchor + (batch["dense_tau"] - anchor) / s[:, None]
        keep = (out["dense_tau"] >= 0) & (out["dense_tau"] <= T - 1)
        out["dense_valid"] = batch["dense_valid"] * keep
        out["dense_v_valid"] = batch["dense_v_valid"] * keep
        out["dense_v"] = batch["dense_v"] * s[:, None, None]
        out["_warp"] = s
    if amp_range != (1.0, 1.0):
        lo, hi = amp_range
        alpha = torch.exp(torch.rand(B, device=dev) * (np.log(hi) - np.log(lo)) + np.log(lo))
        # a tenth of the batch becomes (near) pure rotation: translation ~ 0
        alpha = torch.where(torch.rand(B, device=dev) < 0.1, alpha * 0.05, alpha)
        out["y"] = batch["y"] * alpha[:, None]
        out["dense_y"] = batch["dense_y"] * alpha[:, None, None]
        out["dense_v"] = out.get("dense_v", batch["dense_v"]) * alpha[:, None, None]
        out["_amp"] = alpha
    return out


def sample_stream(d: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Linear interpolation of a (B,T,3) stream at fractional indices (B,M)."""
    T, C = d.shape[1], d.shape[2]
    lo = tau.floor().clamp(0, T - 2).long()
    frac = (tau - lo.to(tau.dtype)).clamp(0.0, 1.0)[..., None]
    idx = lo[..., None].expand(-1, -1, C)
    d_lo = torch.gather(d, 1, idx)
    d_hi = torch.gather(d, 1, idx + 1)
    return d_lo * (1 - frac) + d_hi * frac


TENSOR_KEYS = ("acc", "gyro", "quat", "valid", "interval", "tau_b", "y", "dense_tau",
               "dense_y", "dense_valid", "dense_v", "dense_v_valid",
               "cacc", "cgyro", "cquat", "cvalid", "cinterval")


def to_device(data: dict, device):
    return {k: torch.from_numpy(np.ascontiguousarray(data[k])).to(device)
            for k in TENSOR_KEYS if k in data}


def take(batch: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in batch.items()}


class Model:
    """Network plus its featurizers; one object per checkpoint."""

    def __init__(self, net: PhysNet, feat: Featurizer, cfeat: Featurizer | None, anchor: int):
        self.net, self.feat, self.cfeat, self.anchor = net, feat, cfeat, anchor

    def stream_full(self, batch: dict, aug: dict | None = None):
        """Returns displacement d(t), camera velocity v(t), stillness logit and log-variance."""
        feats, R, v_pre = self.feat(batch, aug)
        cfeats = cvalid = None
        if self.cfeat is not None:
            cfeats, _, _ = self.cfeat(batch, aug)
            cvalid = batch["cvalid"]
        return self.net(feats, R, self.anchor, v_pre, batch["valid"], cfeats, cvalid)

    def stream(self, batch: dict, aug: dict | None = None) -> torch.Tensor:
        return self.stream_full(batch, aug)[0]

    def predict(self, data: dict, batch_size: int = 128) -> np.ndarray:
        self.net.eval()
        out = []
        n = len(data["y"])
        with torch.no_grad():
            for k in range(0, n, batch_size):
                idx = torch.arange(k, min(k + batch_size, n), device=data["y"].device)
                b = take(data, idx)
                d = self.stream(b)
                out.append(sample_stream(d, b["tau_b"][:, None])[:, 0].cpu().numpy())
        return np.concatenate(out)

    def predict_dense(self, data: dict, batch_size: int = 128) -> np.ndarray:
        """Predicted displacement (B,M,3) at every dense supervision time of each window."""
        return self.predict_streams(data, batch_size)["d"]

    def predict_streams(self, data: dict, batch_size: int = 128) -> dict:
        """Displacement, camera velocity, stillness probability and variance at the
        dense supervision times of each window, plus the pair end displacement."""
        self.net.eval()
        out = {k: [] for k in ("d", "v", "still", "var", "end")}
        n = len(data["y"])
        with torch.no_grad():
            for k in range(0, n, batch_size):
                idx = torch.arange(k, min(k + batch_size, n), device=data["y"].device)
                b = take(data, idx)
                d, vel, still, logvar = self.stream_full(b)
                tau = b["dense_tau"]
                out["d"].append(sample_stream(d, tau).cpu().numpy())
                out["v"].append(sample_stream(vel, tau).cpu().numpy())
                out["end"].append(sample_stream(d, b["tau_b"][:, None])[:, 0].cpu().numpy())
                out["still"].append(
                    torch.sigmoid(sample_stream(still, tau))[..., 0].cpu().numpy()
                    if still is not None else np.zeros(tau.shape, np.float32))
                out["var"].append(
                    torch.exp(sample_stream(logvar, tau).clamp(-6.0, 6.0)).cpu().numpy() / 1e4
                    if logvar is not None else np.ones(tau.shape + (3,), np.float32))
        return {k: np.concatenate(v) for k, v in out.items()}


STILL_MM_S = 3.0   # confidently still
MOVE_MM_S = 8.0    # confidently moving; the band in between is not supervised


def still_targets(batch: dict):
    """Binary stillness labels at the dense supervision times, with a mask.

    Frames whose reference speed falls between STILL_MM_S and MOVE_MM_S are
    excluded so that label noise at the boundary does not train the gate.
    """
    speed = batch["dense_v"].norm(dim=-1) * 1000.0
    valid = batch["dense_valid"] * batch["dense_v_valid"]
    label = (speed < STILL_MM_S).to(speed.dtype)
    mask = valid * ((speed < STILL_MM_S) | (speed > MOVE_MM_S)).to(speed.dtype)
    return label, mask


def loss_fn(d, vel, still, logvar, batch, dense_weight, end_weight, vel_weight=0.0,
            still_weight=0.0, nll_weight=0.0, beta=0.5):
    """Smooth L1 in centimetres over dense chessboard frames and the pair end,
    plus optional velocity-stream, stillness and heteroscedastic terms."""
    end = sample_stream(d, batch["tau_b"][:, None])[:, 0]
    total = end_weight * F.smooth_l1_loss(end * 100.0, batch["y"] * 100.0, beta=beta)
    if dense_weight > 0:
        pred = sample_stream(d, batch["dense_tau"])
        per = F.smooth_l1_loss(pred * 100.0, batch["dense_y"] * 100.0, beta=beta, reduction="none")
        w = batch["dense_valid"][..., None]
        total = total + dense_weight * (per * w).sum() / (w.sum() * 3).clamp(min=1.0)
    if vel_weight > 0:
        pred_v = sample_stream(vel, batch["dense_tau"])
        per = F.smooth_l1_loss(pred_v * 100.0, batch["dense_v"] * 100.0, beta=1.0, reduction="none")
        w = (batch["dense_valid"] * batch["dense_v_valid"])[..., None]
        total = total + vel_weight * (per * w).sum() / (w.sum() * 3).clamp(min=1.0)
    if still_weight > 0 and still is not None:
        label, mask = still_targets(batch)
        logit = sample_stream(still, batch["dense_tau"])[..., 0]
        per = F.binary_cross_entropy_with_logits(logit, label, reduction="none")
        total = total + still_weight * (per * mask).sum() / mask.sum().clamp(min=1.0)
    if nll_weight > 0 and logvar is not None:
        pred = sample_stream(d, batch["dense_tau"]) * 100.0
        lv = sample_stream(logvar, batch["dense_tau"]).clamp(-6.0, 6.0)
        target = batch["dense_y"] * 100.0
        per = 0.5 * ((pred - target) ** 2 * torch.exp(-lv) + lv)
        w = batch["dense_valid"][..., None]
        total = total + nll_weight * (per * w).sum() / (w.sum() * 3).clamp(min=1.0)
    return total


def run_phase(model: Model, train_t: dict, val_t: dict, val_y: np.ndarray, weight, aug,
              dense_weight, args, rng, epochs: int, lr: float, patience: int, label: str,
              fixed: bool = False):
    """One optimisation phase with validation-based checkpoint selection."""
    net = model.net
    device = val_t["y"].device
    n = len(train_t["y"])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, best_epoch, bad = None, None, -1, 0
    history = []
    start = time.time()
    for epoch in range(epochs):
        net.train()
        if weight is not None:
            order = torch.from_numpy(rng.choice(n, n, replace=True, p=weight)).to(device)
        else:
            order = torch.from_numpy(rng.permutation(n)).to(device)
        total = 0.0
        for k in range(0, n, args.batch):
            idx = order[k:k + args.batch]
            b = augment_batch(take(train_t, idx), aug, model.anchor)
            d, vel, still, logvar = model.stream_full(b, aug)
            loss = loss_fn(d, vel, still, logvar, b, dense_weight, args.end_weight,
                           args.vel_weight, args.still_weight, args.nll_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        pred = model.predict(val_t)
        metric = api._metrics(pred, val_y)
        row = {"phase": label, "epoch": epoch, "loss": round(total / n, 4), "val": metric,
               "lever_mm": [round(float(v) * 1000, 1) for v in net.lever.detach().cpu()],
               "sec": round(time.time() - start, 1)}
        history.append(row)
        print(json.dumps(row), flush=True)
        if fixed:
            continue
        if best is None or metric["err_mm"] < best:
            best, best_epoch, bad = metric["err_mm"], epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if fixed:
        return epochs - 1, history[-1]["val"]["err_mm"], history
    net.load_state_dict(best_state)
    return best_epoch, best, history


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_train(args) -> None:
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = int(round((1.5 * args.horizon + 2 * args.context) * HZ))
    anchor = int(round(args.context * HZ))
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    if args.holdout:
        # Leave-one-session-out fold over all non-test sessions.
        pool = split["train"] + split["val"]
        assert args.holdout in pool, f"{args.holdout} is not a train/val session"
        train_names = [n for n in pool if n != args.holdout]
        val_names = [args.holdout]
    else:
        train_names = split["train"] + (split["val"] if args.train_on_val else [])
        val_names = split["val"]
    # Extra sessions (rigid_* rotation/still recordings) are training-only in
    # every fold. Default: the `extra_train` list of the split file.
    extra = [] if args.no_extra else (args.extra_train or split.get("extra_train", []))
    train_names = train_names + [n for n in extra if n not in train_names and n not in val_names]
    train = build_group(train_names, args.context, args.horizon, length,
                        reverse=args.reverse, coarse=args.coarse)
    val = build_group(val_names, args.context, args.horizon, length, coarse=args.coarse)
    print(json.dumps({"train_pairs": int(len(train["y"])), "val_pairs": int(len(val["y"])),
                      "dense_per_window_train": float(train["dense_valid"].sum(1).mean())}), flush=True)
    train_t = to_device(train, device)
    val_t = to_device(val, device)
    n = len(train["y"])

    # Sampling weights: equal weight per recorded second instead of per pair, so
    # 30 fps sessions do not dominate 10 fps sessions.
    if args.balance:
        counts = {name: int((train["session"] == name).sum()) for name in np.unique(train["session"])}
        seconds = {name: float(np.ptp(train["t_pair"][train["session"] == name][:, 0]) + 3.0)
                   for name in counts}
        weight = np.array([seconds[s] / counts[s] for s in train["session"]])
        weight /= weight.sum()
    else:
        weight = None

    sub = torch.arange(0, n, max(1, n // 2000), device=device)
    feat = Featurizer(anchor, args.horizon, use_phys=not args.no_phys)
    feat.fit(take(train_t, sub))
    cfeat = None
    if args.coarse > 0:
        cfeat = Featurizer(int(round(args.coarse * COARSE_HZ)), args.horizon, COARSE_HZ,
                           use_phys=not args.no_phys, prefix="c")
        cfeat.fit(take(train_t, sub))
    net = PhysNet(width=args.width, lever=not args.no_lever, dropout=args.dropout,
                  coarse=args.coarse > 0, still=args.still_weight > 0, gate=args.gate,
                  zupt=args.zupt, logvar=args.nll_weight > 0).to(device)
    if args.gate or args.zupt:
        assert args.still_weight > 0, "--gate/--zupt need --still-weight > 0"
    model = Model(net, feat, cfeat, anchor)
    aug = {
        "acc_bias": args.acc_bias * G0, "acc_noise": args.acc_noise * G0,
        "gyro_bias": np.radians(args.gyro_bias), "gyro_noise": np.radians(args.gyro_noise),
        "ext_rad": np.radians(args.ext_deg), "jitter": args.jitter,
        "warp": args.warp, "amp": (args.amp_min, args.amp_max),
    }
    if args.coarse > 0:
        assert args.warp <= 1.0 and (args.amp_min, args.amp_max) == (1.0, 1.0), \
            "trajectory augmentation is not implemented for the coarse branch"
    dense_weight = 0.0 if args.no_dense else args.dense_weight
    history = []
    pretrain_report = None
    if args.pretrain_corpus:
        manifest = json.loads((Path(args.pretrain_corpus) / "manifest.json").read_text(encoding="utf-8"))
        syn = {}
        for group in ("train", "val"):
            paths = [_corpus_path(args.pretrain_corpus, e["path"]) for e in manifest["imu"][group]]
            syn[group] = build_group(paths, args.context, args.horizon, length, coarse=args.coarse,
                                     cap=args.synthetic_cap, rng=rng)
            print(json.dumps({"synthetic": group, "pairs": int(len(syn[group]["y"]))}), flush=True)
        syn_train_t, syn_val_t = to_device(syn["train"], device), to_device(syn["val"], device)
        pre_epoch, pre_err, pre_hist = run_phase(
            model, syn_train_t, syn_val_t, syn["val"]["y"], None, aug, dense_weight, args,
            rng, epochs=args.pretrain_epochs, lr=args.lr, patience=10, label="synthetic")
        history += pre_hist
        pretrain_report = {"synthetic_train_pairs": int(len(syn["train"]["y"])),
                           "pretrain_best_epoch": pre_epoch, "pretrain_synthetic_val_mm": pre_err,
                           "zero_shot_real_val": api._metrics(model.predict(val_t), val["y"])}
        print(json.dumps({"pretrain": pretrain_report}), flush=True)
        del syn_train_t, syn_val_t
        torch.cuda.empty_cache()
    if args.synthetic_only:
        assert args.pretrain_corpus, "--synthetic-only needs --pretrain-corpus"
        best_epoch = pretrain_report["pretrain_best_epoch"]
    else:
        best_epoch, _best, hist = run_phase(
            model, train_t, val_t, val["y"], weight, aug, dense_weight, args, rng,
            epochs=args.epochs, lr=args.finetune_lr if args.pretrain_corpus else args.lr,
            patience=args.patience, label="real", fixed=args.fixed_epochs > 0)
        history += hist
    pred = model.predict(val_t)
    report = {
        "architecture": "physnet_anchor_frame_velocity_integration",
        "config": vars(args) | {"cmd": "train"},
        "train_pairs": int(n),
        "val_pairs": int(len(val["y"])),
        "val_in_train": bool(args.train_on_val),
        "holdout": args.holdout,
        "train_sessions": train_names,
        "pretrain": pretrain_report,
        "best_epoch": best_epoch,
        "val": api._metrics(pred, val["y"]),
        "val_by_session": {},
        "lever_arm_mm": [round(float(v) * 1000, 2) for v in net.lever.detach().cpu()],
        "parameters": sum(p.numel() for p in net.parameters()),
    }
    for name in np.unique(val["session"]):
        sel = val["session"] == name
        report["val_by_session"][str(name)] = api._metrics(pred[sel], val["y"][sel])
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.tag}_s{args.seed}"
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"model": best_state, "featurizer": feat.state(),
                "coarse_featurizer": cfeat.state() if cfeat else None,
                "width": args.width, "lever": not args.no_lever, "context": args.context,
                "horizon": args.horizon, "coarse": args.coarse,
                "still": args.still_weight > 0, "gate": args.gate, "zupt": args.zupt,
                "logvar": args.nll_weight > 0},
               out / f"physnet_{tag}.pt")
    np.savez(out / f"pred_{tag}.npz", val_pred=pred, val_y=val["y"], val_session=val["session"])
    (out / f"metrics_{tag}.json").write_text(
        json.dumps(report | {"history": history}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"final": report}, ensure_ascii=False), flush=True)


def load_model(path: Path, device) -> Model:
    ck = torch.load(path, map_location=device)
    net = PhysNet(width=ck["width"], lever=ck["lever"], coarse=ck.get("coarse", 0.0) > 0,
                  still=ck.get("still", False), gate=ck.get("gate", False),
                  zupt=ck.get("zupt", False), logvar=ck.get("logvar", False)).to(device)
    net.load_state_dict(ck["model"])
    net.eval()
    feat = Featurizer.from_state(ck["featurizer"], device)
    cfeat = Featurizer.from_state(ck["coarse_featurizer"], device) if ck.get("coarse_featurizer") else None
    return Model(net, feat, cfeat, int(round(ck["context"] * HZ)))


def _tta_predict(models: list[Model], names, context, horizon, length, coarse, shifts):
    """Average over anchor shifts: each shifted window predicts the same pair.

    The displacement d(t_b) - d(t_a) read from a window anchored at t_a' is
    rotated into C_a with the gyro rotation R_{a<-a'} taken from that window.
    """
    preds = []
    reference = None
    for shift in shifts:
        groups = []
        for name in names:
            t, Rg, pg, usable = _load_gt(name)
            ses = Session(name)
            pairs = enumerate_pairs(t, usable, horizon, context, ses.t)
            rows = {k: [] for k in ("acc", "gyro", "quat", "valid", "interval", "tau_b", "tau_a",
                                    "y", "cacc", "cgyro", "cquat", "cvalid", "cinterval")}
            coarse_len = int(round((1.5 * horizon + 2 * coarse) * COARSE_HZ)) if coarse > 0 else 0
            for j, i in pairs:
                ta, tb = t[j] + shift, t[i] + shift
                if ta - context < ses.t[0] or tb + context > ses.t[-1]:
                    ta, tb = t[j], t[i]
                acc_p, gyro_p, quat_p, valid, interval, tau_b, g0 = ses.window(
                    ta, tb, context, length, 1.0)
                rows["tau_b"].append(float((t[i] - g0) * HZ))
                rows["tau_a"].append(float((t[j] - g0) * HZ))
                for key, value in zip(("acc", "gyro", "quat", "valid", "interval"),
                                      (acc_p, gyro_p, quat_p, valid, interval)):
                    rows[key].append(value)
                rows["y"].append((Rg[j].T @ (pg[i] - pg[j])).astype(np.float32))
                if coarse > 0:
                    cw = ses.window(ta, tb, coarse, coarse_len, 1.0, hz=COARSE_HZ, lowpass=True)
                    for key, value in zip(("cacc", "cgyro", "cquat", "cvalid", "cinterval"), cw[:5]):
                        rows[key].append(value)
            data = {k: np.stack(v) if len(v) and np.ndim(v[0]) else np.asarray(v, np.float32)
                    for k, v in rows.items() if len(v)}
            data["session"] = np.array([name] * len(pairs))
            groups.append(data)
        data = {k: np.concatenate([g[k] for g in groups]) for k in groups[0]}
        dev = next(models[0].net.parameters()).device
        data_t = {k: torch.from_numpy(np.ascontiguousarray(data[k])).to(dev)
                  for k in TENSOR_KEYS + ("tau_a",) if k in data}
        out = []
        n = len(data["y"])
        with torch.no_grad():
            for k in range(0, n, 128):
                idx = torch.arange(k, min(k + 128, n), device=dev)
                b = take(data_t, idx)
                acc = 0
                for m in models:
                    d = m.stream(b)
                    db = sample_stream(d, b["tau_b"][:, None])[:, 0]
                    da = sample_stream(d, b["tau_a"][:, None])[:, 0]
                    qa = sample_stream(b["quat"], b["tau_a"][:, None])[:, 0]
                    Ra = quat_to_mat(qa / qa.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                    acc = acc + torch.einsum("bji,bj->bi", Ra, db - da)     # R_{a'<-a}^T (d_b - d_a)
                out.append((acc / len(models)).cpu().numpy())
        preds.append(np.concatenate(out))
        reference = data
    return np.mean(preds, axis=0), reference


def cmd_eval(args) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = int(round((1.5 * args.horizon + 2 * args.context) * HZ))
    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    groups = {"val": split["val"]}
    if args.test:
        groups["test"] = split["test"]
    if args.synthetic_corpus:
        manifest = json.loads((Path(args.synthetic_corpus) / "manifest.json").read_text(encoding="utf-8"))
        for name in ("test", "ood"):
            groups[f"synthetic_{name}"] = [_corpus_path(args.synthetic_corpus, e["path"])
                                           for e in manifest["imu"][name]]
    models = [load_model(Path(p), device) for p in args.models]
    coarse = torch.load(args.models[0], map_location="cpu").get("coarse", 0.0)
    report = {"models": [str(p) for p in args.models], "seeds": len(models),
              "tta_shifts_s": args.tta,
              "lever_arm_mm": [[round(float(v) * 1000, 2) for v in m.net.lever.detach().cpu()]
                               for m in models]}
    arrays = {}
    syn_rng = np.random.default_rng(123)  # same pair subsampling as tools/benchmark_methods.py
    for group, names in groups.items():
        synthetic = group.startswith("synthetic_")
        data = build_group(names, args.context, args.horizon, length, coarse=coarse,
                           cap=160 if synthetic else 0, rng=syn_rng if synthetic else None)
        data_t = to_device(data, device)
        preds = np.stack([m.predict(data_t) for m in models])
        pred = preds.mean(0)
        report[group] = api._metrics(pred, data["y"])
        report[f"{group}_single_seed"] = [api._metrics(p, data["y"]) for p in preds]
        report[f"{group}_by_session"] = {}
        for name in np.unique(data["session"]):
            sel = data["session"] == name
            report[f"{group}_by_session"][str(name)] = api._metrics(pred[sel], data["y"][sel])
        arrays[f"{group}_pred"] = pred
        if args.tta:
            tta_pred, ref = _tta_predict(models, names, args.context, args.horizon, length, coarse,
                                         args.tta)
            assert np.allclose(ref["y"], data["y"], atol=1e-6)
            report[f"{group}_tta"] = api._metrics(tta_pred, data["y"])
            arrays[f"{group}_tta_pred"] = tta_pred
        arrays[f"{group}_y"] = data["y"]
        arrays[f"{group}_pair"] = data["pair"]
        arrays[f"{group}_t_pair"] = data["t_pair"]
        arrays[f"{group}_session"] = data["session"]
        print(json.dumps({group: report[group], "tta": report.get(f"{group}_tta")},
                         ensure_ascii=False), flush=True)
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"eval_{args.tag}.npz", **arrays)
    (out / f"eval_{args.tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("train", "eval"))
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--coarse", type=float, default=0.0,
                    help="seconds of 20 Hz coarse context on each side (0 = off)")
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    ap.add_argument("--out", default="physnet_v1")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=int, default=96)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--fixed-epochs", type=int, default=0,
                    help=">0: train exactly --epochs without validation selection")
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--dense-weight", type=float, default=1.0)
    ap.add_argument("--end-weight", type=float, default=1.0)
    ap.add_argument("--acc-bias", type=float, default=0.005, help="g, per-window constant")
    ap.add_argument("--acc-noise", type=float, default=0.002, help="g, per-sample white")
    ap.add_argument("--gyro-bias", type=float, default=0.2, help="deg/s, per-window constant")
    ap.add_argument("--gyro-noise", type=float, default=0.1, help="deg/s, per-sample white")
    ap.add_argument("--ext-deg", type=float, default=1.0, help="random extrinsic perturbation")
    ap.add_argument("--jitter", type=int, default=0, help="max IMU/camera delay jitter, samples")
    ap.add_argument("--warp", type=float, default=1.0,
                    help="time-warp augmentation: speed factor log-uniform in [1/warp, warp]")
    ap.add_argument("--amp-min", type=float, default=1.0, help="translation amplitude factor range")
    ap.add_argument("--amp-max", type=float, default=1.0)
    ap.add_argument("--vel-weight", type=float, default=0.0,
                    help="weight of the velocity-stream loss against local-fit GT velocity")
    ap.add_argument("--still-weight", type=float, default=0.0,
                    help="weight of the zero-velocity classification loss")
    ap.add_argument("--gate", action="store_true",
                    help="structural ZUPT: multiply the velocity stream by (1 - p_still)")
    ap.add_argument("--zupt", action="store_true",
                    help="two-pass ZUPT: re-anchor the strapdown velocity at detected still samples")
    ap.add_argument("--nll-weight", type=float, default=0.0,
                    help="weight of the heteroscedastic Gaussian NLL term (adds a log-variance head)")
    ap.add_argument("--balance", action="store_true", help="equal sampling weight per second")
    ap.add_argument("--reverse", action="store_true", help="time-reversal augmentation")
    ap.add_argument("--train-on-val", action="store_true",
                    help="final model: add validation sessions to training (no selection)")
    ap.add_argument("--holdout", default="",
                    help="leave-one-session-out: validate on this session, train on the rest")
    ap.add_argument("--extra-train", nargs="*", default=[],
                    help="additional real sessions always added to training (never validated); "
                         "default: split file 'extra_train'")
    ap.add_argument("--no-extra", action="store_true", help="ignore extra_train sessions")
    ap.add_argument("--pretrain-corpus", default="",
                    help="synthetic corpus directory with manifest.json for phase-1 pretraining")
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--synthetic-only", action="store_true",
                    help="train: stop after synthetic pretraining (zero-shot model)")
    ap.add_argument("--synthetic-corpus", default="",
                    help="eval: also score the corpus test/ood splits (160 pairs per session)")
    ap.add_argument("--synthetic-cap", type=int, default=120, help="pairs per synthetic session")
    ap.add_argument("--finetune-lr", type=float, default=5e-4)
    ap.add_argument("--no-dense", action="store_true")
    ap.add_argument("--no-phys", action="store_true")
    ap.add_argument("--no-lever", action="store_true")
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--tta", nargs="*", type=float, default=[],
                    help="eval: anchor shifts in seconds for test-time augmentation")
    ap.add_argument("--test", action="store_true", help="eval: also load the sealed test split")
    args = ap.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
