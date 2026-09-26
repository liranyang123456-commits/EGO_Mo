#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SO(3) / SE(3) helpers for dual-IMU and camera hand-eye."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rpy_deg_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    r, p, y = np.deg2rad([roll, pitch, yaw])
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    # ZYX intrinsic, same as WitMotion
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def quat_to_R(w: float, x: float, y: float, z: float) -> np.ndarray:
    n = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def R_to_rpy_deg(R: np.ndarray) -> list[float]:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    pitch = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    if abs(R[2, 0]) < 0.999:
        roll = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = math.atan2(-R[0, 1], R[1, 1])
    return [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)]


def rot_err_deg(A: np.ndarray, B: np.ndarray) -> float:
    R = np.asarray(A, dtype=np.float64).T @ np.asarray(B, dtype=np.float64)
    c = (np.trace(R) - 1.0) * 0.5
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def log_so3(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    ang = math.acos(c)
    if ang < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64) * 0.5
    return (ang / (2.0 * math.sin(ang))) * np.array(
        [R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64
    )


def orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=np.float64))
    Rm = U @ Vt
    if np.linalg.det(Rm) < 0:
        U[:, -1] *= -1
        Rm = U @ Vt
    return Rm


def kabsch_R(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Find R with dst ≈ R @ src (rows are samples)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    H = src.T @ dst
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def se3(R: np.ndarray, t_m: Iterable[float]) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = orthonormalize(R)
    T[:3, 3] = np.asarray(t_m, dtype=np.float64).reshape(3)
    return T


def se3_inv(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def se3_to_dict(T: np.ndarray) -> dict:
    T = np.asarray(T, dtype=np.float64)
    return {
        "matrix": T.tolist(),
        "R": T[:3, :3].tolist(),
        "t_m": T[:3, 3].tolist(),
        "t_mm": (T[:3, 3] * 1000.0).tolist(),
        "rpy_deg": R_to_rpy_deg(T[:3, :3]),
    }


def handeye_rotation(A_list: list[np.ndarray], B_list: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Solve A ≈ X B X^T by mapping so3 logs: log(A) ≈ X log(B)."""
    src, dst = [], []
    for A, B in zip(A_list, B_list):
        a = log_so3(A)
        b = log_so3(B)
        if np.linalg.norm(a) < 1e-4 or np.linalg.norm(b) < 1e-4:
            continue
        src.append(b)
        dst.append(a)
    if len(src) < 8:
        raise RuntimeError(f"相对旋转太少（{len(src)}），再多绕几个轴转。")
    X = kabsch_R(np.stack(src), np.stack(dst))
    errs = [rot_err_deg(A, X @ B @ X.T) for A, B in zip(A_list, B_list)]
    return orthonormalize(X), float(np.mean(errs)) if errs else float("nan")
