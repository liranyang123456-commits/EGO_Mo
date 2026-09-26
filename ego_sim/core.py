#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics and rendering core for the EGO_Mo synthetic dataset generator."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
CALIB = DATA / "calib_intrinsics_20260922_123120"
SIM_ASSETS = DATA / "sim_assets"
G = 9.80665

# Independent rigid-check estimate, not fitted to trajectory validation/test.
R_CAMERA_IMU = np.array([
    [0.007668, -0.946526, -0.322536],
    [-0.003137, 0.322521, -0.946557],
    [0.999966, 0.008270, -0.000496],
], dtype=np.float64)

IMU_COLUMNS = [
    "timestamp",
    "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "angle_x", "angle_y", "angle_z",
    "mag_x", "mag_y", "mag_z",
    "pressure", "altitude", "temp",
    "quat_w", "quat_x", "quat_y", "quat_z",
]


@dataclass
class SimConfig:
    duration_s: float = 12.0
    camera_fps: float = 30.0
    imu_hz: float = 200.0
    width: int = 1280
    height: int = 720
    seed: int = 7
    motion: str = "mixed"
    translation_mm: float = 30.0
    depth_mm: float = 150.0
    depth_change_mm: float = 20.0
    rotation_deg: float = 8.0
    tracking_gain: float = 0.35
    speed: float = 1.0
    light_level: float = 1.0
    light_flicker: float = 0.08
    noise_scale: float = 1.0
    image_noise_std: float = 2.0
    motion_blur: float = 0.35
    board_motion: bool = False
    board_motion_mm: float = 8.0
    # perturb: a few millimetres on top of a camera that still looks at the board.
    # independent: the board has its own SE(3), and the camera does not follow it.
    board_motion_mode: str = "perturb"
    board_rotation_deg: float = 3.0
    baseline_mm: float = 46.5
    square_mm: float = 3.0
    board_cols: int = 12
    board_rows: int = 9
    imu_lever_arm_mm_x: float = 0.0
    imu_lever_arm_mm_y: float = 0.0
    imu_lever_arm_mm_z: float = 0.0
    render_backend: str = "opencv"
    render_images: bool = True
    # Unit gravity direction in the board frame. None keeps the legacy flat
    # board (gravity along -z). The real rig holds the board tilted; the value
    # measured from chessboard poses and the accelerometer is about
    # (0.0, 0.87, 0.49): the specific force measured on the rig points along (0, -0.87, -0.49).
    gravity_board: tuple[float, float, float] | None = None
    # motion == "replay": drive the camera along a real chessboard trajectory.
    replay_session: str = ""
    replay_offset_s: float = 0.0
    replay_speed: float = 1.0
    replay_amp: float = 1.0
    # motion == "handheld_pause": mean spacing and length of the full stops;
    # each is jittered by +-20 % / +-33 % per pause.
    pause_interval_s: float = 12.5
    pause_duration_s: float = 1.5


@dataclass
class PoseStream:
    t: np.ndarray
    R_W_C: np.ndarray
    p_W_C: np.ndarray
    R_W_B: np.ndarray
    p_W_B: np.ndarray


@dataclass
class SyntheticSequence:
    config: SimConfig
    poses: PoseStream
    imu_usb: np.ndarray
    imu_bt: np.ndarray
    imu_usb_ideal: np.ndarray
    imu_bt_ideal: np.ndarray
    profile: dict


def _load_camera(cam: str, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    raw = json.loads((CALIB / f"camera_calibration_{cam}.json").read_text(encoding="utf-8"))
    K = np.asarray(raw["camera_matrix"], dtype=np.float64)
    dist = np.asarray(raw["distortion_coefficients"], dtype=np.float64)
    src_w, src_h = raw["image_size"]
    K = K.copy()
    K[0, 0] *= width / float(src_w)
    K[0, 2] *= width / float(src_w)
    K[1, 1] *= height / float(src_h)
    K[1, 2] *= height / float(src_h)
    return K, dist


def _look_at(position: np.ndarray, target: np.ndarray, roll: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation, OpenCV camera axes: +x right, +y down, +z forward."""
    z = target - position
    z /= max(np.linalg.norm(z), 1e-12)
    up = np.array([0.0, -1.0, 0.0])
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        up = np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.column_stack((x, y, z))
    if roll:
        R = R @ Rotation.from_rotvec(np.array([0.0, 0.0, roll])).as_matrix()
    return R


def _smooth_random(
    t: np.ndarray,
    rng: np.random.Generator,
    dims: int,
    scale: np.ndarray,
    knot_s: float = 1.5,
) -> np.ndarray:
    knots = np.arange(0.0, t[-1] + knot_s, knot_s)
    values = rng.normal(size=(len(knots), dims)) * scale
    values[0] = 0.0
    values[-1] = 0.0
    return CubicSpline(knots, values, axis=0, bc_type="clamped")(t)


def _handheld_progress(t: np.ndarray, rng: np.random.Generator, pauses: bool,
                       interval: float = 12.5, duration: float = 1.5) -> np.ndarray:
    """Monotone motion 'progress' u(t) with a random, non-periodic speed.

    With pauses, the speed ramps to zero for about `duration` s every about
    `interval` s (default 1--2 s every 10--15 s, the prospective acquisition
    protocol), so the whole rig is truly at rest; without pauses it only slows
    down occasionally, as in continuous recordings.
    """
    dt = np.diff(t, prepend=t[0])
    speed = np.exp(_smooth_random(t, rng, 1, np.array([0.45]), knot_s=2.0)[:, 0])
    gate = np.ones_like(t)
    ramp = 0.25
    if pauses:
        start = float(rng.uniform(0.25, 0.65) * interval)
        while start < t[-1]:
            dur = float(rng.uniform(0.67, 1.33) * duration)
            x = t - start
            plateau = np.clip(np.minimum(x / ramp, (dur - x) / ramp), 0.0, 1.0)
            gate *= 1.0 - (0.5 - 0.5 * np.cos(np.pi * plateau))
            start += dur + float(rng.uniform(0.8, 1.2) * interval)
    else:
        for c in rng.uniform(0.0, t[-1], size=max(1, int(t[-1] / 12))):
            gate *= 1.0 - 0.7 * np.exp(-0.5 * ((t - c) / 0.6) ** 2)
    return np.cumsum(speed * gate * dt)


def _gyro_orientation(session: str):
    """Camera orientation increments from the real USB gyro (bias removed on still samples)."""
    from ego_capture.sync import load_imu

    usb_t, usb = load_imu(DATA / session / "imu_stream.csv")
    gyro = np.radians(usb[:, 3:6])
    still = np.linalg.norm(usb[:, 3:6], axis=1) < 3.0
    if int(still.sum()) >= 50:
        gyro = gyro - gyro[still].mean(0)
    dtheta = 0.5 * (gyro[1:] + gyro[:-1]) * np.diff(usb_t)[:, None]
    steps = Rotation.from_rotvec(dtheta)
    mats = np.empty((len(usb_t), 3, 3))
    cur = Rotation.identity()
    mats[0] = cur.as_matrix()
    for i in range(len(steps)):
        cur = cur * steps[i]
        mats[i + 1] = cur.as_matrix()
    # IMU increments expressed on the camera body: R_C(t) = R_CI Q(t) R_CI^T
    cam = np.einsum("ij,njk,lk->nil", R_CAMERA_IMU, mats, R_CAMERA_IMU)
    return usb_t, cam


def load_replay_trajectory(session: str, smooth_mm: float = 0.5, rotation_source: str = "gyro"):
    """Real camera trajectory of one session as continuous functions of time.

    Position: smoothing spline through usable chessboard poses (PnP noise
    ~0.5 mm). Rotation: by default the real gyro integrated and anchored to
    the chessboard orientation of the first usable frame, because planar-PnP
    tilt noise at 10 fps would otherwise turn into several deg/s of fake rate;
    rotation_source="pnp" uses a smoothing spline through the PnP rotations.
    Returns (t_lo, t_hi, position_fn, rotation_fn, gaps); `gaps` lists usable
    frame gaps longer than 0.6 s.
    """
    from scipy.interpolate import UnivariateSpline

    raw = np.load(DATA / "pose_gt_raw" / f"{session}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{session}.npz")["delay_usb"][0])
    usable = np.flatnonzero(raw["usable"] == 1)
    t = raw["t"][usable].astype(np.float64) - delay
    p = raw["p"][usable].astype(np.float64)
    R = raw["R"][usable].astype(np.float64)
    keep = np.concatenate(([True], np.diff(t) > 1e-4))
    t, p, R = t[keep], p[keep], R[keep]
    pos_spl = [UnivariateSpline(t, p[:, k], k=3, s=len(t) * (smooth_mm / 1000.0) ** 2) for k in range(3)]
    gap_idx = np.flatnonzero(np.diff(t) > 0.6)
    gaps = [(float(t[i]), float(t[i + 1])) for i in gap_idx]

    def position(tq):
        return np.column_stack([s(tq) for s in pos_spl])

    if rotation_source == "gyro":
        usb_t, cam = _gyro_orientation(session)
        slerp = None
        idx0 = int(np.clip(np.searchsorted(usb_t, t[0]), 0, len(usb_t) - 1))
        R_anchor = R[0] @ cam[idx0].T          # world<-camera at t0 times inverse increment

        def rotation(tq):
            idx = np.clip(np.searchsorted(usb_t, tq, side="right") - 1, 0, len(usb_t) - 2)
            frac = np.clip((tq - usb_t[idx]) / (usb_t[idx + 1] - usb_t[idx]), 0.0, 1.0)
            r0 = Rotation.from_matrix(cam[idx])
            r1 = Rotation.from_matrix(cam[idx + 1])
            step = (r0.inv() * r1).as_rotvec() * frac[:, None]
            inc = (r0 * Rotation.from_rotvec(step)).as_matrix()
            return np.einsum("ij,njk->nik", R_anchor, inc)

        t_lo, t_hi = max(float(t[0]), float(usb_t[0])), min(float(t[-1]), float(usb_t[-1]))
    else:
        rotvec = Rotation.from_matrix(np.einsum("ij,njk->nik", R[0].T, R)).as_rotvec()
        rot_spl = [UnivariateSpline(t, rotvec[:, k], k=3, s=len(t) * np.radians(0.8) ** 2) for k in range(3)]

        def rotation(tq):
            rv = np.column_stack([s(tq) for s in rot_spl])
            return np.einsum("ij,njk->nik", R[0], Rotation.from_rotvec(rv).as_matrix())

        t_lo, t_hi = float(t[0]), float(t[-1])
    return t_lo, t_hi, position, rotation, gaps


def _replay_poses(config: SimConfig, t: np.ndarray) -> PoseStream:
    t_lo, t_hi, position, rotation, _gaps = load_replay_trajectory(config.replay_session)
    src = config.replay_offset_s + t * config.replay_speed
    if src[0] < t_lo or src[-1] > t_hi:
        raise ValueError(
            f"replay window [{src[0]:.2f}, {src[-1]:.2f}] outside session range [{t_lo:.2f}, {t_hi:.2f}]"
        )
    xyz = position(src)
    centre = xyz.mean(0)
    xyz = centre + config.replay_amp * (xyz - centre)
    R_W_C = rotation(src)
    return PoseStream(
        t=t, R_W_C=R_W_C, p_W_C=xyz,
        R_W_B=np.repeat(np.eye(3)[None], len(t), axis=0), p_W_B=np.zeros((len(t), 3)),
    )


def generate_poses(config: SimConfig, timestamps: np.ndarray | None = None) -> PoseStream:
    rng = np.random.default_rng(config.seed)
    t = (
        np.asarray(timestamps, dtype=np.float64)
        if timestamps is not None
        else np.arange(0.0, config.duration_s, 1.0 / config.imu_hz)
    )
    if config.motion == "replay":
        return _replay_poses(config, t)
    w = 2.0 * np.pi * config.speed / 5.0
    a = config.translation_mm / 1000.0
    dz = config.depth_change_mm / 1000.0
    depth = config.depth_mm / 1000.0
    board_center = np.array([
        0.5 * (config.board_cols - 2) * config.square_mm / 1000.0,
        0.5 * (config.board_rows - 2) * config.square_mm / 1000.0,
        0.0,
    ])
    if config.motion == "lateral":
        xyz = np.column_stack((a * np.sin(w * t), 0.15 * a * np.sin(0.5 * w * t), np.full_like(t, depth)))
    elif config.motion == "depth":
        xyz = np.column_stack((0.1 * a * np.sin(w * t), 0.1 * a * np.sin(0.7 * w * t), depth + dz * np.sin(w * t)))
    elif config.motion == "circle":
        xyz = np.column_stack((a * np.cos(w * t), 0.75 * a * np.sin(w * t), depth + 0.25 * dz * np.sin(0.5 * w * t)))
    elif config.motion == "random_spline":
        offset = _smooth_random(t, rng, 3, np.array([a, 0.8 * a, 0.6 * dz]))
        xyz = offset + np.array([0.0, 0.0, depth])
        xyz[:, 2] = np.clip(xyz[:, 2], 0.13, 0.40)
    elif config.motion == "slow_far":
        u = np.clip(t / max(t[-1], 1e-6), 0.0, 1.0)
        smooth = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        xyz = np.column_stack((
            2.0 * a * (smooth - 0.5),
            0.45 * a * np.sin(np.pi * smooth),
            depth + dz * (smooth - 0.5),
        ))
    elif config.motion == "aggressive_6dof":
        offset = _smooth_random(
            t, rng, 3, np.array([a, 0.9 * a, 0.8 * max(dz, a)]), knot_s=0.45
        )
        xyz = offset + np.array([0.0, 0.0, depth])
        xyz[:, 2] = np.clip(xyz[:, 2], 0.10, 0.80)
    elif config.motion == "pure_rotation":
        xyz = np.column_stack((
            0.03 * a * np.sin(0.3 * w * t),
            0.03 * a * np.sin(0.4 * w * t),
            np.full_like(t, depth),
        ))
    elif config.motion == "translation_xyz":
        xyz = np.column_stack((
            a * np.sin(w * t),
            0.85 * a * np.sin(0.71 * w * t + 0.7),
            depth + dz * np.sin(0.53 * w * t + 1.2),
        ))
    elif config.motion == "pitch_sweep":
        xyz = np.column_stack((
            0.08 * a * np.sin(0.4 * w * t),
            0.08 * a * np.sin(0.3 * w * t),
            np.full_like(t, depth),
        ))
    elif config.motion == "stop_go":
        raw = np.sin(w * t)
        held = np.sign(raw) * np.maximum((np.abs(raw) - 0.35) / 0.65, 0.0) ** 2
        xyz = np.column_stack((
            a * held,
            0.55 * a * np.sign(np.sin(0.63 * w * t)) * np.maximum(
                (np.abs(np.sin(0.63 * w * t)) - 0.4) / 0.6, 0.0
            ) ** 2,
            depth + 0.35 * dz * held,
        ))
    elif config.motion in ("handheld_pause", "handheld_free"):
        # Non-periodic handheld motion driven by a random-speed progress
        # variable; rotation follows the same progress, so pauses stop both.
        u = _handheld_progress(t, rng, pauses=config.motion == "handheld_pause",
                               interval=config.pause_interval_s,
                               duration=config.pause_duration_s)
        knot = float(rng.uniform(0.8, 1.8))
        offset = _smooth_random(u, rng, 3, np.array([a, 0.8 * a, 0.6 * max(dz, 0.3 * a)]),
                                knot_s=knot)
        xyz = offset + np.array([0.0, 0.0, depth])
        xyz[:, 2] = np.clip(xyz[:, 2], 0.12, 0.45)
        handheld_rot = _smooth_random(u, rng, 3, np.full(3, np.radians(config.rotation_deg)),
                                      knot_s=knot)
    elif config.motion == "spiral":
        u = np.clip(t / max(t[-1], 1e-6), 0.0, 1.0)
        radius = a * (0.2 + 0.8 * u)
        xyz = np.column_stack((
            radius * np.cos(2.2 * w * t),
            0.75 * radius * np.sin(2.2 * w * t),
            depth + dz * (u - 0.5) + 0.2 * dz * np.sin(0.5 * w * t),
        ))
    else:
        xyz = np.column_stack((
            a * (0.65 * np.sin(w * t) + 0.25 * np.sin(2.3 * w * t)),
            0.7 * a * np.sin(0.73 * w * t + 0.4),
            depth + dz * (0.55 * np.sin(0.52 * w * t) + 0.2 * np.sin(1.7 * w * t)),
        ))

    independent_board = config.board_motion and config.board_motion_mode == "independent"
    if independent_board:
        bm = config.board_motion_mm / 1000.0
        br = np.radians(config.board_rotation_deg)
        p_board = _smooth_random(t, rng, 3, np.full(3, bm), knot_s=1.2)
        board_rotvec = _smooth_random(t, rng, 3, np.full(3, br), knot_s=1.2)
        R_W_B = Rotation.from_rotvec(board_rotvec).as_matrix()
    elif config.board_motion:
        bm = config.board_motion_mm / 1000.0
        p_board = np.column_stack((
            bm * np.sin(0.45 * w * t),
            0.6 * bm * np.sin(0.61 * w * t + 0.5),
            0.2 * bm * np.sin(0.39 * w * t),
        ))
        board_euler = np.column_stack((
            np.radians(2.0) * np.sin(0.43 * w * t),
            np.radians(2.0) * np.sin(0.37 * w * t),
            np.radians(3.0) * np.sin(0.31 * w * t),
        ))
        R_W_B = Rotation.from_euler("xyz", board_euler).as_matrix()
    else:
        p_board = np.zeros((len(t), 3))
        R_W_B = np.repeat(np.eye(3)[None], len(t), axis=0)

    amp = np.radians(config.rotation_deg)
    independent = np.column_stack((
        amp * 0.45 * np.sin(0.83 * w * t + 0.2),
        amp * 0.45 * np.sin(0.67 * w * t + 1.0),
        amp * 0.65 * np.sin(0.91 * w * t + 0.6),
    ))
    if config.motion == "random_spline":
        independent += _smooth_random(t, rng, 3, np.full(3, amp * 0.35))
    elif config.motion == "aggressive_6dof":
        independent = _smooth_random(t, rng, 3, np.full(3, amp), knot_s=0.4)
    elif config.motion in ("handheld_pause", "handheld_free"):
        independent = handheld_rot
    elif config.motion == "pure_rotation":
        independent = np.column_stack((
            amp * np.sin(0.83 * w * t),
            amp * np.sin(0.61 * w * t + 0.8),
            amp * np.sin(0.97 * w * t + 1.3),
        ))
    elif config.motion == "pitch_sweep":
        independent = np.column_stack((
            amp * np.sin(w * t),
            0.08 * amp * np.sin(0.7 * w * t),
            0.05 * amp * np.sin(0.4 * w * t),
        ))
    elif config.motion in {"slow_far", "translation_xyz", "stop_go", "spiral"}:
        independent *= 0.25
    xyz[:, :2] += board_center[:2]
    R_W_C = np.empty((len(t), 3, 3))
    fixed_target = board_center
    R_fixed = _look_at(xyz[0], fixed_target)
    for i in range(len(t)):
        target = fixed_target if independent_board else p_board[i] + R_W_B[i] @ board_center
        look = _look_at(xyz[i], target)
        correction = Rotation.from_matrix(R_fixed.T @ look).as_rotvec()
        base = R_fixed @ Rotation.from_rotvec(config.tracking_gain * correction).as_matrix()
        R_W_C[i] = base @ Rotation.from_euler("xyz", independent[i]).as_matrix()
    return PoseStream(t=t, R_W_C=R_W_C, p_W_C=xyz, R_W_B=R_W_B, p_W_B=p_board)


def _profile() -> dict:
    path = SIM_ASSETS / "imu_profile_usb.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run tools/fit_imu_sim_profile.py first: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _quantize(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.round(x / q.reshape(1, 3)) * q.reshape(1, 3)


def simulate_imu(
    t: np.ndarray,
    R_W_S: np.ndarray,
    p_W_S: np.ndarray,
    profile: dict,
    rng: np.random.Generator,
    noise_scale: float,
    ideal: bool = False,
    gravity_dir: tuple[float, float, float] | None = None,
) -> np.ndarray:
    dt = 1.0 / float(profile["sample_rate_hz"])
    velocity = np.gradient(p_W_S, t, axis=0, edge_order=2)
    accel_world = np.gradient(velocity, t, axis=0, edge_order=2)
    g_dir = np.asarray(gravity_dir if gravity_dir is not None else (0.0, 0.0, -1.0), dtype=np.float64)
    gravity = G * g_dir / np.linalg.norm(g_dir)
    specific = np.einsum("nji,nj->ni", R_W_S, accel_world - gravity) / G

    rel = np.einsum("nji,njk->nik", R_W_S[:-1], R_W_S[1:])
    rotvec = Rotation.from_matrix(rel).as_rotvec()
    gyro = np.vstack((rotvec[0], rotvec)) / np.diff(t, prepend=t[0] - dt)[:, None]
    gyro = np.degrees(gyro)

    ap = profile["accelerometer"]
    gp = profile["gyroscope"]
    if ideal:
        acc_meas = specific
        gyro_meas = gyro
    else:
        acc_white_cov = np.asarray(ap["white_noise_covariance"]) * noise_scale ** 2
        gyro_white_cov = np.asarray(gp["white_noise_covariance"]) * noise_scale ** 2
        acc_walk_cov = np.asarray(ap["bias_random_walk_covariance_per_1s"]) * noise_scale ** 2
        gyro_walk_cov = np.asarray(gp["bias_random_walk_covariance_per_1s"]) * noise_scale ** 2
        acc_bias = np.cumsum(
            rng.multivariate_normal(np.zeros(3), acc_walk_cov * dt, len(specific)),
            axis=0,
        )
        gyro_bias = np.asarray(gp["static_bias"]) + np.cumsum(
            rng.multivariate_normal(np.zeros(3), gyro_walk_cov * dt, len(gyro)),
            axis=0,
        )
        acc_meas = specific + acc_bias + rng.multivariate_normal(
            np.zeros(3), acc_white_cov, len(specific)
        )
        gyro_meas = gyro + gyro_bias + rng.multivariate_normal(
            np.zeros(3), gyro_white_cov, len(gyro)
        )
        # Match the configured WitMotion full-scale ranges.
        acc_meas = np.clip(acc_meas, -16.0, 16.0)
        gyro_meas = np.clip(gyro_meas, -2000.0, 2000.0)
        acc_meas = _quantize(acc_meas, np.asarray(ap["quantization"]))
        gyro_meas = _quantize(gyro_meas, np.asarray(gp["quantization"]))

    rot = Rotation.from_matrix(R_W_S)
    euler = rot.as_euler("xyz", degrees=True)
    quat_xyzw = rot.as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    field_world = np.array([-7600.0, -1300.0, -2700.0])
    mag = np.einsum("nji,j->ni", R_W_S, field_world)
    mag += rng.normal(0.0, 5.0 * noise_scale, mag.shape)
    pressure = np.full((len(t), 1), 1004.15)
    altitude = np.full((len(t), 1), 77.55)
    temp = np.full((len(t), 1), float(profile["temperature_c"]["mean"]))
    return np.column_stack((acc_meas, gyro_meas, euler, mag, pressure, altitude, temp, quat_wxyz))


def build_sequence(config: SimConfig) -> SyntheticSequence:
    profile = _profile()
    time_rng = np.random.default_rng(config.seed + 500)
    count = max(2, int(round(config.duration_s * config.imu_hz)))
    nominal_dt = 1.0 / config.imu_hz
    jitter_std = float(profile["sample_dt_std_ms"]) / 1000.0
    dts = nominal_dt + time_rng.normal(0.0, jitter_std, count - 1)
    dts = np.clip(dts, 0.55 * nominal_dt, 1.45 * nominal_dt)
    timestamps = np.concatenate(([0.0], np.cumsum(dts)))
    poses = generate_poses(config, timestamps)
    rng = np.random.default_rng(config.seed + 1000)
    lever = np.array([
        config.imu_lever_arm_mm_x,
        config.imu_lever_arm_mm_y,
        config.imu_lever_arm_mm_z,
    ]) / 1000.0
    R_W_I = np.einsum("nij,jk->nik", poses.R_W_C, R_CAMERA_IMU)
    p_W_I = poses.p_W_C + np.einsum("nij,j->ni", poses.R_W_C, lever)
    g_dir = config.gravity_board
    imu_usb = simulate_imu(poses.t, R_W_I, p_W_I, profile, rng, config.noise_scale, gravity_dir=g_dir)
    imu_usb_ideal = simulate_imu(
        poses.t, R_W_I, p_W_I, profile,
        np.random.default_rng(config.seed + 1001), 0.0, ideal=True, gravity_dir=g_dir,
    )
    imu_bt = simulate_imu(
        poses.t,
        poses.R_W_B,
        poses.p_W_B,
        profile,
        np.random.default_rng(config.seed + 2000),
        config.noise_scale,
        gravity_dir=g_dir,
    )
    imu_bt_ideal = simulate_imu(
        poses.t, poses.R_W_B, poses.p_W_B, profile,
        np.random.default_rng(config.seed + 2001), 0.0, ideal=True, gravity_dir=g_dir,
    )
    return SyntheticSequence(
        config=config,
        poses=poses,
        imu_usb=imu_usb,
        imu_bt=imu_bt,
        imu_usb_ideal=imu_usb_ideal,
        imu_bt_ideal=imu_bt_ideal,
        profile=profile,
    )


class StereoRenderer:
    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self.K0, self.d0 = _load_camera("cam0", config.width, config.height)
        self.K1, self.d1 = _load_camera("cam1", config.width, config.height)
        self.texture = self._checker_texture()
        self._maps = {
            "cam0": self._distortion_map(self.K0, self.d0),
            "cam1": self._distortion_map(self.K1, self.d1),
        }

    def _checker_texture(self) -> np.ndarray:
        px = 70
        h = self.config.board_rows * px
        w = self.config.board_cols * px
        image = np.empty((h, w, 3), dtype=np.uint8)
        for row in range(self.config.board_rows):
            for col in range(self.config.board_cols):
                value = 235 if (row + col) % 2 == 0 else 18
                image[row * px:(row + 1) * px, col * px:(col + 1) * px] = value
        return image

    def _distortion_map(self, K: np.ndarray, dist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        yy, xx = np.mgrid[0:self.config.height, 0:self.config.width].astype(np.float32)
        points = np.stack((xx, yy), axis=-1).reshape(-1, 1, 2)
        undistorted = cv2.undistortPoints(points, K, dist, P=K).reshape(self.config.height, self.config.width, 2)
        return undistorted[:, :, 0].astype(np.float32), undistorted[:, :, 1].astype(np.float32)

    def _background(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        small = rng.normal(0.0, 1.0, (30, 40)).astype(np.float32)
        smooth = cv2.resize(small, (self.config.width, self.config.height), interpolation=cv2.INTER_CUBIC)
        smooth = cv2.GaussianBlur(smooth, (0, 0), 18)
        smooth = (smooth - smooth.min()) / max(float(np.ptp(smooth)), 1e-6)
        base = np.empty((self.config.height, self.config.width, 3), dtype=np.float32)
        base[:, :, 0] = 42 + 35 * smooth
        base[:, :, 1] = 48 + 28 * smooth
        base[:, :, 2] = 78 + 55 * smooth
        return base

    def render(
        self,
        R_W_C: np.ndarray,
        p_W_C: np.ndarray,
        R_W_B: np.ndarray,
        p_W_B: np.ndarray,
        index: int,
        right: bool = False,
        angular_speed_dps: float = 0.0,
    ) -> np.ndarray:
        K, dist = (self.K1, self.d1) if right else (self.K0, self.d0)
        if right:
            p_W_C = p_W_C + R_W_C @ np.array([self.config.baseline_mm / 1000.0, 0.0, 0.0])
        square = self.config.square_mm / 1000.0
        # Board frame matches build_pose_gt.py: origin at the first inner
        # corner; the printed board extends one square beyond that origin.
        corners_B = np.array([
            [-square, -square, 0.0],
            [(self.config.board_cols - 1) * square, -square, 0.0],
            [(self.config.board_cols - 1) * square, (self.config.board_rows - 1) * square, 0.0],
            [-square, (self.config.board_rows - 1) * square, 0.0],
        ])
        corners_W = (R_W_B @ corners_B.T).T + p_W_B
        R_C_W = R_W_C.T
        t_C_W = -R_C_W @ p_W_C
        rvec, _ = cv2.Rodrigues(R_C_W)
        projected, _ = cv2.projectPoints(corners_W, rvec, t_C_W, K, np.zeros(5))
        dst = projected.reshape(4, 2).astype(np.float32)
        tex_h, tex_w = self.texture.shape[:2]
        src = np.array([[0, 0], [tex_w - 1, 0], [tex_w - 1, tex_h - 1], [0, tex_h - 1]], np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(self.texture, H, (self.config.width, self.config.height))
        mask = cv2.warpPerspective(np.full((tex_h, tex_w), 255, np.uint8), H, (self.config.width, self.config.height))
        background = self._background(self.config.seed + index // 12 + (17 if right else 0))

        yy, xx = np.mgrid[0:self.config.height, 0:self.config.width]
        gradient = (
            1.0
            + 0.18 * (xx / max(self.config.width - 1, 1) - 0.5)
            - 0.12 * (yy / max(self.config.height - 1, 1) - 0.5)
        )
        flicker = 1.0 + self.config.light_flicker * np.sin(index * 0.17)
        board = warped.astype(np.float32) * self.config.light_level * flicker * gradient[:, :, None]
        image = background
        m = mask.astype(np.float32)[:, :, None] / 255.0
        image = image * (1.0 - m) + board * m

        radius = ((xx - self.config.width / 2) / self.config.width) ** 2 + ((yy - self.config.height / 2) / self.config.height) ** 2
        image *= np.clip(1.0 - 0.75 * radius, 0.45, 1.0)[:, :, None]
        blur = int(round(self.config.motion_blur * angular_speed_dps / 20.0))
        if blur >= 1:
            kernel = 2 * blur + 1
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
        rng = np.random.default_rng(self.config.seed * 100000 + index + (50000 if right else 0))
        image += rng.normal(0.0, self.config.image_noise_std, image.shape)
        image = np.clip(image, 0, 255).astype(np.uint8)
        map_x, map_y = self._maps["cam1" if right else "cam0"]
        return cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def _write_imu(path: Path, timestamps: np.ndarray, values: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(IMU_COLUMNS)
        for t, row in zip(timestamps, values):
            writer.writerow([f"{t:.6f}", *[f"{float(v):.9f}" for v in row]])


def _blender_executable() -> Path:
    found = shutil.which("blender")
    candidates = [
        Path(found) if found else None,
        Path(r"C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"),
        Path(r"C:\Program Files\Blender Foundation\Blender\blender.exe"),
    ]
    for path in candidates:
        if path and path.is_file():
            return path
    raise FileNotFoundError("Blender not found. Install BlenderFoundation.Blender with winget.")


def _matrix4(R: np.ndarray, p: np.ndarray) -> list[list[float]]:
    matrix = np.eye(4)
    matrix[:3, :3] = R
    matrix[:3, 3] = p
    return matrix.tolist()


def _render_with_blender(
    sequence: SyntheticSequence,
    renderer: StereoRenderer,
    out: Path,
    frame_idx: np.ndarray,
    progress=None,
) -> None:
    config = sequence.config
    # OpenCV camera (+x right, +y down, +z forward) to Blender camera
    # (+x right, +y up, -z forward).
    cv_to_blender = np.diag([1.0, -1.0, -1.0])
    frames = []
    for k in frame_idx:
        R_left = sequence.poses.R_W_C[k] @ cv_to_blender
        p_left = sequence.poses.p_W_C[k]
        p_right = p_left + sequence.poses.R_W_C[k] @ np.array([config.baseline_mm / 1000.0, 0.0, 0.0])
        frames.append({
            "left_matrix": _matrix4(R_left, p_left),
            "right_matrix": _matrix4(R_left, p_right),
            "board_matrix": _matrix4(sequence.poses.R_W_B[k], sequence.poses.p_W_B[k]),
        })
    job = {
        "config": asdict(config),
        "K0": renderer.K0.tolist(),
        "K1": renderer.K1.tolist(),
        "left_dir": str((out / "cam0" / "images").resolve()),
        "right_dir": str((out / "cam1" / "images").resolve()),
        "frames": frames,
    }
    job_path = out / "blender_job.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    script = Path(__file__).with_name("blender_render.py")
    command = [
        str(_blender_executable()),
        "--background",
        "--python",
        str(script),
        "--",
        "--job",
        str(job_path),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    done = 0
    log_lines = []
    assert process.stdout is not None
    for line in process.stdout:
        log_lines.append(line.rstrip())
        if line.startswith("BLENDER_FRAME "):
            done += 1
            if progress:
                progress(done, len(frame_idx))
    code = process.wait()
    (out / "blender_render.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    if code or done != len(frame_idx):
        tail = "\n".join(log_lines[-20:])
        raise RuntimeError(
            f"Blender render failed: exit={code}, frames={done}/{len(frame_idx)}\n{tail}"
        )

    # Blender renders an ideal pinhole image. Apply the measured lens
    # distortion afterward so both backends share the same camera model.
    for cam, maps in (("cam0", renderer._maps["cam0"]), ("cam1", renderer._maps["cam1"])):
        map_x, map_y = maps
        for path in sorted((out / cam / "images").glob("*.jpg")):
            image = cv2.imread(str(path))
            distorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            cv2.imwrite(str(path), distorted, [cv2.IMWRITE_JPEG_QUALITY, 92])


def export_sequence(
    sequence: SyntheticSequence,
    output_root: Path = DATA / "synthetic",
    progress=None,
) -> Path:
    config = sequence.config
    name = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}_s{config.seed}"
    out = output_root / name
    for cam in ("cam0", "cam1"):
        (out / cam).mkdir(parents=True, exist_ok=True)
        if config.render_images:
            (out / cam / "images").mkdir(parents=True, exist_ok=True)
    renderer = StereoRenderer(config) if config.render_images else None
    t = sequence.poses.t
    frame_t = np.arange(0.0, min(config.duration_s, float(t[-1])), 1.0 / config.camera_fps)
    frame_idx = np.searchsorted(t, frame_t)
    frame_idx = np.clip(frame_idx, 1, len(t) - 1)
    left_idx = frame_idx - 1
    choose_left = np.abs(t[left_idx] - frame_t) <= np.abs(t[frame_idx] - frame_t)
    frame_idx = np.where(choose_left, left_idx, frame_idx)
    timestamp0 = time.time()
    timestamps = timestamp0 + t
    camera_timestamps = timestamp0 + frame_t
    if config.render_images and config.render_backend == "blender":
        assert renderer is not None
        _render_with_blender(sequence, renderer, out, frame_idx, progress)
    elif config.render_images:
        assert renderer is not None
        gyro_norm = np.linalg.norm(sequence.imu_usb[:, 3:6], axis=1)
        for number, k in enumerate(frame_idx):
            left = renderer.render(
                sequence.poses.R_W_C[k], sequence.poses.p_W_C[k],
                sequence.poses.R_W_B[k], sequence.poses.p_W_B[k],
                number, False, float(gyro_norm[k]),
            )
            right = renderer.render(
                sequence.poses.R_W_C[k], sequence.poses.p_W_C[k],
                sequence.poses.R_W_B[k], sequence.poses.p_W_B[k],
                number, True, float(gyro_norm[k]),
            )
            cv2.imwrite(str(out / "cam0" / "images" / f"{number:06d}.jpg"), left, [cv2.IMWRITE_JPEG_QUALITY, 92])
            cv2.imwrite(str(out / "cam1" / "images" / f"{number:06d}.jpg"), right, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if progress:
                progress(number + 1, len(frame_idx))
    elif progress:
        progress(len(frame_idx), len(frame_idx))
    for cam in ("cam0", "cam1"):
        (out / cam / "times.txt").write_text(
            "\n".join(f"{v:.6f}" for v in camera_timestamps) + "\n",
            encoding="utf-8",
        )
    _write_imu(out / "imu_stream.csv", timestamps, sequence.imu_usb)
    _write_imu(out / "imu_bt.csv", timestamps, sequence.imu_bt)
    _write_imu(out / "imu_stream_ideal.csv", timestamps, sequence.imu_usb_ideal)
    _write_imu(out / "imu_bt_ideal.csv", timestamps, sequence.imu_bt_ideal)

    R_B_W = np.transpose(sequence.poses.R_W_B[frame_idx], (0, 2, 1))
    R_B_C = np.einsum("nij,njk->nik", R_B_W, sequence.poses.R_W_C[frame_idx])
    p_B_C = np.einsum(
        "nij,nj->ni",
        R_B_W,
        sequence.poses.p_W_C[frame_idx] - sequence.poses.p_W_B[frame_idx],
    )
    np.savez_compressed(
        out / "ground_truth.npz",
        t=camera_timestamps,
        R=R_B_C.astype(np.float32),
        p=p_B_C.astype(np.float32),
        R_W_C=sequence.poses.R_W_C[frame_idx].astype(np.float32),
        p_W_C=sequence.poses.p_W_C[frame_idx].astype(np.float32),
        R_W_B=sequence.poses.R_W_B[frame_idx].astype(np.float32),
        p_W_B=sequence.poses.p_W_B[frame_idx].astype(np.float32),
        imu_t=timestamps,
        imu_usb_ideal=sequence.imu_usb_ideal.astype(np.float32),
        imu_bt_ideal=sequence.imu_bt_ideal.astype(np.float32),
        imu_R_W_C=sequence.poses.R_W_C.astype(np.float32),
        imu_p_W_C=sequence.poses.p_W_C.astype(np.float32),
        imu_R_W_B=sequence.poses.R_W_B.astype(np.float32),
        imu_p_W_B=sequence.poses.p_W_B.astype(np.float32),
        image_size=np.array([config.width, config.height]),
    )
    metadata = {
        "kind": "synthetic_stereo_imu",
        "name": name,
        "config": asdict(config),
        "imu_profile": sequence.profile,
        "camera_calibration": str(CALIB),
        "R_camera_imu": R_CAMERA_IMU.tolist(),
        "truth_convention": "R_B_C camera orientation in board frame; p_B_C camera center in board frame (m)",
        "provisional": {
            "stereo_extrinsic": "nominal identity rotation and 46.5 mm +X baseline",
            "imu_lever_arm": "user-configured; default zero until measured",
            "renderer": config.render_backend,
        },
    }
    (out / "session_meta.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return out
