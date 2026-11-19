#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Virtual capture day for the revision experiments, in the digital twin.

The twin renders the left camera and simulates both IMUs; the guided-capture
logic of ego_capture/revision.py (StopRun, CaliperRun, LivePose, the live
stillness and hold checks, the plan file and its verdicts) runs unchanged on
a simulated clock, fed by the live chessboard detector of the capture
program. Every recording is written in the format of the capture program
into a sandbox data root (default datasets/revision_sim), and the
post-recording checks of the program run on it:

  caliper_<day>_*  9 traversals, board on the caliper jaw at about 150 mm
  rigid_<day>_*    step 9, rotation-rich handheld motion; the day's
                   camera<-IMU rotation goes to rigidity.json
  traj_<day>_*     the stop experiment in the order C, P, P, C

The simulated operator sets each caliper position only approximately, locks
the jaw (which shifts it slightly), reads the vernier (0.02-mm graduation,
+0.02-mm zero error) and now and then bumps the table during a hold. In the
stop experiment the camera stops after a reaction time at every beat and is
held with residual tremor, braced on most stops. The IMU is mounted 6 deg
away from the nominal mount, so the injection of the day's rotation in the
analysis is exercised. Not modelled: lever arm, camera-IMU delay, board
motion blur, board flatness.

    python tools/sim_revision_capture.py                 # full day (about 40 min)
    python tools/sim_revision_capture.py --quick         # smoke test
    python tools/revision_pipeline.py --data-root datasets/revision_sim
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time as _time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ego_capture.revision as rv
import ego_sim.core as sim
import tools.build_pose_gt as bp
import tools.check_rigidity as cr
import tools.sim_reference_validation as svr
from ego_capture.cameras import detect_board, refine_corners
from ego_capture.sync import align_session

DATA = ROOT / "datasets"
VERNIER_MM = 0.02
FPS = 10.0                 # frame rate of the real recordings at 1280x720 with JPEG saving
IMU_HZ, BT_HZ = 200.0, 186.0
JPEG_QUALITY = 85          # ego_capture/cameras.py
PSF_PX, NOISE = 1.5, 2.0   # matches the real edge width (Section V-B)
DAY0 = datetime(2026, 11, 8, 9, 0, 0)
GRAVITY_BOARD = (0.0, 0.87, 0.49)   # stop and rigid sessions: world = board frame
GRAVITY_CAMERA = (0.0, 1.0, 0.0)    # caliper: world = camera frame, camera level
TREMOR_DPS = {"braced": 0.2, "unbraced": 1.6, "moving": 0.8}


class Clock:
    def __init__(self, t: float) -> None:
        self.t = float(t)

    def __call__(self) -> float:
        return self.t


CLOCK = Clock(DAY0.timestamp())


def log(msg: str) -> None:
    print(f"[{datetime.fromtimestamp(CLOCK()).strftime('%H:%M:%S')}] {msg}", flush=True)


def sandbox(root: Path, fresh: bool) -> None:
    if fresh and root.exists():
        shutil.rmtree(root)
    (root / "pose_gt").mkdir(parents=True, exist_ok=True)
    (root / "pose_gt_raw").mkdir(exist_ok=True)
    rv.DATA_ROOT = root
    rv.PLAN_PATH = root / "revision_experiment_plan.json"
    rv.CALIPER_PLAN_CSV = root / "reference_caliper_plan.csv"
    rv.CLOCK, rv.SOUND, rv.THREADED_METRONOME = CLOCK, False, False
    rv.camera_config.CONFIG = root / "active_camera.json"     # the twin renders the endoscope camera
    bp.DATA, bp.OUT = root, root / "pose_gt"
    cr.DATA = root


def day_mount(deg: float, seed: int = 3) -> np.ndarray:
    axis = np.random.default_rng(seed).normal(size=3)
    axis /= np.linalg.norm(axis)
    return rv.NOMINAL_R_CI @ Rotation.from_rotvec(np.radians(deg) * axis).as_matrix()


# --------------------------------------------------------------------------
# sensors
# --------------------------------------------------------------------------
class ImuNoise:
    """The noise model of ego_sim.core.simulate_imu, drawn once per recording
    so that samples can be produced while the clock advances."""

    def __init__(self, profile: dict, rate: float, seconds: float, rng, scale: float = 1.0) -> None:
        n = int(rate * seconds) + 16
        dt = 1.0 / rate
        ap, gp = profile["accelerometer"], profile["gyroscope"]
        mvn = lambda cov: rng.multivariate_normal(np.zeros(3), np.asarray(cov), n)  # noqa: E731
        self.acc = np.cumsum(mvn(np.asarray(ap["bias_random_walk_covariance_per_1s"]) * scale ** 2 * dt), 0) \
            + mvn(np.asarray(ap["white_noise_covariance"]) * scale ** 2)
        self.gyro = np.asarray(gp["static_bias"]) \
            + np.cumsum(mvn(np.asarray(gp["bias_random_walk_covariance_per_1s"]) * scale ** 2 * dt), 0) \
            + mvn(np.asarray(gp["white_noise_covariance"]) * scale ** 2)
        self.qa, self.qg = np.asarray(ap["quantization"]), np.asarray(gp["quantization"])
        self.mag = rng.normal(0.0, 5.0 * scale, (n, 3))
        self.temp = float(profile["temperature_c"]["mean"])

    def rows(self, k: int, R_W_S: np.ndarray, f_g: np.ndarray, w_dps: np.ndarray) -> np.ndarray:
        m = len(f_g)
        acc = np.clip(f_g + self.acc[k:k + m], -16.0, 16.0)
        gyro = np.clip(w_dps + self.gyro[k:k + m], -2000.0, 2000.0)
        acc = np.round(acc / self.qa) * self.qa
        gyro = np.round(gyro / self.qg) * self.qg
        rot = Rotation.from_matrix(R_W_S)
        quat = rot.as_quat()[:, [3, 0, 1, 2]]
        mag = np.einsum("nji,j->ni", R_W_S, np.array([-7600.0, -1300.0, -2700.0])) + self.mag[k:k + m]
        const = np.tile([1004.15, 77.55, self.temp], (m, 1))
        return np.column_stack((acc, gyro, rot.as_euler("xyz", degrees=True), mag, const, quat))


def kinematics(pose_fn, ts: np.ndarray, g_dir, h: float = 0.002):
    """Sensor orientation, specific force [g] and body rate [deg/s] by central differences."""
    R0, p0 = pose_fn(ts)
    Rm, pm = pose_fn(ts - h)
    Rp, pp = pose_fn(ts + h)
    acc = (pp - 2.0 * p0 + pm) / h ** 2
    w = Rotation.from_matrix(np.einsum("nji,njk->nik", Rm, Rp)).as_rotvec() / (2.0 * h)
    g = sim.G * np.asarray(g_dir, float) / np.linalg.norm(g_dir)
    return R0, np.einsum("nji,nj->ni", R0, acc - g) / sim.G, np.degrees(w)


class ImuStream:
    """One simulated IMU. It serves the live checks through tail(), like the
    ImuWorker of the capture program, and is written to the session CSV."""

    def __init__(self, rate: float, t0: float, seconds: float, profile: dict, rng) -> None:
        self.rate, self.t_next = rate, t0
        self.noise = ImuNoise(profile, rate, seconds, rng)
        cap = int(rate * seconds) + 16
        self.t = np.zeros(cap)
        self.data = np.zeros((cap, 19))
        self.n = 0

    def extend(self, ts: np.ndarray, R: np.ndarray, f: np.ndarray, w: np.ndarray) -> None:
        m = len(ts)
        self.t[self.n:self.n + m] = ts
        self.data[self.n:self.n + m] = self.noise.rows(self.n, R, f, w)
        self.n += m

    def advance(self, t_now: float, pose_fn, g_dir, mount: np.ndarray) -> None:
        if t_now < self.t_next:
            return
        m = int(np.floor((t_now - self.t_next) * self.rate)) + 1
        ts = self.t_next + np.arange(m) / self.rate
        self.t_next = ts[-1] + 1.0 / self.rate

        def sensor(tt):
            R, p = pose_fn(tt)
            return R @ mount, p
        R, f, w = kinematics(sensor, ts, g_dir)
        self.extend(ts, R, f, w)

    def tail(self, n: int = 2000):
        k = int(np.searchsorted(self.t[:self.n], CLOCK(), side="right"))
        lo = max(0, k - n)
        return [(float(self.t[i]), tuple(self.data[i, 0:3]), tuple(self.data[i, 3:6])) for i in range(lo, k)]

    @property
    def hz(self) -> float:
        return self.rate

    def write(self, path: Path) -> None:
        sim._write_imu(path, self.t[:self.n], self.data[:self.n])


class Grabber:
    """What LivePose reads from the capture program's grabber: the latest
    detection of the live detector, made every 0.4 s on the newest frame."""

    width, height = 1280, 720

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.frame_l = np.zeros((720, 1280), np.uint8)
        self.state = (False, None, (11, 8), 0.0)
        self.last = -1e9

    def offer(self, stamp: float, gray: np.ndarray) -> None:
        if stamp - self.last < 0.4:
            return
        self.last = stamp
        found, corners, pattern = detect_board(gray)
        if found and corners is not None:
            corners = refine_corners(gray, corners)
            found = corners is not None
        self.state = (bool(found), corners, pattern, float(stamp))

    def overlay(self, _side: str):
        return self.state


class Camera:
    """Left camera of the rig, in grey levels: the image formation of the
    twin's StereoRenderer with the corrections of the caliper simulation
    (anti-aliased board texture, converged distortion map), then optical blur
    and sensor noise. Constant terms are computed once, and the background is
    a plain desk plane behind the board: the twin's tiled ground carries
    optical flow but defeats the fast live detector of the capture program."""

    W, H = 1280, 720

    def __init__(self, seed: int, depth_mm: float, motion_blur: float, desk_z: float) -> None:
        cfg = sim.SimConfig(width=self.W, height=self.H, seed=seed, image_noise_std=0.0,
                            light_flicker=0.08, motion_blur=motion_blur)
        r = svr._renderer(cfg, depth_mm)
        self.cfg, self.K, self.desk_z = cfg, r.K0, desk_z
        self.tex = cv2.cvtColor(r.texture, cv2.COLOR_BGR2GRAY)
        self.tex_mask = np.full(self.tex.shape, 255, np.uint8)
        rng = np.random.default_rng(seed + 98)
        smooth = cv2.resize(rng.normal(0.0, 1.0, (16, 16)).astype(np.float32), (512, 512),
                            interpolation=cv2.INTER_CUBIC)
        self.desk = np.clip(160.0 + 12.0 * smooth + rng.normal(0.0, 3.0, (512, 512)), 0, 255).astype(np.uint8)
        yy, xx = np.mgrid[0:self.H, 0:self.W].astype(np.float32)
        gradient = 1.0 + 0.18 * (xx / (self.W - 1) - 0.5) - 0.12 * (yy / (self.H - 1) - 0.5)
        radius = ((xx - self.W / 2) / self.W) ** 2 + ((yy - self.H / 2) / self.H) ** 2
        self.gradient = gradient.astype(np.float32)
        self.vignette = np.clip(1.0 - 0.75 * radius, 0.45, 1.0).astype(np.float32)
        self.map1, self.map2 = cv2.convertMaps(*r._maps["cam0"], cv2.CV_16SC2)
        self.noise = [rng.standard_normal((self.H, self.W), dtype=np.float32) * NOISE for _ in range(24)]
        self.rng = rng
        self.index = 0
        self._desk_cache = None
        sq = cfg.square_mm / 1000.0
        self.corners_B = np.array([[-sq, -sq, 0.0], [(cfg.board_cols - 1) * sq, -sq, 0.0],
                                   [(cfg.board_cols - 1) * sq, (cfg.board_rows - 1) * sq, 0.0],
                                   [-sq, (cfg.board_rows - 1) * sq, 0.0]])

    def _warp(self, src_img, pts_W, rvec, tvec):
        proj, _ = cv2.projectPoints(pts_W, rvec, tvec, self.K, np.zeros(5))
        h, w = src_img.shape[:2]
        src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
        Hm = cv2.getPerspectiveTransform(src, proj.reshape(4, 2).astype(np.float32))
        return Hm

    def _desk_image(self, rvec, tvec, fixed: bool) -> np.ndarray:
        if fixed and self._desk_cache is not None:
            return self._desk_cache
        half = 2.0
        plane = np.array([[-half, -half, self.desk_z], [half, -half, self.desk_z],
                          [half, half, self.desk_z], [-half, half, self.desk_z]])
        Hm = self._warp(self.desk, plane, rvec, tvec)
        img = cv2.warpPerspective(self.desk, Hm, (self.W, self.H), borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
        if fixed:
            self._desk_cache = img
        return img

    def shoot(self, R_W_C, p_W_C, R_W_B, p_W_B, w_dps: float = 0.0, fixed: bool = False) -> np.ndarray:
        R_C_W = np.asarray(R_W_C).T
        tvec = -R_C_W @ np.asarray(p_W_C)
        rvec, _ = cv2.Rodrigues(R_C_W)
        corners_W = (np.asarray(R_W_B) @ self.corners_B.T).T + np.asarray(p_W_B)
        Hm = self._warp(self.tex, corners_W, rvec, tvec)
        board = cv2.warpPerspective(self.tex, Hm, (self.W, self.H)).astype(np.float32)
        m = cv2.warpPerspective(self.tex_mask, Hm, (self.W, self.H)).astype(np.float32) / 255.0
        flicker = 1.0 + self.cfg.light_flicker * np.sin(self.index * 0.17)
        img = self._desk_image(rvec, tvec, fixed) * (1.0 - m) + board * (flicker * self.gradient) * m
        img *= self.vignette
        blur = int(round(self.cfg.motion_blur * w_dps / 20.0))
        if blur >= 1:
            img = cv2.GaussianBlur(img, (2 * blur + 1, 2 * blur + 1), 0)
        img = cv2.remap(img, self.map1, self.map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        img = cv2.GaussianBlur(img, (0, 0), PSF_PX)
        k = int(self.rng.integers(len(self.noise)))
        img += np.roll(self.noise[k], int(self.rng.integers(self.W)), axis=1)
        self.index += 1
        return np.clip(img, 0, 255).astype(np.uint8)


class Recording:
    """Session folder in the format of the capture program."""

    def __init__(self, root: Path, prefix: str, step: str, meta_extra: dict) -> None:
        self.name = f"{prefix}_{datetime.fromtimestamp(CLOCK()).strftime('%Y%m%d_%H%M%S')}"
        self.dir = root / self.name
        (self.dir / "cam0" / "images").mkdir(parents=True)
        (self.dir / "checkerboard.yaml").write_text(
            "# GP050-3-12*9\ntarget_type: 'checkerboard'\ntargetCols: 11   # inner corners\ntargetRows: 8\n"
            "rowSpacingMeters: 0.003000\ncolSpacingMeters: 0.003000\n", encoding="utf-8")
        meta = {"rig": "stereo_endoscope_plus_dual_imu", "board": "GP050-3-12*9", "inner_corners": [11, 8],
                "square_mm": 3.0, "step": step, "cam_left_id": 0, "cam_right_id": 1,
                "imu_usb_port": "SIM", "imu_usb_baud": 921600, "imu_bt_id": "SIM", "imu_bt_kind": "ble",
                "target_imu_hz": 200, "requested": {"width": 1280, "height": 720, "fps": 15},
                "save_jpeg": True, "no_zoom": True, "fixed_extrinsics": True, "stereo_baseline_mm": 46.5,
                "lens_diameter_mm": 5.5, "created_at": datetime.fromtimestamp(CLOCK()).isoformat(timespec="seconds"),
                "protocol_id": "simulated", "prospective_sealed": False, "simulated": True,
                "intrinsics_dir": rv.camera_config.LEGACY_CALIB, **meta_extra}
        (self.dir / "session_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        self.times: list[float] = []

    def frame(self, stamp: float, img: np.ndarray) -> None:
        cv2.imwrite(str(self.dir / "cam0" / "images" / f"{len(self.times):06d}.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        self.times.append(stamp)

    def close(self, usb: ImuStream, bt: ImuStream) -> dict:
        (self.dir / "cam0" / "times.txt").write_text("".join(f"{t:.6f}\n" for t in self.times), encoding="utf-8")
        usb.write(self.dir / "imu_stream.csv")
        bt.write(self.dir / "imu_bt.csv")
        # Frames and IMU samples share the twin's exact clock: the delay is known
        # (zero) rather than estimated from a video the twin does not write.
        sync = align_session(self.dir)
        for cam in sync.get("cameras", {}).values():
            cam.update(delay_usb_s=0.0, delay_usb_reliable=True, delay_bt_s=0.0, delay_bt_reliable=True,
                       delay_known=True)
        write_json(self.dir / "sync_delay.json", sync)
        summary = {"frames": len(self.times), "seconds": float(self.times[-1] - self.times[0]),
                   "calib_left": 0, "calib_right": 0, "session_dir": str(self.dir), "drops": 0,
                   "sync": sync, "usb_hz": IMU_HZ, "bt_hz": BT_HZ,
                   "usb_samples": usb.n, "bt_samples": bt.n, "usb_reconnects": 0, "bt_reconnects": 0,
                   "simulated": True}
        (self.dir / "capture_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------
# caliper standard
# --------------------------------------------------------------------------
def _min_jerk(x):
    x = np.clip(x, 0.0, 1.0)
    return x ** 3 * (10.0 - 15.0 * x + 6.0 * x ** 2)


def _window(t, t0, t1, ramp):
    x = np.clip(np.minimum((t - t0) / ramp, (t1 - t) / ramp), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * x)


class CaliperWorld:
    """Board taped to the sliding jaw; jaw position s(t) in mm along `axis`.
    Hand contact shakes the board slightly, a bump of the table makes it ring."""

    def __init__(self, R0: np.ndarray, p_zero: np.ndarray, axis: np.ndarray, s0: float) -> None:
        self.R0, self.p_zero, self.axis, self.s0 = R0, p_zero, axis, s0
        self.steps: list[tuple[float, float, float]] = []
        self.shakes: list[tuple] = []
        self.bumps: list[tuple] = []

    def s(self, t) -> np.ndarray:
        t = np.atleast_1d(np.asarray(t, float))
        out = np.full(len(t), self.s0)
        for t0, T, d in self.steps:
            out += d * _min_jerk((t - t0) / T)
        return out

    def pose(self, t):
        t = np.atleast_1d(np.asarray(t, float))
        dp, dr = np.zeros((len(t), 3)), np.zeros((len(t), 3))
        for t0, t1, ap, ar, f, ph in self.shakes:
            if t1 + 0.5 < t[0] or t0 - 0.5 > t[-1]:
                continue
            env = _window(t, t0, t1, 0.2)[:, None]
            dp += env * ap * np.sin(2 * np.pi * f * t[:, None] + ph)
            dr += env * ar * np.sin(2 * np.pi * f * t[:, None] + ph + 1.0)
        for tb, ap, ar, f, tau in self.bumps:
            x = np.maximum(t - tb, 0.0)
            ring = np.exp(-x / tau) * np.sin(2 * np.pi * f * x) * (1.0 - np.exp(-x / 0.01))
            dp += ring[:, None] * ap
            dr += ring[:, None] * ar
        p = self.p_zero + np.outer(self.s(t), self.axis) / 1000.0 + (self.R0 @ dp.T).T / 1000.0
        R = self.R0 @ Rotation.from_rotvec(dr).as_matrix()
        return R, p


def caliper_trial(root: Path, plan: dict, entry: dict, world: CaliperWorld, camera: Camera,
                  R_ci: np.ndarray, profile: dict, rng, zero_error: float, bump_p: float) -> dict:
    info = plan["caliper_info"]
    run = rv.CaliperRun(entry, info["zero_reading_mm"], info)
    rec = Recording(root, "caliper", "caliper", run.meta())
    t0 = CLOCK()
    usb = ImuStream(IMU_HZ, t0, 900.0, profile, rng)
    bt = ImuStream(BT_HZ, t0, 900.0, profile, rng)
    still_cam = lambda tt: (np.repeat(np.eye(3)[None], len(np.atleast_1d(tt)), 0),  # noqa: E731
                            np.zeros((len(np.atleast_1d(tt)), 3)))
    grabber, pose = Grabber(), rv.LivePose()
    run.begin(t0)
    log(f"{rec.name}: {entry['axis']}{entry['trial']} " + " -> ".join(f"{v:g}" for v in entry["targets_mm"]))
    t_frame = t0
    confirm_at, retry, prev_state, holds, rejected, streak = None, False, run.state, 0, 0, 0
    while run.state != "done":
        now = CLOCK()
        if run.state == "move" and confirm_at is None:
            target = run.target
            cur = float(world.s(now)[0])
            react = rng.uniform(0.3, 0.8)
            if retry:
                t_read = rng.uniform(2.0, 4.0)
                world.shakes.append((now + react, now + react + t_read, rng.normal(0, 0.012, 3),
                                     np.radians(rng.normal(0, 0.02, 3)), rng.uniform(2, 5), rng.uniform(0, 6.3, 3)))
                confirm_at = now + react + t_read
            else:
                desired = target + (rng.uniform(0.0, 0.25) if target == 0 else rng.uniform(-0.3, 0.3))
                t_move = rng.uniform(0.5, 0.9) if run.fast else rng.uniform(1.2, 2.5)
                t_read = rng.uniform(4.0, 7.0)
                t_lock = now + react + t_move
                world.steps.append((now + react, t_move, desired - cur))
                world.steps.append((t_lock, 0.3, rng.normal(0.0, 0.008)))
                world.shakes.append((now + react, t_lock + t_read, rng.normal(0, 0.03, 3),
                                     np.radians(rng.normal(0, 0.06, 3)), rng.uniform(2, 5), rng.uniform(0, 6.3, 3)))
                confirm_at = t_lock + t_read
            retry = False
        if confirm_at is not None and now >= confirm_at:
            true = float(world.s(now)[0]) + zero_error
            reading = round(round((true + rng.normal(0.0, 0.006)) / VERNIER_MM) * VERNIER_MM, 2)
            err = run.confirm(reading, now)
            if err:
                raise RuntimeError(f"operator reading rejected: {err}")
            confirm_at = None
        t_next = now + 0.1
        usb.advance(t_next, still_cam, GRAVITY_CAMERA, R_ci)
        bt.advance(t_next, world.pose, GRAVITY_CAMERA, np.eye(3))
        while t_frame <= t_next:
            R_B, p_B = world.pose(t_frame)
            img = camera.shoot(np.eye(3), np.zeros(3), R_B[0], p_B[0], fixed=True)
            rec.frame(t_frame, img)
            grabber.offer(t_frame, img)
            t_frame += 1.0 / FPS + rng.normal(0.0, 0.003)
        CLOCK.t = t_next
        pose.update(grabber)
        msgs = run.tick(CLOCK(), pose, bt, usb)
        if run.state == "hold" and prev_state != "hold":
            holds += 1
            if rng.random() < bump_p:
                world.bumps.append((CLOCK() + rng.uniform(0.5, 5.0), rng.normal(0, 0.12, 3),
                                    np.radians(rng.normal(0, 0.4, 3)), rng.uniform(5, 8), 0.25))
                world.steps.append((world.bumps[-1][0], 0.05, rng.normal(0.0, 0.03)))
        for m in msgs:
            if m.startswith("rejected:"):
                rejected += 1
                streak += 1
                retry = True
                log(f"  hold rejected ({m.split(':', 1)[1]}); re-reading")
                if streak > 5:
                    raise RuntimeError(f"{rec.name}: six rejected holds in a row at {run.target:g} mm")
            elif m == "accepted":
                streak = 0
        prev_state = run.state
    events = run.summary()
    write_json(rec.dir / "caliper_events.json", events)
    summary = rec.close(usb, bt)
    qc = rv.qc_caliper(rec.name, events)
    write_json(rec.dir / "qc_result.json", qc)
    rv.record_result(plan, entry, rec.name, qc, events, "caliper")
    fit = qc.get("trial_fit") or {}
    log(f"  {'pass' if qc['ok'] else 'FAIL ' + '; '.join(qc['problems'])}: {summary['frames']} frames, "
        f"{holds} holds ({rejected} rejected), line fit scale {fit.get('scale')}, rms {fit.get('rms_mm')} mm")
    return qc


def caliper_day(root: Path, plan: dict, R_ci: np.ndarray, profile: dict, seed: int, bump_p: float,
                axes: tuple[str, ...]) -> None:
    rng = np.random.default_rng(seed)
    zero_error = 0.02
    plan["caliper_info"] = {"type": "vernier", "resolution_mm": 0.02, "zero_reading_mm": zero_error,
                            "caliper_id": "sim-0-150", "operator": "virtual operator",
                            "saved_at": datetime.fromtimestamp(CLOCK()).isoformat(timespec="seconds")}
    rv.save_plan(plan)
    directions = {"X": np.array([1.0, 0, 0]), "Y": np.array([0, 1.0, 0]), "Z": np.array([0, 0, -1.0])}
    cfg = sim.SimConfig(width=1280, height=720)
    for axis in axes:
        CLOCK.t += 300.0                       # mounting the caliper for this axis
        tilt = rng.uniform(18.0, 26.0)
        R0, p_centre = svr._board_pose(150.0, tilt, cfg)
        p_zero = p_centre - directions[axis] * 0.020          # travel centred in the image
        world = CaliperWorld(R0, p_zero, directions[axis], s0=rng.uniform(0.0, 0.2))
        camera = Camera(seed + ord(axis), 150.0, 0.0, desk_z=0.5)
        attempts = 0
        while True:
            entry = rv.next_entry(plan, "caliper")
            if entry is None or entry["axis"] != axis:
                break
            attempts += 1
            if attempts > 3 * rv.CALIPER_TRIALS:
                raise RuntimeError(f"axis {axis}: too many repeated trials")
            entry["status"] = "recording"
            caliper_trial(root, plan, entry, world, camera, R_ci, profile, rng, zero_error, bump_p)
            CLOCK.t += 60.0


# --------------------------------------------------------------------------
# rigidity check (step 9)
# --------------------------------------------------------------------------
def handheld_sensors(poses, t_rel: np.ndarray, cfg: sim.SimConfig, R_ci: np.ndarray, profile: dict,
                     seed: int, t0: float, tremor_dps: np.ndarray | None = None):
    """Camera orientation with hand tremor, and both IMU streams, for a
    handheld recording in front of the resting board."""
    R_W_C = poses.R_W_C
    if tremor_dps is not None:
        dt = float(np.median(np.diff(t_rel)))
        rng = np.random.default_rng(seed + 77)
        band = gaussian_filter1d(rng.normal(size=(len(t_rel), 3)), 0.015 / dt, axis=0)
        band -= gaussian_filter1d(band, 0.15 / dt, axis=0)
        band /= band.std(0, keepdims=True) * np.sqrt(3.0)
        theta = np.cumsum(np.radians(tremor_dps)[:, None] * band * dt, axis=0)
        R_W_C = R_W_C @ Rotation.from_rotvec(theta).as_matrix()
    usb = ImuStream(IMU_HZ, t0, t_rel[-1] + 5.0, profile, np.random.default_rng(seed))
    vals = sim.simulate_imu(t_rel, R_W_C @ R_ci, poses.p_W_C, profile, np.random.default_rng(seed + 1000),
                            cfg.noise_scale, gravity_dir=cfg.gravity_board)
    usb.t[:len(t_rel)], usb.data[:len(t_rel)], usb.n = t0 + t_rel, vals, len(t_rel)
    tb = np.arange(0.0, t_rel[-1], 1.0 / BT_HZ)
    bt = ImuStream(BT_HZ, t0, t_rel[-1] + 5.0, profile, np.random.default_rng(seed + 1))
    vals = sim.simulate_imu(tb, np.repeat(np.eye(3)[None], len(tb), 0), np.zeros((len(tb), 3)), profile,
                            np.random.default_rng(seed + 2000), cfg.noise_scale, gravity_dir=cfg.gravity_board)
    bt.t[:len(tb)], bt.data[:len(tb)], bt.n = t0 + tb, vals, len(tb)
    return R_W_C, usb, bt


def _film(rec: Recording, camera: Camera, grabber: Grabber, t_abs, R_W_C, p_W_C, R_W_B, p_W_B, gyro_dps,
          rng, t_end: float) -> None:
    t_frame = t_abs[0]
    while t_frame <= t_end:
        k = min(int(np.searchsorted(t_abs, t_frame)), len(t_abs) - 1)
        img = camera.shoot(R_W_C[k], p_W_C[k], R_W_B[k], p_W_B[k], float(np.linalg.norm(gyro_dps[k])))
        rec.frame(t_frame, img)
        grabber.offer(t_frame, img)
        t_frame += 1.0 / FPS + rng.normal(0.0, 0.003)


def rigidity_session(root: Path, R_ci: np.ndarray, profile: dict, seed: int, seconds: float = 60.0) -> dict:
    cfg = sim.SimConfig(duration_s=seconds, seed=seed, motion="random_spline", translation_mm=20.0,
                        depth_mm=160.0, depth_change_mm=25.0, rotation_deg=14.0, tracking_gain=0.2,
                        speed=1.0, gravity_board=GRAVITY_BOARD)
    t_rel = np.arange(0.0, seconds, 1.0 / IMU_HZ)
    poses = sim.generate_poses(cfg, t_rel)
    rec = Recording(root, "rigid", "rigid", {"theme": "", "split": ""})
    t_abs = CLOCK() + t_rel
    R_W_C, usb, bt = handheld_sensors(poses, t_rel, cfg, R_ci, profile, seed, t_abs[0])
    camera = Camera(seed, 160.0, 0.35, desk_z=-0.002)
    _film(rec, camera, Grabber(), t_abs, R_W_C, poses.p_W_C, poses.R_W_B, poses.p_W_B,
          usb.data[:usb.n, 3:6], np.random.default_rng(seed), t_abs[-1] - 0.2)
    CLOCK.t = t_abs[-1]
    rec.close(usb, bt)
    pose = bp.process_session(rec.name)
    rig = cr.check(rec.name)
    write_json(rec.dir / "qc_result.json", {"session": rec.name, "usable_frac": round(pose["pnp_still"] / max(pose["frames"], 1), 3),
                                            "reproj_p50_px": pose.get("reproj_p50"), "rigidity": rig.get("verdict", ""),
                                            "verdict": "合格" if rig.get("verdict") == "rigid" else "不合格：IMU 刚性不合格"})
    write_json(rec.dir / "rigidity.json", rig)
    ang = rv.rotation_angle_deg(np.asarray(rig["R_camera_imu"]), R_ci) if rig.get("R_camera_imu") else None
    log(f"{rec.name}: rigidity {rig.get('verdict')} ({rig.get('quality')}), estimate {ang if ang is None else round(ang, 2)} deg "
        f"from the true mount, true mount {rv.rotation_angle_deg(rv.NOMINAL_R_CI, R_ci):.1f} deg from nominal")
    return rig


# --------------------------------------------------------------------------
# stop experiment
# --------------------------------------------------------------------------
def stop_recording(root: Path, plan: dict, entry: dict, R_ci: np.ndarray, profile: dict, seed: int,
                   unbraced_p: float, practice_speed: dict) -> dict:
    rng = np.random.default_rng(seed)
    total = rv.REST_START_S + rv.MOTION_S + rv.REST_END_S + 1.0
    t_rel = np.arange(0.0, total, 1.0 / IMU_HZ)
    t_motion, t_rest_end = rv.REST_START_S, rv.REST_START_S + rv.MOTION_S
    beats = []
    k = 1
    while t_motion + k * rv.BEAT_S <= t_rest_end - 0.5:
        beats.append(t_motion + k * rv.BEAT_S)
        k += 1
    stops = []
    tremor = np.full(len(t_rel), TREMOR_DPS["moving"])
    rest = (t_rel < t_motion) | (t_rel > t_rest_end)
    tremor[rest] = TREMOR_DPS["braced"]

    def progress(t, prng, pauses=False, interval=12.5, duration=1.5):
        dt = np.diff(t, prepend=t[0])
        speed = np.exp(sim._smooth_random(t, prng, 1, np.array([0.45]), knot_s=2.0)[:, 0])
        gate = _window(t, t_motion, t_rest_end, 0.25)
        if entry["condition"] == "C":
            for c in prng.uniform(t_motion, t_rest_end, size=max(1, int(rv.MOTION_S / 12))):
                gate *= 1.0 - 0.7 * np.exp(-0.5 * ((t - c) / 0.6) ** 2)
        else:
            for b in beats:
                a, z = b + prng.uniform(0.25, 0.45), b + rv.HOLD_S + prng.uniform(0.2, 0.35)
                gate *= 1.0 - _window(t, a, z, 0.25)
                stops.append((a, z))
        return np.cumsum(speed * gate * dt)

    cfg = sim.SimConfig(duration_s=total, seed=seed,
                        motion="handheld_pause" if entry["condition"] == "P" else "handheld_free",
                        translation_mm=practice_speed["translation_mm"], depth_mm=practice_speed["depth_mm"],
                        depth_change_mm=practice_speed["depth_change_mm"], rotation_deg=practice_speed["rotation_deg"],
                        tracking_gain=0.15, noise_scale=1.0, gravity_board=GRAVITY_BOARD)
    saved = sim._handheld_progress
    sim._handheld_progress = progress
    try:
        poses = sim.generate_poses(cfg, t_rel)
    finally:
        sim._handheld_progress = saved
    for a, z in stops:
        inside = (t_rel >= a + 0.25) & (t_rel <= z - 0.25)
        tremor[inside] = TREMOR_DPS["unbraced" if rng.random() < unbraced_p else "braced"]
    run = rv.StopRun(entry)
    rec = Recording(root, "traj", "stopexp", run.meta())
    t_abs = CLOCK() + t_rel
    R_W_C, s_usb, s_bt = handheld_sensors(poses, t_rel, cfg, R_ci, profile, seed, t_abs[0], tremor)
    camera, grabber = Camera(seed, cfg.depth_mm, 0.35, desk_z=-0.002), Grabber()
    detections = []

    class Recorder(Grabber):
        def offer(self, stamp, gray):
            super().offer(stamp, gray)
            if self.state[3] == stamp:
                detections.append(self.state)

    rec_grabber = Recorder()
    _film(rec, camera, rec_grabber, t_abs, R_W_C, poses.p_W_C, poses.R_W_B, poses.p_W_B,
          s_usb.data[:s_usb.n, 3:6], rng, t_abs[0] + total - 0.6)
    pose = rv.LivePose()
    run.begin(t_abs[0])
    di = 0
    now = t_abs[0]
    while run.phase != "done":
        now += 0.1
        CLOCK.t = now
        while di < len(detections) and detections[di][3] <= now:
            grabber.state = detections[di]
            di += 1
        pose.update(grabber)
        run.tick(now, rv.still_now(s_usb, run.acc_ref), pose)
    events = run.summary()
    write_json(rec.dir / "stop_events.json", events)
    rec.close(s_usb, s_bt)
    qc = rv.qc_stop(rec.name, events, 0)
    write_json(rec.dir / "qc_result.json", qc)
    rv.record_result(plan, entry, rec.name, qc, events, "stop")
    se = qc["stop_events"]
    log(f"{rec.name}: {entry['label']} {'pass' if qc['ok'] else 'FAIL ' + '; '.join(qc['problems'])}; "
        f"usable {qc['usable_frac']:.0%}, reproj {qc['reproj_p50']:.2f} px, IMU stops {se['stops']} "
        f"({se['stops_per_min']:.1f}/min), beats {se.get('beats_matched')}/{se.get('beats')}, "
        f"speed {se.get('median_moving_speed_mm_s')} mm/s")
    return qc


def stop_day(root: Path, plan: dict, R_ci: np.ndarray, profile: dict, seed: int, unbraced_p: float) -> None:
    style = {"translation_mm": 28.0, "depth_mm": 160.0, "depth_change_mm": 35.0, "rotation_deg": 4.0}
    k = 0
    while True:
        entry = rv.next_entry(plan, "stop")
        if entry is None:
            break
        entry["status"] = "recording"
        stop_recording(root, plan, entry, R_ci, profile, seed + 17 * k, unbraced_p, style)
        CLOCK.t += 120.0
        k += 1
        if k > 3 * len(plan["stop"]):
            raise RuntimeError("too many repeated recordings")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DATA / "revision_sim")
    ap.add_argument("--only", choices=("caliper", "stop"))
    ap.add_argument("--mount-deg", type=float, default=6.0, help="IMU mount of the day vs. the nominal one")
    ap.add_argument("--bump-p", type=float, default=0.06, help="probability of a table bump per caliper hold")
    ap.add_argument("--unbraced-p", type=float, default=0.08, help="share of stops held without bracing")
    ap.add_argument("--blocks", type=int, default=1, help="C,P,P,C blocks of the stop experiment")
    ap.add_argument("--quick", action="store_true", help="smoke test: two caliper trials per axis on X and Z, "
                    "one C and one P recording of 60-s motion")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--keep", action="store_true", help="keep an existing sandbox (add recordings)")
    args = ap.parse_args()
    t_wall = _time.time()
    sandbox(args.root, fresh=not args.keep)
    profile = sim._profile()
    R_ci = day_mount(args.mount_deg)
    plan = rv.load_plan() if args.keep and rv.PLAN_PATH.is_file() else rv.default_plan()
    for _ in range(args.blocks - 1):
        rv.append_stop_block(plan)
    axes = ("X", "Y", "Z")
    if args.quick:
        rv.MOTION_S = 60.0
        plan["caliper"] = [e for e in plan["caliper"] if e["axis"] in ("X", "Z") and e["trial"] <= 2]
        plan["stop"] = plan["stop"][:2]
        axes = ("X", "Z")
    rv.save_plan(plan)
    CLOCK.t = DAY0.timestamp()
    if args.only in (None, "caliper"):
        caliper_day(args.root, plan, R_ci, profile, args.seed, args.bump_p, axes)
    if args.only in (None, "stop"):
        CLOCK.t = max(CLOCK.t, DAY0.replace(hour=13, minute=30).timestamp())
        rigidity_session(args.root, R_ci, profile, args.seed + 500)
        CLOCK.t += 300.0
        stop_day(args.root, plan, R_ci, profile, args.seed + 900, args.unbraced_p)
    truth = {"imu_mount_deg_from_nominal": args.mount_deg, "R_camera_imu_true": R_ci.tolist(),
             "caliper_zero_error_mm": 0.02, "bump_p": args.bump_p, "unbraced_p": args.unbraced_p,
             "quick": args.quick, "seed": args.seed, "wall_minutes": round((_time.time() - t_wall) / 60.0, 1)}
    write_json(args.root / "simulation_truth.json", truth)
    log(f"done in {truth['wall_minutes']} min; sandbox {args.root}")


if __name__ == "__main__":
    main()
