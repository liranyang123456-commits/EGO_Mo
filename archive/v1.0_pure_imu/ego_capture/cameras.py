#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Camera grabbers with queued recording and freeze watchdog."""

from __future__ import annotations

import csv
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

cv2.setNumThreads(1)

from .affinity import PRIOR_ABOVE, PRIOR_NORMAL, pin_current_thread
from .session import FRAMES_COLUMNS, INNER, SQUARE_MM, BOARD_NAME


def fourcc_str(cap: cv2.VideoCapture) -> str:
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(index))
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    try:
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    except Exception:
        pass
    return cap


def find_board(gray: np.ndarray) -> tuple[bool, Optional[np.ndarray], tuple[int, int]]:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    for pattern in (INNER, (INNER[1], INNER[0])):
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if (not ok) and hasattr(cv2, "findChessboardCornersSB"):
            try:
                ok, corners = cv2.findChessboardCornersSB(gray, pattern)
            except Exception:
                ok = False
        if ok and corners is not None:
            return True, corners.astype(np.float32), pattern
    return False, None, INNER


def detect_board(gray: np.ndarray) -> tuple[bool, Optional[np.ndarray], tuple[int, int]]:
    """Find a full board and return corners in the original image.

    The live view calls this off the grab thread. A fast reject runs first.
    Dark endoscope frames get a second pass with local contrast, still off the
    grab thread so a slow search cannot stall that camera.
    """
    height, width = gray.shape[:2]
    long_side = max(height, width)
    view = gray
    scale_x = scale_y = 1.0
    if long_side > 960:
        factor = 960.0 / float(long_side)
        new_w = max(1, int(round(width * factor)))
        new_h = max(1, int(round(height * factor)))
        view = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        scale_x = width / float(new_w)
        scale_y = height / float(new_h)
    fast = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    slow = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found = _search_patterns(view, fast)
    if found is None:
        try:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(view)
        except Exception:
            clahe = None
        if clahe is not None:
            found = _search_patterns(clahe, slow)
    if found is None:
        return False, None, INNER
    corners, pattern = found
    return True, _scale_corners(corners, scale_x, scale_y), pattern


def _search_patterns(image: np.ndarray, flags: int):
    for pattern in (INNER, (INNER[1], INNER[0])):
        ok, corners = cv2.findChessboardCorners(image, pattern, flags)
        if ok and corners is not None and len(corners) == pattern[0] * pattern[1]:
            return corners, pattern
    return None


def _scale_corners(corners: np.ndarray, scale_x: float, scale_y: float) -> np.ndarray:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2).copy()
    pts[:, 0, 0] *= scale_x
    pts[:, 0, 1] *= scale_y
    return pts


def refine_corners(gray: np.ndarray, corners: np.ndarray) -> Optional[np.ndarray]:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2).copy()
    height, width = gray.shape[:2]
    xy = pts.reshape(-1, 2)
    inside = (
        np.all(xy[:, 0] >= 3) and np.all(xy[:, 1] >= 3)
        and np.all(xy[:, 0] <= width - 4) and np.all(xy[:, 1] <= height - 4)
    )
    if not inside:
        return pts
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.05)
    try:
        cv2.cornerSubPix(gray, pts, (5, 5), (-1, -1), criteria)
    except cv2.error:
        return pts
    return pts


def chessboard_object_points() -> np.ndarray:
    cols, rows = INNER
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= SQUARE_MM
    return objp


def calibrate_from_corners(samples: list[np.ndarray], image_size: tuple[int, int]) -> Optional[dict[str, Any]]:
    if len(samples) < 12:
        return None
    objp = chessboard_object_points()
    obj_pts = [objp] * len(samples)
    img_pts = [c.reshape(-1, 1, 2).astype(np.float32) for c in samples]
    ok, k, dist, _r, _t = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)
    if not ok:
        return None
    mean_err = 0.0
    for i, corners in enumerate(img_pts):
        proj, _ = cv2.projectPoints(objp, _r[i], _t[i], k, dist)
        mean_err += float(np.sqrt(np.mean((proj - corners) ** 2)))
    mean_err /= max(len(img_pts), 1)
    return {
        "calibration_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chessboard": BOARD_NAME,
        "chessboard_size": list(INNER),
        "square_size_mm": SQUARE_MM,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "num_images": len(samples),
        "camera_matrix": k.tolist(),
        "distortion_coefficients": dist.reshape(-1).tolist(),
        "reprojection_error_pixels": mean_err,
        "parameters": {
            "fx": float(k[0, 0]), "fy": float(k[1, 1]),
            "cx": float(k[0, 2]), "cy": float(k[1, 2]),
        },
    }


def _fps(times: deque[float]) -> float:
    if len(times) < 2:
        return 0.0
    dt = times[-1] - times[0]
    return (len(times) - 1) / dt if dt > 1e-6 else 0.0


@dataclass
class LiveStats:
    fps_l: float = 0.0
    fps_r: float = 0.0
    size_l: str = "-"
    size_r: str = "-"
    fourcc_l: str = "-"
    fourcc_r: str = "-"
    board_l: bool = False
    board_r: bool = False
    rec_frames: int = 0
    rec_seconds: float = 0.0
    drops: int = 0
    cam_reopens: int = 0


@dataclass
class _RecItem:
    side: str
    frame: np.ndarray
    stamp: float
    board: bool
    corners: Optional[np.ndarray]
    imu_usb: tuple[int, Optional[dict[str, Any]]]
    imu_bt: tuple[int, Optional[dict[str, Any]]]


class Grabber:
    def __init__(self, imu_usb, imu_bt) -> None:
        self.imu_usb = imu_usb
        self.imu_bt = imu_bt
        self.cap_l: Optional[cv2.VideoCapture] = None
        self.cap_r: Optional[cv2.VideoCapture] = None
        self.id_l = -1
        self.id_r = -1
        self.width = 1280
        self.height = 720
        self.fps_req = 30
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.frame_l: Optional[np.ndarray] = None
        self.frame_r: Optional[np.ndarray] = None
        self.stats = LiveStats()
        self.recording = False
        self.save_jpeg = True
        self.session_dir = ""
        self._qs: dict[str, queue.Queue] = {
            "l": queue.Queue(maxsize=45),
            "r": queue.Queue(maxsize=45),
        }
        self._writer_thread: Optional[threading.Thread] = None
        self._writers: dict[str, Any] = {}
        self._t0 = 0.0
        self._n = 0
        self._times_l: deque[float] = deque(maxlen=40)
        self._times_r: deque[float] = deque(maxlen=40)
        self.need_board = False
        self.calib_l: list[np.ndarray] = []
        self.calib_r: list[np.ndarray] = []
        self._last_frame_l = 0.0
        self._last_frame_r = 0.0
        self._pending: dict[str, Optional[tuple[float, np.ndarray]]] = {"l": None, "r": None}
        self._detect_t = {"l": 0.0, "r": 0.0}
        self._detect_state: dict[str, dict[str, Any]] = {
            "l": {"board": False, "corners": None, "pat": INNER, "stamp": 0.0},
            "r": {"board": False, "corners": None, "pat": INNER, "stamp": 0.0},
        }
        self._img_i = {"l": 0, "r": 0}
        self._fourcc = {"l": "-", "r": "-"}
        self._fourcc_t = {"l": 0.0, "r": 0.0}

    def open(self, id_l: int, id_r: int, width: int, height: int, fps: int) -> None:
        self.close()
        self.id_l, self.id_r = int(id_l), int(id_r)
        self.width, self.height, self.fps_req = int(width), int(height), int(fps)
        self.cap_l = open_camera(self.id_l, self.width, self.height, self.fps_req)
        if self.cap_l is None or not self.cap_l.isOpened():
            raise RuntimeError(f"左目相机 {self.id_l} 打不开（请确认没被其他软件占用）")
        self.cap_r = None
        if self.id_r >= 0 and self.id_r != self.id_l:
            self.cap_r = open_camera(self.id_r, self.width, self.height, self.fps_req)
            if self.cap_r is None or not self.cap_r.isOpened():
                self.cap_r = None
        self._stop.clear()
        self._pending = {"l": None, "r": None}
        self._detect_t = {"l": 0.0, "r": 0.0}
        self._detect_state = {
            "l": {"board": False, "corners": None, "pat": INNER, "stamp": 0.0},
            "r": {"board": False, "corners": None, "pat": INNER, "stamp": 0.0},
        }
        self._threads = [
            threading.Thread(target=self._run_cam, args=("l",), daemon=True, name="cam-l"),
            threading.Thread(target=self._run_detect, args=("l",), daemon=True, name="det-l"),
        ]
        if self.cap_r is not None:
            self._threads.append(threading.Thread(target=self._run_cam, args=("r",), daemon=True, name="cam-r"))
            self._threads.append(threading.Thread(target=self._run_detect, args=("r",), daemon=True, name="det-r"))
        for t in self._threads:
            t.start()

    def close(self) -> None:
        self.stop_recording()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.8)
        self._threads = []
        for cap in (self.cap_l, self.cap_r):
            if cap is not None:
                try:
                    cap.release()
                except Exception:
                    pass
        self.cap_l = self.cap_r = None

    def start_recording(self, session_dir: str, declared_fps: float, save_jpeg: bool = True) -> None:
        self.stop_recording()
        os.makedirs(os.path.join(session_dir, "cam0"), exist_ok=True)
        if save_jpeg:
            os.makedirs(os.path.join(session_dir, "cam0", "images"), exist_ok=True)
        self.session_dir = session_dir
        self.save_jpeg = bool(save_jpeg)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        size = None
        with self._lock:
            if self.frame_l is not None:
                h, w = self.frame_l.shape[:2]
                size = (w, h)
        if size is None:
            size = (self.width, self.height)
        self._writers["l"] = cv2.VideoWriter(
            os.path.join(session_dir, "cam0", "video.avi"), fourcc, float(declared_fps), size
        )
        if self.cap_r is not None:
            os.makedirs(os.path.join(session_dir, "cam1"), exist_ok=True)
            if save_jpeg:
                os.makedirs(os.path.join(session_dir, "cam1", "images"), exist_ok=True)
            self._writers["r"] = cv2.VideoWriter(
                os.path.join(session_dir, "cam1", "video.avi"), fourcc, float(declared_fps), size
            )
        self._writers["csv"] = open(os.path.join(session_dir, "frames.csv"), "w", newline="", encoding="utf-8")
        writer = csv.writer(self._writers["csv"])
        writer.writerow(FRAMES_COLUMNS)
        self._writers["row"] = writer
        self._writers["times0"] = open(os.path.join(session_dir, "cam0", "times.txt"), "w", encoding="utf-8")
        if self.cap_r is not None:
            self._writers["times1"] = open(os.path.join(session_dir, "cam1", "times.txt"), "w", encoding="utf-8")
        self.calib_l, self.calib_r = [], []
        self._img_i = {"l": 0, "r": 0}
        self._n = 0
        self._t0 = time.time()
        self.stats.drops = 0
        for side in ("l", "r"):
            pending = self._qs[side]
            while True:
                try:
                    pending.get_nowait()
                except queue.Empty:
                    break
        self.recording = True
        if self.imu_usb is not None:
            self.imu_usb.begin_csv(os.path.join(session_dir, "imu_stream.csv"))
        if self.imu_bt is not None:
            self.imu_bt.begin_csv(os.path.join(session_dir, "imu_bt.csv"))
        self._writer_thread = threading.Thread(target=self._run_writer, daemon=True, name="rec-writer")
        self._writer_thread.start()

    def stop_recording(self) -> dict[str, Any]:
        summary = {
            "frames": self._n,
            "seconds": time.time() - self._t0 if self._t0 else 0.0,
            "calib_left": len(self.calib_l),
            "calib_right": len(self.calib_r),
            "session_dir": self.session_dir,
            "drops": self.stats.drops,
        }
        self.recording = False
        if self.imu_usb is not None:
            self.imu_usb.end_csv()
        if self.imu_bt is not None:
            self.imu_bt.end_csv()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=3.0)
        self._writer_thread = None
        for key in ("l", "r"):
            writer = self._writers.get(key)
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass
        for key in ("csv", "times0", "times1"):
            fp = self._writers.get(key)
            if fp is not None:
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass
        self._writers = {}
        return summary

    def _reopen_cap(self, side: str) -> Optional[cv2.VideoCapture]:
        idx = self.id_l if side == "l" else self.id_r
        old = self.cap_l if side == "l" else self.cap_r
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        cap = open_camera(idx, self.width, self.height, self.fps_req)
        if side == "l":
            self.cap_l = cap
        else:
            self.cap_r = cap
        with self._lock:
            self.stats.cam_reopens += 1
        return cap if cap.isOpened() else None

    def _run_cam(self, side: str) -> None:
        pin_current_thread("cam_l" if side == "l" else "cam_r", PRIOR_ABOVE)
        times = self._times_l if side == "l" else self._times_r
        stale = 0
        while not self._stop.is_set():
            cap = self.cap_l if side == "l" else self.cap_r
            if cap is None:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            stamp = time.time()
            if not ok or frame is None:
                stale += 1
                if stale >= 40:
                    cap = self._reopen_cap(side)
                    stale = 0
                time.sleep(0.005)
                continue
            stale = 0
            times.append(stamp)
            fps = _fps(times)
            if stamp - self._fourcc_t[side] > 2.0:
                self._fourcc[side] = fourcc_str(cap)
                self._fourcc_t[side] = stamp
            frame = frame.copy()
            if self.need_board and stamp - self._detect_t[side] >= 0.4:
                self._detect_t[side] = stamp
                with self._lock:
                    self._pending[side] = (stamp, frame)
            with self._lock:
                state = self._detect_state[side]
                age = stamp - float(state["stamp"])
                board = bool(state["board"]) and 0.0 <= age < 1.2
                corners = state["corners"]
            if self.recording:
                usb_snap = self.imu_usb.snapshot() if self.imu_usb is not None else (-1, None)
                bt_snap = self.imu_bt.snapshot() if self.imu_bt is not None else (-1, None)
            else:
                usb_snap = (-1, None)
                bt_snap = (-1, None)
            with self._lock:
                if side == "l":
                    self.frame_l = frame
                    self.stats.fps_l = fps
                    self.stats.size_l = f"{frame.shape[1]}x{frame.shape[0]}"
                    self.stats.fourcc_l = self._fourcc[side]
                    self._last_frame_l = stamp
                    self.stats.board_l = board
                else:
                    self.frame_r = frame
                    self.stats.fps_r = fps
                    self.stats.size_r = f"{frame.shape[1]}x{frame.shape[0]}"
                    self.stats.fourcc_r = self._fourcc[side]
                    self._last_frame_r = stamp
                    self.stats.board_r = board
                if self.recording:
                    self.stats.rec_frames = self._n
                    self.stats.rec_seconds = stamp - self._t0
            if self.recording:
                self._enqueue(side, _RecItem(side, frame, stamp, board, corners, usb_snap, bt_snap))

    def overlay(self, side: str) -> tuple[bool, Optional[np.ndarray], tuple[int, int], float]:
        with self._lock:
            state = self._detect_state[side]
            return bool(state["board"]), state["corners"], state["pat"], float(state["stamp"])

    def _enqueue(self, side: str, item: _RecItem) -> None:
        pending = self._qs[side]
        try:
            pending.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            pending.get_nowait()
        except queue.Empty:
            pass
        try:
            pending.put_nowait(item)
        except queue.Full:
            pass
        with self._lock:
            self.stats.drops += 1

    def _run_writer(self) -> None:
        pin_current_thread("writer", PRIOR_NORMAL)
        while self.recording or self._qs["l"].qsize() or self._qs["r"].qsize():
            wrote = False
            for side in ("l", "r"):
                try:
                    item: _RecItem = self._qs[side].get_nowait()
                except queue.Empty:
                    continue
                wrote = True
                self._write_item(item)
            if not wrote:
                if not self.recording:
                    break
                time.sleep(0.002)

    def _write_item(self, item: _RecItem) -> None:
        writer = self._writers.get(item.side)
        if writer is not None:
            try:
                writer.write(item.frame)
            except Exception:
                pass
        if self.save_jpeg:
            try:
                folder = "cam0" if item.side == "l" else "cam1"
                idx = self._img_i[item.side]
                self._img_i[item.side] = idx + 1
                jpg = os.path.join(self.session_dir, folder, "images", f"{idx:06d}.jpg")
                cv2.imwrite(jpg, item.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            except Exception:
                pass
        times_key = "times0" if item.side == "l" else "times1"
        tfp = self._writers.get(times_key)
        if tfp is not None:
            try:
                tfp.write(f"{item.stamp:.6f}\n")
            except Exception:
                pass
        if item.side != "l" or "row" not in self._writers:
            return
        try:
            usb_idx, usb_s = item.imu_usb
            bt_idx, bt_s = item.imu_bt
            primary = usb_s if usb_s is not None else bt_s
            primary_idx = usb_idx if usb_s is not None else bt_idx
            imu_ts = float(primary["timestamp"]) if primary is not None else 0.0
            gyro_n = acc_n = 0.0
            if primary is not None:
                gyro_n = float(np.linalg.norm(primary["gyro"]))
                acc_n = float(np.linalg.norm(primary["acc"]))
            usb_ts = float(usb_s["timestamp"]) if usb_s is not None else 0.0
            bt_ts = float(bt_s["timestamp"]) if bt_s is not None else 0.0
            self._writers["row"].writerow(
                [
                    self._n, f"{item.stamp:.6f}",
                    primary_idx, f"{imu_ts:.6f}" if imu_ts else "",
                    f"{(item.stamp - imu_ts):.6f}" if imu_ts else "",
                    usb_idx, f"{usb_ts:.6f}" if usb_ts else "",
                    f"{(item.stamp - usb_ts):.6f}" if usb_ts else "",
                    bt_idx, f"{bt_ts:.6f}" if bt_ts else "",
                    f"{(item.stamp - bt_ts):.6f}" if bt_ts else "",
                    int(self.stats.board_l), int(self.stats.board_r),
                    f"{gyro_n:.4f}", f"{acc_n:.4f}",
                ]
            )
            self._n += 1
            if self._n % 30 == 0:
                self._writers["csv"].flush()
        except Exception:
            pass

    def _run_detect(self, side: str) -> None:
        pin_current_thread("det_l" if side == "l" else "det_r", PRIOR_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                item = self._pending[side]
                self._pending[side] = None
            if item is None:
                time.sleep(0.02)
                continue
            stamp, frame = item
            board = False
            corners = None
            pat = INNER
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                board, corners, pat = detect_board(gray)
                if board and corners is not None:
                    corners = refine_corners(gray, corners)
                    board = corners is not None
            except Exception:
                board, corners, pat = False, None, INNER
            with self._lock:
                self._detect_state[side] = {
                    "board": board,
                    "corners": corners,
                    "pat": pat,
                    "stamp": stamp,
                }
                if side == "l":
                    self.stats.board_l = board
                else:
                    self.stats.board_r = board
                if self.recording and board and corners is not None:
                    bucket = self.calib_l if side == "l" else self.calib_r
                    if len(bucket) < 80:
                        bucket.append(np.asarray(corners).reshape(-1, 2).copy())
