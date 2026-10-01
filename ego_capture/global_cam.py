#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed 4K global camera: two-board detection and queued recording.

The 4K camera is fixed and watches the scene from outside. Two rigs move in
its view: the stereo+USB-IMU rig carrying the small board (7x5 squares of
5 mm, 6x4 inner corners) and the BLE-IMU board (GP050, 12x9 squares of 3 mm,
11x8 inner corners). Detecting both boards per frame gives the global pose of
both rigs in the fixed camera frame -- the global reference that the
stereo-only setup lacked.

Detection runs on a worker thread so the grab thread never stalls. The two
patterns differ in both dimensions, so a full-grid match is board-specific;
each board is additionally tracked inside a region of interest around its
last position, which keeps the two boards from being confused when they come
close, and falls back to a full-frame search after a board is lost.
"""

from __future__ import annotations

import csv
import os
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

cv2.setNumThreads(1)

from .affinity import PRIOR_ABOVE, PRIOR_NORMAL, pin_current_thread
from .cameras import open_camera
from .session import INNER, INNER2

G4K_WIDTH = 3840
G4K_HEIGHT = 2160
DETECT_MAX_SIDE = 1280  # search resolution; corners are mapped back to 4K
ROI_MARGIN = 1.6        # ROI = last bounding box scaled by this factor
LOST_AFTER = 12         # misses before falling back to a full-frame search

BOARDS = {
    "A": {"inner": INNER2, "label": "rig 6x4"},   # on the stereo+IMU rig
    "B": {"inner": INNER, "label": "GP050 11x8"},  # with the BLE IMU
}

BOARD_CSV_COLUMNS = [
    "frame_idx", "stamp",
    "boardA_found", "boardA_pattern", "boardA_bbox",
    "boardB_found", "boardB_pattern", "boardB_bbox",
]


def _find_pattern(gray: np.ndarray, inner: tuple[int, int]):
    """Search both orientations of the inner-corner grid."""
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    for pattern in (inner, (inner[1], inner[0])):
        ok, corners = cv2.findChessboardCorners(gray, pattern, flags)
        if ok and corners is not None and len(corners) == pattern[0] * pattern[1]:
            return corners.astype(np.float32), pattern
    return None


def _bbox_of(corners: np.ndarray) -> tuple[float, float, float, float]:
    pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
    return (
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    )


class TwoBoardDetector:
    """Detects both chessboards in one 4K frame with ROI tracking."""

    def __init__(self) -> None:
        self._bbox: dict[str, Optional[tuple[float, float, float, float]]] = {"A": None, "B": None}
        self._miss: dict[str, int] = {"A": LOST_AFTER, "B": LOST_AFTER}

    def detect(self, gray: np.ndarray) -> dict[str, dict[str, Any]]:
        return {key: self._detect_one(gray, key) for key in BOARDS}

    def _detect_one(self, gray: np.ndarray, key: str) -> dict[str, Any]:
        inner = BOARDS[key]["inner"]
        height, width = gray.shape[:2]
        scale = min(1.0, DETECT_MAX_SIDE / float(max(height, width)))
        small = gray
        if scale < 1.0:
            small = cv2.resize(
                gray,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        sh, sw = small.shape[:2]

        roi = self._bbox[key]
        use_roi = roi is not None and self._miss[key] < LOST_AFTER
        if use_roi:
            x0, y0, x1, y1 = [v * scale for v in roi]
            cx, cy = (x0 + x1) * 0.5, (y0 + y1) * 0.5
            hw, hh = (x1 - x0) * ROI_MARGIN * 0.5, (y1 - y0) * ROI_MARGIN * 0.5
            rx0 = int(max(0, cx - hw))
            ry0 = int(max(0, cy - hh))
            rx1 = int(min(sw, cx + hw))
            ry1 = int(min(sh, cy + hh))
            crop = small[ry0:ry1, rx0:rx1]
            found = _find_pattern(crop, inner) if crop.size else None
            if found is not None:
                corners, pattern = found
                corners = corners.reshape(-1, 2).copy()
                corners[:, 0] += rx0
                corners[:, 1] += ry0
                return self._accept(key, corners, pattern, scale, gray)
        found = _find_pattern(small, inner)
        if found is None:
            self._miss[key] += 1
            return {"found": False, "corners": None, "pattern": inner, "bbox": None}
        corners, pattern = found
        return self._accept(key, corners.reshape(-1, 2), pattern, scale, gray)

    def _accept(self, key, corners, pattern, scale, gray_full):
        full = corners.copy()
        if scale < 1.0:
            full[:, 0] /= scale
            full[:, 1] /= scale
        self._bbox[key] = _bbox_of(full)
        self._miss[key] = 0
        return {
            "found": True,
            "corners": full.astype(np.float32),
            "pattern": pattern,
            "bbox": self._bbox[key],
        }


@dataclass
class GlobalStats:
    fps: float = 0.0
    size: str = "-"
    board_a: bool = False
    board_b: bool = False
    rec_frames: int = 0
    rec_seconds: float = 0.0
    drops: int = 0


class _RecItem:
    __slots__ = ("frame", "stamp", "boards")

    def __init__(self, frame, stamp, boards):
        self.frame = frame
        self.stamp = stamp
        self.boards = boards


class GlobalGrabber:
    """Owns the fixed 4K camera: preview, two-board detection, recording."""

    def __init__(self) -> None:
        self.cam_id = -1
        self.cap: Optional[cv2.VideoCapture] = None
        self.width = G4K_WIDTH
        self.height = G4K_HEIGHT
        self.fps_req = 30
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._lock = threading.Lock()
        self.frame: Optional[np.ndarray] = None
        self.stats = GlobalStats()
        self.recording = False
        self.save_jpeg = True
        self.session_dir = ""
        self._detector = TwoBoardDetector()
        self._pending: Optional[tuple[float, np.ndarray]] = None
        self._detect_state: dict[str, Any] = {"A": {"found": False}, "B": {"found": False}, "stamp": 0.0}
        self._queue: queue.Queue = queue.Queue(maxsize=30)
        self._writer_thread: Optional[threading.Thread] = None
        self._writers: dict[str, Any] = {}
        self._t0 = 0.0
        self._n = 0
        self._img_i = 0
        self._times: deque[float] = deque(maxlen=40)
        self._last_frame_t = 0.0
        self.calib: list[np.ndarray] = []  # GP050 corners for 4K intrinsics

    # ------------------------------------------------------------------ open
    def open(self, cam_id: int, fps: int) -> None:
        self.close()
        self.cam_id = int(cam_id)
        self.fps_req = int(fps)
        cap = open_camera(self.cam_id, self.width, self.height, self.fps_req)
        if cap is None or not cap.isOpened():
            raise RuntimeError(f"4K 相机 {self.cam_id} 打不开（确认没被占用、且支持 3840×2160）")
        self.cap = cap
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if (actual_w, actual_h) != (self.width, self.height):
            # Some UVC cameras negotiate a lower mode; record what we got.
            self.width, self.height = actual_w, actual_h
        self._stop.clear()
        self._pending = None
        self._detector = TwoBoardDetector()
        self._threads = [
            threading.Thread(target=self._run_cam, daemon=True, name="cam-4k"),
            threading.Thread(target=self._run_detect, daemon=True, name="det-4k"),
        ]
        for t in self._threads:
            t.start()

    def close(self) -> None:
        self.stop_recording()
        self._stop.set()
        for t in self._threads:
            t.join(timeout=1.8)
        self._threads = []
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None

    # -------------------------------------------------------------- recording
    def start_recording(self, session_dir: str, declared_fps: float, save_jpeg: bool = True) -> None:
        self.stop_recording()
        folder = os.path.join(session_dir, "cam2")
        os.makedirs(folder, exist_ok=True)
        if save_jpeg:
            os.makedirs(os.path.join(folder, "images"), exist_ok=True)
        self.session_dir = session_dir
        self.save_jpeg = bool(save_jpeg)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self._writers["v"] = cv2.VideoWriter(
            os.path.join(folder, "video.avi"), fourcc, float(declared_fps), (self.width, self.height)
        )
        self._writers["times"] = open(os.path.join(folder, "times.txt"), "w", encoding="utf-8")
        self._writers["boards"] = open(os.path.join(folder, "boards.csv"), "w", newline="", encoding="utf-8")
        writer = csv.writer(self._writers["boards"])
        writer.writerow(BOARD_CSV_COLUMNS)
        self._writers["boards_row"] = writer
        self._img_i = 0
        self._n = 0
        self._t0 = time.time()
        self.stats.drops = 0
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self.recording = True
        self._writer_thread = threading.Thread(target=self._run_writer, daemon=True, name="rec-4k")
        self._writer_thread.start()

    def stop_recording(self) -> dict[str, Any]:
        summary = {
            "frames": self._n,
            "seconds": time.time() - self._t0 if self._t0 else 0.0,
            "session_dir": self.session_dir,
            "drops": self.stats.drops,
            "resolution": [self.width, self.height],
        }
        self.recording = False
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=5.0)
        self._writer_thread = None
        writer = self._writers.get("v")
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        for key in ("times", "boards"):
            fp = self._writers.get(key)
            if fp is not None:
                try:
                    fp.flush()
                    fp.close()
                except Exception:
                    pass
        self._writers = {}
        return summary

    # ------------------------------------------------------------------ loop
    def _run_cam(self) -> None:
        pin_current_thread("cam_4k", PRIOR_ABOVE)
        stale = 0
        while not self._stop.is_set():
            cap = self.cap
            if cap is None:
                time.sleep(0.05)
                continue
            ok, frame = cap.read()
            stamp = time.time()
            if not ok or frame is None:
                stale += 1
                if stale >= 40:
                    self._reopen()
                    stale = 0
                time.sleep(0.005)
                continue
            stale = 0
            self._times.append(stamp)
            fps = self._fps()
            frame = frame.copy()
            # Hand the newest frame to the detector; the detector sets the pace.
            with self._lock:
                self._pending = (stamp, frame)
                state = self._detect_state
                age = stamp - float(state["stamp"])
                board_a = bool(state["A"].get("found")) and 0.0 <= age < 1.5
                board_b = bool(state["B"].get("found")) and 0.0 <= age < 1.5
                boards = {
                    "A": {
                        "found": board_a,
                        "pattern": state["A"].get("pattern", INNER2),
                        "bbox": state["A"].get("bbox"),
                    },
                    "B": {
                        "found": board_b,
                        "pattern": state["B"].get("pattern", INNER),
                        "bbox": state["B"].get("bbox"),
                    },
                }
                self.frame = frame
                self.stats.fps = fps
                self.stats.size = f"{frame.shape[1]}x{frame.shape[0]}"
                self.stats.board_a = board_a
                self.stats.board_b = board_b
                self._last_frame_t = stamp
                if self.recording:
                    self.stats.rec_frames = self._n
                    self.stats.rec_seconds = stamp - self._t0
            if self.recording:
                self._enqueue(_RecItem(frame, stamp, boards))

    def _reopen(self) -> None:
        old = self.cap
        if old is not None:
            try:
                old.release()
            except Exception:
                pass
        self.cap = open_camera(self.cam_id, self.width, self.height, self.fps_req)

    def _fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / dt if dt > 1e-6 else 0.0

    def _enqueue(self, item: _RecItem) -> None:
        try:
            self._queue.put_nowait(item)
            return
        except queue.Full:
            pass
        try:
            self._queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass
        with self._lock:
            self.stats.drops += 1

    def _run_detect(self) -> None:
        pin_current_thread("det_4k", PRIOR_NORMAL)
        while not self._stop.is_set():
            with self._lock:
                item = self._pending
                self._pending = None
            if item is None:
                time.sleep(0.02)
                continue
            stamp, frame = item
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                result = self._detector.detect(gray)
            except Exception:
                result = {"A": {"found": False}, "B": {"found": False}}
            with self._lock:
                self._detect_state = {"A": result["A"], "B": result["B"], "stamp": stamp}
                self.stats.board_a = bool(result["A"].get("found"))
                self.stats.board_b = bool(result["B"].get("found"))
                if self.recording and result["B"].get("found") and result["B"].get("corners") is not None:
                    if len(self.calib) < 120:
                        self.calib.append(np.asarray(result["B"]["corners"], dtype=np.float32).reshape(-1, 2).copy())

    def _run_writer(self) -> None:
        pin_current_thread("writer_4k", PRIOR_NORMAL)
        while self.recording or self._queue.qsize():
            try:
                item: _RecItem = self._queue.get_nowait()
            except queue.Empty:
                if not self.recording:
                    break
                time.sleep(0.002)
                continue
            self._write_item(item)

    def _write_item(self, item: _RecItem) -> None:
        writer = self._writers.get("v")
        if writer is not None:
            try:
                writer.write(item.frame)
            except Exception:
                pass
        if self.save_jpeg:
            try:
                idx = self._img_i
                self._img_i = idx + 1
                jpg = os.path.join(self.session_dir, "cam2", "images", f"{idx:06d}.jpg")
                cv2.imwrite(jpg, item.frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
            except Exception:
                pass
        tfp = self._writers.get("times")
        if tfp is not None:
            try:
                tfp.write(f"{item.stamp:.6f}\n")
            except Exception:
                pass
        row_writer = self._writers.get("boards_row")
        if row_writer is not None:
            try:
                a = item.boards.get("A", {})
                b = item.boards.get("B", {})
                row_writer.writerow(
                    [
                        self._n,
                        f"{item.stamp:.6f}",
                        int(bool(a.get("found"))),
                        "x".join(str(v) for v in a.get("pattern", (0, 0))),
                        ";".join(f"{v:.0f}" for v in (a.get("bbox") or (0, 0, 0, 0))),
                        int(bool(b.get("found"))),
                        "x".join(str(v) for v in b.get("pattern", (0, 0))),
                        ";".join(f"{v:.0f}" for v in (b.get("bbox") or (0, 0, 0, 0))),
                    ]
                )
            except Exception:
                pass
        self._n += 1
        if self._n % 30 == 0:
            fp = self._writers.get("boards")
            if fp is not None:
                try:
                    fp.flush()
                except Exception:
                    pass

    # ---------------------------------------------------------------- preview
    def overlay(self) -> tuple[Optional[np.ndarray], Optional[tuple[int, int]]]:
        """Corners of the rig board (A) for the shared preview helper."""
        with self._lock:
            state = self._detect_state
            if state["A"].get("found") and state["A"].get("corners") is not None:
                return state["A"]["corners"], state["A"].get("pattern", INNER2)
        return None, None

    def preview_frame(self) -> Optional[np.ndarray]:
        """Latest frame with both boards drawn (A cyan, B magenta)."""
        with self._lock:
            frame = None if self.frame is None else self.frame.copy()
            state = dict(self._detect_state)
        if frame is None:
            return None
        for key, color in (("A", (255, 220, 0)), ("B", (220, 0, 220))):
            info = state.get(key) or {}
            corners = info.get("corners")
            if info.get("found") and corners is not None:
                pattern = info.get("pattern", BOARDS[key]["inner"])
                try:
                    cv2.drawChessboardCorners(frame, pattern, corners.reshape(-1, 1, 2), True)
                except cv2.error:
                    pass
                bbox = info.get("bbox")
                if bbox is not None:
                    x0, y0, x1, y1 = [int(v) for v in bbox]
                    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 3)
                    cv2.putText(
                        frame, key, (x0, max(0, y0 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3, cv2.LINE_AA,
                    )
        return frame
