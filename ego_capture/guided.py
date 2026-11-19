#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live guidance for two calibration recordings (no GUI code here).

intrinsics  The operator moves the board until the cells of a 6 x 4 image
            grid are covered and each side of the board has been turned
            towards the camera; the recording then stops by itself. All
            detections are kept, and the calibration uses a subset that
            covers every reached cell and mixes positions and tilts.
rigid       A 24-s script (rest, nod, shake, roll, rest) shown as cues with
            tones; the recording stops at its end.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .session import INNER

GRID = (6, 4)
# Opposite edges of the GP050 corner grid differ by 2.5% when the board is
# turned by about 15 deg at 25 cm (its 30 x 21 mm grid is small against the distance).
TILT_RATIO = 0.025
TILT_MIN = 2
MIN_SAMPLES = 30
MIN_SECONDS = 10.0
TILT_NAMES = {"L": "左边", "R": "右边", "U": "上边", "D": "下边"}

RIGID_SCRIPT = (
    (3.0, "静止", "握稳相机，整块棋盘在画面里"),
    (6.0, "点头 ×2", "相机上下转，约 ±20°，每次 1–2 秒"),
    (6.0, "摇头 ×2", "相机左右转，约 ±20°"),
    (6.0, "转动 ×2", "绕镜头方向来回拧，约 ±25°"),
    (3.0, "静止", "停住，马上结束"),
)
RIGID_SECONDS = sum(d for d, _c, _h in RIGID_SCRIPT)


def beep(freq: int, ms: int) -> None:
    def _run() -> None:
        try:
            import winsound
            winsound.Beep(int(freq), int(ms))
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


def tilt(corners: np.ndarray, cols: int = INNER[0], rows: int = INNER[1]) -> str:
    """Sides of the board turned towards the camera ('L', 'R', 'U', 'D' or a
    pair), from the foreshortening of opposite edges of the corner grid."""
    xy = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    p00, p10, p01, p11 = xy[0], xy[cols - 1], xy[(rows - 1) * cols], xy[-1]
    out = ""
    for (a0, a1), (b0, b1) in (((p00, p01), (p10, p11)), ((p00, p10), (p01, p11))):
        la, lb = np.linalg.norm(a1 - a0), np.linalg.norm(b1 - b0)
        if min(la, lb) < 1e-6 or abs(la / lb - 1.0) < TILT_RATIO:
            continue
        near, far = ((a0 + a1) / 2, (b0 + b1) / 2) if la > lb else ((b0 + b1) / 2, (a0 + a1) / 2)
        upright = abs((a1 - a0)[1]) > abs((a1 - a0)[0])
        out += ("L" if near[0] < far[0] else "R") if upright else ("U" if near[1] < far[1] else "D")
    return out


def describe(corners: np.ndarray, image_size: tuple[int, int]) -> tuple[set, int, str]:
    """(grid cells reached by the corners, cell of the board centre, tilt)."""
    xy = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    w, h = image_size
    gx, gy = GRID
    ix = np.clip((xy[:, 0] * gx / w).astype(int), 0, gx - 1)
    iy = np.clip((xy[:, 1] * gy / h).astype(int), 0, gy - 1)
    cx, cy = xy.mean(axis=0)
    centre = min(int(cy * gy / h), gy - 1) * gx + min(int(cx * gx / w), gx - 1)
    return set((iy * gx + ix).tolist()), int(centre), tilt(xy)


@dataclass
class CoverageTracker:
    image_size: tuple[int, int]
    n: int = 0
    cells: set = field(default_factory=set)
    tilts: dict = field(default_factory=lambda: {k: 0 for k in "LRUD"})

    def update(self, samples: list) -> None:
        for corners in samples[self.n:]:
            cells, _centre, t = describe(corners, self.image_size)
            self.cells |= cells
            for k in t:
                self.tilts[k] += 1
        self.n = len(samples)

    @property
    def fraction(self) -> float:
        return len(self.cells) / float(GRID[0] * GRID[1])

    def grid(self) -> np.ndarray:
        g = np.zeros((GRID[1], GRID[0]), dtype=np.uint8)
        for c in self.cells:
            g[c // GRID[0], c % GRID[0]] = 1
        return g

    def missing_tilts(self) -> list[str]:
        return [k for k in "LRUD" if self.tilts[k] < TILT_MIN]

    def done(self, seconds: float, coverage_min: float) -> bool:
        return (self.fraction >= coverage_min and not self.missing_tilts()
                and self.n >= MIN_SAMPLES and seconds >= MIN_SECONDS)

    def text(self, coverage_min: float) -> str:
        need = int(np.ceil(coverage_min * GRID[0] * GRID[1] - 1e-9))
        sides = "  ".join(f"{TILT_NAMES[k]}{'✓' if self.tilts[k] >= TILT_MIN else '✗'}" for k in "LRUD")
        return (f"内参：拿着棋盘在画面里移动，把预览上的灰格子都变绿（移过去停一下）。"
                f"已覆盖 {len(self.cells)}/{GRID[0] * GRID[1]} 格，要 {need} 格。\n"
                f"再把棋盘的每一边各朝相机转一下（约 15–30°）：{sides}。样本 {self.n}。够了会自动停止。")


def select_samples(samples: list, image_size: tuple[int, int], max_n: int = 80, seed: int = 0) -> list:
    """Calibration subset: first the detections that reach every covered grid
    cell, then the rest spread over (centre cell, tilt) groups and over time."""
    samples = list(samples)
    if len(samples) <= max_n:
        return samples
    info = [describe(c, image_size) for c in samples]
    chosen, covered = [], set()
    left = set(range(len(samples)))
    while left:
        best = max(left, key=lambda i: len(info[i][0] - covered))
        gain = info[best][0] - covered
        if not gain:
            break
        chosen.append(best)
        covered |= gain
        left.discard(best)
    groups: dict = {}
    for i in np.random.default_rng(seed).permutation(sorted(left)):
        groups.setdefault((info[int(i)][1], info[int(i)][2]), []).append(int(i))
    keys = sorted(groups)
    while len(chosen) < max_n and keys:
        for key in list(keys):
            if len(chosen) >= max_n:
                break
            if groups[key]:
                chosen.append(groups[key].pop())
            else:
                keys.remove(key)
    return [samples[i] for i in sorted(chosen)]


def rigid_cue(elapsed: float) -> Optional[tuple[int, str, str, float]]:
    """(phase, cue, hint, seconds left in the phase), or None after the script."""
    t = 0.0
    for k, (d, cue, hint) in enumerate(RIGID_SCRIPT):
        if elapsed < t + d:
            return k, cue, hint, t + d - elapsed
        t += d
    return None
