#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stereo endoscope + dual IMU capture GUI."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

from .affinity import describe_plan
from .cameras import Grabber, LiveStats, calibrate_from_corners, open_camera
from .imu import ImuWorker, list_serial_candidates, pick_usb_port, scan_ble
from .calib import solve_step, summarize
from .session import (
    BOARD_NAME,
    INNER,
    SQUARE_MM,
    SQUARES,
    LEGACY_THEMES,
    PHASES,
    RIGID_SPLIT,
    STEPS,
    THEME_PLAN,
    THEME_SPLITS,
    TRAJ_THEMES,
    active_protocol_id,
    disk_free_gb,
    now_id,
    rig_banner,
    session_prefix,
    write_checkerboard_yaml,
    write_json,
)

try:
    from PIL import Image, ImageFont, ImageTk
except ImportError as exc:  # pragma: no cover
    raise SystemExit("需要 Pillow: pip install pillow") from exc

import tkinter as tk
from tkinter import messagebox, ttk

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_ROOT = os.path.join(ROOT_DIR, "datasets")
RIG_STATE = os.path.join(DATA_ROOT, "rig_state.json")
USB_BAUD = 921600
BT_SPP_BAUD = 115200
RECORD_STEPS = {"imu_static", "imu_dyn", "intrinsics", "board_bt", "ego",
                "recon", "rigid", "traj", "stage_ref", "stereo_ext"}
BOARD_STEPS = {"intrinsics", "board_bt", "rigid", "traj", "stage_ref",
               "stereo_ext"}
COLLECTION_LOG = os.path.join(DATA_ROOT, "collection_log.csv")
LOG_FIELDS = [
    "session", "time", "resolution", "fps", "theme", "split", "seconds", "frames",
    "board_still", "usable_frac", "reproj_p50_px", "delay_ms",
    "corr_tilt_x", "corr_tilt_y", "corr_roll", "rigidity", "verdict",
]


def _theme_label(code: str) -> str:
    for c, name, _how in TRAJ_THEMES:
        if c == code:
            return f"{c} {name}"
    if code in LEGACY_THEMES:
        return f"{code} {LEGACY_THEMES[code]}"
    return code


def _theme_target(code: str) -> int:
    return len(THEME_PLAN.get(code, THEME_SPLITS))


def _theme_counts() -> dict[str, int]:
    counts = {c: 0 for c, _n, _h in TRAJ_THEMES}
    if not os.path.isdir(DATA_ROOT):
        return counts
    for name in os.listdir(DATA_ROOT):
        if not name.startswith("traj_"):
            continue
        meta_path = os.path.join(DATA_ROOT, name, "session_meta.json")
        try:
            with open(meta_path, encoding="utf-8") as handle:
                code = json.load(handle).get("theme")
        except Exception:
            continue
        if code in counts:
            counts[code] += 1
    return counts


def _next_theme() -> tuple[str, str]:
    """Round-robin over training themes first; test sessions are proposed once
    every training theme has at least two recordings, so a test recording is
    never the first session of a day."""
    counts = _theme_counts()
    for c, _n, _h in TRAJ_THEMES:
        if c != "T" and counts[c] < min(2, _theme_target(c)):
            return c, _split_for(c, counts[c])
    for c, _n, _h in TRAJ_THEMES:
        if counts[c] < _theme_target(c):
            return c, _split_for(c, counts[c])
    return TRAJ_THEMES[0][0], "train"


def _split_for(code: str, index: int) -> str:
    """Predeclared split of the index-th recording of a theme (decided before capture)."""
    if code == "J" or code == "T":
        return "test"
    plan = THEME_PLAN.get(code, THEME_SPLITS)
    return plan[index] if index < len(plan) else "train"


def _font(size: int, bold: bool = False):
    return ("Microsoft YaHei UI", size, "bold" if bold else "normal")


def _yahei_pil(size: int):
    for path in (
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyhbd.ttc",
        r"C:\Windows\Fonts\simhei.ttf",
    ):
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def bgr_to_photo(frame: np.ndarray, max_w: int, max_h: int, corners: np.ndarray | None = None, pattern: tuple[int, int] | None = None) -> ImageTk.PhotoImage:
    h, w = frame.shape[:2]
    scale = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    view = cv2.resize(frame, (nw, nh))
    if corners is not None and pattern is not None:
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2) * scale
        try:
            cv2.drawChessboardCorners(view, pattern, pts.reshape(-1, 1, 2), True)
        except cv2.error:
            pass
    rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
    return ImageTk.PhotoImage(Image.fromarray(rgb))


class CaptureApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("双目内镜 + 双 IMU 三系标定  |  USB / 蓝牙 / 左右目")
        self.geometry("1540x940")
        self.configure(bg="#1b2030")
        self.imu_usb = ImuWorker("usb")
        self.imu_bt = ImuWorker("bt")
        self.grabber = Grabber(self.imu_usb, self.imu_bt)
        self.step = "scan"
        self.done: set[str] = set()
        self.gate_pass = False
        self.photo_l = None
        self.photo_r = None
        self.thumb_photos: list[Any] = []
        self.scan_thumbs: dict[int, np.ndarray] = {}
        self.found_cams: list[int] = []
        self.ble_devices: list[tuple[str, str]] = []
        self._opening = False
        self._gating = False
        self._build()
        self._hydrate_done_from_disk()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(120, self._tick)

    def _latest_json(self, prefix: str, filename: str) -> Optional[dict[str, Any]]:
        newest: Optional[dict[str, Any]] = None
        newest_mtime = -1.0
        if not os.path.isdir(DATA_ROOT):
            return None
        for name in os.listdir(DATA_ROOT):
            if not name.startswith(prefix):
                continue
            path = os.path.join(DATA_ROOT, name, filename)
            if not os.path.isfile(path):
                continue
            mtime = os.path.getmtime(path)
            if mtime <= newest_mtime:
                continue
            try:
                with open(path, encoding="utf-8") as handle:
                    newest = json.load(handle)
                newest_mtime = mtime
            except Exception:
                continue
        return newest

    def _hydrate_done_from_disk(self) -> None:
        if self._latest_json("align_imu_static_", "calib_result.json"):
            self.done.add("imu_static")
        if self._latest_json("align_imu_dyn_", "calib_result.json"):
            self.done.add("imu_dyn")
        if self._latest_json("calib_intrinsics_", "camera_calibration_cam0.json"):
            self.done.add("intrinsics")
        board = self._latest_json("calib_board_bt_", "calib_result.json")
        if board and board.get("T_C0_Ibt"):
            self.done.add("board_bt")
        if os.path.isfile(RIG_STATE):
            try:
                with open(RIG_STATE, encoding="utf-8") as handle:
                    state = json.load(handle)
            except Exception:
                state = {}
            if state.get("T_C0_Ibt") and state.get("T_C0_Iusb"):
                self.done.add("compose")
                self.done.add("gate")
        ego = self._latest_json("ego_seq_", "capture_summary.json")
        if ego and float(ego.get("seconds") or 0) >= 50:
            self.done.add("ego")
        recon = self._latest_json("recon_seq_", "capture_summary.json")
        if recon and float(recon.get("seconds") or 0) >= 50:
            self.done.add("recon")
        rigid = self._latest_json("rigid_", "qc_result.json")
        if rigid and rigid.get("rigidity") == "rigid":
            self.done.add("rigid")
        traj_n = 0
        if os.path.isdir(DATA_ROOT):
            for name in os.listdir(DATA_ROOT):
                if not name.startswith("traj_"):
                    continue
                summary_path = os.path.join(DATA_ROOT, name, "capture_summary.json")
                if not os.path.isfile(summary_path):
                    continue
                try:
                    with open(summary_path, encoding="utf-8") as handle:
                        summary = json.load(handle)
                except Exception:
                    continue
                if float(summary.get("seconds") or 0) >= 50:
                    traj_n += 1
        if traj_n >= 3:
            self.done.add("traj")
        self._refresh_step_styles()
        if self.done:
            self._log("磁盘已有结果，侧栏标绿：" + "、".join(sorted(self.done)))

    def _build(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        left = tk.Frame(self, bg="#141824", width=288)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)
        tk.Label(left, text="采集步骤", bg="#141824", fg="#9ecbff", font=_font(16, True)).pack(pady=(16, 4))
        tk.Label(
            left, text="先几何 · 再残差 · 后应用",
            bg="#141824", fg="#7d8aa8", font=_font(9),
        ).pack(pady=(0, 8))
        self.step_btns: dict[str, tk.Button] = {}
        titles = {key: title for key, title, _hint in STEPS}
        for phase, keys in PHASES:
            tk.Label(
                left, text=phase, bg="#141824", fg="#8eb4e8",
                font=_font(10, True), anchor="w",
            ).pack(fill=tk.X, padx=14, pady=(8, 2))
            for key in keys:
                btn = tk.Button(
                    left, text=titles[key], font=_font(11),
                    bg="#2a3148", fg="#e8eefc", bd=0, pady=5, anchor="w",
                    command=lambda k=key: self._goto(k),
                )
                btn.pack(fill=tk.X, padx=12, pady=3)
                self.step_btns[key] = btn

        right = tk.Frame(self, bg="#1b2030")
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right,
            text=rig_banner() + f"    数据  {DATA_ROOT}",
            bg="#1e3a5f", fg="#ffe08a", font=_font(11, True),
            padx=16, pady=8, anchor="w",
        ).pack(fill=tk.X, padx=12, pady=(12, 0))

        self.hint = tk.Label(
            right, text=STEPS[0][2], bg="#243049", fg="#f4f7ff",
            font=_font(13), wraplength=1180, justify="left", padx=16, pady=12,
        )
        self.hint.pack(fill=tk.X, padx=12, pady=12)

        hw = tk.Frame(right, bg="#1b2030")
        hw.pack(fill=tk.X, padx=12)
        tk.Label(hw, text="分辨率", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(side=tk.LEFT)
        self.res_var = tk.StringVar(value="1280x720")
        ttk.Combobox(hw, textvariable=self.res_var, values=["1920x1080", "1280x720", "640x480"], width=12, state="readonly").pack(side=tk.LEFT, padx=6)
        tk.Label(hw, text="目标帧率", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(side=tk.LEFT, padx=(12, 0))
        self.fps_var = tk.StringVar(value="30")
        ttk.Combobox(hw, textvariable=self.fps_var, values=["30", "25", "20", "15"], width=6, state="readonly").pack(side=tk.LEFT, padx=6)
        tk.Label(hw, text="USB IMU", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(side=tk.LEFT, padx=(12, 0))
        self.usb_var = tk.StringVar(value="")
        self.usb_combo = ttk.Combobox(hw, textvariable=self.usb_var, width=34, state="readonly")
        self.usb_combo.pack(side=tk.LEFT, padx=6)
        tk.Label(hw, text="蓝牙 IMU", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(side=tk.LEFT, padx=(12, 0))
        self.ble_var = tk.StringVar(value="")
        self.ble_combo = ttk.Combobox(hw, textvariable=self.ble_var, width=28, state="readonly")
        self.ble_combo.pack(side=tk.LEFT, padx=6)
        tk.Label(hw, text="预览", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(side=tk.LEFT, padx=(8, 0))
        self.preview_var = tk.StringVar(value="低")
        ttk.Combobox(hw, textvariable=self.preview_var, values=["低", "中", "高"], width=5, state="readonly").pack(side=tk.LEFT, padx=6)

        theme_row = tk.Frame(right, bg="#1b2030")
        theme_row.pack(fill=tk.X, padx=12, pady=(6, 0))
        tk.Label(theme_row, text="主题", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(side=tk.LEFT)
        self.theme_var = tk.StringVar(value=_theme_label(TRAJ_THEMES[0][0]))
        theme_box = ttk.Combobox(
            theme_row, textvariable=self.theme_var,
            values=[_theme_label(c) for c, _n, _h in TRAJ_THEMES], width=16, state="readonly",
        )
        theme_box.pack(side=tk.LEFT, padx=6)
        theme_box.bind("<<ComboboxSelected>>", lambda _e: self._update_theme_info())
        self.theme_info = tk.Label(theme_row, text="", bg="#1b2030", fg="#ffe08a", font=_font(10), anchor="w")
        self.theme_info.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.status = tk.Label(right, text="未连接", bg="#1b2030", fg="#7fd99a", font=_font(10), anchor="w")
        self.status.pack(fill=tk.X, padx=16, pady=(6, 0))
        imu_row = tk.Frame(right, bg="#1b2030")
        imu_row.pack(fill=tk.X, padx=12, pady=(4, 0))
        self.lbl_usb = tk.Label(
            imu_row, text="USB IMU 未连接", bg="#14261c", fg="#b7f0c8",
            font=("Consolas", 10), justify="left", anchor="nw",
        )
        self.lbl_usb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.lbl_ble = tk.Label(
            imu_row, text="蓝牙 IMU 未连接", bg="#142033", fg="#c5dcff",
            font=("Consolas", 10), justify="left", anchor="nw",
        )
        self.lbl_ble.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))
        self.lbl_delta = tk.Label(
            right, text="两路静止时加速度都会接近 (0, 0, 1)g，这是重力，不是同一路数据。看磁力计和航向角。",
            bg="#1b2030", fg="#c9b458", font=_font(10), anchor="w",
        )
        self.lbl_delta.pack(fill=tk.X, padx=16, pady=(2, 0))

        prev = tk.Frame(right, bg="#1b2030")
        prev.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)
        self.lbl_l = tk.Label(prev, text="左目\n点「扫描设备」", bg="#0f1320", fg="#8ea0c8", font=_font(14))
        self.lbl_l.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 6))
        self.lbl_r = tk.Label(prev, text="右目\n点「扫描设备」", bg="#0f1320", fg="#8ea0c8", font=_font(14))
        self.lbl_r.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 0))

        self.thumbs = tk.Frame(right, bg="#1b2030")
        self.thumbs.pack(fill=tk.X, padx=12)

        btns = tk.Frame(right, bg="#1b2030")
        btns.pack(fill=tk.X, padx=12, pady=10)
        self._mk_btn(btns, "扫描设备", self._scan, "#3d5afe").pack(side=tk.LEFT, padx=4)
        self._mk_btn(btns, "IMU 轨迹", self._open_imu_view, "#0b7285").pack(side=tk.LEFT, padx=4)
        self._mk_btn(btns, "打开相机+双IMU", self._open_devices, "#2b6cb0").pack(side=tk.LEFT, padx=4)
        self.start_btn = self._mk_btn(btns, "开始本步录制", self._start_step, "#2f9e44")
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = self._mk_btn(btns, "停止并保存", self._stop_step, "#c92a2a")
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        self._mk_btn(btns, "下一步", self._next_step, "#7048e8").pack(side=tk.LEFT, padx=4)
        self._mk_btn(btns, "解算三系外参", self._solve_compose, "#b08800").pack(side=tk.LEFT, padx=4)

        self.log = tk.Text(right, height=7, bg="#101522", fg="#d7e0f2", font=("Consolas", 10), bd=0)
        self.log.pack(fill=tk.X, padx=12, pady=(0, 12))
        self._log(f"棋盘 {BOARD_NAME}：{SQUARES[0]}×{SQUARES[1]} 格，边长 {SQUARE_MM:.1f} mm，内角点 {INNER[0]}×{INNER[1]}。")
        self._log("标定顺序：门禁 → 双IMU贴紧静止 → 双IMU一起动 → 双目内参 → 蓝牙IMU贴棋盘给双目看 → 三系投影。")
        self._log("参考：手眼 AX=XB（导航文档）以及旧工作 1d0149c5 的 PnP / R_CI。USB=相机刚体，蓝牙先对齐再当棋盘上的移动系。")
        self._refresh_step_styles()

    def _open_imu_view(self) -> None:
        view = getattr(self, "_imu_view", None)
        if view is not None:
            try:
                if view.winfo_exists():
                    view.lift()
                    return
            except Exception:
                pass
        from .imu_view import ImuTraceWindow

        self._imu_view = ImuTraceWindow(self, self.imu_usb, self.imu_bt)

    def _mk_btn(self, parent, text, cmd, color) -> tk.Button:
        return tk.Button(
            parent, text=text, command=cmd, font=_font(12, True),
            bg=color, fg="white", bd=0, padx=14, pady=8, activebackground=color,
        )

    def _log(self, msg: str) -> None:
        self.log.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log.see("end")

    def _goto(self, key: str) -> None:
        self.step = key
        hint = next(h for k, _t, h in STEPS if k == key)
        self.hint.config(text=hint)
        self.grabber.need_board = key in BOARD_STEPS
        if key == "rigid":
            self.res_var.set("1280x720")
            self.fps_var.set("15")
            if self.grabber.cap_l is not None and (self.grabber.width, self.grabber.height) != (1280, 720):
                self._log(
                    f"相机还开在 {self.grabber.width}×{self.grabber.height}。刚性检查推荐 720p；"
                    "实际 8–10 fps 也可，程序比较约 1 秒的累计转角。"
                )
        elif key == "traj":
            self.res_var.set("1280x720")
            self.fps_var.set("15")
            if self.grabber.cap_l is not None and (self.grabber.width, self.grabber.height) != (1280, 720):
                self._log(
                    f"相机还开在 {self.grabber.width}×{self.grabber.height}。v2 轨迹统一用 1280×720（实际约 10 fps，"
                    "重投影 0.2 px），请再点一次「打开相机+双IMU」。"
                )
        elif key == "stage_ref":
            self.res_var.set("1280x720")
            self.fps_var.set("15")
            self._log(
                "已知位移验证：相机静止 8–12 秒；停止后把本段会话名填入 "
                "datasets/reference_stage_plan.csv 对应的位置。"
            )
        elif key == "stereo_ext":
            self.res_var.set("1280x720")
            self.fps_var.set("15")
            self._log(
                "双目外参：每个棋盘姿态必须静止保持 2 秒；不要边移动边采。"
            )
        if key == "traj":
            code, _split = _next_theme()
            self.theme_var.set(_theme_label(code))
            if "rigid" not in self.done:
                self._log("今天还没通过 IMU 刚性检查（第 9 步）。建议先录一段刚性检查。")
        self._update_theme_info()
        self._refresh_step_styles()

    def _current_theme(self) -> tuple[str, str]:
        code = (self.theme_var.get() or "A").split(" ")[0]
        return code, _split_for(code, _theme_counts().get(code, 0))

    def _update_theme_info(self) -> None:
        if self.step != "traj":
            self.theme_info.config(text="（主题只用于第 10 步轨迹）")
            return
        code, split = self._current_theme()
        counts = _theme_counts()
        how = next((h for c, _n, h in TRAJ_THEMES if c == code), "")
        done = sum(min(counts.get(c, 0), _theme_target(c)) for c, _n, _h in TRAJ_THEMES)
        total = sum(_theme_target(c) for c, _n, _h in TRAJ_THEMES)
        target = _theme_target(code)
        self.theme_info.config(
            text=(f"{how}。本主题已录 {counts.get(code, 0)}/{target}，这一段记为 {split}"
                  f"{'（封存，不可用于训练/选择）' if split == 'test' else ''}。全部进度 {done}/{total}。")
        )

    def _refresh_step_styles(self) -> None:
        for key, btn in self.step_btns.items():
            if key == self.step:
                btn.config(bg="#3d8bfd", fg="white")
            elif key in self.done:
                btn.config(bg="#2b8a3e", fg="white")
            else:
                btn.config(bg="#2a3148", fg="#e8eefc")

    def _parse_res(self) -> tuple[int, int, int]:
        w, h = self.res_var.get().split("x")
        return int(w), int(h), int(self.fps_var.get())

    def _scan(self) -> None:
        if getattr(self, "_scanning", False):
            return
        self._scanning = True
        self._log("正在扫描摄像头、串口和蓝牙 IMU…")

        def _work() -> None:
            cams: list[int] = []
            thumbs: dict[int, np.ndarray] = {}
            for idx in range(6):
                cap = open_camera(idx, 640, 480, 15)
                if not cap.isOpened():
                    cap.release()
                    continue
                ok, frame = cap.read()
                cap.release()
                if ok and frame is not None:
                    cams.append(idx)
                    thumbs[idx] = frame
            ports = list_serial_candidates()
            ble: list[tuple[str, str]] = []
            try:
                import asyncio
                ble = asyncio.run(scan_ble(4.0))
            except Exception as exc:
                ble_err = str(exc)
            else:
                ble_err = ""

            def _done() -> None:
                self._scanning = False
                self.found_cams = cams
                self.scan_thumbs = thumbs
                self.ble_devices = ble
                self._render_thumbs()
                usb_vals = [lab for _d, lab, kind in ports if kind != "bt_spp"] or [lab for _d, lab, _k in ports]
                self.usb_combo["values"] = usb_vals or ["(未找到 USB 串口)"]
                picked = pick_usb_port(ports)
                if picked:
                    for item in self.usb_combo["values"]:
                        if str(item).startswith(picked):
                            self.usb_var.set(item)
                            break
                spp = [lab for _d, lab, kind in ports if kind == "bt_spp"]
                ble_vals = [f"{addr}  {name}" for addr, name in ble] + spp
                self.ble_combo["values"] = ble_vals or ["(未找到蓝牙 IMU，确认 WTSDCL 已开机且已配对)"]
                if ble_vals:
                    prefer = next((v for v in ble_vals if "wtsdcl" in v.lower() or "901" in v.lower() or "bwt" in v.lower()), ble_vals[0])
                    self.ble_var.set(prefer)
                self._log(f"相机 {cams}；串口 {[p[0] for p in ports]}；BLE {ble or ble_err or '无'}")

            self.after(0, _done)

        threading.Thread(target=_work, daemon=True).start()

    def _render_thumbs(self) -> None:
        for child in self.thumbs.winfo_children():
            child.destroy()
        self.thumb_photos = []
        if not self.found_cams:
            self._log("没有找到摄像头。检查 Hub / 是否被占用。")
            return
        tk.Label(self.thumbs, text="点缩略图指定左目 / 右目：", bg="#1b2030", fg="#9aa7c7", font=_font(10)).pack(anchor="w")
        row = tk.Frame(self.thumbs, bg="#1b2030")
        row.pack(fill=tk.X)
        for idx in self.found_cams:
            photo = bgr_to_photo(self.scan_thumbs[idx], 180, 120)
            self.thumb_photos.append(photo)
            box = tk.Frame(row, bg="#243049")
            box.pack(side=tk.LEFT, padx=6, pady=4)
            tk.Label(box, image=photo, bg="#243049").pack()
            tk.Label(box, text=f"相机 {idx}", bg="#243049", fg="white", font=_font(10)).pack()
            tk.Button(box, text="设为左目", command=lambda i=idx: self._set_cam("l", i), bg="#1c7ed6", fg="white", bd=0).pack(fill=tk.X)
            tk.Button(box, text="设为右目", command=lambda i=idx: self._set_cam("r", i), bg="#ae3ec9", fg="white", bd=0).pack(fill=tk.X)

    def _set_cam(self, side: str, index: int) -> None:
        if side == "l":
            self.grabber.id_l = index
            self._log(f"左目 = 相机 {index}")
        else:
            self.grabber.id_r = index
            self._log(f"右目 = 相机 {index}")
        if self.grabber.id_l == self.grabber.id_r and self.grabber.id_l >= 0:
            messagebox.showwarning("相机冲突", "左右目不能是同一路。")

    def _usb_port(self) -> str:
        raw = self.usb_var.get().strip()
        if not raw or raw.startswith("("):
            return ""
        return raw.split()[0]

    def _ble_target(self) -> tuple[str, str]:
        raw = self.ble_var.get().strip()
        if not raw or raw.startswith("("):
            return "", ""
        token = raw.split()[0]
        if token.upper().startswith("COM"):
            return "spp", token
        return "ble", token

    def _open_devices(self) -> None:
        if self.grabber.id_l < 0:
            messagebox.showerror("未指定相机", "请先扫描，并用鼠标点一个缩略图设为左目。")
            return
        if self._opening:
            return
        self._opening = True
        w, h, fps = self._parse_res()
        usb = self._usb_port()
        ble_kind, ble_id = self._ble_target()
        if usb and ble_id and usb.upper() == ble_id.upper():
            messagebox.showerror("IMU 冲突", "USB 和蓝牙不能选同一个 COM 口。")
            return
        right_txt = f" 右{self.grabber.id_r}" if self.grabber.id_r >= 0 and self.grabber.id_r != self.grabber.id_l else "（单目）"
        self._log(f"打开 左{self.grabber.id_l}{right_txt} @ {w}x{h} {fps}fps；USB {usb or '无'}；BLE {ble_id or '无'}…")

        def _work() -> None:
            err = None
            try:
                if usb:
                    self.imu_usb.start_serial(usb, USB_BAUD, "usb")
                if ble_id:
                    if ble_kind == "spp":
                        self.imu_bt.start_serial(ble_id, BT_SPP_BAUD, "bt_spp")
                    else:
                        self.imu_bt.start_ble(ble_id)
                self.grabber.open(self.grabber.id_l, self.grabber.id_r, w, h, fps)
            except Exception as exc:  # noqa: BLE001
                err = exc

            def _done() -> None:
                self._opening = False
                if err is not None:
                    messagebox.showerror("打开失败", str(err))
                    return
                self.done.add("scan")
                self._refresh_step_styles()
                mode = "双目" if self.grabber.cap_r is not None else "单目"
                got = self.grabber.cap_l.get(cv2.CAP_PROP_FPS) if self.grabber.cap_l is not None else 0
                self._log(f"已打开 {mode}。驱动回报 {got:.0f} fps。等 1–2 秒看 IMU 频率。")
                self._log(describe_plan())
                if got and got < 20:
                    self._log("驱动回报低于 20 fps。轨迹采集若实测也低于 20 fps，改用 640×480（约 30 fps）。")
                self.after(1600, self._log_imu_distinct)

            self.after(0, _done)

        threading.Thread(target=_work, daemon=True).start()

    def _start_step(self) -> None:
        if self.grabber.cap_l is None:
            self._open_devices()
            return
        if self.step == "scan":
            self._goto("gate")
            return
        if self.step == "gate":
            self._run_gate()
            return
        if self.step == "compose":
            self._solve_compose()
            return
        kind = session_prefix(self.step)
        if self.step not in RECORD_STEPS:
            return
        if disk_free_gb(DATA_ROOT) < 2.0:
            messagebox.showerror("磁盘空间不足", "数据盘剩余 < 2 GB，先清出空间再录。")
            return
        session = os.path.join(DATA_ROOT, f"{kind}_{now_id()}")
        os.makedirs(session, exist_ok=True)
        write_checkerboard_yaml(os.path.join(session, "checkerboard.yaml"))
        ble_kind, ble_id = self._ble_target()
        meta = {
            "rig": "stereo_endoscope_plus_dual_imu",
            "board": BOARD_NAME,
            "inner_corners": list(INNER),
            "square_mm": SQUARE_MM,
            "step": self.step,
            "cam_left_id": self.grabber.id_l,
            "cam_right_id": self.grabber.id_r,
            "imu_usb_port": self.imu_usb.port,
            "imu_usb_baud": self.imu_usb.baud,
            "imu_bt_id": ble_id,
            "imu_bt_kind": self.imu_bt.kind or ble_kind,
            "target_imu_hz": 200,
            "requested": {"width": self.grabber.width, "height": self.grabber.height, "fps": self.grabber.fps_req},
            "save_jpeg": True,
            "no_zoom": True,
            "fixed_extrinsics": True,
            "stereo_baseline_mm": 46.5,
            "lens_diameter_mm": 5.5,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        if self.step == "traj":
            meta["theme"], meta["split"] = self._current_theme()
            meta["protocol_id"] = active_protocol_id()
            meta["prospective_sealed"] = bool(
                meta["theme"] == "T" and meta["protocol_id"]
            )
        elif self.step == "rigid":
            # Rigidity recordings double as rotation/still training data.
            meta["theme"], meta["split"] = "R", RIGID_SPLIT
        elif self.step == "stage_ref":
            meta["theme"], meta["split"] = "S", "reference_validation"
        elif self.step == "stereo_ext":
            meta["theme"], meta["split"] = "E", "calibration"
        write_json(os.path.join(session, "session_meta.json"), meta)
        self.grabber.need_board = self.step in BOARD_STEPS
        if self.step == "traj":
            self._log(f"本段主题 {_theme_label(meta['theme'])}，分组 {meta['split']}。")
            if (self.grabber.width, self.grabber.height) != (1280, 720):
                self._log(f"当前相机是 {self.grabber.width}×{self.grabber.height}。v2 轨迹统一 1280×720，当天不要混用分辨率。")
        self.grabber.start_recording(session, float(self.fps_var.get()), save_jpeg=True)
        self._log(f"开始录制 {session}")
        self.start_btn.config(state=tk.DISABLED)

    def _run_gate(self) -> None:
        if self._gating:
            return
        self._gating = True
        self._log("门禁 8 秒：不要拔线，看相机帧率和两路 IMU Hz…")
        t0 = time.time()
        hist_l: list[float] = []
        hist_r: list[float] = []
        hist_usb: list[float] = []
        hist_bt: list[float] = []
        shapes = ["", ""]

        def _poll() -> None:
            with self.grabber._lock:
                stats = self.grabber.stats
                f0, f1 = self.grabber.frame_l, self.grabber.frame_r
            hist_l.append(stats.fps_l)
            hist_r.append(stats.fps_r)
            hist_usb.append(self.imu_usb.hz)
            hist_bt.append(self.imu_bt.hz)
            if f0 is not None:
                shapes[0] = f"{f0.shape[1]}x{f0.shape[0]}"
            if f1 is not None:
                shapes[1] = f"{f1.shape[1]}x{f1.shape[0]}"
            if time.time() - t0 < 8.0:
                self.after(50, _poll)
                return
            self._gating = False
            fps_l = float(np.median(hist_l[-20:])) if hist_l else 0.0
            fps_r = float(np.median(hist_r[-20:])) if hist_r else 0.0
            usb_hz = float(np.median(hist_usb[-20:])) if hist_usb else 0.0
            bt_hz = float(np.median(hist_bt[-20:])) if hist_bt else 0.0
            fps_need = max(int(self.fps_var.get()) - 5, 18)
            stereo = self.grabber.cap_r is not None
            cam_ok = fps_l >= fps_need and ((not stereo) or fps_r >= fps_need)
            usb_ok = (not self.imu_usb.port) or usb_hz >= 80
            bt_ok = (not self.imu_bt.port) or bt_hz >= 80
            imu_ok = (self.imu_usb.port or self.imu_bt.port) and usb_ok and bt_ok
            size_ok = bool(shapes[0])
            self.gate_pass = bool(size_ok and cam_ok and imu_ok)
            msg = (
                f"门禁：左 {shapes[0]} {fps_l:.1f}fps"
                + (f"，右 {shapes[1]} {fps_r:.1f}fps" if stereo else "（单目）")
                + f"，USB {usb_hz:.0f}Hz，BLE {bt_hz:.0f}Hz。"
            )
            if self.gate_pass:
                self.done.add("gate")
                self._log(msg + " 通过。")
                messagebox.showinfo("门禁通过", msg + "\n下一步：两枚 IMU 贴紧，整套静止。")
                self._goto("imu_static")
            else:
                self._log(msg + " 未通过。")
                extra = "1080p 双目共 Hub 容易掉帧，可改 1280x720。蓝牙 IMU 需开机且靠近电脑。"
                if not imu_ok:
                    extra += f" USB误差={self.imu_usb.last_error or '-'} BLE误差={self.imu_bt.last_error or '-'}。"
                if messagebox.askyesno("门禁未过", msg + "\n\n" + extra + "\n\n仍要继续？"):
                    self.done.add("gate")
                    self._goto("imu_static")
            self._refresh_step_styles()

        self.after(50, _poll)

    def _stop_step(self) -> None:
        if not self.grabber.recording:
            return
        summary = self.grabber.stop_recording()
        self.start_btn.config(state=tk.NORMAL)
        session = summary.get("session_dir") or ""
        if session:
            self._log("正在按画面运动和陀螺估计各路延迟，并把 IMU 插值到每帧时刻…")
            self.update_idletasks()
            try:
                from .sync import align_session

                summary["sync"] = align_session(session)
                for name, item in summary["sync"].get("cameras", {}).items():
                    self._log(
                        f"{name} 延迟 USB {item['delay_usb_s']*1000:.1f} ms"
                        f"（相关 {item['delay_usb_corr']:.2f}），"
                        f"蓝牙 {item['delay_bt_s']*1000:.1f} ms"
                        f"（相关 {item['delay_bt_corr']:.2f}）。"
                        f"对齐 {item['usb_ok']}/{item['frames']} 帧。"
                    )
            except Exception as exc:
                self._log(f"时刻对齐失败：{exc}")
        extra = ""
        if summary.get("drops"):
            extra = f" 丢帧 {summary['drops']}"
        self._log(
            f"已保存 {summary['frames']} 帧 / {summary['seconds']:.1f}s，"
            f"棋盘 左{summary['calib_left']} 右{summary['calib_right']}。"
            f" USB {self.imu_usb.sample_count} 样，BLE {self.imu_bt.sample_count} 样。{extra}"
        )
        if session:
            write_json(os.path.join(session, "capture_summary.json"), {
                **summary,
                "usb_hz": self.imu_usb.hz,
                "bt_hz": self.imu_bt.hz,
                "usb_samples": self.imu_usb.sample_count,
                "bt_samples": self.imu_bt.sample_count,
                "usb_reconnects": self.imu_usb.reconnects,
                "bt_reconnects": self.imu_bt.reconnects,
            })
        if self.step == "intrinsics" and session:
            self._finish_intrinsics(session, summary)
        elif self.step in {"imu_static", "imu_dyn", "board_bt"} and session:
            self._solve_after_record(session, self.step, summary)
        elif self.step in {"ego", "recon"}:
            self.done.add(self.step)
        elif self.step in {"rigid", "traj", "stage_ref"} and session:
            self._qc_after_record(session, self.step, summary)
        self._refresh_step_styles()

    def _qc_after_record(self, session: str, step: str, summary: dict[str, Any]) -> None:
        """Solve the chessboard for this session, check IMU rigidity, append to the collection log."""
        name = os.path.basename(os.path.normpath(session))
        self._log(f"正在检查 {name}：逐帧解棋盘并比对陀螺，约半分钟到一分钟…")
        try:
            with open(os.path.join(session, "session_meta.json"), encoding="utf-8") as handle:
                meta = json.load(handle)
        except Exception:
            meta = {}

        def _work() -> None:
            err = None
            pose = rig = None
            try:
                import sys as _sys

                if ROOT_DIR not in _sys.path:
                    _sys.path.insert(0, ROOT_DIR)
                from tools.build_pose_gt import process_session
                from tools.check_rigidity import check

                pose = process_session(name)
                rig = check(name)
            except Exception as exc:  # noqa: BLE001
                err = exc

            def _done() -> None:
                if err is not None or pose is None:
                    self._log(f"自动检查失败：{err}")
                    return
                frames = max(int(pose.get("frames") or 0), 1)
                still = int(pose.get("pnp_still") or 0)
                frac = still / frames
                p50 = pose.get("reproj_p50")
                seconds = float(summary.get("seconds") or 0.0)
                fps = frames / seconds if seconds > 0 else 0.0
                width, height = pose.get("size") or [0, 0]
                px_limit = 0.4 if width >= 1280 else 1.2
                problems = []
                if step == "traj" and meta.get("theme") != "J" and frac < 0.70:
                    problems.append(f"棋盘可用帧 {frac:.0%} < 70%")
                if step == "stage_ref" and frac < 0.80:
                    problems.append(f"棋盘可用帧 {frac:.0%} < 80%")
                if p50 is not None and p50 > px_limit:
                    problems.append(f"重投影中位数 {p50:.2f} px > {px_limit}")
                if summary.get("drops"):
                    problems.append(f"丢帧 {summary['drops']}")
                rigid_ok = rig.get("verdict") == "rigid"
                if step == "rigid" and not rigid_ok:
                    problems.append("IMU 刚性不合格")
                verdict = "合格" if not problems else "不合格：" + "；".join(problems)
                row = {
                    "session": name,
                    "time": datetime.now().isoformat(timespec="seconds"),
                    "resolution": f"{width}x{height}",
                    "fps": round(fps, 1),
                    "theme": meta.get("theme", ""),
                    "split": meta.get("split", ""),
                    "seconds": round(seconds, 1),
                    "frames": frames,
                    "board_still": still,
                    "usable_frac": round(frac, 3),
                    "reproj_p50_px": None if p50 is None else round(p50, 3),
                    "delay_ms": pose.get("delay_usb_ms"),
                    "corr_tilt_x": rig.get("corr_tilt_x", ""),
                    "corr_tilt_y": rig.get("corr_tilt_y", ""),
                    "corr_roll": rig.get("corr_roll", ""),
                    "rigidity": rig.get("verdict", ""),
                    "verdict": verdict,
                }
                write_json(os.path.join(session, "qc_result.json"), row)
                self._append_collection_log(row)
                rig_zh = {
                    "rigid": "可用",
                    "NOT rigid": "不合格",
                    "NOT rigid on all axes": "不合格",
                    "too few moving pairs": "转动太少，无法判断",
                    "insufficient axis excitation": "某个轴转动太少",
                }.get(rig.get("verdict", ""), rig.get("verdict", "-"))
                quality_zh = {
                    "high": "高质量",
                    "acceptable": "边缘合格",
                    "failed": "不合格",
                }.get(rig.get("quality", ""), "")
                corr = (
                    f"倾角 x {rig.get('corr_tilt_x', '-')}、倾角 y {rig.get('corr_tilt_y', '-')}、"
                    f"光轴 {rig.get('corr_roll', '-')}；幅值 {rig.get('corr_magnitude', '-')}、"
                    f"比例 {rig.get('rotation_scale', '-')}（{rig_zh}{'，' + quality_zh if quality_zh else ''}）"
                )
                p50_txt = "-" if p50 is None else f"{p50:.2f}"
                text = (
                    f"{name}\n{width}×{height}，{fps:.1f} fps，{seconds:.0f} 秒\n"
                    f"棋盘可用帧 {still}/{frames}（{frac:.0%}），重投影中位数 {p50_txt} px\n"
                    f"陀螺与棋盘相关系数：{corr}\n\n{verdict}"
                )
                self._log(text.replace("\n", " | "))
                if step == "rigid" and rigid_ok:
                    self.done.add("rigid")
                self._refresh_step_styles()
                self._update_theme_info()
                if problems:
                    messagebox.showwarning("本段检查", text + "\n\n建议重录这一段（原文件保留，记录表里已标注）。")
                else:
                    messagebox.showinfo("本段检查", text)

            self.after(0, _done)

        threading.Thread(target=_work, daemon=True).start()

    def _append_collection_log(self, row: dict[str, Any]) -> None:
        import csv

        new = not os.path.isfile(COLLECTION_LOG)
        with open(COLLECTION_LOG, "a", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
            if new:
                writer.writeheader()
            writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})

    def _solve_after_record(self, session: str, step: str, summary: dict[str, Any]) -> None:
        need = {"imu_static": 25, "imu_dyn": 40, "board_bt": 80}.get(step, 0)
        if summary.get("seconds", 0) < need:
            messagebox.showwarning("偏短", f"这一步建议 ≥{need} 秒。可以再录一条，或仍用当前数据解算。")
        self._log(f"正在解算 {step} …")

        def _work() -> None:
            err = None
            result = None
            try:
                result = solve_step(session, step, RIG_STATE)
            except Exception as exc:  # noqa: BLE001
                err = exc

            def _done() -> None:
                if err is not None:
                    self._log(f"解算失败：{err}")
                    messagebox.showerror("解算失败", str(err))
                    return
                text = summarize(result or {})
                self._log(text.replace("\n", " | "))
                self.done.add(step)
                self._refresh_step_styles()
                nxt = {"imu_static": "imu_dyn", "imu_dyn": "intrinsics", "board_bt": "compose"}.get(step)
                messagebox.showinfo("本步解算完成", text)
                if nxt:
                    self._goto(nxt)

            self.after(0, _done)

        threading.Thread(target=_work, daemon=True).start()

    def _solve_compose(self) -> None:
        try:
            result = solve_step(DATA_ROOT, "compose", RIG_STATE)
        except Exception as exc:  # noqa: BLE001
            self._log(f"三系投影失败：{exc}")
            messagebox.showerror("三系投影失败", str(exc))
            return
        write_json(os.path.join(DATA_ROOT, "coordinate_frames.json"), result)
        text = summarize(result)
        self._log(text.replace("\n", " | "))
        self.done.add("compose")
        self._refresh_step_styles()
        messagebox.showinfo("三系投影", text + f"\n\n已写 {RIG_STATE}")

    def _finish_intrinsics(self, session: str, summary: dict[str, Any]) -> None:
        with self.grabber._lock:
            frame = self.grabber.frame_l
        if frame is None:
            return
        h, w = frame.shape[:2]
        for name, samples in (("cam0", self.grabber.calib_l), ("cam1", self.grabber.calib_r)):
            payload = calibrate_from_corners(samples, (w, h))
            if payload is None:
                self._log(f"{name} 内参样本不足（{len(samples)}）。")
                continue
            path = os.path.join(session, f"camera_calibration_{name}.json")
            write_json(path, payload)
            self._log(f"{name} 内参 RMS {payload['reprojection_error_pixels']:.3f} px → {path}")
        need_right = self.grabber.cap_r is not None
        left_ok = summary["calib_left"] >= 15
        right_ok = summary["calib_right"] >= 15 if need_right else True
        if left_ok and right_ok:
            self.done.add("intrinsics")
            messagebox.showinfo("内参完成", "内参已写入。下一步：蓝牙 IMU 贴到棋盘上，给双目看。")
            self._goto("board_bt")
        else:
            messagebox.showwarning("内参不够", "左目要经常看到整块 GP050。再录一条慢扫。")

    def _next_step(self) -> None:
        keys = [k for k, _t, _h in STEPS]
        idx = keys.index(self.step)
        if idx + 1 < len(keys):
            self._goto(keys[idx + 1])

    def _fmt_imu_card(self, worker: ImuWorker, title: str) -> str:
        idx, sample = worker.snapshot()
        port = worker.port or worker.address or "-"
        if sample is None:
            err = worker.last_error or "等待数据"
            return f"{title}  {port}\n无数据  {err}"
        a, g, ang, m = sample["acc"], sample["gyro"], sample["angle"], sample["mag"]
        return (
            f"{title}  {port}   {worker.kind or '-'}   {worker.hz:.1f} Hz   n={idx}\n"
            f"acc  {a[0]:+8.4f} {a[1]:+8.4f} {a[2]:+8.4f} g\n"
            f"gyro {g[0]:+8.3f} {g[1]:+8.3f} {g[2]:+8.3f} °/s\n"
            f"RPY  {ang[0]:+8.3f} {ang[1]:+8.3f} {ang[2]:+8.3f} °\n"
            f"mag  {m[0]:+8.1f} {m[1]:+8.1f} {m[2]:+8.1f}"
        )

    def _log_imu_distinct(self) -> None:
        _, su = self.imu_usb.snapshot()
        _, sb = self.imu_bt.snapshot()
        if su is None and sb is None:
            return
        if su is None:
            self._log("USB 尚无数据；蓝牙已出数。")
            return
        if sb is None:
            self._log("蓝牙尚无数据；USB 已出数。")
            return
        self._log(
            f"USB  RPY({su['angle'][0]:+.2f},{su['angle'][1]:+.2f},{su['angle'][2]:+.2f}) "
            f"mag({su['mag'][0]:+.0f},{su['mag'][1]:+.0f},{su['mag'][2]:+.0f})"
        )
        self._log(
            f"BLE  RPY({sb['angle'][0]:+.2f},{sb['angle'][1]:+.2f},{sb['angle'][2]:+.2f}) "
            f"mag({sb['mag'][0]:+.0f},{sb['mag'][1]:+.0f},{sb['mag'][2]:+.0f})"
        )

    def _tick(self) -> None:
        with self.grabber._lock:
            f_l, f_r = self.grabber.frame_l, self.grabber.frame_r
            stats = LiveStats(**self.grabber.stats.__dict__)
        rec = f"  REC {stats.rec_frames}帧 {stats.rec_seconds:.0f}s  drop={stats.drops}" if self.grabber.recording else ""
        board = ""
        if self.grabber.need_board:
            board = f"  棋盘 左{'✓' if stats.board_l else '✗'} 右{'✓' if stats.board_r else '✗'}"
        self.status.config(
            text=(
                f"左 {stats.size_l} {stats.fps_l:.1f}fps {stats.fourcc_l}   "
                f"右 {stats.size_r} {stats.fps_r:.1f}fps {stats.fourcc_r}"
                f"{board}{rec}"
            )
        )
        self.lbl_usb.config(text=self._fmt_imu_card(self.imu_usb, "USB IMU"))
        self.lbl_ble.config(text=self._fmt_imu_card(self.imu_bt, "蓝牙 IMU"))
        _, su = self.imu_usb.snapshot()
        _, sb = self.imu_bt.snapshot()
        if su is not None and sb is not None:
            da = [sb["acc"][i] - su["acc"][i] for i in range(3)]
            dang = [sb["angle"][i] - su["angle"][i] for i in range(3)]
            self.lbl_delta.config(
                text=(
                    f"BLE−USB  Δacc({da[0]:+.3f},{da[1]:+.3f},{da[2]:+.3f})g   "
                    f"ΔRPY({dang[0]:+.2f},{dang[1]:+.2f},{dang[2]:+.2f})°    "
                    f"两路独立；静止时 acc 都接近重力，请看 mag / 航向。"
                )
            )
        if f_l is not None:
            self.photo_l = bgr_to_photo(f_l, 700, 400, *self._overlay_args("l"))
            self.lbl_l.config(image=self.photo_l, text="")
        if f_r is not None:
            self.photo_r = bgr_to_photo(f_r, 700, 400, *self._overlay_args("r"))
            self.lbl_r.config(image=self.photo_r, text="")
        delay = {"低": 150, "中": 100, "高": 60}.get(self.preview_var.get(), 120)
        self.after(delay, self._tick)

    def _overlay_args(self, side: str):
        board, corners, pattern, stamp = self.grabber.overlay(side)
        if board and corners is not None and time.time() - stamp < 0.8:
            return corners, pattern
        return None, None

    def _on_close(self) -> None:
        try:
            self.grabber.close()
            self.imu_usb.stop()
            self.imu_bt.stop()
        finally:
            self.destroy()


def main() -> None:
    os.makedirs(DATA_ROOT, exist_ok=True)
    app = CaptureApp()
    app.mainloop()
