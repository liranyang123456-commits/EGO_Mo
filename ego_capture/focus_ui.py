#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Focus lock for an autofocus camera.

The intrinsic calibration is valid for one focus position only, so the focus
is set once at the working distance, locked, saved in
datasets/active_camera.json and re-applied whenever the camera is opened.
Step 4 (intrinsics) is then recorded with this focus, and every later
recording stores the focus the driver reports.
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import messagebox
from typing import Optional

import cv2

from . import camera_config
from .cameras import board_sharpness

BG, PANEL, FG = "#151a26", "#202838", "#e9eefb"


def _font(size: int, bold: bool = False):
    return ("Microsoft YaHei UI", size, "bold" if bold else "normal")


class FocusDialog(tk.Toplevel):
    def __init__(self, app) -> None:
        super().__init__(app)
        self.app = app
        self.title("对焦与锁焦")
        self.configure(bg=BG)
        self.geometry("620x520")
        lo, mid, hi = camera_config.work_distance_mm(app.grabber.width or 1280)
        tk.Label(self, text="对焦与锁焦（内参只对一个焦距有效）", bg=BG, fg="#9ec5ff", font=_font(14, True)).pack(
            anchor="w", padx=12, pady=(12, 4))
        tk.Label(self, bg=BG, fg=FG, font=_font(10), justify="left", wraplength=590, text=(
            f"1. 把棋盘放在实验的工作距离处（约 {mid / 10:.0f} cm，内参标定前按旧相机估计），整块棋盘在画面中间。\n"
            "2. 点「自动对焦一次并锁定」：相机自己对焦 3 秒后关闭自动对焦，并记下焦距。\n"
            "3. 看清晰度读数；必要时用滑条微调，到最大值附近为止。\n"
            "4. 点「保存为实验焦距」。之后先录第 4 步内参，再做实验；整个实验期间不要再改焦距。\n"
            "相机不报告焦距时，用「打开驱动设置」取消自动对焦并手动设置，再保存。")).pack(anchor="w", padx=12)
        box = tk.Frame(self, bg=PANEL)
        box.pack(fill=tk.X, padx=12, pady=10)
        self.lbl_state = tk.Label(box, text="-", bg=PANEL, fg=FG, font=_font(13, True), anchor="w")
        self.lbl_state.pack(fill=tk.X, padx=10, pady=(8, 2))
        self.lbl_sharp = tk.Label(box, text="-", bg=PANEL, fg=FG, font=_font(13, True), anchor="w")
        self.lbl_sharp.pack(fill=tk.X, padx=10, pady=(2, 8))
        self.var_focus = tk.DoubleVar(value=float(app.grabber.focus_state.get("focus") or 0.0))
        tk.Scale(self, from_=0, to=1023, resolution=1, orient=tk.HORIZONTAL, variable=self.var_focus,
                 label="焦距（驱动单位）", length=580, bg=BG, fg=FG, troughcolor=PANEL,
                 highlightthickness=0).pack(padx=12)
        row = tk.Frame(self, bg=BG)
        row.pack(fill=tk.X, padx=12, pady=8)
        for text, cmd, colour in (("自动对焦一次并锁定", self._auto_once, "#2f9e44"),
                                  ("应用滑条焦距", self._apply_slider, "#1971c2"),
                                  ("打开驱动设置", self._settings, "#495057"),
                                  ("保存为实验焦距", self._save, "#e8590c")):
            tk.Button(row, text=text, command=cmd, bg=colour, fg="white", bd=0, padx=10, pady=6,
                      font=_font(11, True)).pack(side=tk.LEFT, padx=4)
        self.lbl_saved = tk.Label(self, text="", bg=BG, fg="#ffe08a", font=_font(10), justify="left", wraplength=590)
        self.lbl_saved.pack(anchor="w", padx=12, pady=4)
        self._best: Optional[float] = None
        self._show_saved()
        self.after(300, self._tick)

    def _show_saved(self) -> None:
        cfg = camera_config.load()
        cal = "旧内窥镜标定（必须重新标定）" if camera_config.is_legacy() else cfg.get("calib_dir")
        self.lbl_saved.config(text=f"已保存的实验焦距：{cfg.get('focus', '未保存')}；当前内参：{cal}")

    def _tick(self) -> None:
        if not self.winfo_exists():
            return
        g = self.app.grabber
        st = dict(g.focus_state)
        af = st.get("autofocus")
        fv = st.get("focus")
        af_txt = "未知" if af is None else ("开" if camera_config.autofocus_on(af) else "关")
        self.lbl_state.config(text=f"自动对焦：{af_txt}    焦距读数：{'不报告' if fv is None else f'{fv:g}'}",
                              fg="#ff8787" if camera_config.autofocus_on(af) else FG)
        board, corners, _pat, _stamp = g.overlay("l")
        with g._lock:
            frame = None if g.frame_l is None else g.frame_l.copy()
        sharp = None
        if board and frame is not None:
            sharp = board_sharpness(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), corners)
        if sharp is not None:
            self._best = sharp if self._best is None else max(self._best, sharp)
            self.lbl_sharp.config(text=f"棋盘清晰度：{sharp:.0f}    （本次最高 {self._best:.0f}）")
        else:
            self.lbl_sharp.config(text="棋盘清晰度：画面里没有检出整块棋盘")
        g.request("read")
        self.after(400, self._tick)

    def _auto_once(self) -> None:
        g = self.app.grabber
        g.focus_target = None          # the focus guard would undo the autofocus
        g.request("autofocus", 1)
        self.lbl_state.config(text="自动对焦中，3 秒后锁定……")

        def lock() -> None:
            g.request("autofocus", 0)
            self.after(400, self._adopt)
        self.after(3000, lock)

    def _adopt(self) -> None:
        fv = self.app.grabber.focus_state.get("focus")
        if fv is not None:
            self.var_focus.set(fv)
            self.app.grabber.focus_target = float(fv)
        self.app._log(f"自动对焦后锁定：焦距读数 {fv}")

    def _apply_slider(self) -> None:
        v = float(self.var_focus.get())
        self.app.grabber.focus_target = v
        self.app.grabber.request("focus", v)

    def _settings(self) -> None:
        self.app.grabber.focus_target = None   # keep the guard from undoing changes made in the dialog
        self.app.grabber.request("settings")
        self.app._log("在驱动设置里取消自动对焦并调好焦距后，回到这里点「保存为实验焦距」。")

    def _save(self) -> None:
        g = self.app.grabber
        st = dict(g.focus_state)
        if camera_config.autofocus_on(st.get("autofocus")):
            messagebox.showerror("自动对焦还开着", "先点「自动对焦一次并锁定」，或在驱动设置里取消自动对焦。", parent=self)
            return
        fv = st.get("focus")
        if fv is None and not messagebox.askyesno(
                "相机不报告焦距", "这台相机不报告焦距，程序无法核对焦距是否变化。\n"
                "确认已在驱动设置里取消自动对焦并固定了焦距？", parent=self):
            return
        if not camera_config.is_legacy() and camera_config.calibration_focus() is not None \
                and fv is not None and abs(fv - camera_config.calibration_focus()) > camera_config.FOCUS_TOLERANCE:
            if not messagebox.askyesno("和内参标定的焦距不同",
                                       "现在的焦距和当前内参标定时不同。保存后必须重新录第 4 步内参。继续？", parent=self):
                return
        g.focus_target = None if fv is None else float(fv)
        camera_config.save(focus=fv, focus_reported=fv is not None, camera_index=g.id_l,
                           image_size=[g.width, g.height], locked_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self.app._log(f"实验焦距已保存：{fv}。下一步：用这个焦距录第 4 步内参。")
        self._show_saved()
