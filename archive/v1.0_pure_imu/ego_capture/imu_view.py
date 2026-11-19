#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live IMU axis traces and separate 3D trajectories for USB and Bluetooth."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from .mapping.inertial import initialize_from_static, propagate, wit_to_si

GUIDE = (
    "准备：不重录第 2–4 步。棋盘固定，蓝牙粘在板上，采集中不碰板。USB IMU 仍在双目上。\n"
    "点侧栏「9 轨迹×3」后，分辨率应已是 640×480、30 帧。扫描并打开相机和双 IMU。\n"
    "每条都这样录，三条之间可以关掉程序：\n"
    "1. 看本窗口。两路频率接近 200 Hz，松手时三维轨迹几乎不动，再点「开始本步录制」。\n"
    "2. 完全不动 8 秒。\n"
    "3. 慢速平移约 20 秒，棋盘一直在画面里。\n"
    "4. 平滑地绕三个轴转约 20 秒。\n"
    "5. 快速运动约 15 秒，棋盘仍要能看见。\n"
    "6. 让棋盘出画约 5 秒，再移回来对准约 10 秒。点「停止并保存」。\n"
    "第 1、2 条用于训练和验证。第 3 条整段只做测试，不要照着前两条重做同一路径。"
)


class _Track:
    def __init__(self) -> None:
        self.state = None
        self.last_t = 0.0
        self.t: list[float] = []
        self.p: list[np.ndarray] = []
        self.ready = False

    def reset(self) -> None:
        self.state = None
        self.last_t = 0.0
        self.t.clear()
        self.p.clear()
        self.ready = False

    def push(self, samples: list[tuple]) -> None:
        fresh = [s for s in samples if s[0] > self.last_t + 1e-4]
        if len(fresh) < 2:
            return
        acc_g = np.array([s[1] for s in fresh], dtype=np.float64)
        gyro_dps = np.array([s[2] for s in fresh], dtype=np.float64)
        if not self.ready:
            gyro_n = np.linalg.norm(gyro_dps, axis=1)
            if len(fresh) < 40 or float(np.median(gyro_n)) > 3.0:
                self.last_t = fresh[-1][0]
                return
            acc_si, gyro_si = wit_to_si(acc_g, gyro_dps)
            self.state = initialize_from_static(acc_si, gyro_si)
            self.ready = True
            self.last_t = fresh[-1][0]
            self.t.append(self.last_t)
            self.p.append(self.state.p.copy())
            return
        acc_si, gyro_si = wit_to_si(acc_g, gyro_dps)
        t = np.array([s[0] for s in fresh], dtype=np.float64)
        for i in range(len(fresh)):
            dt = float(t[i] - self.last_t) if i == 0 else float(t[i] - t[i - 1])
            if dt <= 0 or dt > 0.05:
                self.last_t = float(t[i])
                continue
            propagate(self.state, gyro_si[i], acc_si[i], dt)
            self.last_t = float(t[i])
            if i % 4 == 0:
                self.t.append(self.last_t)
                self.p.append(self.state.p.copy())
        if len(self.p) > 4000:
            self.t = self.t[-4000:]
            self.p = self.p[-4000:]


class ImuTraceWindow(tk.Toplevel):
    def __init__(self, master, imu_usb, imu_bt) -> None:
        super().__init__(master)
        self.title("IMU 轴向与三维轨迹")
        self.geometry("1180x820")
        self.configure(bg="#1b2030")
        self.imu_usb = imu_usb
        self.imu_bt = imu_bt
        self.usb = _Track()
        self.bt = _Track()
        self._closed = False
        self._lines: dict[str, tuple] = {}
        self.protocol("WM_DELETE_WINDOW", self._close)

        guide = tk.Text(self, height=8, bg="#141824", fg="#d5def2", font=("Microsoft YaHei UI", 10), bd=0)
        guide.pack(fill=tk.X, padx=8, pady=(8, 0))
        guide.insert("1.0", GUIDE)
        guide.configure(state=tk.DISABLED)

        bar = tk.Frame(self, bg="#1b2030")
        bar.pack(fill=tk.X, padx=8, pady=4)
        tk.Button(bar, text="重置轨迹", command=self._reset, bg="#364258", fg="white", bd=0).pack(side=tk.LEFT)
        self.status = tk.Label(bar, text="等待 IMU…", bg="#1b2030", fg="#9ecbff", font=("Microsoft YaHei UI", 10))
        self.status.pack(side=tk.LEFT, padx=12)

        try:
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure
            import matplotlib
            matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei UI", "SimHei"]
            matplotlib.rcParams["axes.unicode_minus"] = False
        except Exception as exc:
            tk.Label(self, text=f"需要 matplotlib 才能画图：{exc}", bg="#1b2030", fg="#ffb4b4").pack()
            return

        fig = Figure(figsize=(11.4, 7.2), dpi=100, facecolor="#1b2030")
        grid = fig.add_gridspec(2, 4, width_ratios=[1.1, 1.1, 1.0, 1.0])
        self.ax_ua = fig.add_subplot(grid[0, 0])
        self.ax_ug = fig.add_subplot(grid[1, 0])
        self.ax_ba = fig.add_subplot(grid[0, 1])
        self.ax_bg = fig.add_subplot(grid[1, 1])
        self.ax_u3 = fig.add_subplot(grid[0, 2:], projection="3d")
        self.ax_b3 = fig.add_subplot(grid[1, 2:], projection="3d")
        self._style_time(self.ax_ua, "USB 加速度 (g)")
        self._style_time(self.ax_ug, "USB 角速度 (°/s)")
        self._style_time(self.ax_ba, "蓝牙 加速度 (g)")
        self._style_time(self.ax_bg, "蓝牙 角速度 (°/s)")
        self._style_3d(self.ax_u3, "USB 轨迹 (m)")
        self._style_3d(self.ax_b3, "蓝牙 轨迹 (m)")
        fig.tight_layout()
        self.canvas = FigureCanvasTkAgg(fig, master=self)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.after(200, self._tick)

    def _style_time(self, ax, title: str) -> None:
        ax.set_facecolor("#141824")
        ax.set_title(title, color="#d5def2", fontsize=10)
        ax.tick_params(colors="#8ea0c8", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#2c3650")
        ax.grid(True, color="#2c3650", linewidth=0.4)

    def _style_3d(self, ax, title: str) -> None:
        ax.set_facecolor("#141824")
        ax.set_title(title, color="#d5def2", fontsize=10)
        ax.tick_params(colors="#8ea0c8", labelsize=7)
        ax.set_xlabel("X (m)", color="#8ea0c8", fontsize=8)
        ax.set_ylabel("Y (m)", color="#8ea0c8", fontsize=8)
        ax.set_zlabel("Z (m)", color="#8ea0c8", fontsize=8)

    def _close(self) -> None:
        self._closed = True
        self.destroy()

    def _reset(self) -> None:
        self.usb.reset()
        self.bt.reset()
        for key in ("u3", "b3"):
            line = self._lines.pop(key, None)
            if line is not None:
                line.remove()

    def _plot_axes(self, key: str, ax, samples: list[tuple], col: int, title: str) -> None:
        if len(samples) < 2:
            return
        t0 = samples[-1][0]
        t = np.array([s[0] - t0 for s in samples])
        y = np.array([s[col] for s in samples])
        lines = self._lines.get(key)
        if lines is None:
            lines = (
                ax.plot(t, y[:, 0], color="#ff6b6b", lw=0.8, label="x")[0],
                ax.plot(t, y[:, 1], color="#69db7c", lw=0.8, label="y")[0],
                ax.plot(t, y[:, 2], color="#74c0fc", lw=0.8, label="z")[0],
            )
            ax.legend(loc="upper right", fontsize=7, frameon=False, labelcolor="#d5def2")
            ax.set_title(title, color="#d5def2", fontsize=10)
            self._lines[key] = lines
        for i, line in enumerate(lines):
            line.set_data(t, y[:, i])
        ax.set_xlim(max(-8.0, float(t[0])), 0.0)
        ymin = float(np.min(y))
        ymax = float(np.max(y))
        pad = max(0.05, 0.1 * (ymax - ymin))
        ax.set_ylim(ymin - pad, ymax + pad)

    def _plot_3d(self, key: str, ax, track: _Track, title: str) -> None:
        ax.set_title(title if track.ready else title + "（等待静默）", color="#d5def2", fontsize=10)
        if len(track.p) < 2:
            return
        p = np.stack(track.p)
        line = self._lines.get(key)
        if line is None:
            line = ax.plot(p[:, 0], p[:, 1], p[:, 2], color="#ffd43b", lw=1.0)[0]
            self._lines[key] = line
        else:
            line.set_data(p[:, 0], p[:, 1])
            line.set_3d_properties(p[:, 2])
        span = np.ptp(p, axis=0)
        center = p[-1]
        radius = max(0.05, float(np.max(span)) * 0.6)
        ax.set_xlim(center[0] - radius, center[0] + radius)
        ax.set_ylim(center[1] - radius, center[1] + radius)
        ax.set_zlim(center[2] - radius, center[2] + radius)

    def _tick(self) -> None:
        if self._closed or not hasattr(self, "canvas"):
            return
        usb = self.imu_usb.tail(1600)
        bt = self.imu_bt.tail(1600)
        self.usb.push(usb)
        self.bt.push(bt)
        self._plot_axes("ua", self.ax_ua, usb, 1, "USB 加速度 (g)")
        self._plot_axes("ug", self.ax_ug, usb, 2, "USB 角速度 (°/s)")
        self._plot_axes("ba", self.ax_ba, bt, 1, "蓝牙 加速度 (g)")
        self._plot_axes("bg", self.ax_bg, bt, 2, "蓝牙 角速度 (°/s)")
        self._plot_3d("u3", self.ax_u3, self.usb, "USB 轨迹 (m)")
        self._plot_3d("b3", self.ax_b3, self.bt, "蓝牙 轨迹 (m)")
        self.canvas.draw_idle()
        usb_hz = getattr(self.imu_usb, "hz", 0.0)
        bt_hz = getattr(self.imu_bt, "hz", 0.0)
        self.status.config(
            text=f"USB {usb_hz:.0f} Hz  {'已初始化' if self.usb.ready else '等待静默'}    "
                 f"蓝牙 {bt_hz:.0f} Hz  {'已初始化' if self.bt.ready else '等待静默'}"
        )
        self.after(250, self._tick)
