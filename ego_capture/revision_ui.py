#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guided window for the revision experiments (stop experiment, vernier caliper).

Opened from the capture app (steps 16 and 17). The window drives the app's
existing camera/IMU threads through two app methods, ``begin_session`` and
``end_session``. ``python -m ego_capture.revision_ui --demo stop|caliper``
runs the same window on a simulated rig, for rehearsal without hardware.
"""

from __future__ import annotations

import json
import os
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk
from typing import Any, Optional

from . import revision as rv

BG, PANEL, FG, DIM = "#1b2030", "#243049", "#f4f7ff", "#9aa7c7"
COLORS = {"rest": "#495057", "motion": "#1c5d99", "stop": "#c92a2a", "go": "#2b8a3e",
          "warn": "#d9480f", "move": "#1c5d99", "settle": "#7048e8", "hold": "#2b8a3e",
          "done": "#2b8a3e", "idle": "#343a40"}

def mounting(axis: str, distance: tuple[float, float, float]) -> str:
    lo, mid, hi = (v / 10.0 for v in distance)
    return {
        "X": f"X 方向：主尺用胶带牢牢固定在桌上（推量爪时主尺不能跟着转）；棋盘隔一层硬纸板贴在活动量爪上，"
             f"蓝牙 IMU 贴在同一量爪上。棋盘可以平放、相机斜向下看，也可以竖着正对相机。单目相机在约 {mid:.0f} cm 外，"
             "用书垫高、胶带固定，主尺与画面水平方向大致平行。之后 Y、Z 只改卡尺的摆放，相机不要再动。",
        "Y": rv.MOUNT_FIX["Y"] + f"距离仍约 {mid:.0f} cm。",
        "Z": rv.MOUNT_FIX["Z"] + f"40 mm 行程要都在 {lo:.0f}–{hi:.0f} cm 内。",
    }[axis]


def _font(size: int, bold: bool = False):
    return ("Microsoft YaHei UI", size, "bold" if bold else "normal")


class RevisionPanel(tk.Toplevel):
    def __init__(self, app, mode: str, demo: bool = False) -> None:
        super().__init__(app)
        self.app, self.mode, self.demo = app, mode, demo
        self.title("修稿补充实验 — " + ("停顿对照" if mode == "stop" else "游标卡尺位移标准"))
        self.geometry("1080x860")
        self.configure(bg=BG)
        self.plan = rv.load_plan()
        self.pose = rv.LivePose()
        self.run: Any = None
        self.entry: Optional[dict[str, Any]] = None
        self.session = ""
        self.recording = False
        self.busy = False
        self.reference_speed: Optional[float] = self.plan.get("reference_speed_mm_s")
        self._mount_ok = False
        self._rest_acc: deque = deque(maxlen=60)
        self.practice = False
        self._quality: list[dict[str, Any]] = []
        self._quality_beat: Optional[float] = None
        self._build()
        self._refresh_plan()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.after(100, self._tick)

    # ------------------------------------------------------------------ layout
    def _build(self) -> None:
        top = tk.Frame(self, bg=BG)
        top.pack(fill=tk.X, padx=12, pady=(10, 4))
        self.lbl_title = tk.Label(top, text="", bg=BG, fg="#9ecbff", font=_font(15, True), anchor="w")
        self.lbl_title.pack(side=tk.LEFT, fill=tk.X, expand=True)

        cols = ("label", "detail", "session", "status")
        style = ttk.Style(self)
        import tkinter.font as tkfont
        row_font = tkfont.Font(family="Microsoft YaHei UI", size=10)
        style.configure("Plan.Treeview", font=row_font, rowheight=int(row_font.metrics("linespace") * 1.35))
        style.configure("Plan.Treeview.Heading", font=("Microsoft YaHei UI", 10, "bold"))
        self.tree = ttk.Treeview(self, columns=cols, show="headings", style="Plan.Treeview",
                                 height=6 if self.mode == "stop" else 9)
        heads = {"label": "段", "detail": "内容", "session": "会话", "status": "状态"}
        widths = {"label": 70, "detail": 360, "session": 230, "status": 360}
        for c in cols:
            self.tree.heading(c, text=heads[c])
            self.tree.column(c, width=widths[c], anchor="w")
        self.tree.pack(fill=tk.X, padx=12)

        self.cue = tk.Label(self, text="", bg=COLORS["idle"], fg="white", font=_font(30, True),
                            pady=14, anchor="center")
        self.cue.pack(fill=tk.X, padx=12, pady=(10, 0))
        self.cue_detail = tk.Label(self, text="", bg=PANEL, fg=FG, font=_font(13), wraplength=1020,
                                   justify="left", padx=12, pady=6, anchor="w")
        self.cue_detail.pack(fill=tk.X, padx=12)
        self.cue_detail.bind("<Configure>", lambda e: self.cue_detail.config(wraplength=max(e.width - 24, 200)))

        ind = tk.Frame(self, bg=BG)
        ind.pack(fill=tk.X, padx=12, pady=8)
        self.ind: dict[str, tuple[tk.Label, tk.Label]] = {}
        names = (("imu", "IMU 状态"), ("board", "棋盘可见（10 s）"), ("motion", "速度 / 3 s 行程"),
                 ("score", "停顿")) if self.mode == "stop" else \
            (("imu", "卡尺 + 相机 IMU"), ("board", "棋盘可见（10 s）"), ("pos", "相对第一停顿的位移"),
             ("score", "角点抖动（2 s）/ 距离"))
        for key, title in names:
            box = tk.Frame(ind, bg=PANEL)
            box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
            tk.Label(box, text=title, bg=PANEL, fg=DIM, font=_font(10)).pack(anchor="w", padx=8, pady=(6, 0))
            val = tk.Label(box, text="-", bg=PANEL, fg=FG, font=_font(16, True), anchor="w")
            val.pack(anchor="w", padx=8, pady=(0, 8))
            self.ind[key] = (box, val)

        if self.mode == "caliper":
            info = tk.Frame(self, bg=BG)
            info.pack(fill=tk.X, padx=12, pady=(0, 4))
            ci = self.plan.get("caliper_info", {})
            self.var_res = tk.StringVar(value=str(ci.get("resolution_mm", "0.02")))
            self.var_id = tk.StringVar(value=str(ci.get("caliper_id", "")))
            self.var_op = tk.StringVar(value=str(ci.get("operator", "")))
            self.var_zero = tk.StringVar(value="" if ci.get("zero_reading_mm") is None else str(ci["zero_reading_mm"]))
            for text, var, w in (("分度值 mm", self.var_res, 6), ("卡尺型号/编号", self.var_id, 16),
                                 ("操作者", self.var_op, 8), ("合拢零位读数 mm", self.var_zero, 7)):
                tk.Label(info, text=text, bg=BG, fg=DIM, font=_font(10)).pack(side=tk.LEFT, padx=(8, 2))
                tk.Entry(info, textvariable=var, width=w, font=_font(11)).pack(side=tk.LEFT)
            tk.Button(info, text="保存卡尺信息", command=self._save_caliper_info, bg="#495057", fg="white",
                      bd=0, padx=10, font=_font(10, True)).pack(side=tk.LEFT, padx=10)
            entry = tk.Frame(self, bg=BG)
            entry.pack(fill=tk.X, padx=12, pady=4)
            tk.Label(entry, text="游标实际读数 (mm)", bg=BG, fg="#ffe08a", font=_font(16, True)).pack(side=tk.LEFT)
            self.var_reading = tk.StringVar()
            self.ent_reading = tk.Entry(entry, textvariable=self.var_reading, width=10, font=_font(22, True))
            self.ent_reading.pack(side=tk.LEFT, padx=10)
            self.ent_reading.bind("<Return>", lambda _e: self._confirm_reading())
            tk.Button(entry, text="确认（回车）", command=self._confirm_reading, bg="#2f9e44", fg="white",
                      bd=0, padx=14, pady=6, font=_font(13, True)).pack(side=tk.LEFT)

        btns = tk.Frame(self, bg=BG)
        btns.pack(fill=tk.X, padx=12, pady=6)
        self.btn_start = self._btn(btns, "开始下一段", self._start, "#2f9e44")
        self._btn(btns, "作废当前段", self._abort, "#c92a2a")
        if self.mode == "stop":
            self._btn(btns, "练习 40 秒（不录制）", self._practice, "#1971c2")
            self._btn(btns, "末尾再加一组 C→P→P→C", self._add_block, "#7048e8")
        self._btn(btns, "打开实验说明", self._open_doc, "#495057")

        self.log = tk.Text(self, height=7, bg="#101522", fg="#d7e0f2", font=("Consolas", 10), bd=0)
        self.log.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self._set_idle_cue()

    def _btn(self, parent, text, cmd, color) -> tk.Button:
        b = tk.Button(parent, text=text, command=cmd, bg=color, fg="white", bd=0, padx=14, pady=8,
                      font=_font(12, True), activebackground=color)
        b.pack(side=tk.LEFT, padx=4)
        return b

    def _say(self, msg: str) -> None:
        self.log.insert("end", f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.log.see("end")
        try:
            self.app._log("[修稿实验] " + msg)
        except Exception:
            pass

    def _set_cue(self, text: str, color: str, detail: str = "") -> None:
        self.cue.config(text=text, bg=COLORS.get(color, color))
        self.cue_detail.config(text=detail)

    def _set_ind(self, key: str, text: str, color: Optional[str] = None) -> None:
        if key in self.ind:
            box, val = self.ind[key]
            val.config(text=text)
            bg = color or PANEL
            box.config(bg=bg)
            val.config(bg=bg)
            for child in box.winfo_children():
                child.config(bg=bg)

    # ------------------------------------------------------------------ plan
    @staticmethod
    def _status_text(entry: dict[str, Any]) -> str:
        text = {"todo": "待录", "redo": "需重录", "done": "合格", "recording": "录制中"}.get(
            entry["status"], entry["status"])
        if entry["status"] == "redo" and entry.get("verdict"):
            text += "：" + entry["verdict"]
        return text

    def _refresh_plan(self) -> None:
        self.tree.delete(*self.tree.get_children())
        if self.mode == "stop":
            for e in self.plan["stop"]:
                what = "连续运动，不停" if e["condition"] == "C" else "每 4 s 硬停 ≥1 s"
                self.tree.insert("", "end", values=(e["label"], what, e["session"] or "", self._status_text(e)))
            done = sum(e["status"] == "done" for e in self.plan["stop"])
            self.lbl_title.config(text=f"停顿对照：已完成 {done}/{len(self.plan['stop'])} 段（顺序固定，不要调换）")
        else:
            for e in self.plan["caliper"]:
                what = f"{e['axis']} 方向第 {e['trial']} 次：{e['targets_mm'][0]:g} → {e['targets_mm'][-1]:g} mm" + \
                       ("，快速移动" if e["fast"] else "")
                self.tree.insert("", "end", values=(f"{e['axis']}{e['trial']}", what, e["session"] or "",
                                                    self._status_text(e)))
            done = sum(e["status"] == "done" for e in self.plan["caliper"])
            self.lbl_title.config(text=f"游标卡尺位移标准：已完成 {done}/{len(self.plan['caliper'])} 次往返")

    def _add_block(self) -> None:
        if not messagebox.askyesno(
                "再加一组", "在计划末尾再加 4 段（连续→停顿→停顿→连续），约 15 分钟。\n"
                "每种条件 4 段时，录像间差异可以估计，检出把握明显更高（见实验说明）。\n"
                "加了就要录完，不要根据前几段的结果决定。确定？", parent=self):
            return
        rv.append_stop_block(self.plan)
        rv.save_plan(self.plan)
        self._refresh_plan()
        self._say("计划末尾已加一组 C→P→P→C。")

    def _save_caliper_info(self) -> None:
        try:
            res = float(self.var_res.get())
            zero = float(self.var_zero.get())
        except ValueError:
            messagebox.showerror("卡尺信息", "分度值和零位读数必须是数字（零位：两量爪合拢时的读数，一般 0.00）。", parent=self)
            return
        self.plan["caliper_info"] = {"type": "vernier", "resolution_mm": res, "zero_reading_mm": zero,
                                     "caliper_id": self.var_id.get().strip(), "operator": self.var_op.get().strip(),
                                     "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        rv.save_plan(self.plan)
        self._say(f"卡尺信息已保存：分度值 {res} mm，合拢零位 {zero:+.2f} mm。")
        self._set_idle_cue()

    def _set_idle_cue(self) -> None:
        if self.recording:
            return
        entry = rv.next_entry(self.plan, "stop" if self.mode == "stop" else "caliper")
        if entry is None:
            self._set_cue("全部完成", "done", "计划内的录制都已合格。回到主窗口或运行分析脚本。")
            return
        if self.mode == "stop":
            what = (f"连续段：运动 {rv.MOTION_S:.0f} 秒内一直动，到折返点画弧线转向，不要停住。"
                    if entry["condition"] == "C" else
                    "停顿段：每听到一声高音「嘀」就立刻停稳（手腕或小臂靠住桌沿），约 1.5 秒后听到低音再继续。")
            lo, hi = rv.speed_band(self.reference_speed)
            ref = f"  运动速度要落在 {lo:.0f}–{hi:.0f} mm/s"
            if self.reference_speed:
                ref += f"（参照 {self.reference_speed:.0f}）"
            ref += "，「运动」指标里的「本段」超出范围这一段会作废。"
            first = "" if any(e["status"] == "done" for e in self.plan["stop"]) else \
                "  第一次录之前先点「练习 40 秒」，确认停顿能被判为停稳。"
            self._set_cue(f"下一段：{entry['label']}", "idle",
                          f"{what} 开始后先静止 {rv.REST_START_S:.0f} 秒，结束前静止 {rv.REST_END_S:.0f} 秒，"
                          f"程序会自动停止。{ref}{first}")
        else:
            info = self.plan.get("caliper_info", {})
            need = "" if info.get("zero_reading_mm") is not None else \
                "先填写卡尺信息并点「保存卡尺信息」（零位：装棋盘前两量爪合拢时的读数）。"
            first = entry["trial"] == 1
            path = " → ".join(f"{v:g}" for v in entry["targets_mm"])
            self._set_cue(f"下一段：{entry['axis']}{entry['trial']}", "idle",
                          (mounting(entry["axis"], rv.work_distance_mm(self.app.grabber.width)) + " " if first else
                           "不要重新装夹，直接开始。 ") +
                          f"位置顺序 {path} mm。" +
                          ("本次往返要快速推动量爪（≤1 秒）。 " if entry["fast"] else " ") + need)

    # ------------------------------------------------------------------ actions
    def _preflight(self, entry: dict[str, Any]) -> Optional[str]:
        app = self.app
        if app.grabber.cap_l is None:
            return "相机还没打开：在主窗口先「扫描设备」并「打开相机+双IMU」。"
        if (app.grabber.width, app.grabber.height) != (1280, 720) and not self.demo:
            return f"相机是 {app.grabber.width}×{app.grabber.height}，本实验必须用 1280×720。改分辨率后重新打开相机。"
        if rv.still_now(app.imu_usb) is None:
            return "USB IMU 没有数据（或不到 200 Hz）。"
        if self.mode == "stop" and rv.still_now(app.imu_bt) is None and not self.demo:
            return "蓝牙 IMU 没有数据。停顿实验要把它贴回棋盘背面，用来确认棋盘未动。"
        if not self.demo:
            if rv.camera_config.is_legacy() and app.grabber.cap_r is None:
                return ("当前用的还是旧双目内窥镜的内参。换成单目相机后，先在主窗口「对焦/锁焦」保存实验焦距，"
                        "再用这个焦距录第 4 步内参，然后才能做实验。")
            problem = rv.camera_config.focus_problem(app.grabber.focus_state)
            if problem:
                return problem + "在主窗口「对焦/锁焦」里恢复保存的实验焦距（或重新标定内参）后再开始。"
            cov = rv.camera_config.calibration_coverage()
            if app.grabber.cap_r is None and cov is not None and cov < rv.camera_config.COVERAGE_MIN:
                mid = rv.work_distance_mm(app.grabber.width)[1]
                return (f"当前内参的标定样本只覆盖了画面的 {cov:.0%}（6×4 格，要求 ≥{rv.camera_config.COVERAGE_MIN:.0%}）。"
                        "棋盘落到没标定过的区域时，解算的姿态会偏 0.2–0.3°。先在主窗口第 4 步重录内参："
                        f"焦距不变，棋盘在约 {mid / 10:.0f} cm 处把预览上的灰格子都移绿（约 1 分钟，自动停止），再开始实验。")
            if self.mode == "stop" and len(self._rest_acc) >= 20 and abs(self._acc_ref() - 1.0) > rv.ACC_FAULT_G:
                return rv.acc_bias_message(self._acc_ref())
        now = time.time()
        vis = self.pose.visible_fraction(now, 3.0)
        if vis is not None and vis < 0.5:
            return "左目画面里棋盘检出率太低：先调整距离和角度，让整块棋盘稳定出现在画面里。"
        if self.mode == "caliper":
            live = self.pose.hold_stats(now - 2.0, now)
            if live["n"] == 0:
                return "最近 2 秒没有检出棋盘。"
            lo, mid, hi = rv.work_distance_mm(app.grabber.width)
            if not lo <= live["distance_mm"] <= hi:
                return (f"现在相机到棋盘约 {live['distance_mm']:.0f} mm，应在 {lo:.0f}–{hi:.0f} mm 之间"
                        f"（棋盘在画面里的大小与轨迹录像相同；X、Y 方向建议 {mid:.0f} mm）。挪动相机后再开始。")
        return None

    def _start(self) -> None:
        if self.recording or self.busy:
            return
        kind = "stop" if self.mode == "stop" else "caliper"
        entry = rv.next_entry(self.plan, kind)
        if entry is None:
            messagebox.showinfo("计划", "计划内的录制都已完成。", parent=self)
            return
        problem = self._preflight(entry)
        if problem:
            messagebox.showerror("还不能开始", problem, parent=self)
            return
        if self.mode == "stop":
            rig = None if self.demo else rv.todays_rigidity()
            if rig is None and not self.demo:
                if not messagebox.askyesno(
                        "缺少当天刚性检查",
                        "今天还没有合格的第 9 步刚性检查。分析时要用当天估计的 IMU→相机旋转；"
                        "没有它，冻结模型的输入坐标可能是错的。\n\n仍然开始？", parent=self):
                    return
            elif rig is not None:
                ang = rv.rotation_angle_deg(rv.NOMINAL_R_CI, rig["R_camera_imu"])
                self._say(f"当天刚性检查 {rig['session']}：IMU→相机旋转与 9/24 标定相差 {ang:.1f}°"
                          f"{'（分析会自动使用当天的旋转）' if ang > 2.0 else ''}。")
                if ang > rv.MOUNT_WARN_DEG and not self._mount_ok and not messagebox.askyesno(
                        "IMU 安装与训练数据不同",
                        f"今天的 IMU 安装与冻结模型训练时（9/24）相差 {ang:.0f}°。\n"
                        "分析会用当天的旋转做校正，但 IMU 相对相机的位置（杠杆臂）无法校正，"
                        "会让模型结果变差。\n\n能恢复原来的安装方式就先恢复，再重做第 9 步刚性检查。"
                        "\n\n仍按现在的安装开始？", parent=self):
                    return
                self._mount_ok = True
            self.run = rv.StopRun(entry, acc_ref=self._acc_ref())
        else:
            info = self.plan.get("caliper_info", {})
            if info.get("zero_reading_mm") is None:
                messagebox.showerror("卡尺信息", "先填写并保存卡尺信息（含合拢零位读数）。", parent=self)
                return
            known = rv.known_directions(self.plan, skip=entry)
            rv.save_plan(self.plan)
            self.run = rv.CaliperRun(entry, info["zero_reading_mm"], info, known)
        try:
            self.session = self.app.begin_session("stopexp" if self.mode == "stop" else "caliper",
                                                  self.run.meta())
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("无法开始录制", str(exc), parent=self)
            self.run = None
            return
        self.entry = entry
        entry["status"] = "recording"
        self.recording = True
        self.run.begin(time.time())
        self._refresh_plan()
        self._say(f"开始 {os.path.basename(self.session)}（{entry.get('label') or entry['axis'] + str(entry['trial'])}）")
        if self.mode == "caliper":
            self.ent_reading.focus_set()

    def _confirm_reading(self) -> None:
        if not (self.recording and self.mode == "caliper" and self.run is not None):
            return
        text = self.var_reading.get().strip().replace(",", ".")
        try:
            value = float(text)
        except ValueError:
            messagebox.showerror("读数", f"「{text}」不是数字。", parent=self)
            return
        err = self.run.confirm(value, time.time())
        if err:
            messagebox.showwarning("读数", err, parent=self)
            return
        self.var_reading.set("")
        self._say(f"读数 {value:.2f} mm（目标 {self.run.pending['target_mm']:g} mm），手离开卡尺。")

    def _abort(self) -> None:
        if not self.recording:
            return
        if self.practice:
            self._stop_practice()
            return
        if not messagebox.askyesno("作废当前段", "停止录制并把这一段标为作废？计划中的这一段会保持待录。", parent=self):
            return
        self._abort_now()

    def _stop_practice(self) -> None:
        if getattr(self.run, "metronome", None) is not None:
            self.run.metronome.stop()
        self.recording = self.practice = False
        self.run = None
        self._say("练习已停止。")
        self._set_idle_cue()

    def _abort_now(self, reason: str = "operator") -> None:
        if getattr(self.run, "metronome", None) is not None:
            self.run.metronome.stop()
        self.recording = False
        session, entry = self.session, self.entry

        def done(summary: dict[str, Any]) -> None:
            with open(os.path.join(session, "aborted.json"), "w", encoding="utf-8") as handle:
                json.dump({"aborted_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "reason": reason},
                          handle, ensure_ascii=False)
            self.plan.setdefault("rejected_sessions", []).append(
                {"session": os.path.basename(session), "reason": f"aborted: {reason}"})
            entry["status"], entry["verdict"] = "redo", "已作废"
            rv.save_plan(self.plan)
            try:
                if self.winfo_exists():
                    self._refresh_plan()
                    self._set_idle_cue()
                    self._say(f"已作废 {os.path.basename(session)}。")
            except tk.TclError:
                pass

        self.app.end_session(done)

    def _finish(self) -> None:
        self.recording = False
        self.busy = True
        run, entry, session = self.run, self.entry, self.session
        name = os.path.basename(session)
        self._set_cue("正在检查…", "settle", "时间对齐、逐帧解棋盘和本段质量检查，约 1 分钟。")

        def after_stop(summary: dict[str, Any]) -> None:
            events = run.summary()
            fname = "stop_events.json" if self.mode == "stop" else "caliper_events.json"
            with open(os.path.join(session, fname), "w", encoding="utf-8") as handle:
                json.dump(events, handle, ensure_ascii=False, indent=2)
            if self.demo:
                self._apply_qc(entry, name, {"ok": True, "problems": [], "demo": True}, events)
                return
            known = rv.known_directions(self.plan, skip=entry) if self.mode == "caliper" else []

            def work() -> None:
                try:
                    qc = rv.qc_stop(name, events, int(summary.get("drops") or 0), self.reference_speed) \
                        if self.mode == "stop" else rv.qc_caliper(name, events, known)
                except Exception as exc:  # noqa: BLE001
                    qc = {"ok": False, "problems": [f"检查失败：{exc}"]}
                if summary.get("focus_reapplied"):
                    qc.setdefault("problems", []).append(
                        f"录制中相机焦距被改动 {summary['focus_reapplied']} 次（已自动恢复），图像尺度可能变过")
                    qc["ok"] = False
                self.after(0, lambda: self._apply_qc(entry, name, qc, events))

            threading.Thread(target=work, daemon=True).start()

        self.app.end_session(after_stop)

    def _apply_qc(self, entry: dict[str, Any], name: str, qc: dict[str, Any], events: dict[str, Any]) -> None:
        self.busy = False
        session = os.path.join(rv.DATA_ROOT, name)
        if not self.demo:
            with open(os.path.join(session, "qc_result.json"), "w", encoding="utf-8") as handle:
                json.dump(qc, handle, ensure_ascii=False, indent=2, default=str)
            try:
                self.app._append_collection_log({
                    "session": name, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "theme": "P" if self.mode == "stop" else "S",
                    "split": "stop_experiment" if self.mode == "stop" else "reference_validation",
                    "usable_frac": qc.get("usable_frac"), "reproj_p50_px": qc.get("reproj_p50"),
                    "verdict": "合格" if qc.get("ok") else "不合格：" + "；".join(qc.get("problems", []))})
            except Exception:
                pass
        rv.record_result(self.plan, entry, name, qc, events, self.mode)
        self.reference_speed = self.plan.get("reference_speed_mm_s")
        self._refresh_plan()
        self._set_idle_cue()
        lines = [f"{name}：{'合格' if qc.get('ok') else '不合格'}"]
        if qc.get("usable_frac") is not None:
            lines.append(f"棋盘可用帧 {qc['usable_frac']:.0%}，重投影中位数 {qc.get('reproj_p50') or 0:.2f} px")
        if self.mode == "stop":
            se = qc.get("stop_events") or events
            if se.get("condition") == "P":
                lines.append(f"硬停 {se.get('beats_matched')}/{se.get('beats')} 个节拍（{(se.get('compliance') or 0):.0%}）")
            lines.append(f"IMU 判出停顿 {se.get('stops')} 次（{se.get('stops_per_min') or 0:.1f} 次/分钟），"
                         f"运动速度约 {se.get('median_moving_speed_mm_s') or 0:.0f} mm/s")
        elif qc.get("trial_fit"):
            f = qc["trial_fit"]
            lines.append(f"本次往返直线拟合：比例 {f['scale']:.3f}，残差 {f['rms_mm']:.2f} mm，"
                         f"最差停顿抖动 {f['worst_hold_p95_mm']:.2f} mm")
        lines += qc.get("problems", []) + qc.get("warnings", [])
        self._say(" | ".join(lines))
        (messagebox.showinfo if qc.get("ok") else messagebox.showwarning)(
            "本段检查", "\n".join(lines) + ("" if qc.get("ok") else "\n\n这一段需要重录（文件保留，计划里已标注）。"),
            parent=self)

    def _open_doc(self) -> None:
        path = os.path.join(rv.ROOT_DIR, "docs", "revision_physical_experiments.md")
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except Exception:
            self._say(path)

    # ------------------------------------------------------------------ live loop
    def _tick(self) -> None:
        try:
            self._update()
        except Exception as exc:  # noqa: BLE001
            msg = str(exc)
            if msg != getattr(self, "_tick_error", None):
                self._tick_error = msg
                try:
                    self._say(f"画面刷新跳过一次：{msg}")
                except Exception:
                    pass
        finally:
            try:
                self.after(100, self._tick)
            except tk.TclError:
                pass

    def _update(self) -> None:
        now = time.time()
        app = self.app
        self.pose.update(app.grabber)
        vis = self.pose.visible_fraction(now)
        self._set_ind("board", "-" if vis is None else f"{vis:.0%}",
                      None if vis is None or vis >= 0.8 else COLORS["warn"])
        if self.mode == "stop":
            self._update_stop(now)
        else:
            self._update_caliper(now)

    def _acc_ref(self) -> float:
        """Accelerometer norm of the camera IMU while it was last held without rotation."""
        if len(self._rest_acc) >= 20:
            return float(sorted(self._rest_acc)[len(self._rest_acc) // 2])
        return 1.0

    def _update_stop(self, now: float) -> None:
        run: Optional[rv.StopRun] = self.run if self.recording else None
        still = rv.still_now(self.app.imu_usb, run.acc_ref if run is not None else self._acc_ref())
        if still is None:
            self._set_ind("imu", "无数据", COLORS["warn"])
        else:
            if still["w_max"] < rv.STILL_W_DPS:
                self._rest_acc.append(still["acc_norm"])
            text = ("静止" if still["still"] else "运动") + f"  {still['w_max']:.1f}°/s"
            colour = COLORS["go"] if still["still"] else None
            if run is not None and run.phase == "motion" and now - run.t_motion > 10.0:
                rot = rv.rate_p95(self.app.imu_usb, max(run.t_motion, now - 20.0), now)
                if rot is not None:
                    text += f"  转动p95 {rot:.0f}"
                    if rot < rv.ROTATION_MIN_P95_DPS:
                        colour = COLORS["warn"]
            self._set_ind("imu", text, colour)
        motion = self.pose.motion(now)
        sp, tr = motion["speed_mm_s"], motion["travel_3s_mm"]
        lo, hi = rv.speed_band(self.reference_speed)
        scored = run.running_speed() if run is not None and run.phase == "motion" else None
        text = f"{'-' if sp is None else f'{sp:.0f}'} mm/s  目标 {lo:.0f}-{hi:.0f}"
        colour = None
        if scored is not None:
            text += f"  本段 {scored:.0f}"
            # The first seconds of the median are one or two windows and jump
            # outside the band even when the rest of the take is fine.
            if run is not None and now - run.t_motion > 8.0 and not lo <= scored <= hi:
                colour = COLORS["warn"]
        self._set_ind("motion", f"{text} / {'-' if tr is None else f'{tr:.0f}'} mm", colour)
        if run is None:
            return
        cues = run.tick(now, still, self.pose)
        stats = run.stats(now)
        if run.condition == "P":
            comp = stats.get("compliance")
            self._set_ind("score", f"{stats.get('beats_matched', 0)}/{stats.get('beats', 0)}" +
                          ("" if comp is None else f"  {comp:.0%}"),
                          None if comp is None or comp >= 0.8 else COLORS["warn"])
        else:
            self._set_ind("score", f"误停 {stats['stops']} 次", None if stats["stops"] <= 1 else COLORS["warn"])
        if run.phase == "rest_start":
            self._set_cue(f"静止  {run.t_motion - now:4.1f} s", "rest",
                          f"{run.label}：握稳相机不要动，整块棋盘在左目画面里。听到长音后开始运动。")
        elif run.phase == "motion":
            left = run.t_rest_end - now
            cam_rate = rv.peak_rate(self.app.imu_usb, now - 8.0, now) if now - run.t_motion > 10.0 else None
            if cam_rate is not None and cam_rate < rv.CAMERA_MOVING_DPS and not sp:
                self._set_cue("相机没在动", "warn", "移动的必须是相机（连同 USB IMU），棋盘固定在桌上不动。")
            elif run.condition == "P" and run.metronome is not None and run.metronome.beats and \
                    now - run.metronome.beats[-1] < rv.HOLD_S:
                self._set_cue("停！", "stop", "完全停住，听到低音再走。")
            else:
                last = stats.get("last_beat_matched")
                mark = "" if last is None else ("  上一次停稳 ✓" if last else "  上一次没停稳 ✗")
                if run.condition == "C" and run.still:
                    self._set_cue("你停下了 — 继续动", "warn", "连续段不要停顿，到折返点画弧线转向。")
                else:
                    self._set_cue(f"运动  {left:5.1f} s{mark}", "motion",
                                  "连续运动，不要停。" if run.condition == "C" else
                                  "照常运动；高音一响立刻停稳（手腕靠住桌沿），低音再走。")
        elif run.phase == "rest_end":
            self._set_cue(f"静止  {run.t_done - now:4.1f} s", "rest", "停在原处保持静止，程序会自动结束。")
        if run.metronome is not None and run.metronome.beats:
            b = run.metronome.beats[-1]
            if self._quality_beat != b and now >= b + rv.HOLD_S + 0.3:
                self._quality_beat = b
                q = rv.stop_quality(self.app.imu_usb, b, b + rv.HOLD_S + 0.3, run.acc_ref)
                if q is not None:
                    self._quality.append(q)
                    if self.practice:
                        self._say(f"这次停顿最稳的 0.5 秒：{q['w_max_dps']:.1f}°/s、{q['dacc_max_g'] * 1000:.0f} mg"
                                  + ("  ✓ 停稳" if q["passes"] else "  ✗ 没停稳（要 <1°/s 且 <10 mg）"))
        if "done" in cues:
            if self.practice:
                self._end_practice()
            else:
                self._finish()

    def _practice(self) -> None:
        if self.recording or self.busy:
            return
        if rv.still_now(self.app.imu_usb) is None:
            messagebox.showerror("还不能练习", "USB IMU 没有数据（或不到 200 Hz）。", parent=self)
            return
        self.run = rv.StopRun({"condition": "P", "label": "练习", "order": 0}, practice=True, acc_ref=self._acc_ref())
        self.practice, self.recording, self.entry, self.session = True, True, None, ""
        self._quality, self._quality_beat = [], None
        self.run.begin(time.time())
        self._say("练习开始：40 秒，不录像。每 4 秒一声高音，停稳后 IMU 状态要变绿（<1°/s）。")

    def _end_practice(self) -> None:
        run = self.run
        stats = run.stats(run.t_done + rv.STOP_MATCH_S + 1.0)
        self.recording = self.practice = False
        self.run = None
        good, n = sum(q["passes"] for q in self._quality), len(self._quality)
        w = sorted(q["w_max_dps"] for q in self._quality)
        msg = (f"练习结束：{stats.get('beats_matched', 0)}/{stats.get('beats', 0)} 个节拍被判为硬停，"
               f"{good}/{n} 次停顿达标" + (f"，最稳 0.5 秒的角速度中位数 {w[len(w) // 2]:.1f}°/s" if w else "") + "。")
        self._say(msg)
        self._set_idle_cue()
        tip = "" if n and good >= 0.9 * n else (
            "\n\n停不稳时：高音一响，把握相机的手腕或小臂靠在桌沿上（或用另一只手托住手腕），"
            "保持到低音再走；然后再练一次，达到 9 成以上再正式录（每段漏停超过 3 次就要重录）。")
        messagebox.showinfo("练习结果", msg + tip, parent=self)

    def _update_caliper(self, now: float) -> None:
        rates = [rv.peak_rate(w, now - 1.0, now) for w in (self.app.imu_bt, self.app.imu_usb)]
        rates = [r for r in rates if r is not None]
        shake = max(rates) if rates else None
        self._set_ind("imu", "无数据" if shake is None else f"{shake:.1f}°/s",
                      None if shake is None or shake < rv.BOARD_SHAKE_DPS else COLORS["warn"])
        t, p, _R = self.pose.between(now - 2.0, now)
        if len(p) >= 3:
            live = self.pose.hold_stats(now - 2.0, now)
            jitter, dist = live.get("jitter_px_p95"), live["distance_mm"]
            lo, _mid, hi = rv.work_distance_mm(self.app.grabber.width)
            bad = (jitter is not None and jitter > rv.HOLD_JITTER_PX) or not lo <= dist <= hi
            self._set_ind("score", ("-" if jitter is None else f"{jitter:.2f} px") + f" / {dist:.0f} mm",
                          COLORS["warn"] if bad else None)
        run: Optional[rv.CaliperRun] = self.run if self.recording else None
        if run is not None and run.pauses and len(p):
            import numpy as np
            first = np.asarray(run.pauses[0]["live"].get("median_mm") or p[-1])
            self._set_ind("pos", f"{float(np.linalg.norm(p[-1] - first)):.2f} mm")
        if run is None:
            return
        msgs = run.tick(now, self.pose, self.app.imu_bt, self.app.imu_usb)
        for m in msgs:
            if m.startswith("mounting:"):
                text = m.split(":", 1)[1]
                self._say("装夹方向不对：" + text)
                self._abort_now("wrong mounting: " + text)
                messagebox.showerror("装夹方向不对", text + "\n\n这一段已作废。改好装夹后重新开始这一组。", parent=self)
                return
            if m.startswith("mountwarn:"):
                self._say(m.split(":", 1)[1])
            elif m.startswith("rejected:"):
                self._say("这次保持不合格：" + m.split(":", 1)[1] + "。请重新锁紧、读数并确认同一位置。")
            elif m == "accepted":
                ev = run.pauses[-1]
                self._say(f"{ev['target_mm']:g} mm 处记录完成（读数 {ev['reading_mm']:.2f}，"
                          f"角点抖动 {ev['live'].get('jitter_px_p95', 0):.2f} px）。")
        if run.state == "move":
            how = "快速推动量爪（≤1 秒），" if run.fast else ""
            self._set_cue(f"移到 ≈ {run.target:g} mm", "move",
                          f"{how}推到附近后锁紧螺丝，从侧面读游标（不要挡住镜头），输入实际读数并回车。")
        elif run.state == "settle":
            self._set_cue(f"手离开  {rv.SETTLE_S - (now - run.t_state):3.1f} s", "settle",
                          "手离开卡尺和桌子，等待静止。")
        elif run.state == "hold":
            self._set_cue(f"静止记录  {rv.CAL_HOLD_S - (now - run.t_state):3.1f} s", "hold",
                          "不要碰卡尺和桌子。")
        elif run.state == "done":
            self._finish()

    def _on_close(self) -> None:
        if self.practice:
            self._stop_practice()
        if self.recording and not messagebox.askyesno("关闭", "正在录制。关闭窗口会作废这一段，确定？", parent=self):
            return
        if self.recording:
            self._abort_now()
        try:
            self.app._revision_panel = None
        except Exception:
            pass
        self.destroy()


# --------------------------------------------------------------------------
# simulated rig for rehearsal (python -m ego_capture.revision_ui --demo stop)
# --------------------------------------------------------------------------
class _DemoImu:
    """200-Hz IMU whose motion state is set by the demo operator."""

    def __init__(self, operator) -> None:
        import numpy as np
        self.operator = operator
        self.hz = 200.0
        self._hist: list = []
        self._lock = threading.Lock()
        self._rng = np.random.default_rng(0)
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        import numpy as np
        while True:
            now = time.time()
            moving = self.operator.moving(now)
            gyro = self._rng.normal(0, 8.0 if moving else 0.15, 3)
            acc = np.array([0.0, 0.0, 1.0]) + self._rng.normal(0, 0.05 if moving else 0.002, 3)
            with self._lock:
                self._hist.append((now, tuple(acc), tuple(gyro)))
                self._hist = self._hist[-4000:]
            time.sleep(0.005)

    def tail(self, n: int = 2000):
        with self._lock:
            return list(self._hist[-n:])


class _DemoGrabber:
    """Projects the GP050 corners for a camera moving as the demo operator says."""

    def __init__(self, operator) -> None:
        import numpy as np
        self.operator = operator
        self.width, self.height = 1280, 720
        self.cap_l = object()
        self.recording = False
        self._lock = threading.Lock()
        self.frame_l = np.zeros((720, 1280, 3), np.uint8)
        cal = json.loads(rv.INTRINSICS_JSON.read_text(encoding="utf-8"))
        self.K = np.asarray(cal["camera_matrix"], float)
        self.dist = np.asarray(cal["distortion_coefficients"], float)
        self.need_board = True
        self.focus_state = {"autofocus": 0.0, "focus": 30.0, "t": 0.0}

    def overlay(self, side: str):
        import cv2
        import numpy as np
        now = time.time()
        stamp = now - (now % 0.4)
        offset = self.operator.position(stamp)
        obj = np.zeros((88, 3), np.float32)
        obj[:, :2] = np.mgrid[0:11, 0:8].T.reshape(-1, 2) * 3.0
        rvec = np.array([0.35, -0.2, 0.05])
        tvec = np.array([-15.0, -10.0, 150.0]) - offset
        img, _ = cv2.projectPoints(obj, rvec, tvec, self.K, self.dist)
        img += np.random.default_rng(int(stamp * 10)).normal(0, 0.1, img.shape)
        return True, img.astype(np.float32), (11, 8), stamp


class _DemoOperator:
    """Moves on a random smooth path, obeys the metronome, or follows the caliper."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.panel: Optional[RevisionPanel] = None

    def moving(self, now: float) -> bool:
        p = self.panel
        if p is None or not p.recording or p.run is None:
            return False
        run = p.run
        if self.mode == "caliper":
            return run.state == "move"
        if run.phase != "motion":
            return False
        if run.condition == "P" and run.metronome is not None and run.metronome.beats:
            return not (0.15 < now - run.metronome.beats[-1] < rv.HOLD_S + 0.3)
        return True

    def position(self, now: float):
        import numpy as np
        p = self.panel
        if self.mode == "caliper" and p is not None and p.run is not None:
            run = p.run
            axis = {"X": 0, "Y": 1, "Z": 2}[run.entry["axis"]]
            v = np.zeros(3)
            v[axis] = (run.pending or {}).get("reading_mm", run.target or 0.0) if run.state != "move" else \
                (run.pauses[-1]["reading_mm"] if run.pauses else 0.0)
            return v
        return 30.0 * np.array([np.sin(0.7 * now), np.sin(0.43 * now + 1.0), 0.5 * np.sin(0.31 * now)]) \
            if self.moving(now) or (p is not None and p.recording) else np.zeros(3)


class _DemoApp(tk.Tk):
    def __init__(self, mode: str) -> None:
        super().__init__()
        self.withdraw()
        self.operator = _DemoOperator(mode)
        self.imu_usb = _DemoImu(self.operator)
        self.imu_bt = _DemoImu(_DemoOperator("none"))
        self.grabber = _DemoGrabber(self.operator)
        self.demo_dir = os.path.join(rv.DATA_ROOT, "revision_rehearsal", "gui_demo")
        os.makedirs(self.demo_dir, exist_ok=True)

    def _log(self, msg: str) -> None:
        print(msg, flush=True)

    def begin_session(self, step: str, meta: dict[str, Any]) -> str:
        session = os.path.join(self.demo_dir, f"{step}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(session, exist_ok=True)
        with open(os.path.join(session, "session_meta.json"), "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        self.grabber.recording = True
        return session

    def end_session(self, on_done) -> None:
        self.grabber.recording = False
        self.after(200, lambda: on_done({"drops": 0}))


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", choices=("stop", "caliper"), required=True)
    ap.add_argument("--fast", action="store_true", help="shorten the stop protocol to 40 s for a quick look")
    ap.add_argument("--snapshot", default="", help="save window snapshots to this folder and exit")
    ap.add_argument("--condition", choices=("C", "P"), default="C", help="stop demo: first segment to run")
    ap.add_argument("--practice", action="store_true", help="stop demo with --snapshot: run the 40-s practice")
    args = ap.parse_args()
    if args.fast:
        rv.REST_START_S, rv.MOTION_S, rv.REST_END_S = 4.0, 24.0, 4.0
    rv.camera_config.CONFIG = rv.DATA_ROOT / "revision_rehearsal" / "demo_active_camera.json"  # endoscope intrinsics
    rv.PLAN_PATH = rv.DATA_ROOT / "revision_rehearsal" / f"demo_plan_{args.demo}.json"
    rv.CALIPER_PLAN_CSV = rv.DATA_ROOT / "revision_rehearsal" / "demo_caliper_plan.csv"
    rv.PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    plan = rv.default_plan()
    if args.condition == "P":
        plan["stop"][0]["status"] = "done"
    rv.save_plan(plan)
    app = _DemoApp(args.demo)
    panel = RevisionPanel(app, args.demo, demo=True)
    app.operator.panel = panel
    if args.demo == "caliper":
        panel.var_zero.set("0.00")
        panel.var_id.set("demo-150mm")
        panel._save_caliper_info()
    if args.snapshot:
        _autopilot(app, panel, args.demo, args.snapshot, args.practice)
    app.mainloop()


def _autopilot(app, panel: RevisionPanel, mode: str, folder: str, practice: bool = False) -> None:
    """Scripted run with periodic window snapshots (for testing the GUI)."""
    from PIL import ImageGrab
    os.makedirs(folder, exist_ok=True)
    shots = {"n": 0}

    def snap(tag: str) -> None:
        panel.attributes("-topmost", True)
        panel.lift()
        panel.update()
        time.sleep(0.25)
        panel.update()
        screen = ImageGrab.grab()
        k = screen.size[0] / float(panel.winfo_screenwidth())
        x, y = panel.winfo_rootx(), panel.winfo_rooty()
        w, h = panel.winfo_width(), panel.winfo_height()
        screen.crop((int(x * k), int(y * k), int((x + w) * k), int((y + h) * k))).save(
            os.path.join(folder, f"{shots['n']:02d}_{tag}.png"))
        shots["n"] += 1
        panel.attributes("-topmost", False)

    messagebox.showinfo = lambda *a, **k: None  # type: ignore[assignment]
    messagebox.showwarning = lambda *a, **k: None  # type: ignore[assignment]
    messagebox.showerror = lambda *a, **k: None  # type: ignore[assignment]
    messagebox.askyesno = lambda *a, **k: True  # type: ignore[assignment]
    if practice:
        panel.after(1500, panel._practice)
        for t, tag in ((14.5, "practice_stop"), (47.5, "practice_done")):
            panel.after(int(t * 1000), lambda tag=tag: snap(tag))
        panel.after(49000, app.destroy)
        return
    panel.after(1500, panel._start)
    if mode == "stop":
        for t, tag in ((3.0, "rest"), (12.0, "motion"), (13.7, "beat"), (37.0, "done")):
            panel.after(int(t * 1000), lambda tag=tag: snap(tag))
        panel.after(40000, app.destroy)
    else:
        def feed(k: int) -> None:
            run = panel.run
            if run is None or not panel.recording:
                return
            if run.state == "move" and run.target is not None:
                panel.var_reading.set(f"{run.target + 0.04:.2f}")
                panel._confirm_reading()
                if k in (0, 1):
                    panel.after(800, lambda: snap(f"settle{k}"))
                    panel.after(4500, lambda: snap(f"hold{k}"))
            panel.after(1000, lambda: feed(k + 1))
        panel.after(2500, lambda: snap("start"))
        panel.after(3000, lambda: feed(0))
        panel.after(92000, lambda: snap("end"))
        panel.after(94000, app.destroy)


if __name__ == "__main__":
    main()
