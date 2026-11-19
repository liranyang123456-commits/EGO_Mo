#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physical rig constants: stereo endoscope + dual IMU + chessboard."""

from __future__ import annotations

# 外壳外侧距 5.2 cm；每只镜头直径 5.5 mm，光心相对外壳各内缩一个半径。
# 光心基线 = 52.0 - 2*(5.5/2) = 52.0 - 5.5 = 46.5 mm
# （文档写成 5.2cm-5.5mm/2=46.5mm，算术符号有笔误，数值按 46.5 mm）
HOUSING_OUTER_MM = 52.0
LENS_DIAMETER_MM = 5.5
STEREO_BASELINE_MM = HOUSING_OUTER_MM - LENS_DIAMETER_MM  # 46.5
STEREO_BASELINE_M = STEREO_BASELINE_MM / 1000.0

# 蓝牙 IMU 粘在棋盘上时，IMU 原点相对棋盘原点（内角点 (0,0)）的偏移，单位 mm。
# 未量之前先当 0；量过之后改这里或写进 session_meta。
BOARD_BT_IMU_OFFSET_MM = (0.0, 0.0, 0.0)

FRAMES = {
    "I_usb": "USB 维特 IMU（刚体绑在双目内镜上）",
    "I_bt": "蓝牙 BWT901CL / WTSDCL（先与 USB 对齐，再贴到棋盘）",
    "C0": "左目光心",
    "C1": "右目光心，+X_C0 方向距 C0 为基线 46.5 mm",
    "O": "棋盘 GP050 内角点坐标系，格距 3 mm",
}
