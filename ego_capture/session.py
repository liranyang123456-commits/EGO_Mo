#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session layout for dual-IMU / stereo calibration and later ego/recon capture."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from .rig import HOUSING_OUTER_MM, LENS_DIAMETER_MM, STEREO_BASELINE_MM

BOARD_NAME = "GP050-3-12*9"
SQUARES = (12, 9)
INNER = (11, 8)
SQUARE_MM = 3.0
SQUARE_M = SQUARE_MM / 1000.0

# 第二块棋盘：刚性绑在 双目+USB-IMU 刚体上，供固定的 4K 全局相机观察。
# 7×5 格、每格 5 mm（内角点 6×4），板面约 35×25 mm。
# 注意：7×5 格（奇×奇）存在 180° 旋转歧义，离线解算时用时间连续性消除。
BOARD2_NAME = "RIG-7x5-5mm"
SQUARES2 = (7, 5)
INNER2 = (6, 4)
SQUARE2_MM = 5.0
SQUARE2_M = SQUARE2_MM / 1000.0

FRAMES_COLUMNS = [
    "frame_idx", "frame_timestamp",
    "imu_latest_idx", "imu_latest_timestamp", "imu_time_diff",
    "imu_usb_idx", "imu_usb_timestamp", "imu_usb_dt",
    "imu_bt_idx", "imu_bt_timestamp", "imu_bt_dt",
    "board_left", "board_right",
    "gyro_norm_dps", "acc_norm_g",
]

STEPS = [
    ("scan", "0  扫描设备",
     "点「扫描设备」，用鼠标点缩略图指定左目（右目必选，三系标定需要双目）。确认 USB IMU 和蓝牙 IMU。"),
    ("gate", "1  帧率门禁",
     "双目 + 两路 IMU 一起跑 8 秒。目标 1280×720@30 或 1080p@30；两路 IMU 都应接近 200Hz。"),
    ("imu_static", "2  双IMU静止对齐",
     "把蓝牙 IMU 紧贴 USB IMU（同向、尽量重合），整套放桌上不要动，录 30–45 秒。用重力+磁力求 R(I_usb←I_bt)。"),
    ("imu_dyn", "3  双IMU动态对齐",
     "两枚 IMU 仍刚性贴在一起，端着它们平移一段距离，并绕滚/俯/航三轴转，45–90 秒。用手眼精化旋转。"),
    ("intrinsics", "4  双目内参",
     "拿开蓝牙 IMU。GP050 整板入左右画。远近、四角、倾斜，约 90 秒。"),
    ("board_bt", "5  棋盘+蓝牙IMU",
     "把蓝牙 IMU 粘/绑在棋盘上（采集中不要松）。板在双目里充分运动 2–3 分钟：远近、倾斜、绕轴转。解 T(C0←I_bt)。"),
    ("compose", "6  三系投影",
     "不录新数据。用前面的对齐结果和基线 46.5 mm 组成 I_usb、I_bt、C0、C1 之间的投影。点「开始」即解算。"
     "残差网络不在本步训练：标签够了再离线训 gru_xattn。"),
    ("ego", "7  EgoMotion",
     "三系外参冻结后再采。纹理丰富，前 5 秒就要有多轴旋转+平移，60–90 秒。"),
    ("recon", "8  三维重建",
     "慢速扫目标，重叠约 70%。JPEG 序列给 COLMAP。可用已标定的 T(C0←C1) 做立体。"),
    ("rigid", "9  IMU 刚性检查 + 旋转/静止训练段",
     "每天开工先录这一段，60–90 秒。USB IMU 要用硬支架和螺丝/环氧固定在双目相机外壳上，尽量靠近镜头。"
     "棋盘完整在左目画面里：先静止 5 秒，再慢慢点头 ±20° 三次、摇头 ±20° 三次、绕光轴转 ±25° 三次，每次 1–2 秒，"
     "中间各停 2 秒；最后静止 5 秒。程序比较约 1 秒的转角，并自动估计 IMU 到相机的安装旋转。"
     "三个轴相关 ≥ 0.75、转角幅值相关 ≥ 0.90、比例接近 1 才算可用；三个轴都 ≥ 0.90 记为高质量。"
     "合格的刚性段同时作为常规训练数据（分组 extra_train，纯旋转 + 静止），不进验证/测试。"),
    ("traj", "10 轨迹（按主题）",
     "先通过第 9 步刚性检查。棋盘固定，蓝牙粘在板上且不要碰。轨迹统一用 1280×720（实际约 10 fps）："
     "重投影中位数 0.2 px，比 640×480 的 1 px 精确得多，长会话下 10 fps 的窗口数已足够。"
     "在上方「主题」里选本段动作（程序会推荐下一个并定好分组，录前不要改）。每段约 3 分钟：先静止 8 秒，"
     "按主题连续动作，每 10–15 秒明显停 1–2 秒，最后静止 10 秒；距离 15–40 cm，行程尽量大（10–20 cm），"
     "棋盘始终完整在左目画面里。停止后自动检查并写入 datasets/collection_log.csv。"
     "完整流程见 docs/collection_protocol.md。"),
    ("stage_ref", "11 已知位移参考验证",
     "把相机刚性固定在精密位移台/三轴滑台上，棋盘固定不动。每个计划位置静止录制 8–12 秒，"
     "位置与会话名填写到 datasets/reference_stage_plan.csv。至少覆盖 X/Y/Z 各轴的"
     "0、±5、±10、±20 mm，并完成 3 个独立重复。分析只允许刚体对齐、不拟合尺度；"
     "运行 tools/reference_stage_validation.py analyze 生成独立参考验证报告。"),
    ("stereo_ext", "12 双目外参（静止姿态组）",
     "棋盘和相机都不要连续运动。每个远近/倾斜/四角姿态静止保持 2 秒，再移动到下一姿态；"
     "采集至少 25 个姿态、总时长 60–90 秒。两相机没有硬件同步，只有静止保持才能保证"
     "最近时间戳图像对应同一姿态。停止后运行 tools/calibrate_stereo_extrinsic.py，"
     "必须通过留出外极线、旋转和平移误差门限后才可替换名义 46.5-mm 外参。"),
    ("g4k_intrinsics", "13 4K 全局相机内参",
     "4K 相机固定在全局机位，变焦/对焦已锁定（锁定后全程不要再碰，否则内参作废）。"
     "手持 GP050 棋盘在全局视野里、与实际工作距离一致的位置，远近、四角、倾斜慢慢变换，约 90 秒。"
     "4K 分辨率下小棋盘可稳定检测；若检测率太低，把棋盘移近一些再录一条。"),
    ("g4k_rig", "14 全局双视外参",
     "解算「新棋盘(7×5×5mm) → 双目左目」的固定外参。新棋盘已刚性绑在 双目+IMU 刚体上。"
     "把刚体放进全局视野，使 4K 相机同时看到刚体上的新棋盘和 GP050 棋盘，且双目左目也看到 GP050；"
     "每个姿态静止保持 2 秒再换，采 25 个以上姿态。程序用共享的 GP050 把两台相机联系起来，"
     "解出固定外参并做留出一致性检查。停止后运行 tools/calibrate_global_rig.py。"),
    ("global_traj", "15 全局运动采集",
     "4K 相机固定不动、焦距锁定。刚体 A（双目+USB-IMU+新棋盘）与刚体 B（蓝牙IMU+GP050）"
     "在全局视野里相互运动，两块棋盘都要始终完整可见（这是全局真值）。每段约 3 分钟：先静止 8 秒，"
     "连续动作并每 10–15 秒明显停 1–2 秒，最后静止 10 秒；运动幅度尽量大。"
     "4K 视频很大，确认数据盘剩余 ≥ 15 GB。停止后自动做时刻同步与棋盘覆盖率检查。"),
]

# v2 动作主题（2026-09-25 起）。每段约 3 分钟，强调大幅度、多样化、明显停顿。
# 每个训练主题录 4 段：3 段训练 + 1 段验证；T 为独立测试段，永远封存，至少 4 段。
TRAJ_THEMES = [
    ("K", "大幅混合-慢", "平移 + 俯仰 + 偏航混合，行程 10–20 cm、距离 15–40 cm，慢速，每 10–15 秒停 1–2 秒"),
    ("L", "大幅混合-快", "同 K 但快速（棋盘不糊为限），角速度可到 60–100°/s，仍每 10–15 秒停 1–2 秒"),
    ("M", "长停走", "移动 1–2 秒、完全停住 2 秒，反复；方向、幅度、速度每次都换"),
    ("N", "旋转主导", "点头 / 摇头 ±20–30°、绕光轴 ±30°，平移尽量小于 3 cm，其中插入 3 次 3 秒静止"),
    ("O", "换人换握法", "另一操作者或另一种握法，自由混合运动，含停顿；与 K/L 同幅度"),
    ("T", "独立测试", "自由混合运动 3 分钟，含停顿与大幅动作；每段尽量换日期、操作者或握法，永远封存"),
]
LEGACY_THEMES = {
    "A": "左右平移", "B": "上下平移", "C": "前后推进", "D": "俯仰", "E": "偏航",
    "F": "横滚", "G": "画圈", "H": "停走", "I": "自由快动", "J": "空档（测试）", "R": "刚性检查",
}
THEME_PLAN = {
    "K": ("train", "train", "val", "train"),
    "L": ("train", "train", "val", "train"),
    "M": ("train", "train", "val", "train"),
    "N": ("train", "train", "val", "train"),
    "O": ("train", "train", "val", "train"),
    "T": ("test", "test", "test", "test"),
}
THEME_SPLITS = ("train", "train", "val", "train")
RIGID_SPLIT = "extra_train"
STUDY_PROTOCOL_LOCK = (
    Path(__file__).resolve().parents[1] / "datasets" /
    "study_protocol_v2.lock.json"
)


def active_protocol_id() -> str | None:
    """Return the frozen prospective protocol ID, if installed."""
    try:
        payload = json.loads(STUDY_PROTOCOL_LOCK.read_text(encoding="utf-8"))
        if payload.get("status") == "FROZEN":
            return str(payload["protocol_id"])
    except (OSError, KeyError, json.JSONDecodeError):
        return None
    return None

# 侧栏分组；残差映射在第 6 步之后离线训，不占按钮。
PHASES = [
    ("A 接通", ("scan", "gate")),
    ("B 几何三系", ("imu_static", "imu_dyn", "intrinsics", "board_bt", "compose")),
    ("C 应用采集", ("ego", "recon", "rigid", "traj", "stage_ref", "stereo_ext")),
    ("D 全局视角", ("g4k_intrinsics", "g4k_rig", "global_traj")),
]


def now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_checkerboard_yaml(path: str) -> None:
    text = (
        f"# {BOARD_NAME}\n"
        "target_type: 'checkerboard'\n"
        f"targetCols: {INNER[0]}   # inner corners\n"
        f"targetRows: {INNER[1]}\n"
        f"rowSpacingMeters: {SQUARE_M:.6f}\n"
        f"colSpacingMeters: {SQUARE_M:.6f}\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if hasattr(obj, "tolist"):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_jsonable)


def session_prefix(step: str) -> str:
    return {
        "imu_static": "align_imu_static",
        "imu_dyn": "align_imu_dyn",
        "intrinsics": "calib_intrinsics",
        "board_bt": "calib_board_bt",
        "ego": "ego_seq",
        "recon": "recon_seq",
        "traj": "traj",
        "rigid": "rigid",
        "stage_ref": "stage",
        "stereo_ext": "stereo_ext",
        "g4k_intrinsics": "calib_g4k",
        "g4k_rig": "calib_grig",
        "global_traj": "gtraj",
    }.get(step, "seq")


def write_checkerboard2_yaml(path: str) -> None:
    text = (
        f"# {BOARD2_NAME}\n"
        "target_type: 'checkerboard'\n"
        f"targetCols: {INNER2[0]}   # inner corners\n"
        f"targetRows: {INNER2[1]}\n"
        f"rowSpacingMeters: {SQUARE2_M:.6f}\n"
        f"colSpacingMeters: {SQUARE2_M:.6f}\n"
    )
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def disk_free_gb(path: str) -> float:
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        return free / (1024 ** 3)
    except Exception:
        return 999.0


def rig_banner() -> str:
    return (
        f"标定板 {BOARD_NAME}  12×9×3mm  内角点 11×8    "
        f"双目光心基线 {STEREO_BASELINE_MM:.1f} mm "
        f"(外壳外侧 {HOUSING_OUTER_MM:.0f} mm − 镜头直径 {LENS_DIAMETER_MM:.1f} mm)"
    )
