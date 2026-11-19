#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Guided recordings for the revision experiments (no GUI code here).

Two experiments that need only the existing rig and a vernier caliper
(docs/revision_physical_experiments.md):

  caliper  The GP050 board rides on the sliding jaw of a vernier caliper in
           front of the fixed camera. For every planned position the operator
           sets and locks the jaw, reads the vernier and types the reading;
           the program then records a hands-off hold and stores the hold
           window with the reading (caliper_events.json).
  stopexp  Continuous handheld motion against motion with a firm stop at every
           4-s beat, recorded in a fixed order on one day. The program times
           the rest and motion phases, sounds the beats and checks from the
           USB IMU, with the detector of the analysis, whether each stop was
           firm (stop_events.json).

The camera and IMU threads are the existing ones of the capture app; the
classes below only read them.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from . import camera_config
from .session import SQUARE_MM

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT_DIR / "datasets"
PLAN_PATH = DATA_ROOT / "revision_experiment_plan.json"
CALIPER_PLAN_CSV = DATA_ROOT / "reference_caliper_plan.csv"
INTRINSICS_JSON = camera_config.intrinsics_json(name=camera_config.LEGACY_CALIB)   # twin rendering only

# Clock and sound; the twin rehearsal (tools/sim_revision_capture.py) replaces
# them to drive the same logic in simulated time.
CLOCK = time.time
SOUND = True
THREADED_METRONOME = True

# Stop experiment. The detector is the one of tools/stop_anchor_check.py.
# 80 s of motion per recording: in the twin a 2+2 design still separates the
# conditions (95% half-width 0.096 against a 0.25 gain, 0.067 at 160 s); two
# blocks (4+4) of 80 s match the precision of 2+2 at 160 s.
STOP_ORDER = ("C", "P", "P", "C")
REST_START_S = 5.0
MOTION_S = 80.0
REST_END_S = 5.0
BEAT_S = 4.0
# Handheld pauses in the recordings pass the detector only 23% of the time
# (best 0.5-s window: 2.9 deg/s median); after a ~0.3-s reaction a 1.5-s hold
# leaves room for the 0.5 s of stillness it needs. The twin stops last 1-2 s.
HOLD_S = 1.5
PRACTICE_S = (3.0, 40.0, 2.0)
STOP_MATCH_S = 2.0
STILL_W_DPS = 1.0
STILL_ACC_G = 0.01
STILL_S = 0.5
# The conditions must differ by the stops alone. The 2026-11-08 handheld batch
# confounded the stop with the speed (paused 12-15 mm/s, continuous 36 mm/s).
# A tripod on the desk cannot hold 36 mm/s, and the 19 mm/s tripod recording of
# 2026-11-09 still beat zero motion, so the handheld figure is not the target.
# Every recording stays inside SPEED_ABS_MM_S. Once a continuous recording is
# accepted, the others must also stay within SPEED_BAND of its median speed.
SPEED_ABS_MM_S = (15.0, 30.0)
SPEED_BAND = (0.7, 1.3)

# Vernier caliper standard. The 1-3-mm steps show the resolution, the larger
# ones the scale over the 10-40-mm range of the 3-s labels.
CALIPER_TARGETS = (0.0, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 40.0)
CALIPER_AXES = ("X", "Y", "Z")
CALIPER_TRIALS = 3
SETTLE_S = 1.5
CAL_HOLD_S = 6.0
# A hold counts as hands-off when the detected corners do not move. With camera
# and board at rest, the median corner of a detection lies 0.12-0.24 px (p95)
# from its hold median in 15 recorded intervals, whereas 0.1 mm of lateral
# travel at 150 mm moves it by about 0.7 px. PnP positions are not used for
# this test: their rotation noise grows with distance.
HOLD_JITTER_PX = 0.5
HOLD_MIN_POSES = 8
BOARD_SHAKE_DPS = 3.0
# The three mountings must move the board along clearly different directions of
# the camera frame, and the trials of one axis share their mounting (the
# analysis takes the direction of a trial from the other trials of its axis).
MOUNT_CHECK_MM = 5.0
AXIS_SEPARATION_DEG = 45.0
SAME_MOUNT_DEG = 5.0
MOUNT_FIX = {
    "X": "X 方向：主尺与画面水平方向大致平行。",
    "Y": "Y 方向：相机和棋盘都不动，把卡尺在桌面上原地转 90°，让量爪朝着/背着相机移动；"
         "棋盘竖着正对相机时，也可以改为把相机绕光轴侧放 90°。",
    "Z": "Z 方向：量爪要沿与 X、Y 都垂直的方向移动：相机不动，把卡尺竖起来（主尺竖直，用胶带固定在书堆"
         "或盒子侧面），棋盘重新贴在活动量爪上并朝向相机，量爪上下移动。",
}
# Stop experiment: the camera, not the board, is moved by hand. Stillness is
# judged around the accelerometer norm of the recording's initial rest (the
# camera IMU read 1.009 g on 2026-11-08); an offset beyond ACC_FAULT_G points
# to a fault of the IMU rather than to its calibration.
CAMERA_MOVING_DPS = 3.0
# The camera, not the board, must move. A camera carried on a tripod can
# translate with almost no rotation, so a low camera rate is a fault only when
# the board does not move in the image either; a turning board IMU always is.
BOARD_MOVING_DPS = 8.0
# Angular rate of the camera during the motion phase. The handheld trajectory
# recordings of the study have a 95th-percentile rate of 12-42 deg/s. A
# recording with almost no rotation cannot be aligned (the camera-IMU delay is
# estimated from rotation) and does not test stop anchoring either, whose error
# grows with the rotation between stops, so it would look favourable for a
# reason that has nothing to do with the stops.
ROTATION_MIN_P95_DPS = 8.0
STUDY_ROTATION_P95_DPS = (12.0, 42.0)
ACC_FAULT_G = 0.05


def work_distance_mm(width: int = 1280) -> tuple[float, float, float]:
    """(low, recommended, high) distance at which the board has the same size in
    pixels as in the trajectory recordings (110-150-230 mm with the endoscope)."""
    return camera_config.work_distance_mm(width)

# Camera <- IMU rotation that the frozen models assume for every session
# (tools/train_seq.py R_CAMERA_IMU_NEW, from rigid_20260924_142902). The mount
# was later changed (37 deg on 2026-09-26 afternoon, 11-13 deg for gtraj_*);
# beyond MOUNT_WARN_DEG the lever arm has probably changed as well.
MOUNT_WARN_DEG = 5.0
NOMINAL_R_CI = np.array([
    [0.007668, -0.946526, -0.322536],
    [-0.003137, 0.322521, -0.946557],
    [0.999966, 0.008270, -0.000496],
])


# --------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------
def default_plan(order: tuple[str, ...] = STOP_ORDER) -> dict[str, Any]:
    stop, counts = [], {"C": 0, "P": 0}
    for i, cond in enumerate(order):
        counts[cond] += 1
        stop.append({"order": i + 1, "condition": cond, "label": f"{cond}{counts[cond]}",
                     "status": "todo", "session": None, "verdict": None})
    caliper = []
    for axis in CALIPER_AXES:
        for trial in range(1, CALIPER_TRIALS + 1):
            targets = list(CALIPER_TARGETS if trial % 2 else reversed(CALIPER_TARGETS))
            caliper.append({"axis": axis, "trial": trial, "fast": trial == CALIPER_TRIALS,
                            "targets_mm": targets, "status": "todo", "session": None,
                            "verdict": None})
    return {"version": 1, "created_at": datetime.now().isoformat(timespec="seconds"),
            "stop": stop, "caliper": caliper, "caliper_info": {}, "rejected_sessions": []}


def load_plan() -> dict[str, Any]:
    if PLAN_PATH.is_file():
        try:
            return json.loads(PLAN_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    plan = default_plan()
    save_plan(plan)
    return plan


def save_plan(plan: dict[str, Any]) -> None:
    PLAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    PLAN_PATH.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")


def next_entry(plan: dict[str, Any], kind: str) -> Optional[dict[str, Any]]:
    for entry in plan[kind]:
        if entry["status"] in ("todo", "redo"):
            return entry
    return None


def append_stop_block(plan: dict[str, Any]) -> None:
    """One more block in the order of the first one (C, P, P, C by default)."""
    for cond in STOP_ORDER:
        k = sum(e["condition"] == cond for e in plan["stop"]) + 1
        plan["stop"].append({"order": len(plan["stop"]) + 1, "condition": cond,
                             "label": f"{cond}{k}", "status": "todo", "session": None,
                             "verdict": None})


def record_result(plan: dict[str, Any], entry: dict[str, Any], name: str, qc: dict[str, Any],
                  events: dict[str, Any], mode: str) -> None:
    """Enter the verdict of one recording into the plan (and the caliper CSV)."""
    entry["session"] = name
    if qc.get("ok"):
        entry["status"], entry["verdict"] = "done", "合格"
        if mode == "caliper":
            entry["readings_mm"] = events["readings_mm"]
            entry["travel_dir_camera"] = qc.get("travel_dir_camera")
        if mode == "stop" and entry["condition"] == "C" and plan.get("reference_speed_mm_s") is None:
            plan["reference_speed_mm_s"] = events.get("median_moving_speed_mm_s")
    else:
        entry["status"], entry["verdict"] = "redo", "；".join(qc.get("problems", [])) or "不合格"
        plan.setdefault("rejected_sessions", []).append({"session": name, "reason": qc.get("problems")})
    save_plan(plan)
    if mode == "caliper":
        update_caliper_csv(plan)


def update_caliper_csv(plan: dict[str, Any]) -> None:
    """Keep datasets/reference_caliper_plan.csv in step with the recorded trials."""
    import csv
    info = plan.get("caliper_info", {})
    fields = ["axis", "trial", "readings_mm", "session", "operator", "caliper_id",
              "caliper_resolution_mm", "temperature_c", "notes"]
    with CALIPER_PLAN_CSV.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for e in plan["caliper"]:
            readings = e.get("readings_mm") or e["targets_mm"]
            writer.writerow({
                "axis": e["axis"], "trial": e["trial"],
                "readings_mm": " ".join(f"{v:g}" for v in readings),
                "session": e["session"] if e["status"] == "done" else "",
                "operator": info.get("operator", ""), "caliper_id": info.get("caliper_id", ""),
                "caliper_resolution_mm": info.get("resolution_mm", ""),
                "temperature_c": info.get("temperature_c", ""),
                "notes": "fast moves" if e["fast"] else "",
            })


def travel_direction(R_ref, p_ref, p) -> Optional[np.ndarray]:
    """Unit direction, in the camera frame, of the board's translation between
    two holds (p: camera centre in the board frame; R_ref: camera -> board)."""
    d = -np.asarray(R_ref, float).T @ (np.asarray(p, float) - np.asarray(p_ref, float))
    n = float(np.linalg.norm(d))
    return d / n if n > 1e-9 else None


def axis_angle_deg(u, v) -> float:
    """Angle between two lines; the sign of a travel direction is arbitrary."""
    c = abs(float(np.dot(u, v))) / max(float(np.linalg.norm(u) * np.linalg.norm(v)), 1e-12)
    return float(np.degrees(np.arccos(min(1.0, c))))


def trial_direction(name: str) -> Optional[list[float]]:
    """Travel direction of a recorded caliper trial, from its chessboard labels."""
    try:
        events = json.loads((DATA_ROOT / name / "caliper_events.json").read_text(encoding="utf-8"))
        raw = np.load(DATA_ROOT / "pose_gt_raw" / f"{name}.npz")
    except (OSError, ValueError):
        return None
    t, R, use = raw["t"], raw["R"].astype(np.float64), raw["usable"] == 1
    p = raw["p"].astype(np.float64) * 1000.0
    holds = []
    for ev in events.get("pauses", []):
        idx = np.flatnonzero(use & (t >= ev["t_hold_start"]) & (t <= ev["t_hold_end"]))
        if len(idx) >= 10:
            holds.append((float(ev["reading_mm"]), np.median(p[idx], axis=0), R[idx[0]]))
    if len(holds) < 2:
        return None
    lo, hi = min(holds, key=lambda h: h[0]), max(holds, key=lambda h: h[0])
    if hi[0] - lo[0] < MOUNT_CHECK_MM:
        return None
    u = travel_direction(lo[2], lo[1], hi[1])
    return None if u is None else np.round(u, 4).tolist()


def known_directions(plan: dict[str, Any], skip: Optional[dict[str, Any]] = None) -> list[tuple[str, int, list[float]]]:
    """(axis, trial, direction) of the accepted caliper trials; fills in the
    direction of trials accepted before the plan stored it."""
    out = []
    for e in plan.get("caliper", []):
        if e is skip or e.get("status") != "done" or not e.get("session"):
            continue
        if e.get("travel_dir_camera") is None:
            e["travel_dir_camera"] = trial_direction(e["session"])
        if e["travel_dir_camera"] is not None:
            out.append((e["axis"], int(e["trial"]), e["travel_dir_camera"]))
    return out


def mounting_check(axis: str, u, known) -> tuple[Optional[str], Optional[str]]:
    """(problem, warning) for a trial of `axis` that moves the board along u."""
    problem = warning = None
    for other, trial, v in known:
        ang = axis_angle_deg(u, v)
        if other != axis and ang < AXIS_SEPARATION_DEG and problem is None:
            problem = (f"这一组是 {axis} 方向，但棋盘在相机里的移动方向与已录的 {other}{trial} 只差 {ang:.0f}°"
                       f"（要求 ≥{AXIS_SEPARATION_DEG:.0f}°），还是 {other} 方向的装夹。" + MOUNT_FIX.get(axis, ""))
        elif other == axis and ang > SAME_MOUNT_DEG and warning is None:
            warning = (f"与同方向的 {other}{trial} 相差 {ang:.0f}°：装置被挪动过。同一方向的几次往返要在同一装夹下录，"
                       "否则分析时不能互相作为方向参考。")
    return problem, warning


def acc_bias_message(acc_norm_g: float) -> str:
    return (f"相机上的 USB IMU 静止时读数 {acc_norm_g:.3f} g，偏离 1 g 达 {abs(acc_norm_g - 1.0) * 1000:.0f} mg。"
            f"程序会自动校正 {ACC_FAULT_G * 1000:.0f} mg 以内的偏差，这么大的偏差多半是 IMU 没接好或设置不对："
            "重新插拔 IMU，看主窗口 USB IMU 卡片里静止时 acc 三个数的平方和是否接近 1。")


def camera_motion(name: str, t0: float, t1: float) -> Optional[dict[str, float]]:
    """p95 angular rate [deg/s] of the camera (USB) and board (Bluetooth) IMUs."""
    from .sync import load_imu
    session = DATA_ROOT / name
    out: dict[str, float] = {}
    for key, fname in (("camera_p95_dps", "imu_stream.csv"), ("board_p95_dps", "imu_bt.csv")):
        if not (session / fname).is_file():
            continue
        t, y = load_imu(session / fname)
        sel = (t >= t0) & (t <= t1)
        if sel.sum() >= 50:
            out[key] = float(np.quantile(np.linalg.norm(y[sel, 3:6], axis=1), 0.95))
    return out if "camera_p95_dps" in out else None


def delay_problem(name: str) -> Optional[str]:
    """Labels are shifted by the camera delay found after the recording
    (sync_delay.json); without a reliable value inside the search range the
    labels and the IMU windows do not line up."""
    from .sync import SEARCH_S
    path = DATA_ROOT / name / "sync_delay.json"
    if not path.is_file():
        return None
    cam = (json.loads(path.read_text(encoding="utf-8")).get("cameras") or {}).get("cam0") or {}
    delay = float(cam.get("delay_usb_s") or 0.0)
    if not cam.get("delay_usb_reliable"):
        return (f"相机与 USB IMU 的时间延迟没有估计出来（相关 {float(cam.get('delay_usb_corr') or 0.0):.2f}）："
                "标签和 IMU 对不齐。录像里相机要有明显的转动。")
    if delay <= SEARCH_S[0] + 0.005 or delay >= SEARCH_S[1] - 0.005:
        return f"相机与 USB IMU 的延迟估计 {delay * 1000:.0f} ms 落在搜索范围的边上，可能不对。"
    return None


def speed_band(reference: Optional[float]) -> tuple[float, float]:
    """Accepted median moving speed [mm/s].

    The absolute range is what a tripod pass can repeat. After a continuous
    recording sets the reference, the range shrinks to ±30% of that speed so
    the paused recordings cannot come out slower."""
    lo, hi = SPEED_ABS_MM_S
    if reference:
        lo = max(lo, reference * SPEED_BAND[0])
        hi = min(hi, reference * SPEED_BAND[1])
    if hi < lo:
        return SPEED_ABS_MM_S
    return lo, hi


def speed_problem(events: dict[str, Any], reference: Optional[float]) -> Optional[str]:
    band = speed_band(reference)
    speed = events.get("median_moving_speed_mm_s")
    if speed is None or band[0] <= speed <= band[1]:
        return None
    how = "慢" if speed < band[0] else "快"
    if reference:
        return (f"运动段速度 {speed:.0f} mm/s，比参照的 {reference:.0f} mm/s {how}太多"
                f"（要求 {band[0]:.0f}–{band[1]:.0f} mm/s）：两种条件的速度必须一致，"
                "否则分不清误差变化是停顿造成的还是走得慢造成的。")
    return (f"运动段速度 {speed:.0f} mm/s，不在 {band[0]:.0f}–{band[1]:.0f} mm/s。"
            "低于 15 mm/s 时 3 秒位移太小，模型比不过零运动；高于 30 mm/s 三脚架停不稳，后面的段也对不齐。")


def motion_problem(events: dict[str, Any], motion: Optional[dict[str, float]]) -> Optional[str]:
    """What moved during the motion phase: it has to be the camera.

    Rotation alone does not decide it, because a tripod-borne camera translates
    with little rotation; the board moving in the image is the evidence that the
    camera moved, and a turning board IMU is the evidence that it did not."""
    if motion is None:
        return None
    board = motion.get("board_p95_dps", 0.0)
    cam = motion["camera_p95_dps"]
    if board > BOARD_MOVING_DPS:
        return (f"运动段棋盘上的蓝牙 IMU 在动（p95 {board:.0f}°/s，相机上的 USB IMU p95 {cam:.1f}°/s）："
                "动的是棋盘。棋盘要固定在桌上不动，移动的必须是相机。")
    if cam < CAMERA_MOVING_DPS and not events.get("median_moving_speed_mm_s"):
        return (f"运动段相机几乎没动（USB IMU p95 {cam:.1f}°/s，棋盘在画面里也没有移动）："
                "相机没有被移动，或主窗口选的 COM 口不是相机上的 IMU。")
    return None


def rotation_note(motion: Optional[dict[str, float]]) -> Optional[str]:
    """Low rotation is reported, and rejects a take only when the delay could
    not be estimated. A tripod pass that still aligns should not be discarded
    for landing just under the 8 deg/s hint."""
    if motion is None or motion["camera_p95_dps"] >= ROTATION_MIN_P95_DPS:
        return None
    cam = motion["camera_p95_dps"]
    return (f"运动段转动偏少（p95 {cam:.1f}°/s，建议 ≥{ROTATION_MIN_P95_DPS:.0f}）。"
            "时间延迟已经对上，这一段仍可用；下一遍可以多摇一点云台。")


def continuous_stop_problem(events: dict[str, Any]) -> Optional[str]:
    """Real pauses in a continuous recording.

    A hesitation shorter than 1 s, and the stillness that begins the required
    rest at the end, are not pauses. Counting them made a normal turnaround
    fail the take."""
    if events.get("condition") != "C":
        return None
    t0, t1 = events.get("t_motion"), events.get("t_rest_end")
    if t0 is None or t1 is None:
        if events.get("stops_per_min", 0) > 1.0:
            return (f"连续段出现 {events.get('stops')} 次停顿"
                    f"（{events['stops_per_min']:.1f} 次/分钟）")
        return None
    kept = [s for s in events.get("imu_stops") or []
            if t0 <= s["t_start"] <= t1 - 2.0 and s["t_end"] - s["t_start"] >= 1.0]
    minutes = max(t1 - t0, 1e-6) / 60.0
    rate = len(kept) / minutes
    if rate > 1.0:
        return (f"连续段出现 {len(kept)} 次停顿（{rate:.1f} 次/分钟）。"
                "短于 1 秒的折返和结尾那一下静止不算。")
    return None


# --------------------------------------------------------------------------
# live signals
# --------------------------------------------------------------------------
def imu_window(worker, seconds: float):
    """Last `seconds` of (time, acc [g], gyro [deg/s]) from an ImuWorker."""
    hist = worker.tail(int(seconds * 260) + 20)
    if not hist:
        return None
    t_end = hist[-1][0]
    if CLOCK() - t_end > 0.5:
        return None
    rows = [h for h in hist if h[0] >= t_end - seconds]
    if len(rows) < 3:
        return None
    return (np.array([r[0] for r in rows]), np.array([r[1] for r in rows]),
            np.array([r[2] for r in rows]))


def still_now(worker, acc_ref: float = 1.0) -> Optional[dict[str, Any]]:
    """IMU stillness over the last 0.5 s with the analysis thresholds. acc_ref
    is the accelerometer norm of this IMU at rest: the 10-mg tolerance applies
    around it, so a calibration offset of the IMU does not hide every stop."""
    win = imu_window(worker, STILL_S)
    if win is None:
        return None
    t, acc, gyro = win
    if len(t) < 40 or t[-1] - t[0] < 0.8 * STILL_S:
        return None
    w_max = float(np.linalg.norm(gyro, axis=1).max())
    norm = np.linalg.norm(acc, axis=1)
    dacc = float(np.abs(norm / acc_ref - 1.0).max())
    return {"still": w_max < STILL_W_DPS and dacc < STILL_ACC_G, "w_max": w_max, "dacc_max": dacc,
            "acc_norm": float(norm.mean()), "acc_min": float(norm.min()), "acc_max": float(norm.max()),
            "t": float(t[-1])}


def rest_reference(samples: list[dict[str, Any]]) -> Optional[float]:
    """Accelerometer norm at rest from still_now samples without rotation."""
    norms = [s["acc_norm"] for s in samples if s["w_max"] < STILL_W_DPS]
    return float(np.median(norms)) if len(norms) >= 10 else None


def peak_rate(worker, t0: float, t1: float) -> Optional[float]:
    """Largest angular rate [deg/s] of an IMU between two host times."""
    hist = worker.tail(int((t1 - t0) * 260) + 50)
    rates = [float(np.linalg.norm(h[2])) for h in hist if t0 <= h[0] <= t1]
    return max(rates) if rates else None


def rate_p95(worker, t0: float, t1: float) -> Optional[float]:
    """95th-percentile angular rate [deg/s], the statistic motion_problem scores."""
    hist = worker.tail(int((t1 - t0) * 260) + 50)
    rates = [float(np.linalg.norm(h[2])) for h in hist if t0 <= h[0] <= t1]
    return float(np.percentile(rates, 95)) if len(rates) >= 50 else None


def stop_quality(worker, t0: float, t1: float, acc_ref: float = 1.0) -> Optional[dict[str, float]]:
    """Stillest STILL_S window between t0 and t1: its largest angular rate and
    specific-force deviation, the two quantities the detector thresholds."""
    hist = [h for h in worker.tail(int((t1 - t0 + 1.0) * 260) + 50) if t0 <= h[0] <= t1]
    if len(hist) < 20:
        return None
    t = np.array([h[0] for h in hist])
    w = np.array([float(np.linalg.norm(h[2])) for h in hist])
    a = np.array([abs(float(np.linalg.norm(h[1])) / acc_ref - 1.0) for h in hist])
    best = None
    for i in range(len(t)):
        j = int(np.searchsorted(t, t[i] + STILL_S))
        if j > len(t) or t[min(j, len(t)) - 1] - t[i] < 0.8 * STILL_S:
            break
        cand = (float(w[i:j].max()), float(a[i:j].max()))
        if best is None or max(cand[0] / STILL_W_DPS, cand[1] / STILL_ACC_G) < \
                max(best[0] / STILL_W_DPS, best[1] / STILL_ACC_G):
            best = cand
    if best is None:
        return None
    return {"w_max_dps": best[0], "dacc_max_g": best[1],
            "passes": best[0] < STILL_W_DPS and best[1] < STILL_ACC_G}


class LivePose:
    """Monocular PnP on the corners the capture app detects for its overlay.

    The app searches the left image every 0.4 s; each new detection gives the
    camera centre in the board frame (mm), the quantity whose differences are
    the labels of the study.
    """

    def __init__(self) -> None:
        cal = json.loads(camera_config.intrinsics_json().read_text(encoding="utf-8"))
        self.K0 = np.asarray(cal["camera_matrix"], dtype=np.float64)
        self.dist = np.asarray(cal["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)
        self.size0 = tuple(int(v) for v in cal["image_size"])
        self.poses: deque = deque(maxlen=5000)
        self.corners: deque = deque(maxlen=5000)
        self.checks: deque = deque(maxlen=5000)
        self._last = 0.0

    def _K(self, width: int, height: int) -> np.ndarray:
        K = self.K0.copy()
        if (width, height) != self.size0:
            K[0, 0] *= width / self.size0[0]
            K[0, 2] *= width / self.size0[0]
            K[1, 1] *= height / self.size0[1]
            K[1, 2] *= height / self.size0[1]
        return K

    def update(self, grabber) -> None:
        try:
            board, corners, pattern, stamp = grabber.overlay("l")
        except (AttributeError, TypeError, ValueError):
            return
        if stamp <= self._last:
            return
        self._last = stamp
        self.checks.append((stamp, bool(board)))
        if not board or corners is None or not pattern:
            return
        try:
            with grabber._lock:
                frame = grabber.frame_l
            height, width = frame.shape[:2] if frame is not None else (grabber.height, grabber.width)
            cols, rows = pattern
            pts = np.asarray(corners, np.float32).reshape(-1, 2)
            if len(pts) != cols * rows:
                return
            obj = np.zeros((cols * rows, 3), np.float32)
            obj[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
            obj *= SQUARE_MM
            ok, rvec, tvec = cv2.solvePnP(obj, pts.reshape(-1, 1, 2),
                                          self._K(width, height), self.dist, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                return
            R, _ = cv2.Rodrigues(rvec)
            self.poses.append((stamp, (-R.T @ tvec.reshape(3)).astype(np.float64), R.T.copy()))
            self.corners.append((stamp, tuple(pattern), pts.astype(np.float64)))
        except (cv2.error, ValueError, TypeError, AttributeError):
            return

    def visible_fraction(self, now: float, window_s: float = 10.0) -> Optional[float]:
        flags = [found for stamp, found in self.checks if stamp >= now - window_s]
        return float(np.mean(flags)) if len(flags) >= 3 else None

    def between(self, t0: float, t1: float):
        sel = [(s, p, R) for s, p, R in self.poses if t0 <= s <= t1]
        if not sel:
            return np.zeros(0), np.zeros((0, 3)), np.zeros((0, 3, 3))
        return (np.array([s for s, _p, _R in sel]), np.array([p for _s, p, _R in sel]),
                np.array([R for _s, _p, R in sel]))

    def hold_stats(self, t0: float, t1: float) -> dict[str, Any]:
        """Corner jitter (the stillness test) and PnP spread (reported only)."""
        _t, p, _R = self.between(t0, t1)
        if len(p) == 0:
            return {"n": 0}
        med = np.median(p, axis=0)
        spread = np.linalg.norm(p - med, axis=1)
        out = {"n": int(len(p)), "median_mm": med.round(3).tolist(),
               "distance_mm": round(float(np.linalg.norm(med)), 1),
               "p95_mm": round(float(np.quantile(spread, 0.95)), 3)}
        sets = [(pat, c) for s, pat, c in self.corners if t0 <= s <= t1]
        if sets:
            pattern = max({pat for pat, _c in sets}, key=lambda q: sum(pat == q for pat, _c in sets))
            ref = np.median(np.array([c for pat, c in sets if pat == pattern]), axis=0)
            jitter = []
            for pat, c in sets:
                if pat != pattern:
                    continue
                d = min(np.median(np.linalg.norm(c - ref, axis=1)),
                        np.median(np.linalg.norm(c[::-1] - ref, axis=1)))
                jitter.append(float(d))
            out["jitter_px_p95"] = round(float(np.quantile(jitter, 0.95)), 3)
            out["n_corners"] = len(jitter)
        return out

    def motion(self, now: float, window_s: float = 20.0) -> dict[str, Optional[float]]:
        """Median speed while moving and median 3-s label magnitude, both in mm."""
        t, p, R = self.between(now - window_s, now)
        out: dict[str, Optional[float]] = {"speed_mm_s": None, "travel_3s_mm": None}
        if len(t) < 4:
            return out
        dt = np.diff(t)
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ok = (dt > 0.2) & (dt < 1.0)
        speed = step[ok] / dt[ok]
        moving = speed[speed > 5.0]
        if len(moving):
            out["speed_mm_s"] = float(np.median(moving))
        mags = []
        for i in range(len(t)):
            j = int(np.argmin(np.abs(t - (t[i] - 3.0))))
            if abs(t[i] - t[j] - 3.0) < 0.5:
                mags.append(float(np.linalg.norm(R[j].T @ (p[i] - p[j]))))
        if mags:
            out["travel_3s_mm"] = float(np.median(mags))
        return out


def beep(freq: int, ms: int) -> None:
    """Non-blocking tone; silent where winsound is unavailable."""
    if not SOUND:
        return

    def _run() -> None:
        try:
            import winsound
            winsound.Beep(int(freq), int(ms))
        except Exception:
            pass
    threading.Thread(target=_run, daemon=True).start()


class Metronome:
    """A high tone at every beat (stop) and a low one after the hold (go)."""

    def __init__(self, beat_s: float = BEAT_S, hold_s: float = HOLD_S) -> None:
        self.beat_s, self.hold_s = beat_s, hold_s
        self.beats: list[float] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._next: Optional[float] = None
        self._end = 0.0

    def poll(self, now: float) -> None:
        """Beats that are due, when the metronome runs on the caller's clock."""
        while self._next is not None and self._next <= now and self._next <= self._end - 0.5:
            self.beats.append(self._next)
            self._next += self.beat_s

    def start(self, t_motion: float, t_end: float) -> None:
        if not THREADED_METRONOME:
            self._next, self._end = t_motion + self.beat_s, t_end
            return

        def run() -> None:
            try:
                import winsound
                tone = winsound.Beep
            except Exception:
                tone = None
            k = 1
            while not self._stop.is_set():
                t_beat = t_motion + k * self.beat_s
                if t_beat > t_end - 0.5:
                    break
                wait = t_beat - time.time()
                if wait > 0 and self._stop.wait(wait):
                    break
                self.beats.append(time.time())
                if tone is not None:
                    tone(1600, 150)
                if self._stop.wait(max(self.hold_s - 0.15, 0.0)):
                    break
                if tone is not None:
                    tone(700, 70)
                k += 1
        self._thread = threading.Thread(target=run, daemon=True, name="metronome")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._next = None


# --------------------------------------------------------------------------
# one recording of the stop experiment
# --------------------------------------------------------------------------
class StopRun:
    def __init__(self, entry: dict[str, Any], practice: bool = False, acc_ref: float = 1.0) -> None:
        """acc_ref: accelerometer norm at rest measured before the recording;
        the initial rest of the recording replaces it."""
        self.entry = entry
        self.condition = entry["condition"]
        self.label = entry["label"]
        self.practice = practice
        self.rest_start_s, self.motion_s, self.rest_end_s = \
            PRACTICE_S if practice else (REST_START_S, MOTION_S, REST_END_S)
        self.t0: Optional[float] = None
        self.metronome = Metronome() if self.condition == "P" else None
        self.phase = "idle"
        self.still = False
        self._still_since: Optional[float] = None
        self.stops: list[dict[str, float]] = []
        self.acc_ref = float(acc_ref)
        self.rest_samples: list[dict[str, Any]] = []
        self.motion_speed: list[float] = []

    def meta(self) -> dict[str, Any]:
        return {"theme": "P", "split": "stop_experiment", "stop_protocol": {
            "label": self.label, "condition": self.condition,
            "condition_name": "continuous" if self.condition == "C" else "paused",
            "order": self.entry["order"], "beat_s": BEAT_S if self.condition == "P" else None,
            "hold_s": HOLD_S if self.condition == "P" else None, "rest_start_s": self.rest_start_s,
            "motion_s": self.motion_s, "rest_end_s": self.rest_end_s,
            "detector": {"gyro_max_dps": STILL_W_DPS, "acc_tol_g": STILL_ACC_G, "min_s": STILL_S}}}

    @property
    def t_motion(self) -> float:
        return float(self.t0) + self.rest_start_s

    @property
    def t_rest_end(self) -> float:
        return self.t_motion + self.motion_s

    @property
    def t_done(self) -> float:
        return self.t_rest_end + self.rest_end_s

    def begin(self, t0: float) -> None:
        self.t0 = t0
        self.phase = "rest_start"
        beep(880, 300)

    def tick(self, now: float, still: Optional[dict[str, Any]], pose: LivePose) -> list[str]:
        """Advance the phase; returns cue names for the GUI."""
        cues: list[str] = []
        if self.t0 is None:
            return cues
        if self.metronome is not None:
            self.metronome.poll(now)
        if self.phase == "rest_start" and now >= self.t_motion:
            self.phase = "motion"
            self.acc_ref = rest_reference(self.rest_samples) or self.acc_ref
            beep(1200, 500)
            if self.metronome is not None:
                self.metronome.start(self.t_motion, self.t_rest_end)
            cues.append("motion")
        elif self.phase == "motion" and now >= self.t_rest_end:
            self.phase = "rest_end"
            if self.metronome is not None:
                self.metronome.stop()
            beep(600, 600)
            cues.append("rest_end")
        elif self.phase == "rest_end" and now >= self.t_done:
            self.phase = "done"
            beep(1000, 120)
            beep(1000, 120)
            cues.append("done")
        if still is not None:
            if self.phase == "rest_start":
                self.rest_samples.append(still)
            if still["still"] and not self.still:
                self.still, self._still_since = True, still["t"] - STILL_S
            elif not still["still"] and self.still:
                self.still = False
                self.stops.append({"t_start": float(self._still_since), "t_end": float(still["t"])})
        if self.phase == "motion":
            speed = pose.motion(now, 6.0)["speed_mm_s"]
            if speed is not None:
                self.motion_speed.append(speed)
        return cues

    def running_speed(self) -> Optional[float]:
        """Median moving speed so far: the quantity the speed check scores."""
        return float(np.median(self.motion_speed)) if self.motion_speed else None

    def _motion_stops(self) -> list[dict[str, float]]:
        stops = list(self.stops)
        if self.still and self._still_since is not None:
            stops.append({"t_start": float(self._still_since), "t_end": CLOCK()})
        return [s for s in stops if self.t_motion <= s["t_start"] <= self.t_rest_end]

    def stats(self, now: float) -> dict[str, Any]:
        stops = self._motion_stops()
        minutes = max(min(now, self.t_rest_end) - self.t_motion, 1e-6) / 60.0
        out = {"stops": len(stops), "stops_per_min": len(stops) / minutes if now > self.t_motion else 0.0}
        if self.metronome is not None:
            beats = [b for b in self.metronome.beats if b < now - STOP_MATCH_S]
            hit = [any(b - 0.3 <= s["t_start"] <= b + STOP_MATCH_S for s in stops) for b in beats]
            out.update({"beats": len(beats), "beats_matched": int(sum(hit)),
                        "compliance": float(np.mean(hit)) if hit else None,
                        "last_beat_matched": hit[-1] if hit else None})
        return out

    def summary(self) -> dict[str, Any]:
        stats = self.stats(self.t_done + STOP_MATCH_S + 1.0)
        rs = self.rest_samples
        still = [s["w_max"] < STILL_W_DPS and max(abs(s["acc_min"] / self.acc_ref - 1.0),
                                                   abs(s["acc_max"] / self.acc_ref - 1.0)) < STILL_ACC_G for s in rs]
        rest = {"acc_norm_g": round(float(np.median([s["acc_norm"] for s in rs])), 4) if rs else None,
                "acc_ref_g": round(self.acc_ref, 4),
                "still_fraction": round(float(np.mean(still)), 3) if rs else None}
        return {
            "kind": "stop_protocol_recording", **self.meta()["stop_protocol"],
            "t_start": self.t0, "t_motion": self.t_motion, "t_rest_end": self.t_rest_end,
            "t_done": self.t_done,
            "beat_times": list(self.metronome.beats) if self.metronome is not None else [],
            "imu_stops": self._motion_stops(), "initial_rest": rest,
            "median_moving_speed_mm_s": round(float(np.median(self.motion_speed)), 1)
            if self.motion_speed else None,
            **{k: v for k, v in stats.items() if k != "last_beat_matched"},
        }


# --------------------------------------------------------------------------
# one trial of the caliper standard
# --------------------------------------------------------------------------
class CaliperRun:
    def __init__(self, entry: dict[str, Any], zero_mm: float, caliper_info: dict[str, Any],
                 known: tuple = ()) -> None:
        """known: (axis, trial, direction) of the accepted trials (known_directions)."""
        self.entry = entry
        self.targets = [float(v) for v in entry["targets_mm"]]
        self.fast = bool(entry["fast"])
        self.zero_mm = float(zero_mm)
        self.caliper_info = dict(caliper_info)
        self.known = list(known)
        self.travel_dir: Optional[list[float]] = None
        self.k = 0
        self.state = "move"
        self.t_state = 0.0
        self.pending: Optional[dict[str, Any]] = None
        self.pauses: list[dict[str, Any]] = []
        self.rejected: list[dict[str, Any]] = []
        self.t0: Optional[float] = None

    def meta(self) -> dict[str, Any]:
        return {"theme": "S", "split": "reference_validation", "caliper_protocol": {
            "axis": self.entry["axis"], "trial": self.entry["trial"], "fast_moves": self.fast,
            "targets_mm": self.targets, "zero_reading_mm": self.zero_mm,
            "caliper": self.caliper_info, "settle_s": SETTLE_S, "hold_s": CAL_HOLD_S}}

    @property
    def target(self) -> Optional[float]:
        return self.targets[self.k] if self.k < len(self.targets) else None

    def begin(self, t0: float) -> None:
        self.t0 = t0
        self.state = "move"

    def confirm(self, reading_mm: float, now: float) -> Optional[str]:
        if self.state != "move" or self.target is None:
            return "现在不是输入读数的时候"
        if not -0.5 <= reading_mm <= 150.0:
            return "读数超出卡尺量程"
        if abs(reading_mm - self.target) > 3.0:
            return f"读数 {reading_mm:.2f} mm 离目标 {self.target:g} mm 太远（>3 mm），确认后重新输入"
        self.pending = {"index": self.k, "target_mm": self.target, "reading_mm": round(reading_mm, 3),
                        "t_confirm": now}
        self.state, self.t_state = "settle", now
        return None

    def tick(self, now: float, pose: LivePose, board_imu, cam_imu=None) -> list[str]:
        """board_imu: the Bluetooth IMU on the sliding jaw; cam_imu: the USB IMU
        on the camera. A touch of the caliper or a knock on the camera or the
        table shows in one of them."""
        msgs: list[str] = []
        if self.state == "settle" and now - self.t_state >= SETTLE_S:
            self.state, self.t_state = "hold", now
            beep(1000, 120)
        elif self.state == "hold" and now - self.t_state >= CAL_HOLD_S:
            t0, t1 = self.t_state, now
            stats = pose.hold_stats(t0, t1)
            rates = [peak_rate(w, t0, t1) for w in (board_imu, cam_imu) if w is not None]
            rates = [r for r in rates if r is not None]
            shake = max(rates) if rates else None
            reasons = []
            if stats["n"] < HOLD_MIN_POSES:
                reasons.append(f"棋盘只检出 {stats['n']} 次")
            elif stats.get("jitter_px_p95", 99.0) > HOLD_JITTER_PX:
                reasons.append(f"棋盘角点在保持期间移动了 {stats.get('jitter_px_p95', 99.0):.2f} px（应 ≤{HOLD_JITTER_PX} px）")
            if shake is not None and shake > BOARD_SHAKE_DPS:
                reasons.append(f"IMU 角速度 {shake:.1f}°/s（卡尺或相机被碰到）")
            event = {**(self.pending or {}), "t_hold_start": t0, "t_hold_end": t1,
                     "live": stats, "board_rate_max_dps": shake, "ok": not reasons,
                     "reject_reason": "；".join(reasons)}
            if reasons:
                self.rejected.append(event)
                self.state = "move"
                beep(400, 400)
                msgs.append("rejected:" + event["reject_reason"])
            else:
                self.pauses.append(event)
                self.k += 1
                self.state = "move" if self.k < len(self.targets) else "done"
                beep(1500, 100)
                msgs.append("accepted")
                if self.travel_dir is None:
                    msgs += self._check_mounting(pose)
            self.pending = None
        return msgs

    def _check_mounting(self, pose: LivePose) -> list[str]:
        """Once the jaw has travelled MOUNT_CHECK_MM: 'mounting:<problem>' when
        the board moves along the direction of another axis, 'mountwarn:<text>'
        when the mounting differs from an earlier trial of this axis."""
        first, last = self.pauses[0], self.pauses[-1]
        p0, p1 = first["live"].get("median_mm"), last["live"].get("median_mm")
        if abs(last["reading_mm"] - first["reading_mm"]) < MOUNT_CHECK_MM or p0 is None or p1 is None:
            return []
        _t, _p, R = pose.between(first["t_hold_start"], first["t_hold_end"])
        u = travel_direction(R[0], p0, p1) if len(R) else None
        if u is None:
            return []
        self.travel_dir = np.round(u, 4).tolist()
        problem, warning = mounting_check(self.entry["axis"], u, self.known)
        if problem:
            return ["mounting:" + problem]
        return ["mountwarn:" + warning] if warning else []

    def summary(self) -> dict[str, Any]:
        readings = [p["reading_mm"] for p in self.pauses]
        return {"kind": "vernier_caliper_trial", **self.meta()["caliper_protocol"],
                "t_start": self.t0, "readings_mm": readings,
                "positions_mm": [round(r - self.zero_mm, 3) for r in readings],
                "travel_dir_camera": self.travel_dir,
                "pauses": self.pauses, "rejected_holds": self.rejected}


# --------------------------------------------------------------------------
# post-recording checks (worker thread)
# --------------------------------------------------------------------------
def _tools():
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))


def build_labels(name: str, require_board_still: bool) -> dict[str, Any]:
    """Chessboard poses, jump cleaning and the raw label file for one session."""
    _tools()
    from tools.build_pose_gt import process_session
    from tools.clean_pose_gt import clean
    camera_config.use_for_labels(camera_config.session_calibration(DATA_ROOT / name))
    pose = process_session(name)
    clean(DATA_ROOT / "pose_gt" / f"{name}.npz")
    gt = np.load(DATA_ROOT / "pose_gt" / f"{name}.npz")
    usable = (gt["ok"] == 1) & (gt["reproj"] < 1.5)
    if require_board_still:
        usable &= gt["ble_gyro"] < 8.0
    else:
        usable &= (gt["ble_gyro"] < 8.0) | (gt["ble_gyro"] >= 999.0)
    raw = DATA_ROOT / "pose_gt_raw"
    raw.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(raw / f"{name}.npz", t=gt["t"], R=gt["R"].astype(np.float32),
                        p=gt["p"].astype(np.float32), usable=usable.astype(np.uint8),
                        px=gt["reproj"].astype(np.float32))
    reproj = gt["reproj"][gt["ok"] == 1]
    return {"frames": int(len(usable)), "usable": int(usable.sum()),
            "usable_frac": float(usable.mean()) if len(usable) else 0.0,
            "reproj_p50": float(np.median(reproj)) if len(reproj) else None,
            "size": pose.get("size"), "delay_usb_ms": pose.get("delay_usb_ms")}


def qc_stop(name: str, events: dict[str, Any], drops: int,
            reference_speed: Optional[float] = None) -> dict[str, Any]:
    labels = build_labels(name, require_board_still=True)
    problems: list[str] = []
    warnings: list[str] = []
    if labels["usable_frac"] < 0.70:
        problems.append(f"棋盘可用帧 {labels['usable_frac']:.0%} < 70%")
    if labels["reproj_p50"] is not None and labels["reproj_p50"] > 0.4:
        problems.append(f"重投影中位数 {labels['reproj_p50']:.2f} px > 0.4")
    if drops:
        problems.append(f"丢帧 {drops}")
    rest = events.get("initial_rest") or {}
    if (rest.get("still_fraction") or 0.0) < 0.5:
        problems.append("开头静止段没有被 IMU 判为静止：相机在静止段被晃动了。")
    acc = rest.get("acc_ref_g") or rest.get("acc_norm_g")
    if acc is not None and abs(acc - 1.0) > ACC_FAULT_G:
        problems.append(acc_bias_message(acc))
    motion = None
    if events.get("t_motion") is not None and events.get("t_rest_end") is not None:
        motion = camera_motion(name, events["t_motion"], events["t_rest_end"])
        if (problem := motion_problem(events, motion)) is not None:
            problems.append(problem)
        elif (problem := delay_problem(name)) is not None:
            problems.append(problem)
        elif (note := rotation_note(motion)) is not None:
            warnings.append(note)
    if events.get("condition") == "P" and (events.get("compliance") or 0.0) < 0.8:
        problems.append(f"只有 {events.get('compliance') or 0:.0%} 的节拍被判为硬停（要求 ≥80%）")
    if (problem := continuous_stop_problem(events)) is not None:
        problems.append(problem)
    if (problem := speed_problem(events, reference_speed)) is not None:
        problems.append(problem)
    return {"session": name, **labels, "stop_events": {k: events.get(k) for k in (
        "label", "condition", "stops", "stops_per_min", "beats", "beats_matched", "compliance",
        "median_moving_speed_mm_s")}, "camera_motion": motion, "problems": problems,
            "warnings": warnings, "ok": not problems}


def qc_caliper(name: str, events: dict[str, Any], known: tuple = ()) -> dict[str, Any]:
    """Protocol checks only. The agreement with the caliper is the result of the
    experiment and is reported, never used to accept or reject a trial.
    known: (axis, trial, direction) of the other accepted trials."""
    labels = build_labels(name, require_board_still=False)
    raw = np.load(DATA_ROOT / "pose_gt_raw" / f"{name}.npz")
    t, p, use = raw["t"], raw["p"].astype(np.float64) * 1000.0, raw["usable"] == 1
    gt = np.load(DATA_ROOT / "pose_gt" / f"{name}.npz")
    board_rate = gt["ble_gyro"]
    pauses, problems = [], []
    for ev in events["pauses"]:
        sel = use & (t >= ev["t_hold_start"]) & (t <= ev["t_hold_end"])
        if sel.sum() < 30:
            problems.append(f"{ev['target_mm']:g} mm 处保持期间只有 {int(sel.sum())} 帧可用（要求 ≥30）")
            continue
        rate = board_rate[(gt["t"] >= ev["t_hold_start"]) & (gt["t"] <= ev["t_hold_end"])]
        rate = rate[rate < 999.0]
        if len(rate) and float(rate.max()) > BOARD_SHAKE_DPS:
            problems.append(f"{ev['target_mm']:g} mm 处保持期间棋盘 IMU 达 {float(rate.max()):.1f}°/s（被碰过）")
        q = p[sel]
        med = np.median(q, axis=0)
        pauses.append({"position_mm": ev["reading_mm"] - events["zero_reading_mm"],
                       "frames": int(sel.sum()), "median_mm": med,
                       "p95_mm": float(np.quantile(np.linalg.norm(q - med, axis=1), 0.95))})
    if len(events["pauses"]) < len(events["targets_mm"]):
        problems.append(f"只完成 {len(events['pauses'])}/{len(events['targets_mm'])} 个位置")
    if labels["reproj_p50"] is not None and labels["reproj_p50"] > 0.4:
        problems.append(f"重投影中位数 {labels['reproj_p50']:.2f} px > 0.4（画面糊或对焦不对）")
    direction, warnings = trial_direction(name), []
    if direction is not None:
        problem, warning = mounting_check(events.get("axis", ""), direction, known)
        if problem:
            problems.append(problem)
        if warning:
            warnings.append(warning)
    fit = None
    if len(pauses) >= 3:
        s = np.array([q["position_mm"] for q in pauses])
        m = np.array([q["median_mm"] for q in pauses])
        sc = s - s.mean()
        g = (sc[:, None] * (m - m.mean(0))).sum(0) / float(sc @ sc)
        resid = m - (m.mean(0) + sc[:, None] * g)
        fit = {"scale": round(float(np.linalg.norm(g)), 4),
               "rms_mm": round(float(np.sqrt((resid ** 2).sum(1).mean())), 3),
               "worst_hold_p95_mm": round(max(q["p95_mm"] for q in pauses), 3),
               "distance_mm": round(float(np.median(np.linalg.norm(m, axis=1))), 1)}
    return {"session": name, **labels, "trial_fit": fit, "travel_dir_camera": direction,
            "problems": problems, "warnings": warnings, "ok": not problems,
            "pauses": [{**q, "median_mm": np.round(q["median_mm"], 3).tolist()} for q in pauses]}


def rigidity_motion_hint(name: str) -> Optional[str]:
    """Why a rigidity recording cannot work: the check compares the rotation of
    the camera seen by PnP with the gyroscope of the USB IMU on the camera."""
    from .sync import load_imu
    session = DATA_ROOT / name
    usb_t, usb = load_imu(session / "imu_stream.csv")
    if len(usb_t) < 100:
        return "USB IMU 没有数据。"
    usb_p95 = float(np.quantile(np.linalg.norm(usb[:, 3:6], axis=1), 0.95))
    if usb_p95 >= 3.0:
        problem = delay_problem(name)
        return None if problem is None else problem + "刚性检查用这个延迟对齐棋盘转动和陀螺仪。"
    bt_t, bt = load_imu(session / "imu_bt.csv") if (session / "imu_bt.csv").is_file() else (np.zeros(0), None)
    bt_moving = float(np.mean(np.linalg.norm(bt[:, 3:6], axis=1) > 8.0)) if len(bt_t) else 0.0
    if bt_moving > 0.1:
        return ("相机上的 USB IMU 几乎没转，蓝牙 IMU 却在动：转动的是棋盘/卡尺，不是相机。"
                "第 9 步要手持相机转动，棋盘和卡尺放在桌上不动。")
    return ("相机上的 USB IMU 一直几乎不转。确认相机确实被转动，USB IMU 牢牢固定在相机上，"
            "并且主窗口选的 COM 口就是这只 IMU。")


def todays_rigidity(day: Optional[str] = None) -> Optional[dict[str, Any]]:
    """Latest passing rigidity check of a day (YYYYMMDD) with its R_camera_imu."""
    day = day or datetime.now().strftime("%Y%m%d")
    for folder in sorted(DATA_ROOT.glob(f"rigid_{day}_*"), reverse=True):
        qc = folder / "qc_result.json"
        if not qc.is_file():
            continue
        try:
            verdict = json.loads(qc.read_text(encoding="utf-8")).get("rigidity")
        except json.JSONDecodeError:
            continue
        if verdict != "rigid":
            continue
        full = folder / "rigidity.json"
        if full.is_file():
            data = json.loads(full.read_text(encoding="utf-8"))
        else:
            _tools()
            from tools.check_rigidity import check
            data = check(folder.name)
            full.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if data.get("R_camera_imu"):
            return {"session": folder.name, **data}
    return None


def rotation_angle_deg(Ra, Rb) -> float:
    rel = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    return float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))
