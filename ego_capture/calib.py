#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Solve USB-IMU / BT-IMU / stereo-camera frames from captured sessions."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from . import rig
from .cameras import chessboard_object_points, find_board
from .geometry import (
    handeye_rotation,
    kabsch_R,
    quat_to_R,
    rot_err_deg,
    rpy_deg_to_R,
    se3,
    se3_inv,
    se3_to_dict,
)
from .session import INNER, SQUARE_MM, write_json

GYRO_STATIC = 3.0  # deg/s
ACC_STATIC_DEV = 0.08  # g


def load_imu_csv(path: str) -> np.ndarray:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            try:
                rows.append(
                    [
                        float(rec["timestamp"]),
                        float(rec["acc_x"]), float(rec["acc_y"]), float(rec["acc_z"]),
                        float(rec["gyro_x"]), float(rec["gyro_y"]), float(rec["gyro_z"]),
                        float(rec["angle_x"]), float(rec["angle_y"]), float(rec["angle_z"]),
                        float(rec["mag_x"]), float(rec["mag_y"]), float(rec["mag_z"]),
                        float(rec["quat_w"]), float(rec["quat_x"]), float(rec["quat_y"]), float(rec["quat_z"]),
                    ]
                )
            except (KeyError, ValueError):
                continue
    return np.asarray(rows, dtype=np.float64)


def _R_of_row(row: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = row[13:17]
    if abs(qw) + abs(qx) + abs(qy) + abs(qz) > 0.5:
        return quat_to_R(qw, qx, qy, qz)
    return rpy_deg_to_R(row[7], row[8], row[9])


def _nearest_indices(src_t: np.ndarray, dst_t: np.ndarray, max_dt: float = 0.012) -> list[tuple[int, int]]:
    """Match each src time to the nearest dst sample. dst may be non-monotonic."""
    if len(src_t) == 0 or len(dst_t) == 0:
        return []
    order = np.argsort(dst_t, kind="mergesort")
    dst_s = dst_t[order]
    pos = np.searchsorted(dst_s, src_t)
    n = len(dst_s)
    pairs: list[tuple[int, int]] = []
    for i, t in enumerate(src_t):
        j = int(pos[i])
        cands = [k for k in (j - 1, j) if 0 <= k < n]
        if not cands:
            continue
        k = min(cands, key=lambda x: abs(float(dst_s[x]) - float(t)))
        if abs(float(dst_s[k]) - float(t)) <= max_dt:
            pairs.append((i, int(order[k])))
    return pairs


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def align_imus_static(usb: np.ndarray, bt: np.ndarray) -> dict[str, Any]:
    if usb.size == 0 or bt.size == 0:
        raise RuntimeError("IMU CSV 为空。")
    u_st = usb[np.linalg.norm(usb[:, 4:7], axis=1) < GYRO_STATIC]
    b_st = bt[np.linalg.norm(bt[:, 4:7], axis=1) < GYRO_STATIC]
    if len(u_st) < 200 or len(b_st) < 200:
        raise RuntimeError("静止样本不够。两枚 IMU 贴紧后整套不要动，再录 ≥25 秒。")
    acc_u = _unit(np.median(u_st[:, 1:4], axis=0))
    acc_b = _unit(np.median(b_st[:, 1:4], axis=0))
    mag_u_raw = np.median(u_st[:, 10:13], axis=0)
    mag_b_raw = np.median(b_st[:, 10:13], axis=0)
    mag_std = np.std(b_st[:, 10:13], axis=0)
    mag_alive = float(mag_std[0]) >= 1.0 and float(mag_std[1]) >= 1.0 and float(np.linalg.norm(mag_b_raw)) >= 1.0
    # 两路重力与磁力的夹角可以差十几度（软铁、安装、室内磁场）。
    # 用 Kabsch 同时硬配两个向量，会把夹角差平分进重力残差，所以永远下不到 5°。
    # TRIAD：重力精确对齐（滚转/俯仰），磁力只定绕重力的航向。
    if mag_alive and _horiz_norm(mag_b_raw, acc_b) > 0.15 and _horiz_norm(mag_u_raw, acc_u) > 0.15:
        R = _triad(acc_b, mag_b_raw, acc_u, mag_u_raw)
        method = "triad"
        mag_err = float(np.degrees(np.arccos(np.clip(np.dot(_unit(R @ mag_b_raw), _unit(mag_u_raw)), -1, 1))))
    else:
        R = _vec_align(acc_b, acc_u)
        method = "gravity_only"
        mag_err = -1.0
    grav_err = float(np.degrees(np.arccos(np.clip(np.dot(R @ acc_b, acc_u), -1, 1))))
    incl_u = float(np.degrees(np.arccos(np.clip(np.dot(acc_u, _unit(mag_u_raw)), -1, 1)))) if np.linalg.norm(mag_u_raw) > 1 else -1.0
    incl_b = float(np.degrees(np.arccos(np.clip(np.dot(acc_b, _unit(mag_b_raw)), -1, 1)))) if np.linalg.norm(mag_b_raw) > 1 else -1.0
    T = se3(R, [0.0, 0.0, 0.0])
    return {
        "kind": "imu_static",
        "frames": "I_usb ← I_bt",
        "note": "静止只解滚转/俯仰（重力）和航向（水平磁力）。两路磁倾角不一致时不再惩罚重力。平移不可观，完整旋转以动态手眼为准。",
        "method": method,
        "n_usb_static": int(len(u_st)),
        "n_bt_static": int(len(b_st)),
        "gravity_residual_deg": grav_err,
        "mag_residual_deg": mag_err,
        "mag_inclination_usb_deg": incl_u,
        "mag_inclination_bt_deg": incl_b,
        "still_usb_deg": _acc_jitter_deg(u_st[:, 1:4]),
        "still_bt_deg": _acc_jitter_deg(b_st[:, 1:4]),
        "T_Iusb_Ibt": se3_to_dict(T),
        "acc_usb": acc_u.tolist(),
        "acc_bt": acc_b.tolist(),
    }


def _horiz_norm(v: np.ndarray, g: np.ndarray) -> float:
    g = _unit(g)
    h = np.asarray(v, dtype=np.float64) - float(np.dot(v, g)) * g
    return float(np.linalg.norm(h)) / max(float(np.linalg.norm(v)), 1e-9)


def _acc_jitter_deg(acc: np.ndarray) -> float:
    med = _unit(np.median(acc, axis=0))
    norms = np.linalg.norm(acc, axis=1)
    ok = norms > 0.2
    if not np.any(ok):
        return 180.0
    dots = (acc[ok] @ med) / norms[ok]
    return float(np.degrees(np.median(np.arccos(np.clip(dots, -1.0, 1.0)))))


def _triad(g_src: np.ndarray, m_src: np.ndarray, g_dst: np.ndarray, m_dst: np.ndarray) -> np.ndarray:
    """R maps src→dst. First axis is gravity, so R @ g_src = g_dst."""

    def basis(g: np.ndarray, m: np.ndarray) -> np.ndarray:
        t1 = _unit(g)
        t2 = _unit(np.cross(t1, m))
        t3 = np.cross(t1, t2)
        return np.column_stack([t1, t2, t3])

    return basis(g_dst, m_dst) @ basis(g_src, m_src).T


def _vec_align(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _unit(a)
    b = _unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-8:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]], dtype=np.float64)
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def align_imus_dynamic(usb: np.ndarray, bt: np.ndarray, R0: Optional[np.ndarray] = None) -> dict[str, Any]:
    pairs = _nearest_indices(usb[:, 0], bt[:, 0], max_dt=0.012)
    if len(pairs) < 400:
        raise RuntimeError(
            f"两路 IMU 对上的时刻只有 {len(pairs)} 个。180Hz 对 200Hz 可以用；"
            "请确认两条都在录，并且录够 40 秒。"
        )
    As, Bs = [], []
    step = max(int(0.25 * 200), 20)
    for k in range(0, len(pairs) - step, max(step // 4, 5)):
        i0, j0 = pairs[k]
        i1, j1 = pairs[k + step]
        Ru0, Ru1 = _R_of_row(usb[i0]), _R_of_row(usb[i1])
        Rb0, Rb1 = _R_of_row(bt[j0]), _R_of_row(bt[j1])
        A = Ru1 @ Ru0.T
        B = Rb1 @ Rb0.T
        if np.linalg.norm(usb[i1, 4:7]) < 8 and np.linalg.norm(usb[i0, 4:7]) < 8:
            continue
        As.append(A)
        Bs.append(B)
    X, mean_err = handeye_rotation(As, Bs)
    if R0 is not None and float(np.trace(X.T @ R0)) < 0:
        from .geometry import orthonormalize
        X = orthonormalize(-X)
    T = se3(X, [0.0, 0.0, 0.0])
    return {
        "kind": "imu_dynamic",
        "frames": "I_usb ← I_bt",
        "note": "两枚 IMU 仍贴在一起，做平移+三轴旋转。A≈X B X^T。",
        "n_pairs": int(len(pairs)),
        "n_rel": int(len(As)),
        "handeye_mean_resid_deg": mean_err,
        "T_Iusb_Ibt": se3_to_dict(T),
    }


def stereo_T_C0_C1() -> np.ndarray:
    """C1 optical center in C0: +X by baseline. Parallel axes."""
    return se3(np.eye(3), [rig.STEREO_BASELINE_M, 0.0, 0.0])


def load_camera_k(session_or_root: str) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    paths = []
    p = Path(session_or_root)
    if p.is_dir():
        paths.extend(sorted(p.glob("camera_calibration_cam0.json")))
        paths.extend(sorted(p.parent.glob("calib_intrinsics_*/camera_calibration_cam0.json")))
    if not paths:
        return None, None
    data = json.loads(paths[-1].read_text(encoding="utf-8"))
    K = np.asarray(data.get("camera_matrix"), dtype=np.float64)
    dist = np.asarray(data.get("distortion_coefficients"), dtype=np.float64).reshape(-1, 1)
    return K, dist


def _pnp_from_images(session: str, max_frames: int = 180) -> tuple[list[np.ndarray], list[float], dict[str, Any]]:
    sess = Path(session)
    frames_csv = sess / "frames.csv"
    img_dir = sess / "cam0" / "images"
    times = []
    flags = []
    if frames_csv.exists():
        with frames_csv.open(newline="", encoding="utf-8") as handle:
            for rec in csv.DictReader(handle):
                times.append(float(rec["frame_timestamp"]))
                flags.append(int(float(rec.get("board_left", 0) or 0)))
    K, dist = load_camera_k(session)
    objp = chessboard_object_points()
    used = [i for i, f in enumerate(flags) if f] if flags else list(range(len(times)))
    if len(used) > max_frames:
        sel = np.linspace(0, len(used) - 1, max_frames).astype(int)
        used = [used[i] for i in sel]
    Ts = []
    ts = []
    corners_for_k: list[np.ndarray] = []
    image_size = None
    for i in used:
        jpg = img_dir / f"{i:06d}.jpg"
        if not jpg.exists():
            continue
        bgr = cv2.imread(str(jpg), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        image_size = (bgr.shape[1], bgr.shape[0])
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        ok, corners, pat = find_board(gray)
        if not ok or corners is None:
            continue
        corners_for_k.append(corners.reshape(-1, 2))
        ts.append(times[i] if i < len(times) else 0.0)
        # stash corners; PnP after K
        Ts.append(corners.reshape(-1, 1, 2).astype(np.float32))
    info = {"tried": len(used), "detected": len(Ts), "image_size": list(image_size or [])}
    if K is None:
        if len(corners_for_k) < 12 or image_size is None:
            raise RuntimeError("没有内参，且棋盘样本不足以现场标定。请先做内参慢扫。")
        obj_pts = [objp] * len(corners_for_k)
        img_pts = [c.reshape(-1, 1, 2).astype(np.float32) for c in corners_for_k]
        ok, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, image_size, None, None)
        if not ok:
            raise RuntimeError("现场内参失败。")
        info["K_from_session"] = True
    else:
        info["K_from_session"] = False
    poses = []
    stamps = []
    for corners, t in zip(Ts, ts):
        ok, rvec, tvec = cv2.solvePnP(objp, corners, K, dist, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            continue
        R, _ = cv2.Rodrigues(rvec)
        poses.append(se3(R, (tvec.reshape(3) / 1000.0)))  # tvec in mm → m
        stamps.append(t)
    info["pnp"] = len(poses)
    info["K"] = np.asarray(K, dtype=np.float64).tolist()
    info["dist"] = np.asarray(dist, dtype=np.float64).reshape(-1).tolist()
    return poses, stamps, info


def _imu_R_at(imu: np.ndarray, t: float) -> Optional[np.ndarray]:
    if imu.size == 0:
        return None
    i = int(np.argmin(np.abs(imu[:, 0] - t)))
    if abs(imu[i, 0] - t) > 0.02:
        return None
    return _R_of_row(imu[i])


def handeye_board_imu(poses_T_C_O: list[np.ndarray], stamps: list[float], imu: np.ndarray) -> dict[str, Any]:
    As, Bs = [], []
    for i in range(len(poses_T_C_O) - 1):
        Ri = _imu_R_at(imu, stamps[i])
        Rj = _imu_R_at(imu, stamps[i + 1])
        if Ri is None or Rj is None:
            continue
        A = se3_inv(poses_T_C_O[i]) @ poses_T_C_O[i + 1]  # relative board motion in cam? 
        # Camera pose in board: T_O_C = T_C_O^{-1}
        # Relative camera motion: T_O_C(j)^{-1} T_O_C(i) wait.
        # Standard eye-in-hand if IMU is on camera. Here IMU is on the BOARD.
        # Board-relative camera motion A = T_C_O(i)^{-1} T_C_O(j)
        A = se3_inv(poses_T_C_O[i]) @ poses_T_C_O[i + 1]
        B = Ri.T @ Rj
        As.append(A[:3, :3])
        Bs.append(B)
    if len(As) < 12:
        raise RuntimeError(f"棋盘+IMU 配对只有 {len(As)} 组。板要一直在画面里，并多轴转动。")
    X, mean_err = handeye_rotation(As, Bs)
    # translation of C←Ibt: if IMU ≈ board origin, t ≈ mean t_C_O
    ts = np.stack([T[:3, 3] for T in poses_T_C_O])
    t_mean = np.mean(ts, axis=0)
    offset = np.asarray(rig.BOARD_BT_IMU_OFFSET_MM, dtype=np.float64) / 1000.0
    # t_C_I ≈ t_C_O + R_C_O * t_O_I  (average)
    t_est = []
    for T in poses_T_C_O:
        t_est.append(T[:3, 3] + T[:3, :3] @ offset)
    t_C_I = np.mean(np.stack(t_est), axis=0) if t_est else t_mean
    T_C_I = se3(X, t_C_I)
    return {
        "kind": "board_bt_handeye",
        "frames": "C0 ← I_bt  (蓝牙 IMU 贴在棋盘上)",
        "n_rel": int(len(As)),
        "handeye_mean_resid_deg": mean_err,
        "T_C0_Ibt": se3_to_dict(T_C_I),
        "board_bt_imu_offset_mm": list(rig.BOARD_BT_IMU_OFFSET_MM),
        "mean_t_C0_O_mm": (t_mean * 1000.0).tolist(),
    }


def compose_rig(
    T_Iusb_Ibt: np.ndarray,
    T_C0_Ibt: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
) -> dict[str, Any]:
    T_C0_C1 = stereo_T_C0_C1()
    T_C1_C0 = se3_inv(T_C0_C1)
    T_Ibt_Iusb = se3_inv(T_Iusb_Ibt)
    payload: dict[str, Any] = {
        "kind": "rig_compose",
        "baseline_mm": rig.STEREO_BASELINE_MM,
        "lens_diameter_mm": rig.LENS_DIAMETER_MM,
        "housing_outer_mm": rig.HOUSING_OUTER_MM,
        "frames": rig.FRAMES,
        "T_Iusb_Ibt": se3_to_dict(T_Iusb_Ibt),
        "T_Ibt_Iusb": se3_to_dict(T_Ibt_Iusb),
        "T_C0_C1": se3_to_dict(T_C0_C1),
        "T_C1_C0": se3_to_dict(T_C1_C0),
        "projection": [
            "p_C0 = R_C0_I * p_I + t_C0_I,   u = K p_C0 / z",
            "I 可以是 I_usb 或 I_bt",
        ],
    }
    if T_C0_Ibt is not None:
        T_C0_Iusb = T_C0_Ibt @ T_Ibt_Iusb
        payload["T_C0_Ibt"] = se3_to_dict(T_C0_Ibt)
        payload["T_C0_Iusb"] = se3_to_dict(T_C0_Iusb)
        payload["T_C1_Iusb"] = se3_to_dict(T_C1_C0 @ T_C0_Iusb)
        payload["T_C1_Ibt"] = se3_to_dict(T_C1_C0 @ T_C0_Ibt)
    if K is not None:
        payload["K_C0"] = np.asarray(K, dtype=np.float64).tolist()
    return payload


def _T_from_saved(blob: dict[str, Any], key: str) -> Optional[np.ndarray]:
    node = blob.get(key)
    if not node:
        return None
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64)
    R = np.asarray(node["R"], dtype=np.float64)
    t = np.asarray(node.get("t_m", [0, 0, 0]), dtype=np.float64)
    return se3(R, t)


def solve_step(session: str, step: str, rig_state_path: str) -> dict[str, Any]:
    sess = Path(session)
    state: dict[str, Any] = {}
    if os.path.exists(rig_state_path):
        try:
            state = json.loads(Path(rig_state_path).read_text(encoding="utf-8"))
        except Exception:
            state = {}
    usb_p = sess / "imu_stream.csv"
    bt_p = sess / "imu_bt.csv"
    usb = load_imu_csv(str(usb_p)) if usb_p.exists() else np.zeros((0, 17))
    bt = load_imu_csv(str(bt_p)) if bt_p.exists() else np.zeros((0, 17))

    result: dict[str, Any] = {"session": str(sess), "step": step}
    if step == "imu_static":
        result.update(align_imus_static(usb, bt))
        state["T_Iusb_Ibt_static"] = result["T_Iusb_Ibt"]
        state["T_Iusb_Ibt"] = result["T_Iusb_Ibt"]
    elif step == "imu_dyn":
        R0 = None
        prev = _T_from_saved(state, "T_Iusb_Ibt")
        if prev is not None:
            R0 = prev[:3, :3]
        result.update(align_imus_dynamic(usb, bt, R0=R0))
        state["T_Iusb_Ibt_dynamic"] = result["T_Iusb_Ibt"]
        state["T_Iusb_Ibt"] = result["T_Iusb_Ibt"]
    elif step == "board_bt":
        poses, stamps, info = _pnp_from_images(str(sess))
        result["pnp_info"] = info
        if len(poses) < 15:
            raise RuntimeError(f"有效 PnP 只有 {len(poses)} 帧。蓝牙 IMU 贴在棋盘上，整板要经常完整入画。")
        he = handeye_board_imu(poses, stamps, bt)
        result.update(he)
        state["T_C0_Ibt"] = he["T_C0_Ibt"]
        state["K_C0"] = info.get("K")
        T_Iusb_Ibt = _T_from_saved(state, "T_Iusb_Ibt")
        T_C0_Ibt = _T_from_saved(he, "T_C0_Ibt")
        if T_Iusb_Ibt is None:
            T_Iusb_Ibt = np.eye(4)
        K = np.asarray(info["K"], dtype=np.float64) if info.get("K") else None
        composed = compose_rig(T_Iusb_Ibt, T_C0_Ibt, K)
        result["composed"] = composed
        state.update({k: composed[k] for k in composed if k.startswith("T_") or k in {"K_C0", "baseline_mm"}})
        state["composed"] = composed
    elif step == "compose":
        T_Iusb_Ibt = _T_from_saved(state, "T_Iusb_Ibt")
        T_C0_Ibt = _T_from_saved(state, "T_C0_Ibt")
        if T_Iusb_Ibt is None:
            raise RuntimeError("还没有双 IMU 对齐结果。先做静止对齐和动态对齐。")
        K = np.asarray(state["K_C0"], dtype=np.float64) if state.get("K_C0") else None
        composed = compose_rig(T_Iusb_Ibt, T_C0_Ibt, K)
        result.update(composed)
        state["composed"] = composed
        state.update({k: composed[k] for k in composed if k.startswith("T_") or k in {"K_C0", "baseline_mm"}})
    else:
        raise RuntimeError(f"步骤 {step} 没有自动解算。")

    write_json(str(sess / "calib_result.json"), result)
    write_json(rig_state_path, state)
    result["rig_state"] = rig_state_path
    return result


def summarize(result: dict[str, Any]) -> str:
    lines = [f"步骤 {result.get('step')}  {result.get('kind', '')}"]
    if "gravity_residual_deg" in result:
        mag = result["mag_residual_deg"]
        mag_txt = "航向未用磁力" if mag < 0 else f"磁力倾角残差 {mag:.2f}°（不要求 <5°）"
        still = ""
        if "still_usb_deg" in result:
            still = f"，静止抖动 USB {result['still_usb_deg']:.2f}° / BLE {result['still_bt_deg']:.2f}°"
        lines.append(f"重力残差 {result['gravity_residual_deg']:.2f}°，{mag_txt}{still}")
    if "handeye_mean_resid_deg" in result:
        lines.append(f"手眼相对旋转残差 {result['handeye_mean_resid_deg']:.2f}°  （{result.get('n_rel', '?')} 对）")
    for key in ("T_Iusb_Ibt", "T_C0_Ibt", "T_C0_Iusb"):
        node = result.get(key) or (result.get("composed") or {}).get(key)
        if node:
            rpy = node.get("rpy_deg", [])
            tmm = node.get("t_mm", [])
            lines.append(f"{key}  RPY({rpy[0]:+.2f},{rpy[1]:+.2f},{rpy[2]:+.2f})°  t({tmm[0]:+.1f},{tmm[1]:+.1f},{tmm[2]:+.1f})mm")
    if "baseline_mm" in result or "composed" in result:
        b = result.get("baseline_mm") or result.get("composed", {}).get("baseline_mm")
        lines.append(f"双目光心基线 {b} mm")
    return "\n".join(lines)
