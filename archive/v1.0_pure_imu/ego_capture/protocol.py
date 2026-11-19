#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WitMotion protocol: classic 0x55/11-byte frames and BLE 0x55 0x61 packed frames."""

from __future__ import annotations

import math
import struct
from typing import Any

# Host → module
CMD_UNLOCK = bytes((0xFF, 0xAA, 0x69, 0x88, 0xB5))
CMD_SAVE = bytes((0xFF, 0xAA, 0x00, 0x00, 0x00))
CMD_RATE_200_CLASSIC = bytes((0xFF, 0xAA, 0x03, 0x0B, 0x00))  # BWT901CL / WTSDCL 实测
CMD_RATE_200_BLE_DOC = bytes((0xFF, 0xAA, 0x03, 0x0A, 0x00))  # BLE 文档里的 200Hz
CMD_BANDWIDTH_256 = bytes((0xFF, 0xAA, 0x1F, 0x00, 0x00))
CMD_READ_QUAT = bytes((0xFF, 0xAA, 0x27, 0x51, 0x00))
CMD_READ_MAG = bytes((0xFF, 0xAA, 0x27, 0x3A, 0x00))
CMD_READ_TEMP = bytes((0xFF, 0xAA, 0x27, 0x40, 0x00))

# ACC|GYRO|ANGLE|MAG|QUAT  (no TIME/PRESS/GPS) — 115200 波特下还能撑住 200Hz
RSW_VIO_USB = 0x021E  # bits 1,2,3,4,9
CMD_RSW_VIO_USB = bytes((0xFF, 0xAA, 0x02, RSW_VIO_USB & 0xFF, (RSW_VIO_USB >> 8) & 0xFF))

CLASSIC_TYPES = {
    0x50: "time",
    0x51: "acc",
    0x52: "gyro",
    0x53: "angle",
    0x54: "mag",
    0x56: "pressure",
    0x59: "quat",
}


def i16_le(lo: int, hi: int) -> int:
    v = ((hi & 0xFF) << 8) | (lo & 0xFF)
    return v - 65536 if v >= 32768 else v


def scale_i16(lo: int, hi: int, scale: float) -> float:
    return i16_le(lo, hi) / 32768.0 * scale


def rpy_deg_to_quat(roll: float, pitch: float, yaw: float) -> list[float]:
    """ZYX intrinsic (WitMotion 东北天)."""
    r = math.radians(roll) * 0.5
    p = math.radians(pitch) * 0.5
    y = math.radians(yaw) * 0.5
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def empty_state() -> dict[str, Any]:
    return {
        "acc": [0.0, 0.0, 0.0],
        "gyro": [0.0, 0.0, 0.0],
        "angle": [0.0, 0.0, 0.0],
        "mag": [0.0, 0.0, 0.0],
        "pressure": 0.0,
        "altitude": 0.0,
        "temp": 0.0,
        "quaternion": [1.0, 0.0, 0.0, 0.0],
        "aux": 0.0,
        "source": "",
    }


def snapshot_of(state: dict[str, Any], timestamp: float, source: str) -> dict[str, Any]:
    return {
        "timestamp": float(timestamp),
        "acc": list(state["acc"]),
        "gyro": list(state["gyro"]),
        "angle": list(state["angle"]),
        "mag": list(state["mag"]),
        "pressure": float(state["pressure"]),
        "altitude": float(state["altitude"]),
        "temp": float(state["temp"]),
        "quaternion": list(state["quaternion"]),
        "aux": float(state["aux"]),
        "source": source,
    }


class WitParser:
    """Byte-buffer parser. Emits one fused sample per ACC (classic) or per 0x61 (BLE)."""

    def __init__(self, source: str = "") -> None:
        self.source = source
        self.buf = bytearray()
        self.state = empty_state()
        self.classic_ok = 0
        self.classic_bad = 0
        self.ble61 = 0
        self.ble71 = 0
        self.last_stamp = 0.0

    def feed(self, data: bytes, t_end: float, dt_hint: float = 0.005) -> list[dict[str, Any]]:
        if not data:
            return []
        self.buf.extend(data)
        if len(self.buf) > 65536:
            self.buf = self.buf[-4096:]
        samples: list[dict[str, Any]] = []
        i = 0
        buf = self.buf
        n = len(buf)
        while i < n:
            if buf[i] != 0x55:
                i += 1
                continue
            if i + 1 >= n:
                break
            flag = buf[i + 1]
            if flag == 0x61:
                flen = _ble61_frame_len(buf, i)
                if flen == 0:
                    break
                self._decode_ble61(buf[i : i + flen])
                self.ble61 += 1
                samples.append(snapshot_of(self.state, 0.0, self.source))
                i += flen
                continue
            if flag == 0x71:
                if i + 20 > n:
                    break
                self._decode_ble71(buf[i : i + 20])
                self.ble71 += 1
                i += 20
                continue
            if flag in CLASSIC_TYPES:
                if i + 11 > n:
                    break
                frame = buf[i : i + 11]
                if (sum(frame[:10]) & 0xFF) == frame[10]:
                    kind = CLASSIC_TYPES[flag]
                    self._decode_classic(kind, frame[2:10])
                    self.classic_ok += 1
                    if kind == "acc":
                        samples.append(snapshot_of(self.state, 0.0, self.source))
                    i += 11
                else:
                    self.classic_bad += 1
                    i += 1
                continue
            i += 1
        if i:
            del self.buf[:i]
        if samples:
            n_s = len(samples)
            dt = dt_hint if dt_hint > 1e-6 else 0.005
            # 一包通知里的帧按 dt 回拨。下一包若来得比这包跨度更密，回拨会早于上一包，
            # 时间戳倒退，后面的就近配对会整段失败。保证单调。
            start = t_end - (n_s - 1) * dt
            if start <= self.last_stamp:
                gap = max(t_end - self.last_stamp, 1e-4)
                dt = gap / n_s
                start = self.last_stamp + dt
            for k, sample in enumerate(samples):
                sample["timestamp"] = start + k * dt
            self.last_stamp = samples[-1]["timestamp"]
        return samples

    def _decode_classic(self, kind: str, payload: bytes) -> None:
        if kind == "acc":
            self.state["acc"] = [
                scale_i16(payload[0], payload[1], 16.0),
                scale_i16(payload[2], payload[3], 16.0),
                scale_i16(payload[4], payload[5], 16.0),
            ]
            self.state["temp"] = i16_le(payload[6], payload[7]) / 100.0
        elif kind == "gyro":
            self.state["gyro"] = [
                scale_i16(payload[0], payload[1], 2000.0),
                scale_i16(payload[2], payload[3], 2000.0),
                scale_i16(payload[4], payload[5], 2000.0),
            ]
        elif kind == "angle":
            self.state["angle"] = [
                scale_i16(payload[0], payload[1], 180.0),
                scale_i16(payload[2], payload[3], 180.0),
                scale_i16(payload[4], payload[5], 180.0),
            ]
        elif kind == "mag":
            self.state["mag"] = [
                float(i16_le(payload[0], payload[1])),
                float(i16_le(payload[2], payload[3])),
                float(i16_le(payload[4], payload[5])),
            ]
        elif kind == "pressure":
            pr = struct.unpack_from("<i", payload, 0)[0]
            alt = struct.unpack_from("<i", payload, 4)[0]
            self.state["pressure"] = pr / 100.0
            self.state["altitude"] = alt / 100.0
        elif kind == "quat":
            self.state["quaternion"] = [
                i16_le(payload[0], payload[1]) / 32768.0,
                i16_le(payload[2], payload[3]) / 32768.0,
                i16_le(payload[4], payload[5]) / 32768.0,
                i16_le(payload[6], payload[7]) / 32768.0,
            ]

    def _decode_ble61(self, frame: bytes) -> None:
        p = frame
        self.state["acc"] = [
            scale_i16(p[2], p[3], 16.0),
            scale_i16(p[4], p[5], 16.0),
            scale_i16(p[6], p[7], 16.0),
        ]
        self.state["gyro"] = [
            scale_i16(p[8], p[9], 2000.0),
            scale_i16(p[10], p[11], 2000.0),
            scale_i16(p[12], p[13], 2000.0),
        ]
        self.state["angle"] = [
            scale_i16(p[14], p[15], 180.0),
            scale_i16(p[16], p[17], 180.0),
            scale_i16(p[18], p[19], 180.0),
        ]
        # 28 字节 0x61 的后 8 字节不是磁场。磁场走 0x54 连续帧，或 0x3A 应答。
        ang = self.state["angle"]
        self.state["quaternion"] = rpy_deg_to_quat(ang[0], ang[1], ang[2])

    def _decode_ble71(self, frame: bytes) -> None:
        reg = frame[2] | (frame[3] << 8)
        if reg == 0x51:
            self.state["quaternion"] = [
                i16_le(frame[4], frame[5]) / 32768.0,
                i16_le(frame[6], frame[7]) / 32768.0,
                i16_le(frame[8], frame[9]) / 32768.0,
                i16_le(frame[10], frame[11]) / 32768.0,
            ]
        elif reg == 0x3A:
            self.state["mag"] = [
                float(i16_le(frame[4], frame[5])),
                float(i16_le(frame[6], frame[7])),
                float(i16_le(frame[8], frame[9])),
            ]
        elif reg == 0x40:
            self.state["temp"] = i16_le(frame[4], frame[5]) / 100.0


def _ble61_frame_len(buf: bytearray, i: int) -> int:
    rem = len(buf) - i
    if rem < 20:
        return 0
    if rem >= 28:
        nxt = i + 28
        if rem == 28 or (nxt < len(buf) and buf[nxt] == 0x55) or (rem % 28 == 0):
            return 28
        if rem % 20 == 0 and rem % 28 != 0:
            return 20
        return 28
    return 20


def build_classic_frame(kind: str, payload8: bytes) -> bytes:
    type_map = {v: k for k, v in CLASSIC_TYPES.items()}
    header = bytes((0x55, type_map[kind])) + payload8
    return header + bytes((sum(header) & 0xFF,))
