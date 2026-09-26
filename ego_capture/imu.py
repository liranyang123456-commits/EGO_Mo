#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""USB serial + BLE WitMotion IMU workers with reconnect and 200 Hz config."""

from __future__ import annotations

import asyncio
import csv
import os
import queue
import threading
import time
from collections import deque
from typing import Any, Optional

from serial.tools import list_ports

from .affinity import PRIOR_HIGHEST, pin_current_thread
from .protocol import (
    CMD_BANDWIDTH_256,
    CMD_RATE_200_CLASSIC,
    CMD_READ_MAG,
    CMD_READ_QUAT,
    CMD_RSW_VIO_USB,
    CMD_SAVE,
    CMD_UNLOCK,
    WitParser,
)

IMU_STREAM_COLUMNS = [
    "timestamp", "acc_x", "acc_y", "acc_z",
    "gyro_x", "gyro_y", "gyro_z",
    "angle_x", "angle_y", "angle_z",
    "mag_x", "mag_y", "mag_z",
    "pressure", "altitude", "temp",
    "quat_w", "quat_x", "quat_y", "quat_z",
]

BLE_NOTIFY = "0000ffe4-0000-1000-8000-00805f9a34fb"
BLE_WRITE = "0000ffe9-0000-1000-8000-00805f9a34fb"
TARGET_HZ = 200.0


def classify_port(device: str, description: str) -> str:
    blob = f"{device} {description}".lower()
    if any(k in blob for k in ("bluetooth", "bth", "standard serial over bluetooth")):
        return "bt_spp"
    if any(k in blob for k in ("cp210", "silicon labs", "ch340", "ch910", "usb to uart")):
        return "usb"
    return "unknown"


def list_serial_candidates() -> list[tuple[str, str, str]]:
    out = []
    for p in list_ports.comports():
        kind = classify_port(p.device, p.description)
        out.append((p.device, f"{p.device}  {p.description}", kind))
    return out


def pick_usb_port(cands: list[tuple[str, str, str]]) -> str:
    for device, _label, kind in cands:
        if kind == "usb":
            return device
    return cands[0][0] if cands else ""


async def scan_ble(timeout: float = 5.0) -> list[tuple[str, str]]:
    from bleak import BleakScanner

    devices = await BleakScanner.discover(timeout=timeout)
    hits = []
    for d in devices:
        name = d.name or ""
        blob = f"{name} {d.address}".lower()
        if any(k in blob for k in ("wt", "901", "bwt", "wit", "sdcl")):
            hits.append((d.address, name or d.address))
    return hits


def sample_row(sample: dict[str, Any]) -> list[str]:
    return [
        f"{float(sample['timestamp']):.6f}",
        *[f"{float(sample['acc'][i]):.6f}" for i in range(3)],
        *[f"{float(sample['gyro'][i]):.6f}" for i in range(3)],
        *[f"{float(sample['angle'][i]):.6f}" for i in range(3)],
        *[f"{float(sample['mag'][i]):.6f}" for i in range(3)],
        f"{float(sample['pressure']):.6f}",
        f"{float(sample['altitude']):.6f}",
        f"{float(sample['temp']):.6f}",
        *[f"{float(sample['quaternion'][i]):.6f}" for i in range(4)],
    ]


class ImuWorker:
    def __init__(self, name: str) -> None:
        self.name = name
        self.kind = ""  # usb / ble / bt_spp
        self.port = ""
        self.address = ""
        self.baud = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.latest: Optional[dict[str, Any]] = None
        self.latest_idx = -1
        self.hz = 0.0
        self.ok = False
        self.last_error = ""
        self.last_rx = 0.0
        self._times: deque[float] = deque(maxlen=400)
        self._hist: deque[tuple[float, tuple[float, float, float], tuple[float, float, float]]] = deque(maxlen=8000)
        self._writer = None
        self._fp = None
        self.recording = False
        self.sample_count = 0
        self.reconnects = 0
        self.parser = WitParser(source=name)
        self._rx: queue.SimpleQueue = queue.SimpleQueue()

    def tail(self, n: int = 2000) -> list[tuple[float, tuple[float, float, float], tuple[float, float, float]]]:
        with self._lock:
            if n >= len(self._hist):
                return list(self._hist)
            return list(self._hist)[-n:]

    def snapshot(self) -> tuple[int, Optional[dict[str, Any]]]:
        with self._lock:
            if self.latest is None:
                return -1, None
            src = self.latest
            return self.latest_idx, {
                "timestamp": float(src["timestamp"]),
                "acc": list(src["acc"]),
                "gyro": list(src["gyro"]),
                "angle": list(src["angle"]),
                "mag": list(src["mag"]),
                "pressure": float(src["pressure"]),
                "altitude": float(src["altitude"]),
                "temp": float(src["temp"]),
                "quaternion": list(src["quaternion"]),
                "aux": float(src.get("aux", 0.0)),
                "source": str(src.get("source", self.name)),
            }

    def begin_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.end_csv()
        fp = open(path, "w", newline="", encoding="utf-8")
        writer = csv.writer(fp)
        writer.writerow(IMU_STREAM_COLUMNS)
        with self._lock:
            self._fp = fp
            self._writer = writer
            self.recording = True
            self.sample_count = 0

    def end_csv(self) -> None:
        with self._lock:
            self.recording = False
            fp = self._fp
            self._fp = None
            self._writer = None
        if fp is not None:
            try:
                fp.flush()
                fp.close()
            except Exception:
                pass

    def start_serial(self, port: str, baud: int, kind: str = "usb") -> None:
        self.stop()
        self.kind = kind
        self.port = port
        self.baud = int(baud)
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_serial, daemon=True, name=f"imu-{self.name}")
        self._thread.start()

    def start_ble(self, address: str) -> None:
        self.stop()
        self.kind = "ble"
        self.address = address
        self.port = address
        self.baud = 0
        self._stop.clear()
        self._thread = threading.Thread(target=self._run_ble_thread, daemon=True, name=f"imu-{self.name}")
        self._thread.start()

    def stop(self) -> None:
        self.end_csv()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.5)
        self._thread = None
        with self._lock:
            self.ok = False

    def _accept(self, samples: list[dict[str, Any]]) -> None:
        if not samples:
            return
        now = samples[-1]["timestamp"]
        flush_fp = None
        with self._lock:
            for sample in samples:
                self.latest_idx += 1
                self.latest = sample
                ts = float(sample["timestamp"])
                self._times.append(ts)
                acc = sample["acc"]
                gyro = sample["gyro"]
                self._hist.append((
                    ts,
                    (float(acc[0]), float(acc[1]), float(acc[2])),
                    (float(gyro[0]), float(gyro[1]), float(gyro[2])),
                ))
                if self.recording and self._writer is not None:
                    try:
                        self._writer.writerow(sample_row(sample))
                        self.sample_count += 1
                        if self.sample_count % 100 == 0:
                            flush_fp = self._fp
                    except Exception:
                        pass
            if len(self._times) >= 2:
                dt = self._times[-1] - self._times[0]
                self.hz = (len(self._times) - 1) / dt if dt > 1e-3 else 0.0
            self.ok = True
            self.last_rx = now
            self.last_error = ""
        if flush_fp is not None:
            try:
                flush_fp.flush()
            except Exception:
                pass

    def _pin(self) -> None:
        role = "imu_usb" if self.name == "usb" else "imu_bt"
        pin_current_thread(role, PRIOR_HIGHEST)

    def _run_serial(self) -> None:
        self._pin()
        import serial

        while not self._stop.is_set():
            ser = None
            try:
                ser = serial.Serial(
                    self.port, self.baud, timeout=0.01, write_timeout=0.4,
                    dsrdtr=False, rtscts=False,
                )
                try:
                    ser.reset_input_buffer()
                except Exception:
                    pass
                self._configure_serial(ser)
                self.parser = WitParser(source=self.name)
                while not self._stop.is_set():
                    try:
                        waiting = ser.in_waiting
                        if not waiting:
                            time.sleep(0.001)
                            continue
                        chunk = ser.read(waiting)
                    except Exception as exc:
                        self.last_error = str(exc)
                        break
                    t_end = time.time()
                    samples = self.parser.feed(chunk, t_end, dt_hint=1.0 / TARGET_HZ)
                    self._accept(samples)
            except Exception as exc:
                self.last_error = str(exc)
                with self._lock:
                    self.ok = False
            finally:
                if ser is not None:
                    try:
                        ser.close()
                    except Exception:
                        pass
            if self._stop.is_set():
                break
            self.reconnects += 1
            time.sleep(0.6)

    def _configure_serial(self, ser) -> None:
        for cmd in (CMD_UNLOCK, CMD_BANDWIDTH_256, CMD_RATE_200_CLASSIC, CMD_SAVE):
            try:
                ser.write(cmd)
                time.sleep(0.08)
            except Exception:
                break

    def _run_ble_thread(self) -> None:
        self._pin()
        try:
            asyncio.run(self._run_ble())
        except Exception as exc:
            self.last_error = str(exc)
            with self._lock:
                self.ok = False

    def _clear_rx(self) -> None:
        while True:
            try:
                self._rx.get_nowait()
            except queue.Empty:
                return

    def _drain_rx(self) -> None:
        while True:
            try:
                t_end, raw = self._rx.get_nowait()
            except queue.Empty:
                return
            samples = self.parser.feed(raw, t_end, dt_hint=1.0 / TARGET_HZ)
            if samples:
                self._accept(samples)

    async def _run_ble(self) -> None:
        from bleak import BleakClient

        while not self._stop.is_set():
            client = None
            try:
                client = BleakClient(self.address, timeout=12.0)
                await client.connect()
                await _request_ble_throughput(client)
                self.parser = WitParser(source=self.name)
                loop = asyncio.get_running_loop()

                self._clear_rx()

                def _on(_s, data: bytearray) -> None:
                    self._rx.put((time.time(), bytes(data)))

                await client.start_notify(BLE_NOTIFY, _on)
                # 0x02 打开磁场，让 0x54 跟加速度一起从通知里出来，不靠 0.2 秒轮询。
                for cmd in (CMD_UNLOCK, CMD_RSW_VIO_USB, CMD_RATE_200_CLASSIC, CMD_SAVE):
                    try:
                        await client.write_gatt_char(BLE_WRITE, cmd, response=False)
                        await asyncio.sleep(0.08)
                    except Exception as exc:
                        self.last_error = f"write {exc}"
                last_quat = 0.0
                last_mag = 0.0
                while not self._stop.is_set() and client.is_connected:
                    self._drain_rx()
                    now = time.time()
                    if now - last_quat > 1.0:
                        last_quat = now
                        try:
                            await client.write_gatt_char(BLE_WRITE, CMD_READ_QUAT, response=False)
                        except Exception:
                            pass
                    # 10 Hz 补读。200 Hz 连写会把通知从 200 Hz 压到 120 Hz。
                    if now - last_mag > 0.1:
                        last_mag = now
                        try:
                            await client.write_gatt_char(BLE_WRITE, CMD_READ_MAG, response=False)
                        except Exception:
                            pass
                    await asyncio.sleep(0.002)
            except Exception as exc:
                self.last_error = str(exc)
                with self._lock:
                    self.ok = False
            finally:
                if client is not None:
                    try:
                        if client.is_connected:
                            await client.disconnect()
                    except Exception:
                        pass
            if self._stop.is_set():
                break
            self.reconnects += 1
            await asyncio.sleep(0.8)


async def _request_ble_throughput(client) -> None:
    try:
        from winrt.windows.devices.bluetooth import BluetoothLEPreferredConnectionParameters
    except Exception:
        return
    backend = getattr(client, "_backend", None)
    requester = getattr(backend, "_requester", None) if backend is not None else None
    if requester is None:
        return
    try:
        requester.request_preferred_connection_parameters(
            BluetoothLEPreferredConnectionParameters.throughput_optimized
        )
    except Exception:
        pass
