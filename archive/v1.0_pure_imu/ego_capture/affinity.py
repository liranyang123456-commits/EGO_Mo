#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pin capture threads onto distinct logical processors."""

from __future__ import annotations

import os
import threading

# Win32 thread priorities. TIME_CRITICAL is avoided so the UI stays responsive.
PRIOR_HIGHEST = 2
PRIOR_ABOVE = 1
PRIOR_NORMAL = 0
PRIOR_BELOW = -1

_LABELS = (
    ("cam_l", "左目取图"),
    ("cam_r", "右目取图"),
    ("imu_usb", "USB IMU"),
    ("imu_bt", "蓝牙 IMU"),
    ("det_l", "左目角点"),
    ("det_r", "右目角点"),
    ("writer", "写盘"),
)

_api_ready = False
_api_lock = threading.Lock()


def cpu_count() -> int:
    return os.cpu_count() or 1


def core_order() -> list[int]:
    """Even logical ids first.

    On common Intel and AMD layouts, even and odd ids are hyperthread
    siblings. The four capture roles then land on different physical cores.
    """
    count = cpu_count()
    if count >= 4:
        evens = list(range(0, count, 2))
        odds = list(range(1, count, 2))
        return evens + odds
    return list(range(count))


def cores_for_roles() -> dict[str, int | None]:
    order = core_order()
    out: dict[str, int | None] = {}
    for index, (role, _label) in enumerate(_LABELS):
        out[role] = order[index] if index < len(order) else None
    return out


def describe_plan() -> str:
    cores = cores_for_roles()
    parts = []
    for role, label in _LABELS:
        core = cores[role]
        parts.append(f"{label}→核心{core}" if core is not None else f"{label}不绑核")
    return f"逻辑核心 {cpu_count()} 个。" + "，".join(parts)


def pin_current_thread(role: str, priority: int | None = None) -> int | None:
    core = cores_for_roles().get(role)
    if os.name == "nt":
        return _pin_windows(core, priority)
    if core is None:
        return None
    try:
        os.sched_setaffinity(0, {core})
    except (AttributeError, OSError):
        return None
    return core


def _kernel32():
    global _api_ready
    import ctypes

    kernel = ctypes.windll.kernel32
    with _api_lock:
        if not _api_ready:
            kernel.GetCurrentThread.restype = ctypes.c_void_p
            kernel.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong]
            kernel.SetThreadAffinityMask.restype = ctypes.c_ulonglong
            kernel.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            kernel.SetThreadPriority.restype = ctypes.c_int
            kernel.GetCurrentProcessorNumber.restype = ctypes.c_ulong
            _api_ready = True
    return kernel


def _pin_windows(core: int | None, priority: int | None) -> int | None:
    kernel = _kernel32()
    handle = kernel.GetCurrentThread()
    applied: int | None = None
    if core is not None and 0 <= core < 64:
        mask = 1 << core
        if kernel.SetThreadAffinityMask(handle, mask):
            applied = core
    if priority is not None:
        kernel.SetThreadPriority(handle, int(priority))
    return applied
