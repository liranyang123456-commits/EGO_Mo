#!/usr/bin/env python3
from __future__ import annotations
import time
from ego_capture.imu import ImuWorker

usb = ImuWorker("usb")
bt = ImuWorker("bt")
usb.start_serial("COM3", 921600, "usb")
bt.start_ble("F9:5A:7B:FD:65:A0")
time.sleep(3.0)
for i in range(8):
    time.sleep(0.4)
    _, su = usb.snapshot()
    _, sb = bt.snapshot()
    def f(s):
        if not s:
            return "None"
        return f"acc={s['acc']} gyro={s['gyro']} rpy={s['angle']} mag={s['mag']} src={s.get('source')}"
    same = False
    if su and sb:
        same = su["acc"] == sb["acc"] and su["angle"] == sb["angle"]
    print(f"--- {i} USB {usb.hz:.0f}Hz n={usb.latest_idx}")
    print("   ", f(su))
    print(f"    BLE {bt.hz:.0f}Hz n={bt.latest_idx} err={bt.last_error!r}")
    print("   ", f(sb))
    print("    IDENTICAL" if same else "    different")
usb.stop(); bt.stop()
