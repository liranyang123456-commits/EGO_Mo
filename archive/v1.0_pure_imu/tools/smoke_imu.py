#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Live 3s smoke: USB COM3 + BLE WTSDCL."""

from __future__ import annotations

import time

from ego_capture.imu import ImuWorker


def main() -> None:
    usb = ImuWorker("usb")
    bt = ImuWorker("bt")
    usb.start_serial("COM3", 921600, "usb")
    bt.start_ble("F9:5A:7B:FD:65:A0")
    t0 = time.time()
    while time.time() - t0 < 4.5:
        time.sleep(0.5)
        print(f"t={time.time()-t0:.1f}  USB {usb.hz:.1f}Hz ok={usb.ok} n={usb.latest_idx} err={usb.last_error!r}")
        print(f"         BLE {bt.hz:.1f}Hz ok={bt.ok} n={bt.latest_idx} err={bt.last_error!r}")
    usb.stop()
    bt.stop()
    print("done")


if __name__ == "__main__":
    main()
