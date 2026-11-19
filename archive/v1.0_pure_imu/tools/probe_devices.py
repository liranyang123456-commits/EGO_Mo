#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe USB IMU (COM) and WitMotion BLE (WTSDCL / BWT901CL)."""

from __future__ import annotations

import asyncio
import time
from collections import Counter

from serial.tools import list_ports


def dump_ports() -> None:
    print("=== serial ports ===")
    ports = list(list_ports.comports())
    if not ports:
        print("  (none)")
    for p in ports:
        print(f"  {p.device} | {p.description} | {p.hwid}")


def probe_serial(port: str, baud: int, seconds: float = 1.2) -> dict:
    import serial

    ser = serial.Serial(port, baud, timeout=0.05, write_timeout=0.3)
    ser.reset_input_buffer()
    t0 = time.time()
    buf = bytearray()
    while time.time() - t0 < seconds:
        chunk = ser.read(4096)
        if chunk:
            buf.extend(chunk)
    ser.close()
    n55 = buf.count(0x55)
    types = Counter()
    i = 0
    while i + 2 <= len(buf):
        if buf[i] == 0x55:
            types[buf[i + 1]] += 1
            i += 1
        else:
            i += 1
    return {
        "port": port,
        "baud": baud,
        "bytes": len(buf),
        "headers_0x55": n55,
        "next_byte_hist": {f"0x{k:02X}": v for k, v in types.most_common(12)},
        "head": buf[:24].hex(" "),
    }


async def probe_ble(seconds: float = 4.0) -> None:
    from bleak import BleakClient, BleakScanner

    print("=== BLE scan (4s) ===")
    devices = await BleakScanner.discover(timeout=4.0)
    hits = []
    for d in devices:
        name = d.name or ""
        blob = f"{name} {d.address}".lower()
        interesting = any(k in blob for k in ("wt", "901", "bwt", "wit", "sdcl", "f95a"))
        print(f"  {d.address} | {name!r}{'  <--' if interesting else ''}")
        if interesting:
            hits.append(d)
    if not hits:
        # fallback: known MAC from PnP
        print("  no name match; will try F9:5A:7B:FD:65:A0")
    targets = hits or []
    extra = "F9:5A:7B:FD:65:A0"
    addrs = [d.address for d in targets]
    if extra not in addrs:
        addrs.append(extra)

    for addr in addrs:
        print(f"=== BLE connect {addr} ===")
        try:
            async with BleakClient(addr, timeout=12.0) as client:
                print("  connected", client.is_connected)
                for svc in client.services:
                    print(f"  svc {svc.uuid} {svc.description}")
                    for ch in svc.characteristics:
                        print(f"    ch {ch.uuid} {ch.properties} {ch.description}")

                notify_uuid = None
                write_uuid = None
                for svc in client.services:
                    for ch in svc.characteristics:
                        u = ch.uuid.lower()
                        props = set(ch.properties)
                        if "notify" in props and ("ffe4" in u or notify_uuid is None):
                            if "ffe4" in u:
                                notify_uuid = ch.uuid
                        if ("write" in props or "write-without-response" in props) and (
                            "ffe9" in u or write_uuid is None
                        ):
                            if "ffe9" in u:
                                write_uuid = ch.uuid
                if notify_uuid is None:
                    for svc in client.services:
                        for ch in svc.characteristics:
                            if "notify" in ch.properties:
                                notify_uuid = ch.uuid
                                break
                print("  notify", notify_uuid, "write", write_uuid)

                packets: list[bytes] = []

                def _on(_s, data: bytearray) -> None:
                    packets.append(bytes(data))

                if notify_uuid:
                    await client.start_notify(notify_uuid, _on)
                if write_uuid:
                    # BLE 200 Hz
                    await client.write_gatt_char(write_uuid, bytes([0xFF, 0xAA, 0x03, 0x0A, 0x00]), response=False)
                    await asyncio.sleep(0.05)
                    await client.write_gatt_char(write_uuid, bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]), response=False)
                await asyncio.sleep(seconds)
                print(f"  packets={len(packets)} ~ {len(packets)/seconds:.1f} Hz")
                for pkt in packets[:6]:
                    print(f"    {pkt[:24].hex(' ')}  len={len(pkt)}")
                if packets:
                    hist = Counter(p[:2].hex() for p in packets)
                    print("  prefix", dict(hist.most_common(8)))
        except Exception as exc:
            print("  FAIL", type(exc).__name__, exc)


def main() -> None:
    dump_ports()
    from serial.tools import list_ports as lp

    for p in lp.comports():
        for baud in (921600, 115200):
            try:
                info = probe_serial(p.device, baud, 0.8)
                print("=== serial probe ===", info)
            except Exception as exc:
                print(f"=== serial {p.device}@{baud} FAIL {exc}")

    try:
        asyncio.run(probe_ble(3.0))
    except Exception as exc:
        print("BLE probe failed", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
