#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure WTSDCL BLE packet size / Hz, try 200 Hz + connection params."""

from __future__ import annotations

import asyncio
import time

ADDR = "F9:5A:7B:FD:65:A0"
NOTIFY = "0000ffe4-0000-1000-8000-00805f9a34fb"
WRITE = "0000ffe9-0000-1000-8000-00805f9a34fb"
PPCP = "00002a04-0000-1000-8000-00805f9b34fb"


def i16(lo: int, hi: int) -> int:
    v = (hi << 8) | lo
    return v - 65536 if v >= 32768 else v


async def measure(client, seconds: float) -> tuple[int, list[bytes]]:
    pkts: list[bytes] = []

    def _on(_s, data: bytearray) -> None:
        pkts.append(bytes(data))

    await client.start_notify(NOTIFY, _on)
    await asyncio.sleep(seconds)
    try:
        await client.stop_notify(NOTIFY)
    except Exception:
        pass
    return len(pkts), pkts


async def w(client, payload: bytes) -> None:
    await client.write_gatt_char(WRITE, payload, response=False)
    await asyncio.sleep(0.08)


async def main() -> None:
    from bleak import BleakClient

    async with BleakClient(ADDR, timeout=15.0) as client:
        print("connected", client.is_connected)
        try:
            raw = await client.read_gatt_char(PPCP)
            print("PPCP raw", raw.hex(" "), "len", len(raw))
            if len(raw) >= 8:
                def u16(i):
                    return raw[i] | (raw[i + 1] << 8)
                print("  min_interval", u16(0), "max_interval", u16(2),
                      "latency", u16(4), "timeout", u16(6),
                      "=> ms", u16(0) * 1.25, u16(2) * 1.25)
        except Exception as exc:
            print("PPCP read fail", exc)

        n, pkts = await measure(client, 2.0)
        print(f"default {n} pkts / 2s = {n/2:.1f} Hz")
        if pkts:
            p = pkts[0]
            print("sample", p.hex(" "), "len", len(p))
            if len(p) >= 20 and p[0] == 0x55 and p[1] == 0x61:
                ax = i16(p[2], p[3]) / 32768 * 16
                ay = i16(p[4], p[5]) / 32768 * 16
                az = i16(p[6], p[7]) / 32768 * 16
                wx = i16(p[8], p[9]) / 32768 * 2000
                wy = i16(p[10], p[11]) / 32768 * 2000
                wz = i16(p[12], p[13]) / 32768 * 2000
                r = i16(p[14], p[15]) / 32768 * 180
                pit = i16(p[16], p[17]) / 32768 * 180
                y = i16(p[18], p[19]) / 32768 * 180
                print(f"  acc {ax:.3f} {ay:.3f} {az:.3f} g")
                print(f"  gyro {wx:.2f} {wy:.2f} {wz:.2f}")
                print(f"  rpy {r:.2f} {pit:.2f} {y:.2f}")
                if len(p) >= 28:
                    q = [i16(p[20 + i], p[21 + i]) / 32768 for i in range(0, 8, 2)]
                    mag = [i16(p[20 + i], p[21 + i]) for i in range(0, 6, 2)]
                    print("  as_quat", [round(v, 4) for v in q])
                    print("  as_mag", mag, "tail", p[20:].hex(" "))

        # try unlock + 200Hz classic 0x0B and BLE 0x0A
        for name, cmds in [
            ("unlock+0x0A+save", [
                bytes([0xFF, 0xAA, 0x69, 0x88, 0xB5]),
                bytes([0xFF, 0xAA, 0x03, 0x0A, 0x00]),
                bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]),
            ]),
            ("0x0B+save", [
                bytes([0xFF, 0xAA, 0x03, 0x0B, 0x00]),
                bytes([0xFF, 0xAA, 0x00, 0x00, 0x00]),
            ]),
        ]:
            print("send", name)
            for c in cmds:
                await w(client, c)
            await asyncio.sleep(0.4)
            n, pkts = await measure(client, 2.5)
            print(f"  {name}: {n} / 2.5s = {n/2.5:.1f} Hz  lens={sorted({len(p) for p in pkts})}")

        # winrt throughput hint
        try:
            from winrt.windows.devices.bluetooth import BluetoothLEPreferredConnectionParameters
            print("winrt params available", BluetoothLEPreferredConnectionParameters)
        except Exception as exc:
            print("no winrt", exc)

        backend = getattr(client, "_backend", None) or client
        print("client type", type(client), "attrs", [a for a in dir(client) if "param" in a.lower() or "winrt" in a.lower() or "device" in a.lower()])


if __name__ == "__main__":
    asyncio.run(main())
