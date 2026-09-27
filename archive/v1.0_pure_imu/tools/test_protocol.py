#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ego_capture.protocol import WitParser, build_classic_frame, scale_i16


def test_classic_acc() -> None:
    # ax=0, ay=0, az=+1g approx: 32768/16 = 2048 raw for 1g
    raw_z = 2048
    payload = bytes((0, 0, 0, 0, raw_z & 0xFF, (raw_z >> 8) & 0xFF, 0x20, 0x03))
    frame = build_classic_frame("acc", payload)
    p = WitParser("t")
    samples = p.feed(frame, t_end=1.0, dt_hint=0.005)
    assert len(samples) == 1
    assert abs(samples[0]["acc"][2] - 1.0) < 0.02
    assert abs(samples[0]["temp"] - 8.0) < 0.05


def test_ble61_multi() -> None:
    def pack_az(g: float) -> bytes:
        raw = int(round(g / 16.0 * 32768))
        body = bytearray(28)
        body[0] = 0x55
        body[1] = 0x61
        body[6] = raw & 0xFF
        body[7] = (raw >> 8) & 0xFF
        return bytes(body)

    chunk = pack_az(1.0) + pack_az(0.5) + pack_az(-0.25)
    p = WitParser("ble")
    samples = p.feed(chunk, t_end=10.0, dt_hint=0.005)
    assert len(samples) == 3
    assert abs(samples[0]["acc"][2] - 1.0) < 0.03
    assert abs(samples[1]["acc"][2] - 0.5) < 0.03
    assert abs(samples[2]["acc"][2] + 0.25) < 0.03
    assert abs(samples[2]["timestamp"] - 10.0) < 1e-9
    assert samples[1]["timestamp"] < samples[2]["timestamp"]


if __name__ == "__main__":
    test_classic_acc()
    test_ble61_multi()
    print("protocol tests ok")
