#!/usr/bin/env python3
import numpy as np

from ego_capture.sync import causal_interp, estimate_delay


def test_causal_interp_does_not_read_the_future() -> None:
    t = np.array([0.0, 0.005, 0.010])
    y = np.zeros((3, 19))
    y[:, 0] = [0.0, 1.0, 2.0]
    y[:, 15] = 1.0
    out, ok = causal_interp(t, y, np.array([0.010, 0.012, 0.050]))
    assert ok[0] and abs(out[0, 0] - 2.0) < 1e-9
    assert ok[1] and abs(out[1, 0] - 2.4) < 1e-9
    assert not ok[2]


def test_delay_finds_late_camera() -> None:
    imu_t = np.arange(0.0, 2.0, 0.005)
    gyro = np.exp(-0.5 * ((imu_t - 1.0) / 0.02) ** 2)
    late = 0.03
    t_obs = np.arange(0.2, 1.8, 0.05)
    signal = np.exp(-0.5 * ((t_obs - (1.0 + late)) / 0.02) ** 2)
    found = estimate_delay(t_obs, signal, imu_t, gyro, search_s=0.08, step_s=0.001)
    assert found["reliable"] == 1.0
    assert abs(found["delay_s"] - late) <= 0.002


if __name__ == "__main__":
    test_causal_interp_does_not_read_the_future()
    test_delay_finds_late_camera()
    print("sync tests ok")
