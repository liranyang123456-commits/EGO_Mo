#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from ego_capture.calib import align_imus_static, stereo_T_C0_C1
from ego_capture.geometry import handeye_rotation, kabsch_R, rpy_deg_to_R, rot_err_deg
from ego_capture.rig import STEREO_BASELINE_M, STEREO_BASELINE_MM


def test_baseline() -> None:
    assert abs(STEREO_BASELINE_MM - 46.5) < 1e-9
    T = stereo_T_C0_C1()
    assert abs(T[0, 3] - STEREO_BASELINE_M) < 1e-12


def test_kabsch_and_handeye() -> None:
    X = rpy_deg_to_R(12.0, -7.0, 33.0)
    rng = np.random.default_rng(0)
    src = rng.normal(size=(20, 3))
    src /= np.linalg.norm(src, axis=1, keepdims=True)
    dst = src @ X.T
    R = kabsch_R(src, dst)
    assert rot_err_deg(R, X) < 1e-6
    As, Bs = [], []
    for _ in range(40):
        B = rpy_deg_to_R(*rng.uniform(-40, 40, size=3))
        As.append(X @ B @ X.T)
        Bs.append(B)
    Xh, err = handeye_rotation(As, Bs)
    assert rot_err_deg(Xh, X) < 0.5
    assert err < 0.5


def test_static_gravity() -> None:
    X = rpy_deg_to_R(0, 0, 25)
    n = 400
    t = np.linspace(0, 4, n)
    acc_u = np.tile([0.0, 0.0, 1.0], (n, 1))
    acc_b = (X.T @ acc_u.T).T
    rng = np.random.default_rng(1)
    mag_u = np.tile(_unit([1.0, 0.2, 0.1]), (n, 1))
    mag_u = mag_u * 3000.0 + rng.normal(0, 30.0, size=mag_u.shape)
    mag_b = (X.T @ mag_u.T).T
    usb = np.zeros((n, 17))
    bt = np.zeros((n, 17))
    usb[:, 0] = t
    bt[:, 0] = t
    usb[:, 1:4] = acc_u
    bt[:, 1:4] = acc_b
    usb[:, 10:13] = mag_u
    bt[:, 10:13] = mag_b
    usb[:, 13] = 1.0
    bt[:, 13] = 1.0
    out = align_imus_static(usb, bt)
    R = np.array(out["T_Iusb_Ibt"]["R"])
    assert rot_err_deg(R, X) < 1.0
    assert out["gravity_residual_deg"] < 1.0


def _unit(v):
    a = np.asarray(v, dtype=float)
    return a / np.linalg.norm(a)


if __name__ == "__main__":
    test_baseline()
    test_kabsch_and_handeye()
    test_static_gravity()
    print("calib tests ok")
