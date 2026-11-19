#!/usr/bin/env python3
import torch
from ego_capture.mapping import FrameMappingSystem, build_mapper, geodesic_loss, rot6d_to_R


def test_rot6d() -> None:
    R = torch.eye(3).unsqueeze(0)
    x6 = torch.tensor([[1.0, 0, 0, 0, 1.0, 0]])
    Rh = rot6d_to_R(x6)
    assert torch.allclose(Rh, R, atol=1e-5)
    assert float(geodesic_loss(Rh, R)) < 1e-5


def test_forward() -> None:
    B, T = 2, 20
    usb = torch.randn(B, T, 13)
    bt = torch.randn(B, T, 13)
    for kind in ("mlp", "gru", "gru_xattn"):
        m = build_mapper(kind, seq_len=T, hidden=32)
        sys = FrameMappingSystem(m, torch.eye(3), torch.zeros(3))
        R, t, r6 = sys(usb, bt)
        assert R.shape == (B, 3, 3)
        assert t.shape == (B, 3)
        det = torch.det(R)
        assert torch.allclose(det, torch.ones_like(det), atol=1e-4)


def test_corrector_is_causal_and_gates() -> None:
    from ego_capture.mapping.inertial import MotionTrajectoryCorrector, mix_residual

    net = MotionTrajectoryCorrector(d_model=32, nhead=4, layers=2, dropout=0.0).eval()
    usb = torch.randn(2, 32, 21)
    ble = torch.randn(2, 20, 21)
    usb_t = torch.arange(32).float().expand(2, 32) / 200
    ble_t = torch.arange(20).float().expand(2, 20) / 180
    h1 = net.encode_imu(usb, usb_t, 0)
    cut = 16
    usb2 = usb.clone()
    usb2[:, cut:] = torch.randn_like(usb2[:, cut:])
    h2 = net.encode_imu(usb2, usb_t, 0)
    assert torch.allclose(h1[:, :cut], h2[:, :cut], atol=1e-5)
    past_t = usb_t[:, -1:] - 0.05
    future_t = usb_t[:, -1:] + 0.05
    pnp_a = torch.randn(2, 1, 9)
    pnp_b = torch.randn(2, 1, 9)
    valid = torch.ones(2, 1, dtype=torch.bool)
    logits_future_a = net(usb, ble, cam0=pnp_a, cam0_mask=valid, usb_t=usb_t, ble_t=ble_t, cam0_t=future_t)[0]
    logits_future_b = net(usb, ble, cam0=pnp_b, cam0_mask=valid, usb_t=usb_t, ble_t=ble_t, cam0_t=future_t)[0]
    assert torch.allclose(logits_future_a, logits_future_b, atol=1e-5)
    logits_past_a = net(usb, ble, cam0=pnp_a, cam0_mask=valid, usb_t=usb_t, ble_t=ble_t, cam0_t=past_t)[0]
    logits_past_b = net(usb, ble, cam0=pnp_b, cam0_mask=valid, usb_t=usb_t, ble_t=ble_t, cam0_t=past_t)[0]
    assert not torch.allclose(logits_past_a, logits_past_b, atol=1e-4)
    expert = torch.randn(2, 4, 6)
    still = mix_residual(expert, torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]))
    fast = mix_residual(expert, torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))
    assert torch.allclose(still, torch.zeros_like(still), atol=1e-6)
    assert torch.allclose(fast[:, 3:], torch.zeros(2, 3), atol=1e-6)
    assert not torch.allclose(fast[:, :3], torch.zeros(2, 3))


def test_correction_loss_and_pnp_snap() -> None:
    import numpy as np

    from ego_capture.mapping.inertial import (
        MotionTrajectoryCorrector,
        correct_increment,
        correction_loss,
        initialize_from_static,
        label_motion,
        propagate,
        snap_to_pnp,
    )

    assert label_motion(0.0, 0.0) == 0
    assert label_motion(0.2, 0.1) == 3
    net = MotionTrajectoryCorrector(d_model=32, nhead=4, layers=2, dropout=0.0)
    acc = torch.zeros(4, 16, 3)
    acc[:, :, 2] = 9.81
    gyro = torch.zeros_like(acc)
    bg = torch.zeros(3)
    ba = torch.zeros(3)
    dR, dp, logits, xi, log_var, _ = correct_increment(net, acc, gyro, bg, ba, dt=0.005, horizon=8)
    eye = torch.eye(3).expand(4, 3, 3)
    loss, _stats = correction_loss(
        dR, dp, logits, xi, log_var, eye, torch.zeros_like(dp),
        torch.tensor([0, 1, -1, 3]), torch.tensor([1.0, 1.0, 0.0, 1.0]),
    )
    loss.backward()

    acc_np = np.zeros((40, 3))
    acc_np[:, 2] = 9.81
    gyro_np = np.zeros_like(acc_np)
    gyro_np[:, 2] = 0.02
    state = initialize_from_static(acc_np, gyro_np)
    propagate(state, gyro_np, acc_np, 0.005)
    assert np.linalg.norm(state.p) < 1e-6
    snap_to_pnp(state, np.eye(3), np.array([0.01, 0.0, 0.0]))
    assert abs(state.p[0] - 0.01) < 1e-12


if __name__ == "__main__":
    test_rot6d()
    test_forward()
    test_corrector_is_causal_and_gates()
    test_correction_loss_and_pnp_snap()
    print("mapping tests ok")
