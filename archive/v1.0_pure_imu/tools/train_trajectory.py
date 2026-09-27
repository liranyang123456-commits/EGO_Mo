#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train the trajectory corrector and score it against chessboard pose."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ego_capture.mapping.inertial import (
    DISP_LEN,
    IMU_DIM,
    WINDOW_LEN,
    MotionTrajectoryCorrector,
    apply_correction,
    exp_so3,
    exp_so3_b,
    geodesic_per,
    initialize_from_static,
    label_motion,
)
from ego_capture.sync import load_imu

DATA = ROOT / "datasets"
GT = DATA / "pose_gt"
TEST_NAME = "traj_20260923_023422"
VAL_NAME = "traj_20260923_023241"


def _age(t: np.ndarray, values: np.ndarray) -> np.ndarray:
    age = np.zeros(len(t), dtype=np.float32)
    if len(t) == 0:
        return age
    last = float(t[0])
    prev = values[0].copy()
    for i in range(len(t)):
        if float(np.max(np.abs(values[i] - prev))) > 1e-3:
            last = float(t[i])
            prev = values[i].copy()
        age[i] = float(t[i] - last)
    return age


def _packet(t: np.ndarray, y: np.ndarray) -> np.ndarray:
    feat = np.zeros((len(t), IMU_DIM), dtype=np.float32)
    n = min(y.shape[1], 19)
    feat[:, :n] = y[:, :n]
    feat[:, 19] = _age(t, y[:, 9:12])
    feat[:, 20] = _age(t, y[:, 15:19])
    return feat


def _window(t: np.ndarray, feat: np.ndarray, t_query: float, n: int) -> tuple[np.ndarray, np.ndarray] | None:
    k = int(np.searchsorted(t, t_query, side="right")) - 1
    if k < 10:
        return None
    a = max(0, k - n + 1)
    sl = feat[a:k + 1]
    ts = t[a:k + 1] - t_query
    if len(sl) < n:
        pad = n - len(sl)
        sl = np.vstack([np.repeat(sl[:1], pad, axis=0), sl])
        ts = np.concatenate([np.full(pad, ts[0], dtype=np.float64), ts])
    return sl.astype(np.float32), ts.astype(np.float32)


def _so3_log(R: np.ndarray) -> np.ndarray:
    cos = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    th = float(np.arccos(cos))
    if th < 1e-8:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]], dtype=np.float64)
    return w * (th / (2.0 * np.sin(th)))


def _fit_rotation(dR_imu: np.ndarray, dR_cam: np.ndarray) -> np.ndarray:
    w = np.zeros(3)
    step = min(len(dR_imu), 240)
    imu = dR_imu[:step]
    cam = dR_cam[:step]
    for _ in range(8):
        R = exp_so3(w)
        resid = np.zeros((len(imu), 3))
        for i in range(len(imu)):
            resid[i] = _so3_log(cam[i].T @ R @ imu[i] @ R.T)
        jac = np.zeros((len(imu), 3, 3))
        for axis in range(3):
            dw = np.zeros(3)
            dw[axis] = 1e-4
            Rp = exp_so3(w + dw)
            for i in range(len(imu)):
                jac[i, :, axis] = (_so3_log(cam[i].T @ Rp @ imu[i] @ Rp.T) - resid[i]) / 1e-4
        delta, *_ = np.linalg.lstsq(jac.reshape(-1, 3), -resid.reshape(-1), rcond=None)
        w = w + delta
        if float(np.linalg.norm(delta)) < 1e-6:
            break
    return exp_so3(w)


def _bias(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(120, len(y))
    gyro = y[:n, 3:6] * (np.pi / 180.0)
    acc = y[:n, 0:3] * 9.81
    still = np.linalg.norm(y[:n, 3:6], axis=1) < 3.0
    if int(still.sum()) >= 20:
        gyro = gyro[still]
        acc = acc[still]
    state = initialize_from_static(acc, gyro)
    return state.bg.astype(np.float32), state.ba.astype(np.float32), state.R.astype(np.float32)


def _rot6(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)


def load_split(names: list[str]) -> dict[str, dict]:
    packs = {}
    for name in names:
        path = GT / f"{name}.npz"
        if not path.is_file():
            continue
        gt = np.load(path)
        sess = DATA / name
        usb_t, usb_y = load_imu(sess / "imu_stream.csv")
        bt_t, bt_y = load_imu(sess / "imu_bt.csv")
        if len(usb_t) < WINDOW_LEN or len(bt_t) < 20:
            continue
        packs[name] = {
            "gt": gt,
            "usb_t": usb_t,
            "usb": _packet(usb_t, usb_y),
            "bt_t": bt_t,
            "bt": _packet(bt_t, bt_y),
        "bg": _bias(usb_y)[0],
        "ba": _bias(usb_y)[1],
        "R0": _bias(usb_y)[2],
        }
    return packs


def build_pairs(pack: dict, min_dt: float = 0.12, max_dt: float = 0.30) -> dict[str, np.ndarray] | None:
    gt = pack["gt"]
    t = gt["t"]
    ok = (gt["ok"] == 1) & (gt["ble_gyro"] < 8.0) & (gt["reproj"] < 1.5)
    idx = np.flatnonzero(ok)
    if len(idx) < 8:
        return None
    delay = float(gt["delay_usb"][0])
    tq = t - delay
    usb_w, bt_w, usb_ts, bt_ts = [], [], [], []
    cam, cam_m, cam_t = [], [], []
    dR, dp, state, acc, gyro = [], [], [], [], []
    span = []
    t_start, t_end = [], []
    for i in idx:
        target = tq[i] - 0.2
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = prev[np.argmin(np.abs(tq[prev] - target))]
        dt = float(tq[i] - tq[j])
        if dt < min_dt or dt > max_dt:
            continue
        uw = _window(pack["usb_t"], pack["usb"], float(tq[i]), WINDOW_LEN)
        bw = _window(pack["bt_t"], pack["bt"], float(tq[i]), WINDOW_LEN)
        if uw is None or bw is None:
            continue
        hist = prev[tq[prev] < tq[i] - 0.05][-4:]
        pose = np.zeros((4, 9), dtype=np.float32)
        pt = np.zeros(4, dtype=np.float32)
        mask = np.zeros(4, dtype=np.bool_)
        for s, h in enumerate(hist):
            pose[s, :6] = _rot6(gt["R"][h])
            pose[s, 6:] = gt["p"][h]
            pt[s] = float(tq[h] - tq[i])
            mask[s] = True
        Rj = gt["R"][j]
        dR_cam = Rj.T @ gt["R"][i]
        dp_cam = Rj.T @ (gt["p"][i] - gt["p"][j])
        speed = float(np.linalg.norm(dp_cam) / dt)
        gyr = float(np.linalg.norm(uw[0][-DISP_LEN:, 3:6].mean(axis=0)) * np.pi / 180.0)
        usb_w.append(uw[0])
        bt_w.append(bw[0])
        usb_ts.append(uw[1])
        bt_ts.append(bw[1])
        cam.append(pose)
        cam_m.append(mask)
        cam_t.append(pt)
        dR.append(dR_cam.astype(np.float32))
        dp.append(dp_cam.astype(np.float32))
        state.append(label_motion(speed, gyr))
        acc.append(uw[0][-DISP_LEN:, 0:3] * 9.81)
        gyro.append(uw[0][-DISP_LEN:, 3:6] * (np.pi / 180.0))
        span.append(dt)
        t_start.append(float(tq[j]))
        t_end.append(float(tq[i]))
    if len(dR) < 8:
        return None
    return {
        "usb": np.stack(usb_w),
        "bt": np.stack(bt_w),
        "usb_t": np.stack(usb_ts),
        "bt_t": np.stack(bt_ts),
        "cam": np.stack(cam),
        "cam_m": np.stack(cam_m),
        "cam_t": np.stack(cam_t),
        "dR": np.stack(dR),
        "dp": np.stack(dp),
        "state": np.asarray(state, dtype=np.int64),
        "acc": np.stack(acc).astype(np.float32),
        "gyro": np.stack(gyro).astype(np.float32),
        "bg": pack["bg"],
        "ba": pack["ba"],
        "R0": pack["R0"],
        "span": np.asarray(span, dtype=np.float32),
        "t_start": np.asarray(t_start, dtype=np.float64),
        "t_end": np.asarray(t_end, dtype=np.float64),
    }


def _rotate_labels(pack: dict, R_ci: np.ndarray) -> None:
    Rt = R_ci.T.astype(np.float32)
    pack["dR"] = np.stack([Rt @ R @ R_ci for R in pack["dR"]]).astype(np.float32)


def _loader(pack: dict, shuffle: bool) -> DataLoader:
    tensors = [
        torch.from_numpy(pack["usb"]),
        torch.from_numpy(pack["bt"]),
        torch.from_numpy(pack["usb_t"]),
        torch.from_numpy(pack["bt_t"]),
        torch.from_numpy(pack["cam"]),
        torch.from_numpy(pack["cam_m"]),
        torch.from_numpy(pack["cam_t"]),
        torch.from_numpy(pack["dR"]),
        torch.from_numpy(pack["dp"]),
        torch.from_numpy(pack["state"]),
        torch.from_numpy(pack["acc"]),
        torch.from_numpy(pack["gyro"]),
    ]
    return DataLoader(TensorDataset(*tensors), batch_size=16, shuffle=shuffle, drop_last=False)


def _score(net, loader, bg, ba, R0, coef, device) -> dict[str, float]:
    net.eval()
    rot = pos = base_r = base_p = 0.0
    n = 0
    bg_t = torch.tensor(bg, device=device)
    ba_t = torch.tensor(ba, device=device)
    R0_t = torch.tensor(R0, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]
            usb, bt, usb_t, bt_t, cam, cam_m, cam_t, dR, dp, state, acc, gyro = batch
            dRs, dps = integrate_aligned(acc, gyro, bg_t, ba_t, R0_t)
            logits, xi, log_var, _ = net(usb, bt, cam, None, cam_m, None, usb_t, bt_t, cam_t, None)
            dRc, dpc, dplin = _correct(dRs, dps, xi, coef)
            b = usb.shape[0]
            rot += float(torch.acos(torch.clamp((torch.diagonal(dRc.transpose(1, 2) @ dR, dim1=1, dim2=2).sum(-1) - 1) * 0.5, -1 + 1e-6, 1 - 1e-6)).sum())
            pos += float((dpc - dp).norm(dim=1).sum())
            base_r += float(torch.acos(torch.clamp((torch.diagonal(dRs.transpose(1, 2) @ dR, dim1=1, dim2=2).sum(-1) - 1) * 0.5, -1 + 1e-6, 1 - 1e-6)).sum())
            base_p += float((dplin - dp).norm(dim=1).sum())
            n += b
            _ = logits, log_var, state
    n = max(n, 1)
    return {
        "rot_deg": rot / n * 180.0 / np.pi,
        "pos_mm": pos / n * 1000.0,
        "strap_rot_deg": base_r / n * 180.0 / np.pi,
        "linear_pos_mm": base_p / n * 1000.0,
    }


def integrate_aligned(acc, gyro, bg, ba, R0, dt: float = 0.005):
    """Strapdown that starts in the gravity-aligned frame, then returns the start-frame increment."""
    eye_motion_R = R0.detach().to(dtype=acc.dtype, device=acc.device)
    batch = acc.shape[0]
    R = eye_motion_R.expand(batch, 3, 3).clone()
    v = acc.new_zeros(batch, 3)
    p = acc.new_zeros(batch, 3)
    g = acc.new_tensor([0.0, 0.0, -9.81])
    bg = bg.reshape(1, 3).to(dtype=acc.dtype, device=acc.device)
    ba = ba.reshape(1, 3).to(dtype=acc.dtype, device=acc.device)
    for i in range(acc.shape[1]):
        R = torch.matmul(R, exp_so3_b((gyro[:, i] - bg) * dt))
        a = torch.matmul(R, (acc[:, i] - ba).unsqueeze(-1)).squeeze(-1) + g
        p = p + v * dt + 0.5 * a * (dt * dt)
        v = v + a * dt
    dR = torch.matmul(eye_motion_R.transpose(0, 1).expand(batch, 3, 3), R)
    dp = torch.matmul(eye_motion_R.transpose(0, 1).expand(batch, 3, 3), p.unsqueeze(-1)).squeeze(-1)
    return dR, dp


def _map_dp(dp: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
    ones = torch.ones(dp.shape[0], 1, device=dp.device, dtype=dp.dtype)
    return torch.cat([dp, ones], dim=1) @ coef


def geodesic_per_safe(R_pred: torch.Tensor, R_gt: torch.Tensor) -> torch.Tensor:
    return geodesic_per(R_pred, R_gt) * (180.0 / np.pi)


def _correct(dRs, dps, xi, coef):
    dRc = torch.matmul(dRs, exp_so3_b(xi[:, :3]))
    mapped = _map_dp(dps, coef)
    return dRc, mapped + xi[:, 3:], mapped


def _train_epoch(net, loader, opt, bg, ba, R0, coef, device) -> float:
    net.train()
    total = 0.0
    seen = 0
    bg_t = torch.tensor(bg, device=device)
    ba_t = torch.tensor(ba, device=device)
    R0_t = torch.tensor(R0, device=device)
    for batch in loader:
        batch = [x.to(device) for x in batch]
        usb, bt, usb_t, bt_t, cam, cam_m, cam_t, dR, dp, state, acc, gyro = batch
        dRs, dps = integrate_aligned(acc, gyro, bg_t, ba_t, R0_t)
        logits, xi, log_var, _ = net(usb, bt, cam, None, cam_m, None, usb_t, bt_t, cam_t, None)
        dRc, dpc, dplin = _correct(dRs, dps, xi, coef)
        rot_c = geodesic_per_safe(dRc, dR)
        rot_s = geodesic_per_safe(dRs, dR)
        pos_mm = (dpc - dp).norm(dim=-1) * 1000.0
        base_mm = (dplin - dp).norm(dim=-1) * 1000.0
        loss = pos_mm.mean() + 40.0 * torch.relu(rot_c - rot_s.detach()).mean()
        loss = loss + 5.0 * torch.relu(pos_mm - base_mm.detach()).mean()
        loss = loss + 0.05 * xi[:, :3].pow(2).mean() * 180.0 / np.pi
        if bool((state >= 0).any()):
            loss = loss + 0.2 * torch.nn.functional.cross_entropy(logits, state)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        total += float(loss.item()) * usb.shape[0]
        seen += usb.shape[0]
    return total / max(seen, 1)


def _path_mm(net, loader, bg, ba, R0, coef, device) -> dict[str, float]:
    """Chain non-overlapping 0.2 s increments and compare the path to PnP."""
    net.eval()
    pred_R, pred_p, base_R, base_p, gt_R, gt_p = [], [], [], [], [], []
    bg_t = torch.tensor(bg, device=device)
    ba_t = torch.tensor(ba, device=device)
    R0_t = torch.tensor(R0, device=device)
    with torch.no_grad():
        for batch in loader:
            batch = [x.to(device) for x in batch]
            usb, bt, usb_t, bt_t, cam, cam_m, cam_t, dR, dp, _state, acc, gyro = batch
            dRs, dps = integrate_aligned(acc, gyro, bg_t, ba_t, R0_t)
            _logits, xi, _log_var, _h = net(usb, bt, cam, None, cam_m, None, usb_t, bt_t, cam_t, None)
            dRc, dpc, dplin = _correct(dRs, dps, xi, coef)
            pred_R.append(dRc.cpu().numpy())
            pred_p.append(dpc.cpu().numpy())
            base_R.append(dRs.cpu().numpy())
            base_p.append(dplin.cpu().numpy())
            gt_R.append(dR.cpu().numpy())
            gt_p.append(dp.cpu().numpy())

    def chain(Rs, ps) -> np.ndarray:
        R = np.eye(3)
        p = np.zeros(3)
        out = []
        for i in range(0, len(Rs), 2):
            p = p + R @ ps[i]
            R = R @ Rs[i]
            out.append(p.copy())
        return np.stack(out)

    pr = np.concatenate(pred_R)
    pp = np.concatenate(pred_p)
    br = np.concatenate(base_R)
    bp = np.concatenate(base_p)
    gr = np.concatenate(gt_R)
    gp = np.concatenate(gt_p)
    path_g = chain(gr, gp)
    path_p = chain(pr, pp)
    path_b = chain(br, bp)
    return {
        "path_mm": float(np.sqrt(np.mean(np.sum((path_p - path_g) ** 2, axis=1))) * 1000.0),
        "linear_path_mm": float(np.sqrt(np.mean(np.sum((path_b - path_g) ** 2, axis=1))) * 1000.0),
    }


def _imu_motion(packs: list[dict], device) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotations, positions = [], []
    with torch.no_grad():
        for pack in packs:
            acc = torch.from_numpy(pack["acc"]).to(device)
            gyro = torch.from_numpy(pack["gyro"]).to(device)
            bg = torch.tensor(pack["bg"], device=device)
            ba = torch.tensor(pack["ba"], device=device)
            dR, dp = integrate_aligned(acc, gyro, bg, ba, torch.tensor(pack["R0"], device=device))
            rotations.append(dR.cpu().numpy())
            positions.append(dp.cpu().numpy())
    return rotations, positions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default=str(DATA / "traj_run"))
    args = parser.parse_args()
    index = json.loads((GT / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("pnp_still", 0) >= 30]
    if TEST_NAME not in names:
        raise SystemExit(f"测试会话 {TEST_NAME} 没有足够的静止棋盘位姿")
    train_names = [n for n in names if n not in (TEST_NAME, VAL_NAME)]
    packs = load_split(names)
    built = {name: build_pairs(packs[name]) for name in names if name in packs}
    built = {k: v for k, v in built.items() if v is not None}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_packs = [built[n] for n in train_names if n in built]
    if not train_packs:
        raise SystemExit("训练集为空")
    imu_dR, imu_dp = _imu_motion(train_packs, device)
    cam_dR = np.concatenate([p["dR"] for p in train_packs])
    cam_dp = np.concatenate([p["dp"] for p in train_packs])
    imu_cat = np.concatenate(imu_dR)
    step = max(1, len(imu_cat) // 240)
    R_ci = _fit_rotation(imu_cat[::step], cam_dR[::step])
    design = np.concatenate([np.concatenate(imu_dp), np.ones((len(cam_dp), 1))], axis=1)
    coef_np, *_ = np.linalg.lstsq(design, cam_dp, rcond=None)
    coef = torch.tensor(coef_np, dtype=torch.float32, device=device)
    for pack in built.values():
        _rotate_labels(pack, R_ci)
    val = built.get(VAL_NAME)
    test = built[TEST_NAME]
    net = MotionTrajectoryCorrector().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    val_loader = _loader(val, False) if val is not None else None
    test_loader = _loader(test, False)
    best = 1e9
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        loss = 0.0
        seen = 0
        for pack in train_packs:
            loader = _loader(pack, True)
            loss += _train_epoch(net, loader, opt, pack["bg"], pack["ba"], pack["R0"], coef, device) * pack["dR"].shape[0]
            seen += pack["dR"].shape[0]
        loss /= max(seen, 1)
        val_metrics = _score(net, val_loader, val["bg"], val["ba"], val["R0"], coef, device) if val_loader is not None else {}
        row = {"epoch": epoch, "loss": loss, **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(row)
        score = val_metrics.get("pos_mm", loss)
        print(json.dumps(row), flush=True)
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    if best_state:
        net.load_state_dict(best_state)
    test_metrics = _score(net, test_loader, test["bg"], test["ba"], test["R0"], coef, device)
    test_metrics.update(_path_mm(net, test_loader, test["bg"], test["ba"], test["R0"], coef, device))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "R_ci": R_ci,
        "test": test_metrics,
        "train_names": train_names,
        "val": VAL_NAME,
        "test_name": TEST_NAME,
    }, out / "corrector.pt")
    report = {
        "device": str(device),
        "train_names": train_names,
        "val": VAL_NAME,
        "test": TEST_NAME,
        "train_pairs": int(sum(built[n]["dR"].shape[0] for n in train_names if n in built)),
        "test_pairs": int(test["dR"].shape[0]),
        "test_metrics": test_metrics,
        "history_tail": history[-3:],
    }
    (out / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
