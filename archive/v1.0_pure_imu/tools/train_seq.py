#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Long-context IMU network for the 0.2 s camera step.

The input is the USB stream from `context` seconds before the pair start to
`context` seconds after its end, so an offline label can use motion on both
sides. A mask channel marks the pair interval. Labels are cleaned raw
chessboard poses where both endpoints are usable. Sessions are split before
training; the test sessions are not evaluated until model selection finishes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
from tools.train_trajectory import TEST_NAME, VAL_NAME

DATA = ROOT / "datasets"
HZ = 200.0


LABELS = "pose_gt_raw"
TARGET_S = 0.2
# Estimated without validation/test labels: OLD uses the five 2026-09-23
# training sessions; NEW uses the independent rigid_20260924_142902 check.
R_CAMERA_IMU_OLD = np.array([
    [0.017848, -0.962561, -0.270477],
    [0.067527, 0.271063, -0.960190],
    [0.997558, -0.001126, 0.069836],
], dtype=np.float32)
R_CAMERA_IMU_NEW = np.array([
    [0.007668, -0.946526, -0.322536],
    [-0.003137, 0.322521, -0.946557],
    [0.999966, 0.008270, -0.000496],
], dtype=np.float32)


def _pairs(name: str):
    locked = np.load(DATA / LABELS / f"{name}.npz")
    raw = np.load(DATA / "pose_gt" / f"{name}.npz")
    delay = float(raw["delay_usb"][0])
    t = locked["t"] - delay
    idx = np.flatnonzero(locked["usable"] == 1)
    rows = []
    for i in idx:
        prev = idx[idx < i]
        if len(prev) == 0:
            continue
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - TARGET_S)))])
        dt = float(t[i] - t[j])
        if dt < 0.60 * TARGET_S or dt > 1.50 * TARGET_S:
            continue
        R_j = locked["R"][j].astype(np.float64)
        dp = R_j.T @ (locked["p"][i] - locked["p"][j])
        rows.append((float(t[j]), float(t[i]), dp))
    return rows


def _grid(usb_t, usb, t0, t1, context):
    n = int(round((t1 - t0 + 2 * context) * HZ))
    grid = t0 - context + np.arange(n) / HZ
    feat = np.zeros((n, 8), dtype=np.float32)
    for c in range(6):
        feat[:, c] = np.interp(grid, usb_t, usb[:, c])
    feat[:, 0:3] -= feat[:, 0:3].mean(0, keepdims=True)
    feat[:, 3:6] /= 100.0
    feat[:, 6] = ((grid >= t0) & (grid <= t1)).astype(np.float32)
    feat[:, 7] = (t1 - t0) / TARGET_S
    return feat


def build(name: str, context: float, length: int, camera_aligned: bool = False):
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    if camera_aligned:
        R_ci = R_CAMERA_IMU_NEW if name.startswith("traj_20260924_") else R_CAMERA_IMU_OLD
        usb = usb.copy()
        usb[:, 0:3] = usb[:, 0:3] @ R_ci.T
        usb[:, 3:6] = usb[:, 3:6] @ R_ci.T
    X, Y = [], []
    for t0, t1, dp in _pairs(name):
        if t0 - context < usb_t[0] or t1 + context > usb_t[-1]:
            continue
        feat = _grid(usb_t, usb, t0, t1, context)
        if len(feat) < length:
            feat = np.vstack([feat, np.zeros((length - len(feat), 8), np.float32)])
        X.append(feat[:length])
        Y.append(dp.astype(np.float32))
    if not X:
        return None
    return np.stack(X), np.stack(Y)


class SeqNet(nn.Module):
    def __init__(self, width: int = 96) -> None:
        super().__init__()
        layers = []
        ch = 8
        for d in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(ch, width, 5, padding=2 * d, dilation=d), nn.GELU()]
            ch = width
        self.conv = nn.Sequential(*layers)
        self.gru = nn.GRU(width, width, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * width, width), nn.GELU(), nn.Linear(width, 3))

    def forward(self, x):
        mask = x[:, :, 6:7]
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)
        h, _ = self.gru(h)
        inside = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        pooled = h.mean(1)
        return self.head(torch.cat([inside, pooled], dim=-1))


def _metrics(pred, y):
    err = np.linalg.norm(pred - y, axis=1)
    mag = np.linalg.norm(y, axis=1)
    fast = mag >= np.quantile(mag, 0.67)
    resid = np.sum((pred - y) ** 2)
    base = np.sum((y - y.mean(0)) ** 2)
    return {
        "n": int(len(y)),
        "zero_mm": round(float(mag.mean() * 1000), 2),
        "err_mm": round(float(err.mean() * 1000), 2),
        "fast_mm": round(float(err[fast].mean() * 1000), 2),
        "fast_zero_mm": round(float(mag[fast].mean() * 1000), 2),
        "pred_med_mm": round(float(np.median(np.linalg.norm(pred, axis=1)) * 1000), 2),
        "r2": round(float(1 - resid / max(base, 1e-12)), 3),
    }


def _predict(net, X, device):
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 256):
            out.append(net(torch.from_numpy(X[k:k + 256]).to(device)).cpu().numpy())
    return np.concatenate(out)


def _join(data: dict, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    available = [name for name in names if name in data]
    if not available:
        raise RuntimeError(f"no built sessions from {names}")
    return (
        np.concatenate([data[name][0] for name in available]),
        np.concatenate([data[name][1] for name in available]),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", default="pose_gt_raw")
    ap.add_argument("--out", default="traj_run_v7")
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    ap.add_argument("--evaluate-test", action="store_true")
    ap.add_argument("--camera-aligned", action="store_true")
    ap.add_argument("--horizon", type=float, default=0.2)
    args = ap.parse_args()
    global LABELS, TARGET_S
    LABELS = args.labels
    TARGET_S = args.horizon
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = int(round((1.50 * TARGET_S + 2 * args.context) * HZ))
    index = json.loads((DATA / LABELS / "index.json").read_text(encoding="utf-8"))
    indexed = {row["name"] for row in index if row.get("usable", 0) >= 30}
    split_path = DATA / args.split_file
    if split_path.is_file():
        split = json.loads(split_path.read_text(encoding="utf-8"))
        train_names = [n for n in split["train"] if n in indexed]
        val_names = [n for n in split["val"] if n in indexed]
        test_names = [n for n in split["test"] if n in indexed]
    else:
        train_names = [n for n in indexed if n not in (TEST_NAME, VAL_NAME)]
        val_names = [VAL_NAME]
        test_names = [TEST_NAME]
    names = train_names + val_names + (test_names if args.evaluate_test else [])
    data = {}
    for name in names:
        got = build(name, args.context, length, camera_aligned=args.camera_aligned)
        if got is not None:
            data[name] = got
            print(json.dumps({"built": name, "n": int(len(got[0]))}), flush=True)
    Xtr, Ytr = _join(data, train_names)
    Xval, Yval = _join(data, val_names)
    scale = 0.01
    net = SeqNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    Xt = torch.from_numpy(Xtr)
    Yt = torch.from_numpy(Ytr / scale)
    best, best_state, bad = None, None, 0
    for epoch in range(args.epochs):
        net.train()
        perm = torch.randperm(len(Xt))
        total = 0.0
        for k in range(0, len(perm), 64):
            b = perm[k:k + 64]
            x = Xt[b].clone()
            x[:, :, 0:3] += 0.01 * torch.randn(len(b), 1, 3)
            x[:, :, 3:6] += 0.01 * torch.randn(len(b), 1, 3)
            x[:, :, 0:6] += 0.003 * torch.randn_like(x[:, :, 0:6])
            x = x.to(device)
            y = Yt[b].to(device)
            loss = torch.nn.functional.smooth_l1_loss(net(x), y, beta=0.2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(b)
        sched.step()
        val = _metrics(_predict(net, Xval, device) * scale, Yval)
        print(json.dumps({"epoch": epoch, "loss": round(total / len(Xt), 4), "val": val}), flush=True)
        if best is None or val["err_mm"] < best:
            best = val["err_mm"]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 15:
                break
    net.load_state_dict(best_state)
    report = {
        "context_s": args.context,
        "horizon_s": TARGET_S,
        "seed": args.seed,
        "best_val_mm": best,
        "labels": LABELS,
        "camera_aligned": args.camera_aligned,
        "train_sessions": train_names,
        "val_sessions": val_names,
        "test_sessions": test_names,
        "train_pairs": int(len(Xtr)),
        "val_pairs": int(len(Xval)),
    }
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    horizon_tag = "" if abs(TARGET_S - 0.2) < 1e-9 else f"_h{TARGET_S:g}"
    tag = f"ctx{args.context:g}{horizon_tag}{'_cam' if args.camera_aligned else ''}_s{args.seed}"
    val_pred = _predict(net, Xval, device) * scale
    report["val"] = _metrics(val_pred, Yval)
    report["val_by_session"] = {}
    for name in val_names:
        pred = _predict(net, data[name][0], device) * scale
        report["val_by_session"][name] = _metrics(pred, data[name][1])
    arrays = {"val_pred": val_pred, "val_y": Yval}
    if args.evaluate_test:
        # Use only after context and seeds have been selected on validation.
        Xtest, Ytest = _join(data, test_names)
        test_pred = _predict(net, Xtest, device) * scale
        report["test"] = _metrics(test_pred, Ytest)
        report["test_pairs"] = int(len(Xtest))
        report["test_by_session"] = {}
        for name in test_names:
            pred = _predict(net, data[name][0], device) * scale
            report["test_by_session"][name] = _metrics(pred, data[name][1])
        arrays.update({"test_pred": test_pred, "test_y": Ytest})
        print(json.dumps({"val": report["val"], "test": report["test"]}), flush=True)
    else:
        print(json.dumps({"val": report["val"]}), flush=True)
    np.savez(out / f"pred_{tag}.npz", **arrays)
    torch.save(best_state, out / f"seq_{tag}.pt")
    (out / f"seq_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
