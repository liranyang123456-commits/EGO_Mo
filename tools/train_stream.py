#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""IMU stream in, camera pose stream out.

A window of `window` seconds of 200 Hz USB IMU produces one camera-frame
velocity every 0.05 s. Rotation inside the window is the USB gyro integral.
Position is the sum of those velocities rotated by that attitude. Every pair
of gyro-locked chessboard poses inside the window, 0.1 to 2 s apart,
supervises the relative displacement.
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

from ego_capture.mapping.inertial import exp_so3
from ego_capture.sync import load_imu
from tools.chain_lock import BODY_TO_CAM
from tools.train_trajectory import TEST_NAME, VAL_NAME

DATA = ROOT / "datasets"
HZ = 200
DT = 1.0 / HZ
NODE = 10
MIN_GAP = 0.10
MAX_GAP = 2.0
BUCKETS = ((0.12, 0.30), (0.40, 0.60), (0.90, 1.10), (1.80, 2.00))


def load_session(name: str) -> dict:
    usb_t, usb = load_imu(DATA / name / "imu_stream.csv")
    grid = np.arange(usb_t[0], usb_t[-1], DT)
    feat = np.stack([np.interp(grid, usb_t, usb[:, c]) for c in range(6)], 1)
    gyro = feat[:, 3:6] * (np.pi / 180.0)
    still = np.linalg.norm(feat[:, 3:6], axis=1) < 3.0
    bg = gyro[still].mean(0) if int(still.sum()) >= 50 else np.zeros(3)
    w_cam = (gyro - bg) @ BODY_TO_CAM.T
    C = np.empty((len(grid), 3, 3))
    C[0] = np.eye(3)
    for k in range(1, len(grid)):
        C[k] = C[k - 1] @ exp_so3(w_cam[k] * DT)
    x = np.zeros((len(grid), 6), np.float32)
    x[:, 0:3] = feat[:, 0:3]
    x[:, 3:6] = feat[:, 3:6] / 100.0
    locked = np.load(DATA / "pose_gt_locked" / f"{name}.npz")
    raw = np.load(DATA / "pose_gt" / f"{name}.npz")
    delay = float(raw["delay_usb"][0])
    use = locked["usable"] == 1
    t_lab = locked["t"][use] - delay
    k_lab = np.round((t_lab - grid[0]) / DT).astype(np.int64)
    keep = (k_lab >= 0) & (k_lab < len(grid))
    return {
        "x": x,
        "C": C.astype(np.float32),
        "k": k_lab[keep],
        "R": locked["R"][use][keep].astype(np.float64),
        "p": locked["p"][use][keep].astype(np.float64),
    }


def windows(sess: dict, length: int, stride: int, center: tuple[float, float] | None, cap: int, rng):
    """Each item: input, window attitude, pair sample indices, true displacement, pair gap."""
    out = []
    n = len(sess["x"])
    k = sess["k"]
    for s in range(0, n - length, stride):
        inside = np.flatnonzero((k >= s) & (k < s + length))
        if len(inside) < 2:
            continue
        a_idx, b_idx = np.meshgrid(inside, inside, indexing="ij")
        a_idx = a_idx.reshape(-1)
        b_idx = b_idx.reshape(-1)
        gap = (k[b_idx] - k[a_idx]) * DT
        m = (gap >= MIN_GAP) & (gap <= MAX_GAP)
        if center is not None:
            mid = 0.5 * (k[a_idx] + k[b_idx]) - s
            m &= (mid >= center[0] * length) & (mid < center[1] * length)
        a_idx, b_idx, gap = a_idx[m], b_idx[m], gap[m]
        if len(a_idx) == 0:
            continue
        if cap and len(a_idx) > cap:
            pick = rng.choice(len(a_idx), cap, replace=False)
            a_idx, b_idx, gap = a_idx[pick], b_idx[pick], gap[pick]
        dp = np.einsum("nji,nj->ni", sess["R"][a_idx], sess["p"][b_idx] - sess["p"][a_idx])
        C0 = sess["C"][s]
        att = np.einsum("ji,njk->nik", C0, sess["C"][s:s + length])
        x = sess["x"][s:s + length].copy()
        x[:, 0:3] -= x[:, 0:3].mean(0, keepdims=True)
        out.append({
            "x": x,
            "att": att.astype(np.float32),
            "a": (k[a_idx] - s).astype(np.int64),
            "b": (k[b_idx] - s).astype(np.int64),
            "dp": dp.astype(np.float32),
            "gap": gap.astype(np.float32),
        })
    return out


def collate(items, pairs: int):
    B = len(items)
    L = items[0]["x"].shape[0]
    x = torch.zeros(B, L, 6)
    att = torch.zeros(B, L, 3, 3)
    a = torch.zeros(B, pairs, dtype=torch.long)
    b = torch.zeros(B, pairs, dtype=torch.long)
    dp = torch.zeros(B, pairs, 3)
    gap = torch.zeros(B, pairs)
    mask = torch.zeros(B, pairs)
    for i, it in enumerate(items):
        n = min(pairs, len(it["a"]))
        x[i] = torch.from_numpy(it["x"])
        att[i] = torch.from_numpy(it["att"])
        a[i, :n] = torch.from_numpy(it["a"][:n])
        b[i, :n] = torch.from_numpy(it["b"][:n])
        dp[i, :n] = torch.from_numpy(it["dp"][:n])
        gap[i, :n] = torch.from_numpy(it["gap"][:n])
        mask[i, :n] = 1.0
    return x, att, a, b, dp, gap, mask


class StreamNet(nn.Module):
    def __init__(self, width: int = 96, dropout: float = 0.1) -> None:
        super().__init__()
        self.inp = nn.Conv1d(6, width, 5, padding=2)
        blocks = []
        for d in (1, 2, 4, 8):
            blocks.append(nn.Sequential(
                nn.Conv1d(width, width, 5, padding=2 * d, dilation=d), nn.GELU(), nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)
        self.down = nn.Conv1d(width, width, NODE, stride=NODE)
        self.gru = nn.GRU(width, width, num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Linear(2 * width, 3)

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for blk in self.blocks:
            h = h + blk(h)
        h = self.down(h).transpose(1, 2)
        h, _ = self.gru(h)
        return self.head(h) * 0.05


def positions(v_node, att):
    """Node velocity (camera frame at that time) to window-start camera positions per sample."""
    v = v_node.repeat_interleave(NODE, dim=1)
    L = att.shape[1]
    if v.shape[1] < L:
        v = torch.cat([v, v[:, -1:].expand(-1, L - v.shape[1], -1)], dim=1)
    v = v[:, :L]
    world = torch.einsum("blij,blj->bli", att, v)
    P = torch.cumsum(world * DT, dim=1)
    return torch.cat([torch.zeros_like(P[:, :1]), P[:, :-1]], dim=1)


def pair_pred(v_node, att, a, b):
    P = positions(v_node, att)
    idx_a = a.unsqueeze(-1).expand(-1, -1, 3)
    idx_b = b.unsqueeze(-1).expand(-1, -1, 3)
    Pa = torch.gather(P, 1, idx_a)
    Pb = torch.gather(P, 1, idx_b)
    Ra = att[torch.arange(att.shape[0]).unsqueeze(-1), a]
    return torch.einsum("bnji,bnj->bni", Ra, Pb - Pa)


def evaluate(net, items, device, pairs):
    net.eval()
    preds, truth, gaps = [], [], []
    with torch.no_grad():
        for k in range(0, len(items), 32):
            batch = items[k:k + 32]
            P = max(len(it["a"]) for it in batch)
            x, att, a, b, dp, gap, mask = collate(batch, P)
            v = net(x.to(device))
            pred = pair_pred(v, att.to(device), a.to(device), b.to(device)).cpu()
            m = mask.bool()
            preds.append(pred[m].numpy())
            truth.append(dp[m].numpy())
            gaps.append(gap[m].numpy())
    pred = np.concatenate(preds)
    y = np.concatenate(truth)
    g = np.concatenate(gaps)
    rep = {}
    for lo, hi in BUCKETS:
        m = (g >= lo) & (g <= hi)
        if int(m.sum()) < 5:
            continue
        err = np.linalg.norm(pred[m] - y[m], axis=1)
        mag = np.linalg.norm(y[m], axis=1)
        resid = np.sum((pred[m] - y[m]) ** 2)
        base = np.sum((y[m] - y[m].mean(0)) ** 2)
        fast = mag >= np.quantile(mag, 0.67)
        rep[f"{lo:.2f}-{hi:.2f}s"] = {
            "n": int(m.sum()),
            "zero_mm": round(float(mag.mean() * 1000), 2),
            "err_mm": round(float(err.mean() * 1000), 2),
            "fast_zero_mm": round(float(mag[fast].mean() * 1000), 2),
            "fast_mm": round(float(err[fast].mean() * 1000), 2),
            "r2": round(float(1 - resid / max(base, 1e-12)), 3),
        }
    return rep, (pred, y, g)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pairs", type=int, default=96)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    length = int(round(args.window * HZ)) // NODE * NODE
    index = json.loads((DATA / "pose_gt_locked" / "index.json").read_text(encoding="utf-8"))
    names = [row["name"] for row in index if row.get("usable", 0) >= 30]
    sessions = {n: load_session(n) for n in names}
    train_names = [n for n in names if n not in (VAL_NAME, TEST_NAME)]
    train_items = []
    for n in train_names:
        train_items += windows(sessions[n], length, HZ // 4, None, args.pairs, rng)
    held = {}
    for n in (VAL_NAME, TEST_NAME):
        held[n] = windows(sessions[n], length, HZ, (0.375, 0.625), 0, rng)
    print(json.dumps({"train_windows": len(train_items), "val_windows": len(held[VAL_NAME]),
                      "test_windows": len(held[TEST_NAME])}), flush=True)
    net = StreamNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=2e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best, best_state, bad = None, None, 0
    for epoch in range(args.epochs):
        net.train()
        order = rng.permutation(len(train_items))
        total = 0.0
        for k in range(0, len(order), 16):
            batch = [train_items[i] for i in order[k:k + 16]]
            x, att, a, b, dp, gap, mask = collate(batch, args.pairs)
            x[:, :, 0:3] += 0.01 * torch.randn(x.shape[0], 1, 3)
            x[:, :, 3:6] += 0.01 * torch.randn(x.shape[0], 1, 3)
            x = x + 0.003 * torch.randn_like(x)
            x, att, a, b, dp, gap, mask = (t.to(device) for t in (x, att, a, b, dp, gap, mask))
            pred = pair_pred(net(x), att, a, b)
            w = mask / gap.clamp(min=0.2)
            per = torch.nn.functional.smooth_l1_loss(pred / 0.01, dp / 0.01, beta=0.2, reduction="none").sum(-1)
            loss = (per * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach())
        sched.step()
        rep, _ = evaluate(net, held[VAL_NAME], device, args.pairs)
        score = rep.get("0.12-0.30s", {}).get("err_mm")
        long = rep.get("0.90-1.10s", {}).get("err_mm")
        print(json.dumps({"epoch": epoch, "loss": round(total, 2), "val": rep}), flush=True)
        key = score + (long or 0.0) / 4.0
        if best is None or key < best:
            best = key
            best_state = {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 12:
                break
    net.load_state_dict(best_state)
    out = DATA / "traj_run_v8"
    out.mkdir(parents=True, exist_ok=True)
    tag = f"w{args.window:g}_s{args.seed}"
    report = {"window_s": args.window, "seed": args.seed}
    saved = {}
    for n in (VAL_NAME, TEST_NAME):
        rep, (pred, y, g) = evaluate(net, held[n], device, args.pairs)
        report[n] = rep
        saved[n] = (pred, y, g)
        print(json.dumps({n: rep}), flush=True)
    torch.save(best_state, out / f"stream_{tag}.pt")
    np.savez(out / f"pred_{tag}.npz",
             val_pred=saved[VAL_NAME][0], val_y=saved[VAL_NAME][1], val_g=saved[VAL_NAME][2],
             test_pred=saved[TEST_NAME][0], test_y=saved[TEST_NAME][1], test_g=saved[TEST_NAME][2])
    (out / f"stream_{tag}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
