#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic pretraining followed by real-data fine-tuning.

The sealed real test split is never loaded here. Model files use the same name
as train_seq.py so eval_seq_ensemble.py can evaluate a validation-selected
hybrid ensemble once.
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
import tools.train_seq as api

DATA = ROOT / "datasets"


def _synthetic_build(
    session: Path,
    context: float,
    length: int,
    horizon: float,
    cap: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    gt = np.load(session / "ground_truth.npz")
    usb_t, usb = load_imu(session / "imu_stream.csv")
    t = gt["t"].astype(np.float64)
    candidates = []
    for i in range(1, len(t)):
        prev = np.arange(i)
        j = int(prev[np.argmin(np.abs(t[prev] - (t[i] - horizon)))])
        dt = float(t[i] - t[j])
        if not (0.60 * horizon <= dt <= 1.50 * horizon):
            continue
        if t[j] - context < usb_t[0] or t[i] + context > usb_t[-1]:
            continue
        candidates.append((j, i))
    if cap and len(candidates) > cap:
        selected = np.sort(rng.choice(len(candidates), cap, replace=False))
        candidates = [candidates[k] for k in selected]
    X, Y = [], []
    for j, i in candidates:
        feat = api._grid(usb_t, usb, float(t[j]), float(t[i]), context)
        if len(feat) < length:
            feat = np.vstack((feat, np.zeros((length - len(feat), 8), np.float32)))
        Rj = gt["R"][j].astype(np.float64)
        dp = Rj.T @ (gt["p"][i] - gt["p"][j])
        X.append(feat[:length])
        Y.append(dp.astype(np.float32))
    if not X:
        return None
    return np.stack(X), np.stack(Y)


def _predict(net, X, device):
    net.eval()
    out = []
    with torch.no_grad():
        for k in range(0, len(X), 128):
            out.append(net(torch.from_numpy(X[k:k + 128]).to(device)).cpu().numpy())
    return np.concatenate(out)


def _train_phase(
    net,
    X,
    Y,
    Xval,
    Yval,
    device,
    epochs,
    lr,
    patience,
    seed,
    label,
):
    rng = np.random.default_rng(seed)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y / 0.01)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    best, best_state, best_epoch, bad = None, None, None, 0
    for epoch in range(epochs):
        net.train()
        order = rng.permutation(len(X))
        total = 0.0
        for k in range(0, len(order), 64):
            idx = torch.from_numpy(order[k:k + 64])
            x = Xt[idx].clone()
            x[:, :, 0:3] += 0.012 * torch.randn(len(idx), 1, 3)
            x[:, :, 3:6] += 0.012 * torch.randn(len(idx), 1, 3)
            x[:, :, 0:6] += 0.004 * torch.randn_like(x[:, :, 0:6])
            x, y = x.to(device), Yt[idx].to(device)
            loss = nn.functional.smooth_l1_loss(net(x), y, beta=0.2)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        sched.step()
        pred = _predict(net, Xval, device) * 0.01
        val = api._metrics(pred, Yval)
        print(json.dumps({
            "phase": label, "epoch": epoch,
            "loss": round(total / len(X), 4), "val": val,
        }), flush=True)
        if best is None or val["err_mm"] < best:
            best = val["err_mm"]
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    return best_epoch, best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--context", type=float, default=2.0)
    ap.add_argument("--horizon", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-cap", type=int, default=120)
    ap.add_argument("--pretrain-epochs", type=int, default=35)
    ap.add_argument("--finetune-epochs", type=int, default=60)
    ap.add_argument("--finetune-lr", type=float, default=3e-4)
    ap.add_argument("--replay", type=float, default=0.0)
    ap.add_argument("--reset-head", action="store_true")
    ap.add_argument("--skip-finetune", action="store_true")
    ap.add_argument("--out", default="traj_run_v15")
    ap.add_argument("--split-file", default="trajectory_split_20260924.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    api.LABELS = "pose_gt_raw"
    api.TARGET_S = args.horizon
    length = int(round((1.5 * args.horizon + 2 * args.context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    synth = {}
    for split_name in ("train", "val"):
        parts = []
        for entry in manifest["imu"][split_name]:
            got = _synthetic_build(
                args.corpus / entry["path"],
                args.context, length, args.horizon,
                args.synthetic_cap, rng,
            )
            if got is not None:
                parts.append(got)
        synth[split_name] = (
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
        )
        print(json.dumps({"synthetic": split_name, "pairs": len(synth[split_name][0])}), flush=True)

    split = json.loads((DATA / args.split_file).read_text(encoding="utf-8"))
    real = {}
    for group in ("train", "val"):
        parts = [api.build(name, args.context, length) for name in split[group]]
        real[group] = (
            np.concatenate([part[0] for part in parts if part is not None]),
            np.concatenate([part[1] for part in parts if part is not None]),
        )
        print(json.dumps({"real": group, "pairs": len(real[group][0])}), flush=True)

    net = api.SeqNet().to(device)
    pre_epoch, pre_err = _train_phase(
        net,
        synth["train"][0], synth["train"][1],
        synth["val"][0], synth["val"][1],
        device, args.pretrain_epochs, 1e-3, 10, args.seed, "synthetic",
    )
    pretrained_state = {
        name: value.detach().cpu().clone()
        for name, value in net.state_dict().items()
    }
    if args.skip_finetune:
        out = DATA / args.out
        out.mkdir(parents=True, exist_ok=True)
        tag = f"ctx{args.context:g}_h{args.horizon:g}_p{args.pretrain_epochs}_preonly_s{args.seed}"
        real_val_pred = _predict(net, real["val"][0], device) * 0.01
        report = {
            "architecture": "conv_bigru_synthetic_zero_shot",
            "context_s": args.context,
            "horizon_s": args.horizon,
            "seed": args.seed,
            "synthetic_train_pairs": len(synth["train"][0]),
            "pretrain_best_epoch": pre_epoch,
            "synthetic_val_err_mm": pre_err,
            "real_val": api._metrics(real_val_pred, real["val"][1]),
        }
        torch.save(pretrained_state, out / f"pretrained_{tag}.pt")
        np.savez(
            out / f"zero_shot_pred_{tag}.npz",
            val_pred=real_val_pred,
            val_y=real["val"][1],
        )
        (out / f"zero_shot_metrics_{tag}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps({"final": report}, ensure_ascii=False), flush=True)
        return
    if args.reset_head:
        # Keep temporal IMU features but remove the synthetic displacement
        # calibration before adapting to the real sensor/domain.
        for module in net.head.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
    Xfine, Yfine = real["train"]
    replay_n = int(round(len(Xfine) * args.replay / max(1.0 - args.replay, 1e-6)))
    if replay_n:
        idx = rng.choice(len(synth["train"][0]), replay_n, replace=replay_n > len(synth["train"][0]))
        Xfine = np.concatenate((Xfine, synth["train"][0][idx]))
        Yfine = np.concatenate((Yfine, synth["train"][1][idx]))
    fine_epoch, fine_err = _train_phase(
        net,
        Xfine, Yfine,
        real["val"][0], real["val"][1],
        device, args.finetune_epochs, args.finetune_lr, 15, args.seed + 100, "real",
    )
    val_pred = _predict(net, real["val"][0], device) * 0.01
    report = {
        "architecture": "conv_bigru_synthetic_pretrain",
        "context_s": args.context,
        "horizon_s": args.horizon,
        "seed": args.seed,
        "replay": args.replay,
        "finetune_lr": args.finetune_lr,
        "reset_head": args.reset_head,
        "synthetic_train_pairs": len(synth["train"][0]),
        "real_train_pairs": len(real["train"][0]),
        "pretrain_best_epoch": pre_epoch,
        "pretrain_val_err_mm": pre_err,
        "finetune_best_epoch": fine_epoch,
        "finetune_val_err_mm": fine_err,
        "val": api._metrics(val_pred, real["val"][1]),
    }
    out = DATA / args.out
    out.mkdir(parents=True, exist_ok=True)
    replay_tag = f"r{args.replay:g}"
    tag = (
        f"ctx{args.context:g}_h{args.horizon:g}_{replay_tag}"
        f"_p{args.pretrain_epochs}_lr{args.finetune_lr:g}"
        f"{'_reset' if args.reset_head else ''}_s{args.seed}"
    )
    torch.save(net.state_dict(), out / f"seq_{tag}.pt")
    torch.save(pretrained_state, out / f"pretrained_{tag}.pt")
    np.savez(out / f"pred_{tag}.npz", val_pred=val_pred, val_y=real["val"][1])
    (out / f"metrics_{tag}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"final": report}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
