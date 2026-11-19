#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train residual dual-IMU → camera mapper on a board_bt session."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ego_capture.mapping import build_mapper, geodesic_loss
from ego_capture.mapping.dataset import WindowPack, build_windows, time_split
from ego_capture.mapping.prior import apply_residual
from ego_capture.session import write_json

DEFAULT_SESSION = ROOT / "datasets" / "calib_board_bt_20260922_124351"
DEFAULT_RIG = ROOT / "datasets" / "rig_state.json"
DEFAULT_OUT = ROOT / "datasets" / "mapping_run"


def _forward(mapper, kind: str, usb, bt):
    if kind == "mlp":
        return mapper(torch.cat([usb, bt], dim=-1))
    return mapper(usb, bt)


def _eval(mapper, kind: str, loader: DataLoader, device: torch.device) -> dict[str, float]:
    mapper.eval()
    geo_r, geo_t, pred_r, pred_t, n = 0.0, 0.0, 0.0, 0.0, 0
    with torch.no_grad():
        for batch in loader:
            usb = batch["usb"].to(device)
            bt = batch["bt"].to(device)
            R = batch["R"].to(device)
            t = batch["t"].to(device)
            R0 = batch["R_prior"].to(device)
            t0 = batch["t_prior"].to(device)
            R_res, t_res, _ = _forward(mapper, kind, usb, bt)
            Rp, tp = apply_residual(R0, t0, R_res, t_res)
            B = usb.size(0)
            pred_r += float(geodesic_loss(Rp, R).item()) * B
            pred_t += float((tp - t).norm(dim=1).mean().item()) * B
            geo_r += float(geodesic_loss(R0, R).item()) * B
            geo_t += float((t0 - t).norm(dim=1).mean().item()) * B
            n += B
    n = max(n, 1)
    return {
        "geo_rot_deg": geo_r / n * 180.0 / np.pi,
        "geo_t_mm": geo_t / n * 1000.0,
        "pred_rot_deg": pred_r / n * 180.0 / np.pi,
        "pred_t_mm": pred_t / n * 1000.0,
    }


def train_one(kind: str, pack: dict, train_set: WindowPack, val_set: WindowPack, test_set: WindowPack,
              out_dir: Path, epochs: int, device: torch.device) -> dict:
    seq_len = int(pack["usb"].shape[1])
    mapper = build_mapper(kind, seq_len=seq_len, hidden=128).to(device)
    opt = torch.optim.AdamW(mapper.parameters(), lr=1e-3, weight_decay=1e-4)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)
    best = 1e9
    best_state = None
    history = []
    I = torch.eye(3, device=device).unsqueeze(0)
    for epoch in range(1, epochs + 1):
        mapper.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            usb = batch["usb"].to(device)
            bt = batch["bt"].to(device)
            R = batch["R"].to(device)
            t = batch["t"].to(device)
            R0 = batch["R_prior"].to(device)
            t0 = batch["t_prior"].to(device)
            R_res, t_res, _ = _forward(mapper, kind, usb, bt)
            Rp, tp = apply_residual(R0, t0, R_res, t_res)
            loss = geodesic_loss(Rp, R) + 20.0 * (tp - t).pow(2).mean().sqrt()
            loss = loss + 0.02 * geodesic_loss(R_res, I.expand(usb.size(0), -1, -1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            opt.step()
            running += float(loss.item()) * usb.size(0)
            seen += usb.size(0)
        val = _eval(mapper, kind, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / max(seen, 1), **{f"val_{k}": v for k, v in val.items()}})
        score = val["pred_rot_deg"] + 0.02 * val["pred_t_mm"]
        if score < best:
            best = score
            best_state = {k: v.detach().cpu().clone() for k, v in mapper.state_dict().items()}
    if best_state:
        mapper.load_state_dict(best_state)
    test = _eval(mapper, kind, test_loader, device)
    ckpt = out_dir / f"{kind}.pt"
    torch.save({
        "kind": kind,
        "state_dict": mapper.state_dict(),
        "R_geo": pack["R_geo"],
        "t_geo": pack["t_geo"],
        "mean": train_set.mean,
        "std": train_set.std,
        "seq_len": seq_len,
        "label": "R_C(t)=R_prior(t) R_res,  R_prior from hand-eye + BT relative IMU",
        "test": test,
    }, ckpt)
    mapper.cpu()
    return {"kind": kind, "checkpoint": str(ckpt), "test": test, "best_val_score": best, "history_tail": history[-3:]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", default=str(DEFAULT_SESSION))
    parser.add_argument("--rig", default=str(DEFAULT_RIG))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seq-len", type=int, default=80)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = out / "windows.npz"
    if cache.exists():
        raw = np.load(cache, allow_pickle=False)
        pack = {k: raw[k] for k in raw.files}
    else:
        pack = build_windows(args.session, args.rig, seq_len=args.seq_len)
        np.savez(cache, **pack)
    n = int(pack["usb"].shape[0])
    tr, va, te = time_split(n)
    train_set = WindowPack(pack, tr)
    val_set = WindowPack(pack, va, train_set.mean, train_set.std)
    test_set = WindowPack(pack, te, train_set.mean, train_set.std)
    results = {
        "session": args.session,
        "n_windows": n,
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "device": str(device),
        "models": [],
    }
    print(f"windows={n} train/val/test={len(tr)}/{len(va)}/{len(te)} device={device}", flush=True)
    for kind in ("mlp", "gru", "gru_xattn"):
        print(f"training {kind} ...", flush=True)
        rec = train_one(kind, pack, train_set, val_set, test_set, out, args.epochs, device)
        results["models"].append(rec)
        t = rec["test"]
        print(
            f"  {kind}: geo {t['geo_rot_deg']:.2f}°/{t['geo_t_mm']:.1f}mm  "
            f"pred {t['pred_rot_deg']:.2f}°/{t['pred_t_mm']:.1f}mm",
            flush=True,
        )
    write_json(str(out / "metrics.json"), results)
    if os.path.isfile(args.rig):
        state = json.loads(Path(args.rig).read_text(encoding="utf-8"))
        best = min(results["models"], key=lambda m: m["test"]["pred_rot_deg"])
        state["mapping"] = {
            "kind": best["kind"],
            "checkpoint": best["checkpoint"],
            "test": best["test"],
            "formula": "R_C = R_geo R_res, t_C = t_geo + R_geo t_res",
        }
        write_json(args.rig, state)
    print("wrote", out / "metrics.json", flush=True)


if __name__ == "__main__":
    main()
