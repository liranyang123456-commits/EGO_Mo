#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Matched factorial ablation on the digital-twin corpus.

The synthetic split is where PhysNet differs most from the public
architectures, so the contributions of input representation, supervision and
backbone are separated there. Every configuration sees the same synthetic
pairs (the pair subsampling of tools/train_physnet.py eval), the same
augmentation, optimiser, epoch budget and two seeds.

Factors
  input:       body8   demeaned camera-frame specific force, angular rate,
                       interval and validity masks (no rotation, no integrals)
               anchor24 anchor-frame pre-integrated channels (Featurizer)
  supervision: end     loss on the pair end point only
               dense   loss on every chessboard time in the window
               gate    dense + stillness gate (PhysNet trunk only)
  backbone:    physnet (Conv-BiGRU trunk with velocity integration and lever arm),
               imunet, ronin_resnet, tlio_resnet (regression heads)

The synthetic-only models are also scored, without any real update, on the
held-out real recording and on the prospective recording T1 ("real direct").

    python tools/run_synthetic_factorial.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
import tools.train_seq as api
from tools.run_reviewer_revisions import AnchorIMUNet, AnchorRoNINResNet, AnchorTLIOResNet

DATA = ROOT / "datasets"
CORPUS = DATA / "synthetic_corpus" / "corpus_20260924_181622"
OUT = DATA / "synthetic_factorial.json"
REAL_TEST = "traj_20260923_023422"
REAL_T1 = "traj_20260926_143837"

CONFIGS = [
    ("physnet", "body8", "end"),
    ("physnet", "body8", "dense"),
    ("physnet", "anchor24", "end"),
    ("physnet", "anchor24", "dense"),
    ("physnet", "anchor24", "gate"),
    ("imunet", "body8", "end"),
    ("imunet", "anchor24", "end"),
    ("ronin_resnet", "body8", "end"),
    ("ronin_resnet", "anchor24", "end"),
    ("tlio_resnet", "body8", "end"),
    ("tlio_resnet", "anchor24", "end"),
]

AUG = {
    "acc_bias": 0.005 * phys.G0, "acc_noise": 0.002 * phys.G0,
    "gyro_bias": np.radians(0.2), "gyro_noise": np.radians(0.1),
    "ext_rad": np.radians(1.0), "jitter": 0, "warp": 1.0, "amp": (1.0, 1.0),
}


class BodyFeaturizer:
    """Eight body-frame channels with the augmentation of phys.Featurizer."""

    binary = (6, 7)

    def __init__(self, anchor: int):
        self.anchor = anchor
        self.mean = None
        self.std = None

    def raw(self, batch: dict, aug: dict | None = None):
        acc, gyro, quat = batch["acc"], batch["gyro"], batch["quat"]
        valid, interval = batch["valid"], batch["interval"]
        B, T, _ = acc.shape
        dev = acc.device
        R = phys.quat_to_mat(quat)
        if aug:
            acc = acc + aug["acc_bias"] * torch.randn(B, 1, 3, device=dev)
            acc = acc + aug["acc_noise"] * torch.randn_like(acc)
            bg = aug["gyro_bias"] * torch.randn(B, 1, 3, device=dev)
            gyro = gyro + bg + aug["gyro_noise"] * torch.randn_like(gyro)
            tk = (torch.arange(T, device=dev, dtype=acc.dtype) - self.anchor) / phys.HZ
            R = R @ phys.rodrigues(bg * tk[None, :, None])
            if aug.get("ext_rad", 0.0) > 0:
                eps = phys.rodrigues(aug["ext_rad"] * torch.randn(B, 3, device=dev))[:, None]
                R = eps @ R @ eps.transpose(-1, -2)
                acc = torch.einsum("bij,btj->bti", eps[:, 0], acc)
                gyro = torch.einsum("bij,btj->bti", eps[:, 0], gyro)
        vmask = valid[..., None]
        mean = (acc * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
        lin = (acc - mean[:, None]) * vmask
        feats = torch.cat((lin, gyro * vmask, interval[..., None], valid[..., None]), -1)
        return feats, R, torch.zeros_like(lin)

    def fit(self, batch: dict):
        feats, _, _ = self.raw(batch)
        sel = feats[batch["valid"] > 0.5]
        self.mean = sel.mean(0)
        self.std = sel.std(0).clamp(min=1e-6)
        for c in self.binary:
            self.mean[c], self.std[c] = 0.0, 1.0

    def __call__(self, batch: dict, aug: dict | None = None):
        feats, R, v = self.raw(batch, aug)
        return ((feats - self.mean) / self.std).clamp(-8.0, 8.0), R, v


def make_net(backbone: str, in_ch: int, supervision: str):
    if backbone == "physnet":
        gate = supervision == "gate"
        return phys.PhysNet(width=96, in_ch=in_ch, lever=True, still=gate, gate=gate)
    return {"imunet": AnchorIMUNet, "ronin_resnet": AnchorRoNINResNet,
            "tlio_resnet": AnchorTLIOResNet}[backbone](in_ch)


def forward_end(net, backbone, feats, R, v, batch, anchor):
    if backbone == "physnet":
        d, _vel, _still, _lv = net(feats, R, anchor, v, batch["valid"])
        return phys.sample_stream(d, batch["tau_b"][:, None])[:, 0]
    return net(feats)


def predict(net, backbone, feat, data_t, anchor):
    net.eval()
    out = []
    n = len(data_t["y"])
    with torch.no_grad():
        for k in range(0, n, 128):
            idx = torch.arange(k, min(k + 128, n), device=data_t["y"].device)
            b = phys.take(data_t, idx)
            feats, R, v = feat(b)
            out.append(forward_end(net, backbone, feats, R, v, b, anchor).cpu().numpy())
    return np.concatenate(out)


def train_one(backbone, supervision, feat, train_t, val_t, val_y, anchor, seed, device,
              epochs, patience):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1000)
    in_ch = 8 if isinstance(feat, BodyFeaturizer) else phys.FEATURES
    net = make_net(backbone, in_ch, supervision).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    dense_w = 0.0 if supervision == "end" else 1.0
    still_w = 1.0 if supervision == "gate" else 0.0
    n = len(train_t["y"])
    best, best_state, best_epoch, bad = None, None, -1, 0
    for epoch in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(n)).to(device)
        for k in range(0, n, 64):
            b = phys.take(train_t, order[k:k + 64])
            feats, R, v = feat(b, AUG)
            if backbone == "physnet":
                d, vel, still, logvar = net(feats, R, anchor, v, b["valid"])
                loss = phys.loss_fn(d, vel, still, logvar, b, dense_w, 1.0, 0.0, still_w, 0.0)
            else:
                loss = F.smooth_l1_loss(net(feats) * 100.0, b["y"] * 100.0, beta=0.5)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()
        err = api._metrics(predict(net, backbone, feat, val_t, anchor), val_y)["err_mm"]
        if best is None or err < best:
            best, best_epoch, bad = err, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return net, best_epoch, params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--only", nargs="*", default=[], help="subset of config keys")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    out_json = args.out
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * phys.HZ))
    anchor = int(round(context * phys.HZ))

    manifest = json.loads((CORPUS / "manifest.json").read_text(encoding="utf-8"))

    def paths(group):
        return [phys._corpus_path(CORPUS, e["path"]) for e in manifest["imu"][group]]

    train_rng = np.random.default_rng(0)
    train = phys.build_group(paths("train"), context, horizon, length, cap=120, rng=train_rng)
    val = phys.build_group(paths("val"), context, horizon, length, cap=120, rng=train_rng)
    eval_rng = np.random.default_rng(123)
    test = phys.build_group(paths("test"), context, horizon, length, cap=160, rng=eval_rng)
    ood = phys.build_group(paths("ood"), context, horizon, length, cap=160, rng=eval_rng)
    real_test = phys.build_group([REAL_TEST], context, horizon, length)
    real_t1 = phys.build_group([REAL_T1], context, horizon, length)
    tensors = {name: phys.to_device(d, device)
               for name, d in (("train", train), ("val", val), ("test", test), ("ood", ood),
                               ("real_test", real_test), ("real_t1", real_t1))}

    report = json.loads(out_json.read_text(encoding="utf-8")) if out_json.is_file() else {}
    report["protocol"] = {
        "corpus": str(CORPUS.relative_to(ROOT)),
        "pairs": {k: int(len(d["y"])) for k, d in (("train", train), ("val", val),
                                                   ("test", test), ("ood", ood))},
        "zero_motion_mm": {k: api._metrics(np.zeros_like(d["y"]), d["y"])["err_mm"]
                           for k, d in (("val", val), ("test", test), ("ood", ood),
                                        ("real_test", real_test), ("real_t1", real_t1))},
        "optimizer": "AdamW lr 1e-3 wd 1e-2, cosine, batch 64",
        "epochs": args.epochs, "patience": args.patience, "seeds": args.seeds,
        "augmentation": {k: (list(v) if isinstance(v, tuple) else float(v)) for k, v in AUG.items()},
    }
    print(json.dumps(report["protocol"]), flush=True)

    sub = torch.arange(0, len(train["y"]), max(1, len(train["y"]) // 2000), device=device)
    featurizers = {"anchor24": phys.Featurizer(anchor, horizon, use_phys=True),
                   "body8": BodyFeaturizer(anchor)}
    for f in featurizers.values():
        f.fit(phys.take(tensors["train"], sub))

    arrays = {}
    npz_path = out_json.with_suffix(".npz")
    if npz_path.is_file():
        arrays = dict(np.load(npz_path))
    for backbone, rep, sup in CONFIGS:
        key = f"{backbone}_{rep}_{sup}"
        if args.only and key not in args.only:
            continue
        if key in report.get("results", {}):
            print(f"skip {key}", flush=True)
            continue
        start = time.time()
        feat = featurizers[rep]
        preds = {"val": [], "test": [], "ood": [], "real_test": [], "real_t1": []}
        epochs, params = [], None
        for seed in args.seeds:
            net, best_epoch, params = train_one(backbone, sup, feat, tensors["train"], tensors["val"],
                                                val["y"], anchor, seed, device, args.epochs,
                                                args.patience)
            epochs.append(best_epoch)
            for split in preds:
                preds[split].append(predict(net, backbone, feat, tensors[split], anchor))
        row = {"backbone": backbone, "input": rep, "supervision": sup,
               "parameters": int(params), "best_epochs": epochs,
               "minutes": round((time.time() - start) / 60, 1)}
        ref = {"val": val["y"], "test": test["y"], "ood": ood["y"],
               "real_test": real_test["y"], "real_t1": real_t1["y"]}
        for split, p in preds.items():
            row[f"{split}_single_seed_mm"] = [api._metrics(x, ref[split])["err_mm"] for x in p]
            row[split] = api._metrics(np.mean(p, axis=0), ref[split])
            arrays[f"{key}__{split}"] = np.mean(p, axis=0)
        report.setdefault("results", {})[key] = row
        out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        arrays.update({f"y__{s}": ref[s] for s in ref})
        np.savez(npz_path, **arrays)
        print(json.dumps({key: {k: row[k] for k in ("parameters", "minutes")} |
                          {s: row[s]["err_mm"] for s in ref}}), flush=True)


if __name__ == "__main__":
    main()
