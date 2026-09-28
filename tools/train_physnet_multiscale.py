#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Multi-scale windows: does a shorter horizon reduce the initial-velocity term?

The unobservable part of a window is v(t_a) * dt. A shorter window has a
smaller dt, so the missing term is smaller. This trains PhysNet at 0.5, 1 and
3 s on the same corpus and compares the error per unit time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet_v3 as v3  # noqa: E402
import tools.train_seq as api  # noqa: E402

DATA = ROOT / "datasets"
PROTO = DATA / "synthetic_protocol"


def train_eval(horizon: float, seed: int, corpus: str, device) -> dict:
    api.LABELS, api.TARGET_S = "pose_gt_raw", horizon
    v3.BIAS_MODE = "none"
    root = Path((PROTO / corpus / "latest.txt").read_text(encoding="utf-8").strip())
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    context = 2.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))

    train_names = [(root / e["path"]).resolve() for e in manifest["imu"]["train"]]
    val_names = [(root / e["path"]).resolve() for e in manifest["imu"]["val"]]
    test_names = [(root / e["path"]).resolve() for e in manifest["imu"]["test"]]

    rng = np.random.default_rng(seed)
    train = v3.build_group(train_names, context, horizon, length, cap=120, rng=rng)
    val = v3.build_group(val_names, context, horizon, length, cap=120, rng=rng)
    test = v3.build_group(test_names, context, horizon, length, cap=160,
                          rng=np.random.default_rng(123))
    train_t = v3.to_device(train, device)
    val_t = v3.to_device(val, device)
    test_t = v3.to_device(test, device)

    anchor = int(round(context * v3.HZ))
    feat = v3.Featurizer(anchor, horizon, use_phys=True)
    feat.fit(v3.take(train_t, torch.arange(0, len(train["y"]),
                                           max(1, len(train["y"]) // 2000), device=device)))
    net = v3.PhysNet(width=96, lever=True, still=True, gate=True).to(device)
    model = v3.Model(net, feat, None, anchor)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
    best = None
    best_err = np.inf
    bad = 0
    n = len(train["y"])
    for epoch in range(15):
        net.train()
        order = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, 32):
            idx = order[i:i + 32]
            batch = v3.take(train_t, idx)
            pred = model.stream(batch)
            y = torch.from_numpy(train["y"][idx.cpu().numpy()]).to(device)
            loss = torch.nn.functional.smooth_l1_loss(pred[:, -1], y, beta=0.5)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        net.eval()
        with torch.no_grad():
            pred = model.predict(val_t)
            err = np.linalg.norm(pred - val["y"], axis=1).mean() * 1000
        if err < best_err:
            best_err = err
            best = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= 10:
            break
    net.load_state_dict(best)
    net.eval()
    pred = model.predict(test_t)
    err = np.linalg.norm(pred - test["y"], axis=1).mean() * 1000
    zero = np.linalg.norm(test["y"], axis=1).mean() * 1000
    return {"horizon_s": horizon, "err_mm": round(float(err), 2),
            "zero_mm": round(float(zero), 2),
            "err_per_s": round(float(err) / horizon, 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="pause_4")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for horizon in (0.5, 1.0, 3.0):
        for seed in args.seeds:
            r = train_eval(horizon, seed, args.corpus, device)
            print(json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
