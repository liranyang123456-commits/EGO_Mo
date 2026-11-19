#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Parameter count and inference latency of every architecture, measured alike.

Each model runs on the same 16 held-out-test windows with gradients disabled,
after 20 warm-up passes, averaged over 200 timed passes with CUDA
synchronization. PhysNet time includes its analytic pre-integration features,
which are part of its inference; the public architectures receive the
already-gridded input tensor.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
import tools.train_physnet as phys
from tools.train_external_architectures import make_model

DATA = ROOT / "datasets"
TEST = "traj_20260923_023422"


def _time(fn, device, warm=20, reps=200):
    with torch.no_grad():
        for _ in range(warm):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / reps


def main(batch=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    report = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
              "batch": batch, "warmup": 20, "timed_passes": 200, "models": {}}

    data = phys.to_device(phys.build_group([TEST], context, horizon, length), device)
    b = phys.take(data, torch.arange(batch, device=device))
    model = phys.load_model(DATA / "physnet_cv" / "physnet_cvg_traj_20260923_015507_s0.pt", device)
    model.net.eval()
    sec = _time(lambda: model.stream(b), device)
    report["models"]["physnet"] = {
        "parameters": int(sum(p.numel() for p in model.net.parameters())),
        "ms_per_window": 1000 * sec / batch}

    X, _y = api.build(TEST, context, length)
    x = torch.from_numpy(X[:batch]).to(device)
    for arch in ("ronin_resnet", "ronin_lstm", "tlio_resnet", "imunet"):
        net = make_model(arch).to(device).eval()
        sec = _time(lambda: net(x), device)
        report["models"][arch] = {
            "parameters": int(sum(p.numel() for p in net.parameters())),
            "ms_per_window": 1000 * sec / batch}
    out = DATA / "latency_benchmark.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
