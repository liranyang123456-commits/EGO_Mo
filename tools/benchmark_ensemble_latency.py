#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-window (B=1) GPU latency of the frozen fold ensembles.

The accuracy figures of the held-out recordings come from sixteen fold models
per family, so the online latency of that operating point is measured here
next to a single model: PhysNet featurization plus inference per model, the
IMUNet forward pass per model, both run sequentially, with 20 warm-up and 200
timed passes under CUDA synchronization.

    python tools/benchmark_ensemble_latency.py
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

import tools.train_physnet as phys
import tools.train_seq as api
from tools.train_external_architectures import make_model

DATA = ROOT / "datasets"
TEST = "traj_20260923_023422"


def timed(fn, device, warm=20, reps=200) -> float:
    with torch.no_grad():
        for _ in range(warm):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(reps):
            fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - start) / reps * 1000.0


def main() -> None:
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = json.loads((DATA / "final_model_manifest.json").read_text(encoding="utf-8"))
    paths = [Path(r["path"]) for r in manifest["models_and_configs"]]
    physnet = [phys.load_model(p, device) for p in paths if p.name.startswith("physnet_cvg_")]
    imunet = []
    for p in paths:
        if p.name.startswith("imunet_cvx_"):
            net = make_model("imunet").to(device)
            net.load_state_dict(torch.load(p, map_location=device))
            imunet.append(net.eval())
    length = int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ))
    data_t = phys.to_device(phys.build_group([TEST], 2.0, 3.0, length), device)
    b1 = phys.take(data_t, torch.arange(1, device=device))
    X, _y = api.build(TEST, 2.0, length)
    x1 = torch.from_numpy(X[:1]).to(device)
    for m in physnet:
        m.net.eval()

    report = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "physnet_single_ms": round(timed(lambda: physnet[0].stream(b1), device), 3),
        "physnet_ensemble16_ms": round(timed(lambda: [m.stream(b1) for m in physnet], device), 3),
        "imunet_single_ms": round(timed(lambda: imunet[0](x1), device), 3),
        "imunet_ensemble16_ms": round(timed(lambda: [n(x1) for n in imunet], device), 3),
    }
    report["blend32_ms"] = round(timed(lambda: ([m.stream(b1) for m in physnet], [n(x1) for n in imunet]),
                                       device), 3)
    (DATA / "ensemble_latency.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
