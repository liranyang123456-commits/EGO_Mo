#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Context sweep of PhysNet scored on one common set of windows, with a strictly causal window.

Two properties of the sweep in tools/run_reviewer_revisions.py (Experiment 3)
are addressed here.

1. Its shortest window is padded with ten samples (50 ms) on each side, so it
   still waits 50 ms after t_b. Here a context of exactly zero is added: the
   window is [t_a, t_b), its last sample lies before t_b, and the output delay
   after t_b is computation only.
2. A pair (t_a, t_b) is admissible only if t_a - c and t_b + c lie inside the
   recording, so a shorter context admits more windows near the start and the
   end of each recording, where the camera is often at rest. The stored sweep
   scored every context on its own window set, which makes the errors of
   different contexts not comparable. Here every context is trained on its own
   admissible pairs, as before, and scored both on that set ("own") and on the
   windows admissible at c = 2 s ("common"), which every shorter context also
   contains.

Protocol otherwise identical to Experiment 3: gated PhysNet, curated real pool
(trajectory train + valid rigidity recordings), AdamW (lr 1e-3, weight decay
0.1), cosine schedule, up to 50 epochs, patience 12 on the context's own fixed
validation set, seeds 0 and 1, two-seed ensemble. Timing: mean CPU gridding
over 100 windows; GPU feature computation and forward pass at batch size 1
(20 warm-up passes, 200 timed, CUDA-synchronized).

    python tools/run_context_sweep_common.py --repeats 3
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
import tools.train_seq as api

DATA = ROOT / "datasets"
OUT = DATA / "context_sweep_common.json"
HORIZON = 3.0
CONTEXTS = (0.0, 0.05, 0.5, 1.0, 2.0)
REFERENCE = 2.0          # its admissible windows are the common set
AUG = {
    "acc_bias": 0.005 * phys.G0, "acc_noise": 0.002 * phys.G0,
    "gyro_bias": np.radians(0.2), "gyro_noise": np.radians(0.1),
    "ext_rad": np.radians(1.0), "jitter": 0, "warp": 1.0, "amp": (1.0, 1.0),
}


def keys_of(data: dict) -> list[tuple]:
    return [(str(s), int(a), int(b)) for s, (a, b) in zip(data["session"], data["pair"])]


def last_sample_offset_ms(data: dict) -> dict:
    """Last valid IMU sample of each window relative to t_b (negative: before t_b)."""
    off = (data["valid"].sum(1) - 1.0 - data["tau_b"]) / phys.HZ * 1000.0
    return {"min_ms": round(float(off.min()), 2), "max_ms": round(float(off.max()), 2),
            "windows_with_sample_at_or_after_tb": int((off >= 0).sum())}


def block_ci(err_a: np.ndarray, err_b: np.ndarray, t_pair: np.ndarray, n_boot: int = 2000) -> list[float]:
    """95% block-bootstrap interval of mean(err_b - err_a) in mm over 3-s blocks
    (as tools/summarize_matched_benchmark.py)."""
    rng = np.random.default_rng(0)
    blocks = np.floor((t_pair[:, 0] - t_pair[:, 0].min()) / 3.0).astype(int)
    ids = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in ids]
    diff = err_b - err_a
    stats = [diff[np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])].mean()
             for _ in range(n_boot)]
    return [round(float(v) * 1000, 2) for v in np.quantile(stats, (0.025, 0.975))]


def err_mm(pred: np.ndarray, y: np.ndarray) -> float:
    return round(float(np.linalg.norm(pred - y, axis=1).mean() * 1000), 2)


def run_context(context: float, device, names, common: dict) -> tuple[dict, dict]:
    train_names, val_names, test_names = names
    length = int(round((1.5 * HORIZON + 2 * context) * phys.HZ))
    anchor = int(round(context * phys.HZ))
    train = phys.build_group(train_names, context, HORIZON, length)
    val = phys.build_group(val_names, context, HORIZON, length)
    test = phys.build_group(test_names, context, HORIZON, length)
    train_t, val_t, test_t = (phys.to_device(x, device) for x in (train, val, test))

    sub = torch.arange(0, len(train["y"]), max(1, len(train["y"]) // 2000), device=device)
    feat = phys.Featurizer(anchor, HORIZON, use_phys=True)
    feat.fit(phys.take(train_t, sub))
    val_preds, test_preds, epochs = [], [], []
    for seed in (0, 1):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = phys.PhysNet(width=96, lever=True, still=True, gate=True).to(device)
        model = phys.Model(net, feat, None, anchor)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
        best, best_state, best_epoch, bad = None, None, 0, 0
        n = len(train["y"])
        for epoch in range(50):
            net.train()
            order = torch.from_numpy(rng.permutation(n)).to(device)
            for k in range(0, n, 64):
                b = phys.take(train_t, order[k:k + 64])
                d, vel, still, logvar = model.stream_full(b, AUG)
                loss = phys.loss_fn(d, vel, still, logvar, b, 1.0, 1.0, 0.0, 1.0, 0.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            m = api._metrics(model.predict(val_t), val["y"])["err_mm"]
            if best is None or m < best:
                best, best_epoch, bad = m, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 12:
                    break
        net.load_state_dict(best_state)
        val_preds.append(model.predict(val_t))
        test_preds.append(model.predict(test_t))
        epochs.append(best_epoch)
    val_p, test_p = np.mean(val_preds, axis=0), np.mean(test_preds, axis=0)

    ses = phys.Session(test_names[0])
    t_gt, _, _, us = phys._load_gt(test_names[0])
    pairs = phys.enumerate_pairs(t_gt, us, HORIZON, context, ses.t)
    t0 = time.perf_counter()
    for a, b in pairs[:100]:
        _ = ses.window(t_gt[a], t_gt[b], context, length, 1.0)
    cpu_grid_ms = (time.perf_counter() - t0) / 100 * 1000.0
    b1 = phys.take(test_t, torch.arange(1, device=device))
    with torch.no_grad():
        for _ in range(20):
            _ = model.stream(b1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        s = time.perf_counter()
        for _ in range(200):
            _ = model.stream(b1)
        if device.type == "cuda":
            torch.cuda.synchronize()
        gpu_b1_ms = (time.perf_counter() - s) / 200 * 1000.0

    # Restrict to the windows admissible at the reference context, in its order.
    picked = {}
    for split, data, pred in (("val", val, val_p), ("test", test, test_p)):
        index = {k: i for i, k in enumerate(keys_of(data))}
        missing = [k for k in common[split]["keys"] if k not in index]
        if missing:
            raise AssertionError(f"context {context:g}: {len(missing)} common {split} windows are not admissible")
        sel = np.array([index[k] for k in common[split]["keys"]])
        if not np.allclose(data["y"][sel], common[split]["y"], atol=1e-6):
            raise AssertionError(f"context {context:g}: labels of the common {split} windows differ")
        picked[split] = pred[sel]
    offs = last_sample_offset_ms(test)
    wait_ms = max(0.0, 1000.0 * context) if offs["windows_with_sample_at_or_after_tb"] else 0.0
    row = {
        "context_s": context, "window_samples": length, "best_epochs": epochs,
        "own_windows": {"train": int(len(train["y"])), "val": int(len(val["y"])), "test": int(len(test["y"]))},
        "last_sample_relative_to_tb_test": offs,
        "future_wait_ms": wait_ms,
        "cpu_gridding_ms": round(cpu_grid_ms, 3), "gpu_single_model_b1_ms": round(gpu_b1_ms, 3),
        "two_model_output_delay_ms": round(wait_ms + cpu_grid_ms + 2 * gpu_b1_ms, 2),
        "own": {"val_err_mm": err_mm(val_p, val["y"]), "test_err_mm": err_mm(test_p, test["y"]),
                "val_zero_mm": err_mm(np.zeros_like(val["y"]), val["y"]),
                "test_zero_mm": err_mm(np.zeros_like(test["y"]), test["y"])},
        "common": {"val_err_mm": err_mm(picked["val"], common["val"]["y"]),
                   "test_err_mm": err_mm(picked["test"], common["test"]["y"])},
    }
    print(f"  context={context:g}s: own {row['own']} | common {row['common']} | "
          f"delay {row['two_model_output_delay_ms']} ms | epochs {epochs}", flush=True)
    return row, picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    names = (split["train"] + split["extra_train"], split["val"], split["test"])

    ref_len = int(round((1.5 * HORIZON + 2 * REFERENCE) * phys.HZ))
    common = {}
    for key, group in (("val", names[1]), ("test", names[2])):
        data = phys.build_group(group, REFERENCE, HORIZON, ref_len)
        common[key] = {"keys": keys_of(data), "y": data["y"], "t_pair": data["t_pair"]}
    report = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "contexts_s": list(CONTEXTS),
        "common_windows": {"definition": f"windows admissible at c = {REFERENCE:g} s",
                           "val": len(common["val"]["keys"]), "test": len(common["test"]["keys"]),
                           "val_zero_mm": err_mm(np.zeros_like(common["val"]["y"]), common["val"]["y"]),
                           "test_zero_mm": err_mm(np.zeros_like(common["test"]["y"]), common["test"]["y"])},
        "repeats": [],
    }
    arrays = {}
    for r in range(args.repeats):
        print(f"=== sweep {r + 1}/{args.repeats} ===", flush=True)
        rows, preds = {}, {}
        for c in CONTEXTS:
            rows[f"{c:g}"], preds[c] = run_context(c, device, names, common)
        y, t_pair = common["test"]["y"], common["test"]["t_pair"]
        e_ref = np.linalg.norm(preds[REFERENCE]["test"] - y, axis=1)
        for c in CONTEXTS:
            if c == REFERENCE:
                continue
            e_c = np.linalg.norm(preds[c]["test"] - y, axis=1)
            # positive: the shorter context has the lower error on the common held-out windows
            rows[f"{c:g}"]["common"]["test_gain_over_2s_mm"] = round(float((e_ref - e_c).mean() * 1000), 2)
            rows[f"{c:g}"]["common"]["test_gain_over_2s_ci95_mm"] = block_ci(e_c, e_ref, t_pair)
            arrays[f"sweep{r}__c{c:g}__test"] = preds[c]["test"].astype(np.float32)
        arrays[f"sweep{r}__c{REFERENCE:g}__test"] = preds[REFERENCE]["test"].astype(np.float32)
        report["repeats"].append(rows)
        OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    arrays["test_y"], arrays["test_t_pair"] = common["test"]["y"], common["test"]["t_pair"]
    np.savez_compressed(DATA / "context_sweep_common.npz", **arrays)

    def rng_of(c, split, key):
        v = [rep[f"{c:g}"][split][key] for rep in report["repeats"]]
        return [min(v), max(v)]

    report["summary"] = {f"{c:g}": {
        "common_val_err_mm": rng_of(c, "common", "val_err_mm"),
        "common_test_err_mm": rng_of(c, "common", "test_err_mm"),
        "own_val_err_mm": rng_of(c, "own", "val_err_mm"),
        "own_test_err_mm": rng_of(c, "own", "test_err_mm"),
        "two_model_output_delay_ms": [min(rep[f"{c:g}"]["two_model_output_delay_ms"] for rep in report["repeats"]),
                                      max(rep[f"{c:g}"]["two_model_output_delay_ms"] for rep in report["repeats"])],
    } for c in CONTEXTS}
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
