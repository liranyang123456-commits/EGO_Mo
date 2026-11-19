#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run controlled additional experiments requested by TIM reviewers.

1) Reviewer #2 Comment 1: Controlled 2x2x2 ablation separating:
   - Input preprocessing: 8-channel demeaned body-frame vs 24-channel anchor-frame pre-integrated features
   - Supervision: End-point only vs Dense stream supervision
   - Architecture: Conv-BiGRU (PhysNet trunk, 0.42M) vs IMUNetAdapted (2.83M) vs RoNINResNet (3.85M) vs TLIOResNet (3.85M)
   under matched training data (fixed split with curated extra_train), matched preprocessing (24-ch anchor-frame), and matched ensemble size (2 seeds).

2) Reviewer #2 Comment 2 & Reviewer #1 Comment 2: Controlled 2x2 simulation tests on pause_inf and pause_4:
   - With vs without true initial velocity v(t_a) supplied as input channels
   - With vs without input demeaning (demeaned lin = f^Ca - mu_f vs exact gravity-compensated f^Ca - g^Ca)
   for both open-loop double integration and learned PhysNet.

3) Reviewer #1 Comment 4: Uninterrupted single-anchor trajectory evaluation vs 31-component periodically anchored evaluation on the 115.2-s held-out test recording.

4) Reviewer #1 Comment 5 & Reviewer #2 Comment 4: End-to-end latency breakdown across context lookahead c in {0.0, 0.5, 1.0, 2.0} s, including waiting time, CPU gridding, GPU featurization, and single-model vs 16-model ensemble inference.
"""

from __future__ import annotations

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
import tools.train_physnet_v3 as v3
import tools.train_seq as api
from tools.train_external_architectures import DepthwiseBlock, ResBlock, make_model, predict as predict_ext

DATA = ROOT / "datasets"
OUT_JSON = DATA / "reviewer_revision_experiments.json"


# ---------------------------------------------------------------------------
# Experiment 1: Matched preprocessing (8-ch body vs 24-ch anchor) & supervision
# ---------------------------------------------------------------------------
class AnchorRoNINResNet(nn.Module):
    def __init__(self, in_ch=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(3, 2, 1),
        )
        layers = []
        cin = 64
        for stage, cout in enumerate((64, 128, 256, 512)):
            for block in range(2):
                stride = 2 if stage > 0 and block == 0 else 1
                layers.append(ResBlock(cin, cout, stride))
                cin = cout
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(512, 3)

    def forward(self, feats):
        x = F.interpolate(feats.transpose(1, 2), 400, mode="linear", align_corners=False)
        return self.head(F.adaptive_avg_pool1d(self.body(self.stem(x)), 1).squeeze(-1)) * 0.01


class AnchorTLIOResNet(nn.Module):
    def __init__(self, in_ch=24):
        super().__init__()
        self.encoder = AnchorRoNINResNet(in_ch)
        self.encoder.head = nn.Identity()
        self.mean = nn.Linear(512, 3)

    def forward(self, feats):
        x = F.interpolate(feats.transpose(1, 2), 400, mode="linear", align_corners=False)
        feat = F.adaptive_avg_pool1d(self.encoder.body(self.encoder.stem(x)), 1).squeeze(-1)
        return self.mean(feat) * 0.01


class AnchorIMUNet(nn.Module):
    def __init__(self, in_ch=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(3, 2, 1),
        )
        self.body = nn.Sequential(
            DepthwiseBlock(64, 64), DepthwiseBlock(64, 64),
            DepthwiseBlock(64, 128, 2), DepthwiseBlock(128, 128),
            DepthwiseBlock(128, 256, 2), DepthwiseBlock(256, 256),
            DepthwiseBlock(256, 512, 2), DepthwiseBlock(512, 512),
            DepthwiseBlock(512, 1024, 2), DepthwiseBlock(1024, 1024),
        )
        self.head = nn.Linear(1024, 3)

    def forward(self, feats):
        x = F.interpolate(feats.transpose(1, 2), 200, mode="linear", align_corners=False)
        feature = F.adaptive_avg_pool1d(self.body(self.stem(x)), 1).squeeze(-1)
        return self.head(feature) * 0.01


class DenseIMUNet(nn.Module):
    """IMUNet depthwise-separable backbone outputting a 50-Hz velocity stream
    integrated into a displacement stream, supervised densely like PhysNet."""
    def __init__(self, in_ch=24):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 64, 7, 2, 3, bias=False), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(3, 2, 1),
        )
        self.body = nn.Sequential(
            DepthwiseBlock(64, 64), DepthwiseBlock(64, 64),
            DepthwiseBlock(64, 128, 1), DepthwiseBlock(128, 128),
            DepthwiseBlock(128, 256, 1), DepthwiseBlock(256, 256),
        )
        self.head = nn.Linear(256, 3)
        self.lever = nn.Parameter(torch.zeros(3))

    def forward(self, feats, R, anchor):
        T = feats.shape[1]
        h = self.body(self.stem(feats.transpose(1, 2))).transpose(1, 2)
        v50 = self.head(h) * 0.1
        vel = F.interpolate(v50.transpose(1, 2), size=T, mode="linear", align_corners=False).transpose(1, 2)
        d = torch.cumsum(vel, 1) / phys.HZ
        d = d - d[:, anchor:anchor + 1]
        eye = torch.eye(3, device=R.device, dtype=R.dtype)
        d = d + torch.einsum("btij,j->bti", R - eye, self.lever)
        return d


def run_matched_ablation(device) -> dict:
    print("\n=== Experiment 1: Matched Preprocessing, Supervision, and Architecture ===", flush=True)
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * phys.HZ))
    anchor = int(round(context * phys.HZ))
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    train_names = split["train"] + split["extra_train"]
    val_names = split["val"]
    test_names = split["test"]

    train = phys.build_group(train_names, context, horizon, length)
    val = phys.build_group(val_names, context, horizon, length)
    test = phys.build_group(test_names, context, horizon, length)
    train_t = phys.to_device(train, device)
    val_t = phys.to_device(val, device)
    test_t = phys.to_device(test, device)

    sub = torch.arange(0, len(train["y"]), max(1, len(train["y"]) // 2000), device=device)
    feat24 = phys.Featurizer(anchor, horizon, use_phys=True)
    feat24.fit(phys.take(train_t, sub))

    aug = {
        "acc_bias": 0.005 * phys.G0, "acc_noise": 0.002 * phys.G0,
        "gyro_bias": np.radians(0.2), "gyro_noise": np.radians(0.1),
        "ext_rad": np.radians(1.0), "jitter": 0,
        "warp": 1.0, "amp": (1.0, 1.0),
    }

    # Also build 8-ch body-frame tensors for the exact same train_names/val_names/test_names:
    p_tr = [api.build(n, context, length) for n in train_names]
    Xtr8, ytr8 = np.concatenate([p[0] for p in p_tr]), np.concatenate([p[1] for p in p_tr])
    p_va = [api.build(n, context, length) for n in val_names]
    Xva8, yva8 = np.concatenate([p[0] for p in p_va]), np.concatenate([p[1] for p in p_va])
    p_te = [api.build(n, context, length) for n in test_names]
    Xte8, yte8 = np.concatenate([p[0] for p in p_te]), np.concatenate([p[1] for p in p_te])

    results = {}

    # 1A: 8-channel body-frame + End-point supervision across architectures (2 seeds each, matched train pool)
    from tools.train_external_architectures import train_phase as ext_train_phase
    for arch_name in ("conv_bigru", "ronin_resnet", "tlio_resnet", "imunet"):
        val_preds, test_preds = [], []
        for seed in (0, 1):
            torch.manual_seed(seed)
            if arch_name == "conv_bigru":
                net = api.SeqNet().to(device)
            else:
                net = make_model(arch_name).to(device)
            ext_train_phase(net, Xtr8, ytr8, Xva8, yva8, device, epochs=50, lr=1e-3, seed=seed + 200, patience=12)
            val_preds.append(predict_ext(net, Xva8, device))
            test_preds.append(predict_ext(net, Xte8, device))
        results[f"{arch_name}_body8_endpoint"] = {
            "val_s0": api._metrics(val_preds[0], yva8)["err_mm"],
            "val_s1": api._metrics(val_preds[1], yva8)["err_mm"],
            "val_ens": api._metrics(np.mean(val_preds, axis=0), yva8)["err_mm"],
            "val_r2": api._metrics(np.mean(val_preds, axis=0), yva8)["r2"],
            "test_ens": api._metrics(np.mean(test_preds, axis=0), yte8)["err_mm"],
            "test_r2": api._metrics(np.mean(test_preds, axis=0), yte8)["r2"],
        }
        print(f"  {arch_name}_body8_endpoint:", results[f"{arch_name}_body8_endpoint"], flush=True)

    # 1B: 24-channel anchor-frame + End-point supervision across architectures (2 seeds each)
    for arch_name, cls in (("ronin_resnet", AnchorRoNINResNet),
                           ("tlio_resnet", AnchorTLIOResNet),
                           ("imunet", AnchorIMUNet)):
        val_preds, test_preds = [], []
        for seed in (0, 1):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = cls(24).to(device)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            best, best_state, bad = None, None, 0
            n = len(train["y"])
            for epoch in range(50):
                net.train()
                order = torch.from_numpy(rng.permutation(n)).to(device)
                for k in range(0, n, 64):
                    idx = order[k:k + 64]
                    b = phys.take(train_t, idx)
                    feats, _, _ = feat24(b, aug)
                    pred = net(feats)
                    loss = F.smooth_l1_loss(pred * 100.0, b["y"] * 100.0, beta=0.5)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vp = []
                    for k in range(0, len(val["y"]), 128):
                        idx = torch.arange(k, min(k + 128, len(val["y"])), device=device)
                        feats, _, _ = feat24(phys.take(val_t, idx))
                        vp.append(net(feats).cpu().numpy())
                    vp = np.concatenate(vp)
                m = api._metrics(vp, val["y"])["err_mm"]
                if best is None or m < best:
                    best, bad = m, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 12:
                        break
            net.load_state_dict(best_state)
            net.eval()
            with torch.no_grad():
                for split_t, store in ((val_t, val_preds), (test_t, test_preds)):
                    out = []
                    for k in range(0, len(split_t["y"]), 128):
                        idx = torch.arange(k, min(k + 128, len(split_t["y"])), device=device)
                        feats, _, _ = feat24(phys.take(split_t, idx))
                        out.append(net(feats).cpu().numpy())
                    store.append(np.concatenate(out))
        results[f"{arch_name}_anchor24_endpoint"] = {
            "val_s0": api._metrics(val_preds[0], val["y"])["err_mm"],
            "val_s1": api._metrics(val_preds[1], val["y"])["err_mm"],
            "val_ens": api._metrics(np.mean(val_preds, axis=0), val["y"])["err_mm"],
            "val_r2": api._metrics(np.mean(val_preds, axis=0), val["y"])["r2"],
            "test_ens": api._metrics(np.mean(test_preds, axis=0), test["y"])["err_mm"],
            "test_r2": api._metrics(np.mean(test_preds, axis=0), test["y"])["r2"],
        }
        print(f"  {arch_name}_anchor24_endpoint:", results[f"{arch_name}_anchor24_endpoint"], flush=True)

    # 1C: PhysNet (Conv-BiGRU trunk, 0.42M) with 24-ch anchor-frame + End-point only vs Dense supervision,
    # and DenseIMUNet (24-ch anchor-frame + Dense supervision):
    for mode_name, dense_w, gate_on in (("physnet_anchor24_endpoint", 0.0, False),
                                        ("physnet_anchor24_dense", 1.0, False),
                                        ("physnet_anchor24_dense_gate", 1.0, True)):
        val_preds, test_preds = [], []
        for seed in (0, 1):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = phys.PhysNet(width=96, lever=True, still=gate_on, gate=gate_on).to(device)
            model = phys.Model(net, feat24, None, anchor)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
            best, best_state, bad = None, None, 0
            n = len(train["y"])
            for epoch in range(60):
                net.train()
                order = torch.from_numpy(rng.permutation(n)).to(device)
                for k in range(0, n, 64):
                    idx = order[k:k + 64]
                    b = phys.take(train_t, idx)
                    d, vel, still, logvar = model.stream_full(b, aug)
                    loss = phys.loss_fn(d, vel, still, logvar, b, dense_w, 1.0, 0.0, 1.0 if gate_on else 0.0, 0.0)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                sched.step()
                vp = model.predict(val_t)
                m = api._metrics(vp, val["y"])["err_mm"]
                if best is None or m < best:
                    best, bad = m, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 12:
                        break
            net.load_state_dict(best_state)
            val_preds.append(model.predict(val_t))
            test_preds.append(model.predict(test_t))
        results[mode_name] = {
            "val_s0": api._metrics(val_preds[0], val["y"])["err_mm"],
            "val_s1": api._metrics(val_preds[1], val["y"])["err_mm"],
            "val_ens": api._metrics(np.mean(val_preds, axis=0), val["y"])["err_mm"],
            "val_r2": api._metrics(np.mean(val_preds, axis=0), val["y"])["r2"],
            "test_ens": api._metrics(np.mean(test_preds, axis=0), test["y"])["err_mm"],
            "test_r2": api._metrics(np.mean(test_preds, axis=0), test["y"])["r2"],
        }
        print(f"  {mode_name}:", results[mode_name], flush=True)

    # Also train DenseIMUNet (24-ch + dense supervision):
    val_preds, test_preds = [], []
    for seed in (0, 1):
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        net = DenseIMUNet(24).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
        best, best_state, bad = None, None, 0
        n = len(train["y"])
        for epoch in range(60):
            net.train()
            order = torch.from_numpy(rng.permutation(n)).to(device)
            for k in range(0, n, 64):
                idx = order[k:k + 64]
                b = phys.take(train_t, idx)
                feats, R, _ = feat24(b, aug)
                d = net(feats, R, anchor)
                loss = phys.loss_fn(d, None, None, None, b, 1.0, 1.0)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            sched.step()
            net.eval()
            with torch.no_grad():
                vp = []
                for k in range(0, len(val["y"]), 128):
                    idx = torch.arange(k, min(k + 128, len(val["y"])), device=device)
                    b = phys.take(val_t, idx)
                    feats, R, _ = feat24(b)
                    d = net(feats, R, anchor)
                    vp.append(phys.sample_stream(d, b["tau_b"][:, None])[:, 0].cpu().numpy())
                vp = np.concatenate(vp)
            m = api._metrics(vp, val["y"])["err_mm"]
            if best is None or m < best:
                best, bad = m, 0
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 12:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            for split_t, store in ((val_t, val_preds), (test_t, test_preds)):
                out = []
                for k in range(0, len(split_t["y"]), 128):
                    idx = torch.arange(k, min(k + 128, len(split_t["y"])), device=device)
                    b = phys.take(split_t, idx)
                    feats, R, _ = feat24(b)
                    d = net(feats, R, anchor)
                    out.append(phys.sample_stream(d, b["tau_b"][:, None])[:, 0].cpu().numpy())
                store.append(np.concatenate(out))
    results["imunet_anchor24_dense"] = {
        "val_s0": api._metrics(val_preds[0], val["y"])["err_mm"],
        "val_s1": api._metrics(val_preds[1], val["y"])["err_mm"],
        "val_ens": api._metrics(np.mean(val_preds, axis=0), val["y"])["err_mm"],
        "val_r2": api._metrics(np.mean(val_preds, axis=0), val["y"])["r2"],
        "test_ens": api._metrics(np.mean(test_preds, axis=0), test["y"])["err_mm"],
        "test_r2": api._metrics(np.mean(test_preds, axis=0), test["y"])["r2"],
    }
    print("  imunet_anchor24_dense:", results["imunet_anchor24_dense"], flush=True)
    return results


# ---------------------------------------------------------------------------
# Experiment 2: Controlled 2x2 Simulation Tests (True v0 x Input Demeaning)
# ---------------------------------------------------------------------------
def _add_sim_oracle(data: dict) -> dict:
    v0 = np.zeros((len(data["y"]), 3), np.float32)
    g_ca = np.zeros((len(data["y"]), 3), np.float32)
    R_ci = api.R_CAMERA_IMU_NEW.astype(np.float64)
    for name in np.unique(data["session"]):
        gt = np.load(Path(name) / "ground_truth.npz")
        t_imu = gt["imu_t"].astype(np.float64)
        p_imu = gt["imu_p_W_C"].astype(np.float64)
        R_imu = gt["imu_R_W_C"].astype(np.float64)
        v_w = np.gradient(p_imu, t_imu, axis=0)
        a_w = np.gradient(v_w, t_imu, axis=0)
        f_id = (gt["imu_usb_ideal"][:, 0:3] * v3.G0) @ R_ci.T
        for k in np.flatnonzero(data["session"] == name):
            ta, _tb = data["t_pair"][k]
            ia = int(np.argmin(np.abs(t_imu - ta)))
            v0[k] = R_imu[ia].T @ v_w[ia]
            g_ca[k] = f_id[ia] - (R_imu[ia].T @ a_w[ia])
    data["v0"] = v0
    data["g_ca"] = g_ca
    return data


def _build_custom_feats(batch: dict, anchor: int, horizon: float, demean: bool, use_v0: bool):
    """Build 24-ch (or 27-ch with v0) features with or without window demeaning."""
    acc, gyro, quat = batch["acc"], batch["gyro"], batch["quat"]
    valid, interval = batch["valid"], batch["interval"]
    B, T, _ = acc.shape
    dev = acc.device
    R = v3.quat_to_mat(quat)
    a_rot = torch.einsum("btij,btj->bti", R, acc)
    vmask = valid[..., None]
    if demean:
        g_hat = (a_rot * vmask).sum(1) / vmask.sum(1).clamp(min=1.0)
    else:
        g_hat = batch["g_ca"]  # true gravity reaction in C_a, preserving mean linear acceleration
    lin = (a_rot - g_hat[:, None]) * vmask
    dt = 1.0 / v3.HZ
    v = torch.cumsum(lin, 1) * dt
    v = v - v[:, anchor:anchor + 1]
    if use_v0:
        v = v + batch["v0"][:, None, :] * vmask
    d = torch.cumsum(v, 1) * dt
    d = d - d[:, anchor:anchor + 1]
    gdir = (g_hat / g_hat.norm(dim=-1, keepdim=True).clamp(min=1e-6))[:, None].expand(B, T, 3)
    rot6 = R[..., :, :2].reshape(B, T, 6)
    tcoord = ((torch.arange(T, device=dev, dtype=acc.dtype) - anchor)
              / (v3.HZ * horizon))[None, :, None].expand(B, T, 1)
    feats = torch.cat((lin, gyro, rot6, v, d, gdir, interval[..., None],
                       valid[..., None], tcoord), dim=-1)
    if use_v0:
        feats = torch.cat((feats, batch["v0"][:, None, :].expand(B, T, 3)), dim=-1)
    return feats, R, d


def run_oracle_simulation(device) -> dict:
    print("\n=== Experiment 2: Controlled 2x2 Simulation Tests (True v0 x Demeaning) ===", flush=True)
    v3.BIAS_MODE = "none"
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * v3.HZ))
    anchor = int(round(context * v3.HZ))
    proto_dir = DATA / "synthetic_protocol"
    out = {}

    for cname in ("pause_inf", "pause_4"):
        root = Path((proto_dir / cname / "latest.txt").read_text(encoding="utf-8").strip())
        if not root.is_absolute():
            root = ROOT / root
        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        tr_names = [(root / e["path"]).resolve() for e in man["imu"]["train"]]
        va_names = [(root / e["path"]).resolve() for e in man["imu"]["val"]]
        te_names = [(root / e["path"]).resolve() for e in man["imu"]["test"]]

        train = _add_sim_oracle(v3.build_group(tr_names, context, horizon, length, cap=120, rng=np.random.default_rng(0)))
        val = _add_sim_oracle(v3.build_group(va_names, context, horizon, length, cap=120, rng=np.random.default_rng(0)))
        test = _add_sim_oracle(v3.build_group(te_names, context, horizon, length, cap=160, rng=np.random.default_rng(123)))

        def to_dev(d):
            res = v3.to_device(d, device)
            res["v0"] = torch.from_numpy(d["v0"]).to(device)
            res["g_ca"] = torch.from_numpy(d["g_ca"]).to(device)
            return res

        train_t, val_t, test_t = to_dev(train), to_dev(val), to_dev(test)
        out[cname] = {"zero_mm": float(np.linalg.norm(test["y"], axis=1).mean() * 1000)}

        for demean in (True, False):
            for use_v0 in (False, True):
                key = f"{'demeaned' if demean else 'no_demean'}_{'with_v0' if use_v0 else 'no_v0'}"
                # 1. Open-loop double-integral error on test:
                with torch.no_grad():
                    _, _, d_pre = _build_custom_feats(test_t, anchor, horizon, demean, use_v0)
                    d_end = v3.sample_stream(d_pre, test_t["tau_b"][:, None])[:, 0].cpu().numpy()
                open_loop_mm = float(np.linalg.norm(d_end - test["y"], axis=1).mean() * 1000)

                # 2. Learned PhysNet error (2 seeds):
                in_ch = 27 if use_v0 else 24
                sub = torch.arange(0, len(train["y"]), max(1, len(train["y"]) // 2000), device=device)
                f_sub, _, _ = _build_custom_feats(v3.take(train_t, sub), anchor, horizon, demean, use_v0)
                v_mask = v3.take(train_t, sub)["valid"] > 0.5
                f_mean = f_sub[v_mask].mean(0)
                f_std = f_sub[v_mask].std(0).clamp(min=1e-6)
                for c in (21, 22, 23):
                    f_mean[c], f_std[c] = 0.0, 1.0

                preds = []
                for seed in (0, 1):
                    torch.manual_seed(seed)
                    rng = np.random.default_rng(seed)
                    net = v3.PhysNet(width=96, in_ch=in_ch, lever=True, still=True, gate=True,
                                     vel_scale=0.25 if use_v0 else 0.1).to(device)
                    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
                    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 25)
                    best, best_state = None, None
                    n = len(train["y"])
                    for epoch in range(25):
                        net.train()
                        order = torch.from_numpy(rng.permutation(n)).to(device)
                        for k in range(0, n, 64):
                            idx = order[k:k + 64]
                            b = v3.take(train_t, idx)
                            raw_f, R, _ = _build_custom_feats(b, anchor, horizon, demean, use_v0)
                            feats = ((raw_f - f_mean) / f_std).clamp(-8.0, 8.0)
                            d, vel, still, logvar = net(feats, R, anchor, None, b["valid"])
                            loss = v3.loss_fn(d, vel, still, logvar, b, 1.0, 1.0, 0.0, 1.0, 0.0)
                            opt.zero_grad()
                            loss.backward()
                            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                            opt.step()
                        sched.step()
                        net.eval()
                        with torch.no_grad():
                            vp = []
                            for k in range(0, len(val["y"]), 128):
                                idx = torch.arange(k, min(k + 128, len(val["y"])), device=device)
                                b = v3.take(val_t, idx)
                                raw_f, R, _ = _build_custom_feats(b, anchor, horizon, demean, use_v0)
                                feats = ((raw_f - f_mean) / f_std).clamp(-8.0, 8.0)
                                d, _, _, _ = net(feats, R, anchor, None, b["valid"])
                                vp.append(v3.sample_stream(d, b["tau_b"][:, None])[:, 0].cpu().numpy())
                            vp = np.concatenate(vp)
                        m = float(np.linalg.norm(vp - val["y"], axis=1).mean() * 1000)
                        if best is None or m < best:
                            best = m
                            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                    net.load_state_dict(best_state)
                    net.eval()
                    with torch.no_grad():
                        tp = []
                        for k in range(0, len(test["y"]), 128):
                            idx = torch.arange(k, min(k + 128, len(test["y"])), device=device)
                            b = v3.take(test_t, idx)
                            raw_f, R, _ = _build_custom_feats(b, anchor, horizon, demean, use_v0)
                            feats = ((raw_f - f_mean) / f_std).clamp(-8.0, 8.0)
                            d, _, _, _ = net(feats, R, anchor, None, b["valid"])
                            tp.append(v3.sample_stream(d, b["tau_b"][:, None])[:, 0].cpu().numpy())
                        preds.append(np.concatenate(tp))
                ens_pred = np.mean(preds, axis=0)
                met = api._metrics(ens_pred, test["y"])
                slopes = [round(float(np.polyfit(test["y"][:, ax], ens_pred[:, ax], 1)[0]), 3) for ax in range(3)]
                out[cname][key] = {
                    "open_loop_mm": round(open_loop_mm, 2),
                    "learned_ens_mm": met["err_mm"],
                    "learned_r2": met["r2"],
                    "pred_med_mm": met["pred_med_mm"],
                    "slopes": slopes,
                }
                print(f"  {cname} {key:22s}: open_loop={open_loop_mm:6.2f}mm, learned={met['err_mm']:5.2f}mm, R2={met['r2']:.3f}, slopes={slopes}", flush=True)
    return out


# ---------------------------------------------------------------------------
# Experiment 1, continued: PhysNet trunk on 8-channel body-frame input
# ---------------------------------------------------------------------------
def run_body_frame_physnet(device) -> dict:
    """Table III, Group C, rows "PhysNet, 8-ch, end" and "PhysNet, 8-ch, dense".

    These two rows were first computed with a one-off command on 2026-11-06 and
    entered into reviewer_revision_experiments.json by hand. This function is
    that command, so the rows can be regenerated. Same pool, split and
    augmentation as run_matched_ablation; the trunk has no lever arm, stillness
    head or gate, and trains for up to 60 epochs with patience 12.
    """
    print("\n=== Experiment 1 (continued): PhysNet trunk, 8-channel body-frame input ===", flush=True)
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * phys.HZ))
    anchor = int(round(context * phys.HZ))
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    train = phys.build_group(split["train"] + split["extra_train"], context, horizon, length)
    val = phys.build_group(split["val"], context, horizon, length)
    test = phys.build_group(split["test"], context, horizon, length)
    train_t, val_t, test_t = (phys.to_device(d, device) for d in (train, val, test))
    sub = torch.arange(0, len(train["y"]), max(1, len(train["y"]) // 2000), device=device)
    feat8 = phys.Featurizer(anchor, horizon, use_phys=False)
    feat8.fit(phys.take(train_t, sub))
    aug = {
        "acc_bias": 0.005 * phys.G0, "acc_noise": 0.002 * phys.G0,
        "gyro_bias": np.radians(0.2), "gyro_noise": np.radians(0.1),
        "ext_rad": np.radians(1.0), "jitter": 0, "warp": 1.0, "amp": (1.0, 1.0),
    }
    results = {}
    for key, dense_weight in (("physnet_body_endpoint", 0.0), ("physnet_body_dense", 1.0)):
        val_preds, test_preds = [], []
        for seed in (0, 1):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = phys.PhysNet(width=96, lever=False, still=False, gate=False).to(device)
            model = phys.Model(net, feat8, None, anchor)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60)
            best, best_state, bad = None, None, 0
            n = len(train["y"])
            for _epoch in range(60):
                net.train()
                order = torch.from_numpy(rng.permutation(n)).to(device)
                for k in range(0, n, 64):
                    b = phys.take(train_t, order[k:k + 64])
                    d, vel, still, logvar = model.stream_full(b, aug)
                    loss = phys.loss_fn(d, vel, still, logvar, b, dense_weight, 1.0)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                sched.step()
                m = api._metrics(model.predict(val_t), val["y"])["err_mm"]
                if best is None or m < best:
                    best, bad = m, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 12:
                        break
            net.load_state_dict(best_state)
            val_preds.append(model.predict(val_t))
            test_preds.append(model.predict(test_t))
        val_m = api._metrics(np.mean(val_preds, axis=0), val["y"])
        test_m = api._metrics(np.mean(test_preds, axis=0), test["y"])
        results[key] = {
            "val_s0": api._metrics(val_preds[0], val["y"])["err_mm"],
            "val_s1": api._metrics(val_preds[1], val["y"])["err_mm"],
            "val_ens": val_m["err_mm"], "val_r2": val_m["r2"],
            "test_ens": test_m["err_mm"], "test_r2": test_m["r2"],
        }
        print(f"  {key}:", results[key], flush=True)
    return results


# ---------------------------------------------------------------------------
# Experiment 3: Context / Lookahead Latency vs Accuracy Tradeoff
# ---------------------------------------------------------------------------
def run_latency_context_sweep(device) -> dict:
    print("\n=== Experiment 3: Context Lookahead vs End-to-End Latency ===", flush=True)
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    horizon = 3.0
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    train_names = split["train"] + split["extra_train"]
    val_names = split["val"]
    test_names = split["test"]
    out = {}

    for context in (0.0, 0.5, 1.0, 2.0):
        c_eff = max(context, 0.05)  # 0.05s = 10 samples padding for causal 0-s lookahead
        length = int(round((1.5 * horizon + 2 * c_eff) * phys.HZ))
        anchor = int(round(c_eff * phys.HZ))
        train = phys.build_group(train_names, c_eff, horizon, length)
        val = phys.build_group(val_names, c_eff, horizon, length)
        test = phys.build_group(test_names, c_eff, horizon, length)
        train_t, val_t, test_t = phys.to_device(train, device), phys.to_device(val, device), phys.to_device(test, device)

        sub = torch.arange(0, len(train["y"]), max(1, len(train["y"]) // 2000), device=device)
        feat = phys.Featurizer(anchor, horizon, use_phys=True)
        feat.fit(phys.take(train_t, sub))
        aug = {
            "acc_bias": 0.005 * phys.G0, "acc_noise": 0.002 * phys.G0,
            "gyro_bias": np.radians(0.2), "gyro_noise": np.radians(0.1),
            "ext_rad": np.radians(1.0), "jitter": 0, "warp": 1.0, "amp": (1.0, 1.0),
        }
        val_preds, test_preds = [], []
        for seed in (0, 1):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            net = phys.PhysNet(width=96, lever=True, still=True, gate=True).to(device)
            model = phys.Model(net, feat, None, anchor)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.1)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 50)
            best, best_state, bad = None, None, 0
            n = len(train["y"])
            for epoch in range(50):
                net.train()
                order = torch.from_numpy(rng.permutation(n)).to(device)
                for k in range(0, n, 64):
                    idx = order[k:k + 64]
                    b = phys.take(train_t, idx)
                    d, vel, still, logvar = model.stream_full(b, aug)
                    loss = phys.loss_fn(d, vel, still, logvar, b, 1.0, 1.0, 0.0, 1.0, 0.0)
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                sched.step()
                vp = model.predict(val_t)
                m = api._metrics(vp, val["y"])["err_mm"]
                if best is None or m < best:
                    best, bad = m, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 12:
                        break
            net.load_state_dict(best_state)
            val_preds.append(model.predict(val_t))
            test_preds.append(model.predict(test_t))

        # Timing breakdown for this context length:
        ses = phys.Session(test_names[0])
        t_gt, _, _, us = phys._load_gt(test_names[0])
        pairs = phys.enumerate_pairs(t_gt, us, horizon, c_eff, ses.t)
        t0 = time.perf_counter()
        for a, b in pairs[:100]:
            _ = ses.window(t_gt[a], t_gt[b], c_eff, length, 1.0)
        cpu_grid_ms = (time.perf_counter() - t0) / 100 * 1000.0

        b1 = phys.take(test_t, torch.arange(1, device=device))
        b16 = phys.take(test_t, torch.arange(16, device=device))
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

            s = time.perf_counter()
            for _ in range(200):
                _ = model.stream(b16)
            if device.type == "cuda":
                torch.cuda.synchronize()
            gpu_b16_per_win_ms = (time.perf_counter() - s) / (200 * 16) * 1000.0

        wait_ms = context * 1000.0
        out[f"context_{context:g}s"] = {
            "future_wait_ms": wait_ms,
            "cpu_gridding_ms": round(cpu_grid_ms, 3),
            "gpu_single_model_b1_ms": round(gpu_b1_ms, 3),
            "gpu_single_model_b16_ms": round(gpu_b16_per_win_ms, 3),
            "total_e2e_latency_single_ms": round(wait_ms + cpu_grid_ms + gpu_b1_ms, 2),
            "val_err_mm": api._metrics(np.mean(val_preds, axis=0), val["y"])["err_mm"],
            "val_r2": api._metrics(np.mean(val_preds, axis=0), val["y"])["r2"],
            "test_err_mm": api._metrics(np.mean(test_preds, axis=0), test["y"])["err_mm"],
            "test_r2": api._metrics(np.mean(test_preds, axis=0), test["y"])["r2"],
        }
        print(f"  context={context:g}s:", out[f"context_{context:g}s"], flush=True)
    return out


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {}
    if OUT_JSON.is_file():
        report = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    report["exp1_matched_ablation"] = run_matched_ablation(device) | run_body_frame_physnet(device)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["exp2_oracle_simulation"] = run_oracle_simulation(device)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["exp3_latency_context"] = run_latency_context_sweep(device)
    OUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nSaved all results to", OUT_JSON, flush=True)


if __name__ == "__main__":
    main()
