#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fine-tune the frozen PhysNet ensemble on part of the 2026-11-09 rig.

The files named in datasets/final_model_manifest.json are copied and never
written. Each ensemble member continues from its own checkpoint. In each of
two folds it is shown one continuous and one paused recording from the new
camera, and the comparison is scored only on the pair it did not see.

This analysis was added after the frozen-model result was known. It is not
part of the pre-specified protocol.

    python tools/adapt_new_rig_physnet.py              # + three rejected paused takes
    python tools/adapt_new_rig_physnet.py --balanced   # one continuous, one paused
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

# The sampling order and the augmentation noise are seeded. A rerun of one seed
# is close but not bit-identical: the frozen training code interpolates the
# dense stream on the GPU, and that backward pass has no deterministic
# implementation in PyTorch, so torch.use_deterministic_algorithms does not
# remove the difference.
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.eval_stop_protocol_experiment as esp
import tools.train_physnet as phys
import tools.train_seq as api

DATA = ROOT / "datasets"
MANIFEST = DATA / "final_model_manifest.json"
BACKUP = DATA / "frozen_model_backup_20261109"
RIGID = "rigid_20261109_023120"
# Qualified recordings that did not pass the capture gate, but still have
# chessboard labels. They are training-only in both folds.
EXTRA = ("traj_20261109_022228", "traj_20261109_023237", "traj_20261109_023607")
FOLDS = {
    "ab": {
        "adapt": ("traj_20261109_012026", "traj_20261109_022758") + EXTRA,
        "test_c": ("traj_20261109_024257",),
        "test_p": ("traj_20261109_023932",),
    },
    "ba": {
        "adapt": ("traj_20261109_024257", "traj_20261109_023932") + EXTRA,
        "test_c": ("traj_20261109_012026",),
        "test_p": ("traj_20261109_022758",),
    },
}
CONTEXT, HORIZON = 2.0, 3.0
EPOCHS = 8
LR = 1e-4
N_BOOT = 2000   # block-bootstrap resamples, as in eval_stop_protocol_experiment.py


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def backup_frozen() -> list[Path]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    BACKUP.mkdir(parents=True, exist_ok=True)
    copied = []
    for row in manifest["models_and_configs"]:
        src = Path(row["path"])
        if not src.is_file():
            raise SystemExit(f"missing frozen weight: {src}")
        dest = BACKUP / src.name
        if not dest.is_file() or dest.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dest)
        digest = _sha256(dest)
        if digest != row["sha256"]:
            raise SystemExit(f"backup hash mismatch: {dest.name}")
        if _sha256(src) != row["sha256"]:
            raise SystemExit(f"frozen weight changed: {src}")
        copied.append(src)
    (BACKUP / "manifest_copy.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"backup": str(BACKUP), "models": len(copied)}), flush=True)
    return copied


def rest_scale(name: str) -> float:
    events = json.loads((DATA / name / "stop_events.json").read_text(encoding="utf-8"))
    t, y = phys.load_imu(DATA / name / "imu_stream.csv")
    still = np.linalg.norm(y[:, 3:6], axis=1) < 1.0
    sel = (t >= events["t_start"]) & (t <= events["t_motion"]) & still
    if int(sel.sum()) < 100:
        raise SystemExit(f"{name}: not enough rest samples to scale the accelerometer")
    return float(np.median(np.linalg.norm(y[sel, 0:3], axis=1)))


def _concat_groups(groups: list[dict]) -> dict:
    """Join already-built groups whose dense supervision has different widths.

    The new camera runs at about 13 fps, so a 3 s window holds fewer reference
    frames than the older recordings. The network still sees a fixed IMU length.
    """
    width = max(g["dense_tau"].shape[1] for g in groups)
    merged = {}
    for key in groups[0]:
        if key.startswith("dense_"):
            continue
        merged[key] = np.concatenate([g[key] for g in groups])
    n = len(merged["y"])
    dense_tau = np.zeros((n, width), np.float32)
    dense_y = np.zeros((n, width, 3), np.float32)
    dense_v = np.zeros((n, width, 3), np.float32)
    dense_valid = np.zeros((n, width), np.float32)
    dense_v_valid = np.zeros((n, width), np.float32)
    dense_frame = np.full((n, width), -1, np.int64)
    k = 0
    for group in groups:
        m = group["dense_tau"].shape[1]
        count = len(group["y"])
        dense_tau[k:k + count, :m] = group["dense_tau"]
        dense_y[k:k + count, :m] = group["dense_y"]
        dense_v[k:k + count, :m] = group["dense_v"]
        dense_valid[k:k + count, :m] = group["dense_valid"]
        dense_v_valid[k:k + count, :m] = group["dense_v_valid"]
        dense_frame[k:k + count, :m] = group["dense_frame"]
        k += count
    merged.update(dense_tau=dense_tau, dense_y=dense_y, dense_v=dense_v,
                  dense_valid=dense_valid, dense_v_valid=dense_v_valid,
                  dense_frame=dense_frame)
    return merged


def build_windows(adapt: tuple[str, ...], test_c: tuple[str, ...], test_p: tuple[str, ...],
                  old=None, val=None):
    length = int(round((1.5 * HORIZON + 2 * CONTEXT) * phys.HZ))
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    old_names = split["train"]
    val_names = split["val"]
    forbidden = set(test_c + test_p)
    if forbidden & set(adapt):
        raise SystemExit("a test recording is also in the adaptation set")
    if forbidden & set(old_names + val_names):
        raise SystemExit("a test recording is in the original split")
    nominal = api.R_CAMERA_IMU_NEW.copy()
    if old is None:
        old = phys.build_group(old_names, CONTEXT, HORIZON, length)
    if val is None:
        val = phys.build_group(val_names, CONTEXT, HORIZON, length)
    rig = json.loads((DATA / RIGID / "rigidity.json").read_text(encoding="utf-8"))
    today = np.asarray(rig["R_camera_imu"], np.float64)
    scales = {name: rest_scale(name) for name in adapt}
    original = phys.load_imu

    def scaled(path):
        t, y = original(path)
        name = Path(path).parent.name
        if name in scales:
            y = np.array(y, copy=True)
            y[:, 0:3] /= scales[name]
        return t, y

    phys.load_imu = scaled
    api.R_CAMERA_IMU_NEW = today.astype(np.float32)
    try:
        new = phys.build_group(adapt, CONTEXT, HORIZON, length)
    finally:
        phys.load_imu = original
        api.R_CAMERA_IMU_NEW = nominal
    train = _concat_groups([old, new])
    print(json.dumps({
        "old_pairs": int(len(old["y"])), "new_pairs": int(len(new["y"])),
        "train_pairs": int(len(train["y"])), "val_pairs": int(len(val["y"])),
        "adapt": list(adapt), "test_continuous": list(test_c),
        "test_paused": list(test_p),
        "acc_scale": {k: round(v, 4) for k, v in scales.items()},
    }, ensure_ascii=False), flush=True)
    return train, val, old, val


def finetune(sources: list[Path], train, val, adapt: tuple[str, ...],
             test_c: tuple[str, ...], test_p: tuple[str, ...], out: Path,
             seed: int = 0) -> list[Path]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_t = phys.to_device(train, device)
    val_t = phys.to_device(val, device)
    n = len(train["y"])
    # Half of each epoch is the new rig. The held-out test recordings are absent.
    is_new = np.isin(train["session"], adapt)
    weight = np.ones(n, np.float64)
    if is_new.any() and (~is_new).any():
        weight[is_new] = 0.5 * (~is_new).sum() / (0.5 * int(is_new.sum()))
    weight /= weight.sum()
    args = Namespace(
        weight_decay=0.01, batch=64, end_weight=1.0, vel_weight=0.0,
        still_weight=1.0, nll_weight=0.0,
    )
    aug = {
        "acc_bias": 0.005 * phys.G0, "acc_noise": 0.002 * phys.G0,
        "gyro_bias": np.radians(0.2), "gyro_noise": np.radians(0.1),
        "ext_rad": np.radians(1.0), "jitter": 0, "warp": 1.0, "amp": (1.0, 1.0),
    }
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    for k, src in enumerate(sources):
        model = phys.load_model(src, device)
        # Sampling order (numpy) and augmentation noise (torch) of this member.
        rng = np.random.default_rng(seed)
        torch.manual_seed(1000 * seed + k)
        phys.run_phase(
            model, train_t, val_t, val["y"], weight, aug, 1.0, args, rng,
            epochs=EPOCHS, lr=LR, patience=EPOCHS, label=src.stem, fixed=True)
        ck = torch.load(src, map_location="cpu")
        ck["model"] = {k: v.detach().cpu().clone() for k, v in model.net.state_dict().items()}
        ck["adapted_from"] = str(src)
        ck["adapt_sessions"] = list(adapt)
        ck["adapt_seed"] = seed
        ck["held_out_test"] = {"continuous": list(test_c), "paused": list(test_p)}
        dest = out / src.name
        torch.save(ck, dest)
        saved.append(dest)
        print(json.dumps({"saved": dest.name}), flush=True)
        del model
        torch.cuda.empty_cache()
    return saved


def _score(name: str, models: list, device, today: np.ndarray) -> dict:
    """Gated PhysNet, with and without stop-anchored switching. No IMUNet."""
    import tools.train_physnet_v3 as v3
    import tools.stop_anchor_check as sac

    length = int(round((1.5 * HORIZON + 2 * CONTEXT) * phys.HZ))
    anchor = int(round(CONTEXT * phys.HZ))
    events = json.loads((DATA / name / "stop_events.json").read_text(encoding="utf-8"))
    skip = (events.get("rest_start_s", 8.0) + 2.0, events.get("rest_end_s", 8.0) + 2.0)
    with esp.acc_at_rest(name), esp.imu_mount(today):
        data = phys.build_group([name], CONTEXT, HORIZON, length)
        data_t = phys.to_device(data, device)
        pred = np.mean([m.predict(data_t) for m in models], axis=0)
        v3.BIAS_MODE = "none"
        anchored, span = sac.stop_anchor(v3.build_group([name], CONTEXT, HORIZON, length), anchor)
    switch = span <= esp.MAX_SPAN_S
    t, _r, p, usable = phys._load_gt(name)
    t0, t1 = float(t[usable].min()), float(t[usable].max())
    keep = (data["t_pair"][:, 0] >= t0 + skip[0]) & (data["t_pair"][:, 1] <= t1 - skip[1])
    return {
        "y": data["y"][keep],
        "t_pair": data["t_pair"][keep],
        "preds": {
            "physnet": pred[keep],
            "physnet_switch": np.where(switch[:, None], anchored, pred)[keep],
        },
    }


def evaluate(models: list, adapt: tuple[str, ...], test_c: tuple[str, ...],
             test_p: tuple[str, ...], report_path: Path, seed: int = 0) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loaded = [phys.load_model(p, device) for p in models]
    rig = json.loads((DATA / RIGID / "rigidity.json").read_text(encoding="utf-8"))
    today = np.asarray(rig["R_camera_imu"], np.float64)
    rows = {"continuous": [], "paused": []}
    for cond, names in (("continuous", test_c), ("paused", test_p)):
        for name in names:
            rows[cond].append(_score(name, loaded, device, today))
    cont, paus = esp._pool(rows["continuous"]), esp._pool(rows["paused"])
    report = {
        "models": [p.name for p in models],
        "trained_on": list(adapt),
        "tested_on": {"continuous": list(test_c), "paused": list(test_p)},
        "rigid": RIGID,
        "epochs": EPOCHS, "lr": LR, "seed": seed, "n_boot": N_BOOT,
        "physnet_switch": esp.contrast(cont, paus, "physnet_switch", N_BOOT),
        "physnet": esp.contrast(cont, paus, "physnet", N_BOOT),
        "per_session": {},
    }
    for cond, names in (("continuous", test_c), ("paused", test_p)):
        for name, row in zip(names, rows[cond]):
            zero = float(np.linalg.norm(row["y"], axis=1).mean())
            one = {"condition": cond, "zero_mm": round(zero * 1000, 2)}
            for key in ("physnet", "physnet_switch"):
                err = float(np.linalg.norm(row["preds"][key] - row["y"], axis=1).mean())
                one[key] = {"err_mm": round(err * 1000, 2),
                            "reduction_vs_zero": round(1.0 - err / zero, 3)}
            report["per_session"][name] = one
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--balanced", action="store_true",
                    help="adapt on one continuous and one paused recording only. The default set "
                         "adds three rejected paused takes, so four of its five new recordings "
                         "are paused; this control removes that imbalance")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling order and augmentation noise. Seed 0 writes the reported files; "
                         "other seeds get the suffix _s<seed>")
    ap.add_argument("--suffix", default=None,
                    help="output suffix, to keep a repeat of one seed beside the first run")
    cli = ap.parse_args()
    tag = "bal_" if cli.balanced else ""
    suffix = cli.suffix if cli.suffix is not None else ("" if cli.seed == 0 else f"_s{cli.seed}")
    sources = backup_frozen()
    sources = [p for p in sources if p.name.startswith("physnet_cvg_")]
    if len(sources) != 16:
        raise SystemExit(f"expected 16 PhysNet checkpoints, found {len(sources)}")
    old = val = None
    reports = {}
    for name, fold in FOLDS.items():
        adapt = tuple(n for n in fold["adapt"] if not (cli.balanced and n in EXTRA))
        train, val, old, val = build_windows(adapt, fold["test_c"], fold["test_p"], old, val)
        saved = finetune(
            sources, train, val, adapt, fold["test_c"], fold["test_p"],
            DATA / f"physnet_adapt_20261109_{tag}{name}{suffix}", cli.seed)
        reports[name] = evaluate(
            saved, adapt, fold["test_c"], fold["test_p"],
            DATA / f"stop_eval_20261109_adapted_{tag}{name}{suffix}.json", cli.seed)
    gains = {name: {"physnet": rep["physnet"]["reduction_gain"],
                    "physnet_switch": rep["physnet_switch"]["reduction_gain"]}
             for name, rep in reports.items()}
    print(json.dumps({"held_out_gains": gains}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
