#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Screen real IMU+camera sessions and a slice of the synthetic corpus,
then score public architectures on those sets.

Real sessions are usable only when both cameras, both IMUs, and a chessboard
pose file are present, the USB stream is dense, and the session was not
already rejected as non-rigid. The synthetic slice is one held-out session
per motion from the latest corpus test split.

Public RoNIN and TLIO weights target a different sensor frame, so the
comparison uses the protocol-adapted checkpoints in datasets/external_benchmark.
The sealed real test session is scored once. Training sessions are listed
but not used as a test score.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
from tools.benchmark_methods import _strapdown
from tools.train_external_architectures import make_model, predict
from tools.train_hybrid import _synthetic_build

DATA = ROOT / "datasets"
SPLIT_PATH = DATA / "trajectory_split_20260924.json"
CORPUS_PTR = DATA / "synthetic_corpus" / "latest.txt"
WEIGHTS = DATA / "external_benchmark"
OUT = DATA / "selected_sota_eval.json"
ARCHES = ("ronin_resnet", "ronin_lstm", "tlio_resnet", "imunet")
MOTIONS = (
    "lateral", "depth", "circle", "mixed",
    "translation_xyz", "aggressive_6dof",
)


def _count_jpg(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for p in path.iterdir() if p.suffix.lower() == ".jpg")


def _imu(path: Path) -> dict:
    if not path.is_file():
        return {"n": 0, "hz": 0.0, "gaps_gt_50ms": 99}
    import csv
    ts = []
    with path.open(newline="", encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            try:
                ts.append(float(rec["timestamp"]))
            except (KeyError, ValueError):
                continue
    if len(ts) < 5:
        return {"n": len(ts), "hz": 0.0, "gaps_gt_50ms": 99}
    d = np.diff(np.asarray(ts))
    return {
        "n": len(ts),
        "hz": round((len(ts) - 1) / max(ts[-1] - ts[0], 1e-6), 2),
        "gaps_gt_50ms": int((d > 0.05).sum()),
    }


def screen_real(split: dict) -> list[dict]:
    excluded = {item["name"]: item["reason"] for item in split.get("excluded", [])}
    role = {}
    for key in ("train", "val", "test", "extra_train"):
        for name in split.get(key, []):
            role[name] = key
    rows = []
    for summary_path in sorted(DATA.glob("*/capture_summary.json")):
        name = summary_path.parent.name
        if not (name.startswith("traj_") or name.startswith("rigid_")):
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sess = summary_path.parent
        n0 = _count_jpg(sess / "cam0" / "images")
        n1 = _count_jpg(sess / "cam1" / "images")
        usb = _imu(sess / "imu_stream.csv")
        bt = _imu(sess / "imu_bt.csv")
        pose = (DATA / "pose_gt_raw" / f"{name}.npz").is_file()
        reasons = []
        if name in excluded:
            reasons.append(excluded[name])
        if float(summary.get("seconds") or 0) < 40:
            reasons.append("shorter than 40 s")
        if n0 < 200:
            reasons.append(f"cam0 images {n0}")
        frames = int(summary.get("frames") or 0)
        if frames and n0 < 0.85 * frames:
            reasons.append("cam0 dropped more than 15%")
        if n1 < 200:
            reasons.append(f"cam1 images {n1}")
        if usb["n"] < 1000 or usb["hz"] < 150 or usb["gaps_gt_50ms"] > 5:
            reasons.append(f"usb imu n={usb['n']} hz={usb['hz']} gaps={usb['gaps_gt_50ms']}")
        if bt["n"] < 1000 or bt["hz"] < 120:
            reasons.append(f"bt imu n={bt['n']} hz={bt['hz']}")
        if not pose:
            reasons.append("no pose_gt_raw")
        rows.append({
            "name": name,
            "role": role.get(name),
            "seconds": round(float(summary.get("seconds") or 0), 1),
            "cam0": n0,
            "cam1": n1,
            "usb_hz": usb["hz"],
            "bt_hz": bt["hz"],
            "usable": not reasons,
            "reasons": reasons,
        })
    return rows


def screen_synthetic(corpus: Path) -> list[dict]:
    manifest = json.loads((corpus / "manifest.json").read_text(encoding="utf-8"))
    chosen = []
    seen = set()
    for entry in manifest["imu"]["test"]:
        motion = entry.get("motion")
        if motion not in MOTIONS or motion in seen:
            continue
        path = corpus / entry["path"]
        ok = (path / "ground_truth.npz").is_file() and (path / "imu_stream.csv").is_file()
        if not ok:
            continue
        seen.add(motion)
        chosen.append({
            "name": entry["name"],
            "motion": motion,
            "path": str(path),
        })
    return chosen


def _real_xy(names: list[str], context: float, length: int):
    parts = []
    used = []
    for name in names:
        built = api.build(name, context, length)
        if built is None:
            continue
        parts.append(built)
        used.append(name)
    if not parts:
        return None, []
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
    ), used


def _synth_xy(selected, context, length):
    rng = np.random.default_rng(123)
    parts = []
    used = []
    for item in selected:
        built = _synthetic_build(Path(item["path"]), context, length, 3.0, 80, rng)
        if built is None:
            continue
        parts.append(built)
        used.append(item["name"])
    if not parts:
        return None, []
    return (
        np.concatenate([p[0] for p in parts]),
        np.concatenate([p[1] for p in parts]),
    ), used


def _score_pack(X, y, device):
    scores = {
        "zero_motion": api._metrics(np.zeros_like(y), y),
        "strapdown_zero_velocity": api._metrics(_strapdown(X), y),
    }
    for arch in ARCHES:
        scores[arch] = {}
        for mode, suffix in (("zero_shot", "zero"), ("fine_tuned", "fine")):
            path = WEIGHTS / f"{arch}_{suffix}_s0.pt"
            if not path.is_file():
                scores[arch][mode] = {"missing": str(path)}
                continue
            model = make_model(arch).to(device)
            model.load_state_dict(torch.load(path, map_location=device))
            model.eval()
            pred = predict(model, X, device)
            scores[arch][mode] = api._metrics(pred, y)
    return scores


def main():
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    corpus = Path(CORPUS_PTR.read_text(encoding="utf-8").strip())
    real = screen_real(split)
    synthetic = screen_synthetic(corpus)
    context, length = 2.0, int(round((1.5 * 3.0 + 2 * 2.0) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    real_pack, real_used = _real_xy(split["test"], context, length)
    synth_pack, synth_used = _synth_xy(synthetic, context, length)
    report = {
        "device": str(device),
        "corpus": str(corpus),
        "real_screen": real,
        "real_usable": [r["name"] for r in real if r["usable"]],
        "real_rejected": [r for r in real if not r["usable"]],
        "synthetic_selected": synthetic,
        "synthetic_used": synth_used,
        "sealed_real_test": real_used,
        "public_note": (
            "RoNIN and TLIO are the public architectures. Their published "
            "pedestrian weights are not scored: the frame and the carrier "
            "do not match this rig. zero_shot and fine_tuned are the "
            "protocol-adapted checkpoints."
        ),
        "scores": {},
    }
    if real_pack is not None:
        report["scores"]["sealed_real_test"] = _score_pack(*real_pack, device)
    if synth_pack is not None:
        report["scores"]["synthetic_subset"] = _score_pack(*synth_pack, device)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    usable = len(report["real_usable"])
    print(f"real usable {usable} of {len(real)}; synthetic {len(synth_used)}; device {device}")
    for split_name, block in report["scores"].items():
        print(split_name)
        for key, val in block.items():
            if "err_mm" in val:
                print(f"  {key:28} err {val['err_mm']:8}  r2 {val['r2']}")
            elif isinstance(val, dict) and "fine_tuned" in val:
                fine = val["fine_tuned"]
                zero = val["zero_shot"]
                print(f"  {key:28} zero {zero.get('err_mm')}  fine {fine.get('err_mm')}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
