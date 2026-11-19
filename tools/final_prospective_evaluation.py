#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One-shot final evaluation on prospectively sealed recordings.

The tool has two irreversible stages:
  freeze-models: record model/config hashes before any new test metric is seen.
  run: verify 4/4 prospective sessions and evaluate one prediction bundle,
       then create a FINAL lock. It refuses to overwrite a final result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.train_seq as api

DATA = ROOT / "datasets"
PROTOCOL = DATA / "study_protocol_v2.lock.json"
SPLIT = DATA / "trajectory_split_20260924.json"
MODEL_MANIFEST = DATA / "final_model_manifest.json"
FINAL = DATA / "final_prospective_evaluation.json"
FINAL_LOCK = DATA / "FINAL_EVALUATION_COMPLETE.lock"


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(payload: dict):
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def freeze_models(files: list[Path], output: Path):
    if output.exists():
        raise RuntimeError(f"{output} exists; model set is already frozen")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    rows = []
    for p in files:
        if not p.is_file():
            raise FileNotFoundError(p)
        rows.append({"path": str(p.resolve()), "bytes": p.stat().st_size,
                     "sha256": sha(p)})
    payload = {
        "status": "FROZEN_BEFORE_PROSPECTIVE_TEST",
        "protocol_id": protocol["protocol_id"],
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models_and_configs": rows,
        "analysis": protocol["decision_rules"],
        "metrics": protocol["final_metrics"],
    }
    payload["manifest_sha256"] = canonical_hash(payload)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    print(json.dumps({"output": str(output), "models": len(rows),
                      "manifest_sha256": payload["manifest_sha256"]}, indent=2))


def blocks(t_pair):
    # One block ID per non-overlapping 3-s interval, local to each session.
    return np.floor((t_pair[:, 0] - t_pair[:, 0].min()) / 3.0).astype(int)


def bootstrap_difference(pa, pb, y, session, block, n=10000, seed=20260926):
    """CI of mean error(pb)-mean error(pa); positive means a is better."""
    rng = np.random.default_rng(seed)
    ea = np.linalg.norm(pa - y, axis=1)
    eb = np.linalg.norm(pb - y, axis=1)
    units = [(s, b) for s in np.unique(session)
             for b in np.unique(block[session == s])]
    values = []
    for _ in range(n):
        sampled = rng.choice(len(units), len(units), replace=True)
        idx = np.concatenate([
            np.flatnonzero((session == units[k][0]) & (block == units[k][1]))
            for k in sampled
        ])
        values.append(float((eb[idx] - ea[idx]).mean() * 1000))
    return [float(x) for x in np.quantile(values, (0.025, 0.975))]


def run(predictions: Path, model_manifest: Path, output: Path):
    if FINAL_LOCK.exists() or output.exists():
        raise RuntimeError(
            "FINAL evaluation already exists. Do not overwrite it; acquire a "
            "new prospectively sealed set and freeze a new protocol."
        )
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    manifest = json.loads(model_manifest.read_text(encoding="utf-8"))
    if manifest["protocol_id"] != protocol["protocol_id"]:
        raise RuntimeError("model manifest and study protocol IDs differ")
    if manifest["manifest_sha256"] != canonical_hash({
            k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise RuntimeError("model manifest was modified after freezing")
    for row in manifest["models_and_configs"]:
        p = Path(row["path"])
        if not p.is_file() or sha(p) != row["sha256"]:
            raise RuntimeError(f"frozen model/config changed: {p}")

    status = split.get("prospective_test_status", {})
    required = int(status.get("required", 4))
    sessions = list(status.get("sessions", []))
    if not status.get("complete") or len(sessions) < required:
        raise RuntimeError(
            f"prospective test incomplete: {len(sessions)}/{required} sessions"
        )
    blob = np.load(predictions, allow_pickle=True)
    methods = ["physnet", "imunet", "equal_blend", "cv_calibrated_blend"]
    all_y, all_session, all_block = [], [], []
    all_pred = {m: [] for m in methods}
    per_session = {}
    for name in sessions:
        prefix = f"{name}__"
        needed = [prefix + "y", prefix + "t_pair",
                  *[prefix + m for m in methods]]
        missing = [k for k in needed if k not in blob.files]
        if missing:
            raise RuntimeError(f"{name}: missing prediction arrays {missing}")
        y, tp = blob[prefix + "y"], blob[prefix + "t_pair"]
        per_session[name] = {
            "n": int(len(y)),
            "zero_motion": api._metrics(np.zeros_like(y), y),
        }
        all_y.append(y)
        all_session.append(np.full(len(y), name))
        all_block.append(blocks(tp))
        for method in methods:
            pred = blob[prefix + method]
            if pred.shape != y.shape:
                raise RuntimeError(f"{name}/{method}: shape mismatch")
            all_pred[method].append(pred)
            per_session[name][method] = api._metrics(pred, y)
    y = np.concatenate(all_y)
    session = np.concatenate(all_session)
    block = np.concatenate(all_block)
    pred = {m: np.concatenate(v) for m, v in all_pred.items()}
    pooled = {"n": int(len(y)), "sessions": sessions,
              "zero_motion": api._metrics(np.zeros_like(y), y)}
    for method in methods:
        pooled[method] = api._metrics(pred[method], y)
    comparisons = {
        "physnet_vs_imunet_gain_mm_95ci":
            bootstrap_difference(pred["physnet"], pred["imunet"], y,
                                 session, block),
        "equal_blend_vs_imunet_gain_mm_95ci":
            bootstrap_difference(pred["equal_blend"], pred["imunet"], y,
                                 session, block),
        "physnet_vs_zero_gain_mm_95ci":
            bootstrap_difference(pred["physnet"], np.zeros_like(y), y,
                                 session, block),
    }
    report = {
        "kind": "ONE_SHOT_PROSPECTIVE_FINAL",
        "completed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha(PROTOCOL),
        "model_manifest_sha256": manifest["manifest_sha256"],
        "prediction_bundle": str(predictions.resolve()),
        "prediction_bundle_sha256": sha(predictions),
        "per_session": per_session,
        "pooled": pooled,
        "comparisons": comparisons,
        "claim_rule": protocol["decision_rules"],
    }
    report["result_sha256"] = canonical_hash(report)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                      encoding="utf-8")
    FINAL_LOCK.write_text(json.dumps({
        "protocol_id": protocol["protocol_id"],
        "result": str(output),
        "result_sha256": report["result_sha256"],
        "completed_utc": report["completed_utc"],
        "message": "Do not rerun on these sessions or tune from this result.",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "pooled": pooled,
                      "comparisons": comparisons}, ensure_ascii=False, indent=2))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("freeze-models")
    f.add_argument("files", nargs="+", type=Path)
    f.add_argument("--output", type=Path, default=MODEL_MANIFEST)
    r = sub.add_parser("run")
    r.add_argument("--predictions", type=Path, required=True)
    r.add_argument("--model-manifest", type=Path, default=MODEL_MANIFEST)
    r.add_argument("--output", type=Path, default=FINAL)
    args = p.parse_args()
    if args.command == "freeze-models":
        freeze_models(args.files, args.output)
    else:
        run(args.predictions, args.model_manifest, args.output)


if __name__ == "__main__":
    main()
