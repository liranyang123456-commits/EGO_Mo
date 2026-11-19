#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declared interim look at new recordings with the frozen model set.

Uses exactly the models and blend constants of
datasets/final_model_manifest.json / study_protocol_v2.lock.json (hashes are
verified first) and records every look in the prospective checklist. It does
not replace tools/final_prospective_evaluation.py: after an interim look no
model, threshold or blend may change, otherwise the prospective claim lapses.

Recordings moved to datasets/rejected_prospective/ can be included as
exploratory data (--exploratory); they are linked back temporarily so the
standard reference pipeline can process them, and are never counted as
protocol test sessions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_seq as api
import tools.train_physnet as phys
from tools.train_external_architectures import make_model, predict as predict_external
from tools.eval_session_cv import trajectory_ate

DATA = ROOT / "datasets"
REJECTED = DATA / "rejected_prospective"
CHECKLIST = DATA / "prospective_test_checklist.json"


def _verify_manifest():
    m = json.loads((DATA / "final_model_manifest.json").read_text(encoding="utf-8"))
    for r in m["models_and_configs"]:
        h = hashlib.sha256(Path(r["path"]).read_bytes()).hexdigest()
        if h != r["sha256"]:
            raise RuntimeError(f"frozen file changed: {r['path']}")
    return m


def _link(name):
    """Temporarily expose a rejected recording under datasets/ (junction)."""
    src, dst = REJECTED / name, DATA / name
    if dst.exists():
        return False
    subprocess.run(["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                   check=True, capture_output=True)
    return True


def _unlink(name):
    subprocess.run(["cmd", "/c", "rmdir", str(DATA / name)], check=True, capture_output=True)


def _prepare_reference(name):
    from tools.build_pose_gt import process_session
    from tools.clean_pose_gt import clean
    from tools.prepare_trajectory_data import _export_raw
    if not (DATA / "pose_gt" / f"{name}.npz").is_file():
        process_session(name)
        clean(DATA / "pose_gt" / f"{name}.npz")
    return _export_raw(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="+", required=True)
    ap.add_argument("--exploratory", nargs="*", default=[],
                    help="rejected recordings to score as non-protocol data")
    ap.add_argument("--output", type=Path, default=DATA / "interim_eval.json")
    args = ap.parse_args()
    manifest = _verify_manifest()
    protocol = json.loads((DATA / "study_protocol_v2.lock.json").read_text(encoding="utf-8"))
    w_blend = protocol["frozen_models"]["blend"]["cv_calibrated_physnet_weight"]
    shrink = protocol["frozen_models"]["blend"]["cv_calibrated_shrink"]

    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    paths = [Path(r["path"]) for r in manifest["models_and_configs"]]
    phys_models = [phys.load_model(p, device) for p in paths if p.name.startswith("physnet_cvg_")]
    imu_nets = []
    for p in paths:
        if p.name.startswith("imunet_cvx_"):
            net = make_model("imunet").to(device)
            net.load_state_dict(torch.load(p, map_location=device))
            net.eval()
            imu_nets.append(net)
    if len(phys_models) != 16 or len(imu_nets) != 16:
        raise RuntimeError("expected 16 + 16 frozen models")

    linked = []
    report = {"kind": "INTERIM_LOOK_NOT_FINAL", "created_utc":
              datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "model_manifest_sha256": manifest["manifest_sha256"],
              "blend": {"physnet_weight": w_blend, "shrink": shrink}, "sessions": {}}
    try:
        for name in args.sessions + args.exploratory:
            role = "protocol_test" if name in args.sessions else "exploratory_rejected"
            if role == "exploratory_rejected" and _link(name):
                linked.append(name)
            usable = _prepare_reference(name)
            data = phys.build_group([name], context, horizon, length)
            data_t = phys.to_device(data, device)
            y = data["y"]
            p_phys = np.mean([m.predict(data_t) for m in phys_models], axis=0)
            X, y2 = api.build(name, context, length)
            assert np.allclose(y2, y, atol=1e-6)
            p_imu = np.mean([predict_external(n, X, device) for n in imu_nets], axis=0)
            preds = {
                "zero_motion": np.zeros_like(y),
                "imunet": p_imu,
                "physnet": p_phys,
                "equal_blend": 0.5 * (p_phys + p_imu),
                "cv_calibrated_blend": shrink * (w_blend * p_phys + (1 - w_blend) * p_imu),
            }
            meta = json.loads((DATA / name / "session_meta.json").read_text(encoding="utf-8"))
            res = {"role": role, "resolution": f"{meta['requested']['width']}x{meta['requested']['height']}",
                   "usable_reference_frames": int(usable), "pairs": int(len(y))}
            for k, v in preds.items():
                res[k] = api._metrics(v, y)
            for k in ("physnet", "imunet", "cv_calibrated_blend"):
                res[f"ate_{k}"] = trajectory_ate(name, data, preds[k], None)["single_edge"]
            report["sessions"][name] = res
            print(json.dumps({name: {k: (v["err_mm"], v["r2"]) if isinstance(v, dict) and "err_mm" in v
                                     else v for k, v in res.items()}}, ensure_ascii=False), flush=True)
    finally:
        for name in linked:
            _unlink(name)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    checklist = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    checklist.setdefault("interim_looks", []).append({
        "utc": report["created_utc"], "protocol_sessions": args.sessions,
        "exploratory": args.exploratory, "output": str(args.output),
        "commitment": "no model, threshold or blend change after this look",
    })
    CHECKLIST.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
