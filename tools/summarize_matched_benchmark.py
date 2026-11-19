#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the matched leave-one-session-out benchmark for the revision tables.

Reads the outputs of tools/eval_session_cv.py for the non-gated (cvx) and gated
(cvg) PhysNet fold models with the IMUNet, RoNIN-ResNet and TLIO-ResNet fold
models trained under the same protocol, and adds what eval_session_cv.py does
not compute: fold-ensemble CV errors for every baseline and the zero-motion
trajectory error of every test recording.

    python tools/summarize_matched_benchmark.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_physnet as phys
import tools.train_seq as api
from tools.build_trajectory_comparison import _metrics as ate_metrics, _solve_positions
from tools.train_external_architectures import make_model, predict as predict_external

DATA = ROOT / "datasets"
BASELINES = ("imunet", "ronin_resnet", "tlio_resnet")


def fold_ensemble_cv(device, physnet_dir: str = "physnet_cv", ext_dir: str = "external_benchmark",
                     baseline_tag: str = "cvx", cv_tags: tuple[str, str] = ("cvx", "cvg")) -> dict:
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    folds = split["train"] + split["val"]
    length = int(round((1.5 * 3.0 + 2 * 2.0) * api.HZ))
    errs = {k: [] for k in ("zero", "physnet_cvx", "physnet_cvg", *BASELINES)}
    for ses in folds:
        X, y = api.build(ses, 2.0, length)
        data_t = phys.to_device(phys.build_group([ses], 2.0, 3.0, length), device)
        errs["zero"].append(np.linalg.norm(y, axis=1))
        for key, tag in zip(("physnet_cvx", "physnet_cvg"), cv_tags):
            preds = [phys.load_model(DATA / physnet_dir / f"physnet_{tag}_{ses}_s{s}.pt", device).predict(data_t)
                     for s in (0, 1)]
            errs[key].append(np.linalg.norm(np.mean(preds, 0) - y, axis=1))
        for arch in BASELINES:
            preds = []
            for s in (0, 1):
                net = make_model(arch).to(device)
                net.load_state_dict(torch.load(DATA / ext_dir / f"{arch}_{baseline_tag}_{ses}_s{s}.pt",
                                               map_location=device))
                preds.append(predict_external(net, X, device))
            errs[arch].append(np.linalg.norm(np.mean(preds, 0) - y, axis=1))
    return {k: round(float(np.concatenate(v).mean() * 1000), 2) for k, v in errs.items()}


def zero_ate(session: str) -> dict:
    t, R, p, usable = phys._load_gt(session)
    data = phys.build_group([session], 2.0, 3.0, int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ)))
    node = {int(f): i for i, f in enumerate(usable)}
    edges = np.array([(node[int(a)], node[int(b)]) for a, b in data["pair"]], np.int64)
    traj = _solve_positions(p[usable], R[usable], edges, np.zeros((len(edges), 3)))
    return ate_metrics(traj, p[usable])


def block_ci(better: np.ndarray, worse: np.ndarray, t_pair: np.ndarray, n_boot: int = 2000) -> list[float]:
    """95% block-bootstrap interval of mean(worse - better) in mm over 3-s blocks."""
    rng = np.random.default_rng(0)
    blocks = np.floor((t_pair[:, 0] - t_pair[:, 0].min()) / 3.0).astype(int)
    ids = np.unique(blocks)
    members = [np.flatnonzero(blocks == b) for b in ids]
    diff = worse - better
    stats = [diff[np.concatenate([members[i] for i in rng.integers(0, len(ids), len(ids))])].mean()
             for _ in range(n_boot)]
    return [round(float(v) * 1000, 2) for v in np.quantile(stats, (0.025, 0.975))]


def paired_intervals(cv_dir: Path | None = None) -> dict:
    """Every method against zero motion and against IMUNet on each test recording."""
    root = cv_dir or DATA
    out = {}
    for tag in ("cvx", "cvg"):
        arrays = np.load(root / f"session_cv_matched_{tag}.npz")
        rep = json.loads((root / f"session_cv_matched_{tag}.json").read_text(encoding="utf-8"))
        for ses in rep["test_sessions"]:
            y, t_pair = arrays[f"{ses}__y"], arrays[f"{ses}__t_pair"]
            err = {k: np.linalg.norm(arrays[f"{ses}__{k}"] - y, axis=1)
                   for k in ("physnet", *BASELINES, "equal_blend", "cv_calibrated_blend")}
            zero = np.linalg.norm(y, axis=1)
            row = out.setdefault(ses, {})
            for k, e in err.items():
                name = {"physnet": "physnet" if tag == "cvx" else "physnet_gate",
                        "equal_blend": "equal_blend" if tag == "cvx" else "equal_blend_gate",
                        "cv_calibrated_blend": "calibrated_blend" if tag == "cvx" else "calibrated_blend_gate"}.get(k, k)
                row[name] = {"gain_over_zero_ci95_mm": block_ci(e, zero, t_pair),
                             "gain_over_imunet_ci95_mm": None if k == "imunet" else block_ci(e, err["imunet"], t_pair)}
    return out


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-dir", type=Path, default=None,
                    help="folder with session_cv_matched_cvx/cvg.{json,npz} (default: datasets/)")
    ap.add_argument("--output", type=Path, default=DATA / "matched_benchmark_summary.json")
    ap.add_argument("--physnet-dir", default="physnet_cv")
    ap.add_argument("--ext-dir", default="external_benchmark")
    ap.add_argument("--baseline-tag", default="cvx")
    ap.add_argument("--cv-tags", nargs=2, default=["cvx", "cvg"])
    args = ap.parse_args()
    cv_dir = args.cv_dir or DATA
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = {"cv_fold_ensemble_mm": fold_ensemble_cv(
        device, args.physnet_dir, args.ext_dir, args.baseline_tag, tuple(args.cv_tags)), "test_sessions": {}}
    for tag in ("cvx", "cvg"):
        rep = json.loads((cv_dir / f"session_cv_matched_{tag}.json").read_text(encoding="utf-8"))
        for ses, r in rep["test_sessions"].items():
            row = out["test_sessions"].setdefault(ses, {"n": r["n"], "zero_motion": r["zero_motion"],
                                                        "zero_ate": zero_ate(ses)})
            name = "physnet" if tag == "cvx" else "physnet_gate"
            row[name] = r["physnet_cv_ensemble"] | {
                "ate_single_edge_mm": r["trajectory_physnet"]["single_edge"]["ate_rmse_mm"],
                "ate_dense_mm": r["trajectory_physnet"]["dense_graph"]["ate_rmse_mm"],
                "minus_zero_ci95_mm": r["physnet_minus_zero_mm_95ci_blocks"],
                **{f"minus_{a}_ci95_mm": r[f"physnet_minus_{a}_mm_95ci_blocks"] for a in BASELINES}}
            row[f"equal_blend{'' if tag == 'cvx' else '_gate'}"] = r["equal_blend_physnet_imunet"] | {
                "ate_single_edge_mm": r["trajectory_equal_blend"]["single_edge"]["ate_rmse_mm"]}
            row[f"calibrated_blend{'' if tag == 'cvx' else '_gate'}"] = r["cv_calibrated_blend"] | {
                "minus_imunet_ci95_mm": r["cv_calibrated_blend_minus_imunet_mm_95ci_blocks"]}
            for a in BASELINES:
                row[a] = r[f"{a}_cv_ensemble"] | {
                    "ate_single_edge_mm": r[f"trajectory_{a}"]["single_edge"]["ate_rmse_mm"]}
        out.setdefault("test_pooled", {})[tag] = rep.get("test_pooled")
    out["paired_intervals"] = paired_intervals(cv_dir)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out["cv_fold_ensemble_mm"]))
    for ses, row in out["test_sessions"].items():
        print(ses, "zero ATE", row["zero_ate"]["ate_rmse_mm"])


if __name__ == "__main__":
    main()
