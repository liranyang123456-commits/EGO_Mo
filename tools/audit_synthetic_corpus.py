#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit every sequence in a generated synthetic corpus."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture.sync import load_imu
from tools.validate_synthetic import validate


def _audit_imu(session: Path) -> dict:
    meta = json.loads((session / "session_meta.json").read_text(encoding="utf-8"))
    gt = np.load(session / "ground_truth.npz")
    t, imu = load_imu(session / "imu_stream.csv")
    rate = (len(t) - 1) / max(t[-1] - t[0], 1e-9)
    ortho = float(np.max(np.abs(
        np.einsum("nji,njk->nik", gt["R_W_C"], gt["R_W_C"]) - np.eye(3)
    )))
    return {
        "session": session.name,
        "imu_samples": int(len(t)),
        "gt_frames": int(len(gt["t"])),
        "imu_rate_hz": round(rate, 3),
        "orthogonality": ortho,
        "acc_norm_minmax": np.round(
            [np.linalg.norm(imu[:, :3], axis=1).min(), np.linalg.norm(imu[:, :3], axis=1).max()],
            3,
        ).tolist(),
        "gyro_max_dps": round(float(np.linalg.norm(imu[:, 3:6], axis=1).max()), 2),
        "passed": bool(
            abs(rate - meta["config"]["imu_hz"]) < 2.0
            and ortho < 1e-5
            and np.isfinite(imu).all()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    args = ap.parse_args()
    manifest = json.loads((args.corpus / "manifest.json").read_text(encoding="utf-8"))
    imu_rows = []
    visual_rows = []
    seeds = {}
    for split, entries in manifest["imu"].items():
        seeds[split] = []
        for entry in entries:
            session = args.corpus / entry["path"]
            row = _audit_imu(session)
            row.update({"split": split, "motion": entry["motion"]})
            imu_rows.append(row)
            seeds[split].append(entry["config"]["seed"])
    for split, entries in manifest["visual"].items():
        for entry in entries:
            row = validate(args.corpus / entry["path"])
            row.update({"split": split, "motion": entry["motion"], "backend": "opencv"})
            visual_rows.append(row)
    for entry in manifest["blender"]:
        row = validate(args.corpus / entry["path"])
        row.update({"split": "representative", "motion": entry["motion"], "backend": "blender"})
        visual_rows.append(row)
    overlaps = {}
    split_names = list(seeds)
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            overlaps[f"{a}-{b}"] = len(set(seeds[a]) & set(seeds[b]))
    detections = []
    corner_errors = []
    for row in visual_rows:
        got, total = (int(x) for x in row["checkerboard_detection_sample"].split("/"))
        detections.append(got / max(total, 1))
        if row["corner_truth_rmse_px_median"] is not None:
            corner_errors.append(row["corner_truth_rmse_px_median"])
    summary = {
        "imu_sequences": len(imu_rows),
        "visual_sequences": len(visual_rows),
        "imu_passed": int(sum(row["passed"] for row in imu_rows)),
        "visual_passed": int(sum(row["passed"] for row in visual_rows)),
        "detection_rate_median": None if not detections else round(float(np.median(detections)), 3),
        "detection_rate_min": None if not detections else round(float(np.min(detections)), 3),
        "corner_rmse_px_median": None if not corner_errors else round(float(np.median(corner_errors)), 3),
        "corner_rmse_px_p90": None if not corner_errors else round(float(np.quantile(corner_errors, 0.9)), 3),
        "seed_overlap": overlaps,
        "passed": bool(
            all(row["passed"] for row in imu_rows)
            and all(row["passed"] for row in visual_rows)
            and all(value == 0 for value in overlaps.values())
        ),
    }
    payload = {"summary": summary, "imu": imu_rows, "visual": visual_rows}
    (args.corpus / "audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
