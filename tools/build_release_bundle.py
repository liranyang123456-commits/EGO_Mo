#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a deterministic, integrity-checked software/results release ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RELEASE = ROOT / "release"

ROOT_FILES = [
    "README.md", "REPRODUCIBILITY.md", "CITATION.cff", "LICENSE",
    ".zenodo.json", "requirements.txt", "capture_app.py",
    "run_capture.bat", "run_simulator.bat",
]
RESULT_FILES = [
    "trajectory_split_paper.json", "trajectory_split_20260924.json",
    "study_protocol_v2.lock.json",
    "method_benchmark_20260924.json",
    "session_cv_benchmark.json", "session_cv_gate_benchmark.json",
    "uncertainty_val.json", "zupt_bound.json", "pure_imu_pose_benchmark.json",
    "reference_audit.json", "stereo_extrinsic_measured.json",
    "summary_table.json", "twin_realistic_summary.json",
    "twin_protocol_sweep.json", "twin_motion_class_eval.json",
    "twin_imunet_synthetic.json", "latency_benchmark.json",
]
CODE_DIRS = ["ego_capture", "ego_sim", "tools"]
PAPER_GLOBS = ["*.tex", "*.bib", "REVISION_NOTES.md", "figures/*.pdf"]


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def session_manifest():
    split_path = DATA / "trajectory_split_20260924.json"
    if not split_path.is_file():
        return {}
    split = json.loads(split_path.read_text(encoding="utf-8"))
    out = {"protocol_id": split.get("protocol_id"), "groups": {}}
    for group in ("train", "val", "test", "extra_train"):
        rows = []
        for name in split.get(group, []):
            folder = DATA / name
            meta = folder / "session_meta.json"
            rows.append({
                "session": name,
                "group": group,
                "metadata_sha256": sha(meta) if meta.is_file() else None,
                "files_present": sorted(
                    str(p.relative_to(folder)).replace("\\", "/")
                    for p in folder.rglob("*") if p.is_file()
                ),
            })
        out["groups"][group] = rows
    out["prospective_test_status"] = split.get("prospective_test_status")
    return out


def collect():
    files = []
    for name in ROOT_FILES:
        p = ROOT / name
        if p.is_file():
            files.append(p)
    for directory in CODE_DIRS:
        for p in (ROOT / directory).rglob("*.py"):
            if "__pycache__" not in p.parts:
                files.append(p)
    for pattern in PAPER_GLOBS:
        files.extend((ROOT / "paper").glob(pattern))
    for name in RESULT_FILES:
        p = DATA / name
        if p.is_file():
            files.append(p)
    return sorted(set(files), key=lambda p: str(p.relative_to(ROOT)).lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()
    RELEASE.mkdir(exist_ok=True)
    output = args.output or RELEASE / f"EGO_Mo_v{args.version}.zip"
    files = collect()
    manifest = {
        "name": "EGO_Mo", "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "raw_data_included": False,
        "note": "Raw sessions are described by data_manifest.json and require a separate archive.",
        "files": [{
            "path": str(p.relative_to(ROOT)).replace("\\", "/"),
            "bytes": p.stat().st_size, "sha256": sha(p),
        } for p in files],
    }
    manifest_bytes = json.dumps(manifest, ensure_ascii=False, indent=2).encode()
    data_bytes = json.dumps(session_manifest(), ensure_ascii=False, indent=2).encode()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=9) as z:
        for p in files:
            info = zipfile.ZipInfo(str(p.relative_to(ROOT)).replace("\\", "/"))
            info.date_time = (2026, 9, 26, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, p.read_bytes())
        z.writestr("MANIFEST.json", manifest_bytes)
        z.writestr("DATA_MANIFEST.json", data_bytes)
    report = {
        "output": str(output), "bytes": output.stat().st_size,
        "sha256": sha(output), "files": len(files) + 2,
        "raw_data_included": False,
    }
    (RELEASE / "release_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
