#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run the analyses behind the physical results of the revision and compare
every number with the stored file that the manuscript checks read.

Each analysis starts again from the recordings, the labels and the frozen
models and writes into datasets/reproducibility_check/. The stored files are
not changed. Analyses that train networks (twin factorial, 2x2 oracle, context
sweep, rig adaptation) are not repeated here; they have their own scripts.

    python tools/reproduce_revision_results.py            # all
    python tools/reproduce_revision_results.py stop_tripod caliper_static
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = DATA / "reproducibility_check"
CAL = "datasets/pilot_caliper_20261108/selfcal"
CAL_PLAN = f"{CAL}/reference_caliper_plan_onedir.csv"

# name -> (stored file, command with {out} for the fresh file)
JOBS: dict[str, tuple[str, list[str]]] = {
    "stop_handheld": ("stop_protocol_experiment.json", [
        "tools/eval_stop_protocol_experiment.py", "--continuous", "traj_20261108_160843",
        "--paused", "traj_20261108_163523", "traj_20261108_165541",
        "--rigid", "rigid_20261108_160717", "--n-boot", "2000", "--output", "{out}"]),
    "stop_handheld_rotation": ("stop_protocol_experiment_rotation.json", [
        "tools/stop_rotation_diagnostic.py", "--result", "{dir}/stop_handheld.npz"]),
    "stop_tripod": ("stop_eval_20261109_batch.json", [
        "tools/eval_stop_protocol_experiment.py",
        "--continuous", "traj_20261109_012026", "traj_20261109_024257",
        "--paused", "traj_20261109_022758", "traj_20261109_023932",
        "--rigid", "rigid_20261109_023120", "--n-boot", "2000", "--output", "{out}"]),
    "stop_tripod_rotation": ("stop_eval_20261109_batch_rotation.json", [
        "tools/stop_rotation_diagnostic.py", "--result", "{dir}/stop_tripod.npz"]),
    "stop_tripod_tilt": ("stop_anchor_tilt_20261109.json", [
        "tools/stop_anchor_tilt.py", "--output", "{out}"]),
    "caliper_static": ("reference_stage_validation.json", [
        "tools/reference_stage_validation.py", "analyze", "--mode", "caliper",
        "--data-root", CAL, "--plan", CAL_PLAN, "--output", "{out}"]),
    "caliper_dynamic_board": ("reference_caliper_dynamic.json", [
        "tools/reference_caliper_dynamic.py", "--data-root", CAL, "--plan", CAL_PLAN,
        "--quantity", "board", "--output", "{out}"]),
    "caliper_dynamic_camera": ("reference_caliper_dynamic_camera.json", [
        "tools/reference_caliper_dynamic.py", "--data-root", CAL, "--plan", CAL_PLAN,
        "--quantity", "camera", "--output", "{out}"]),
    "board_imu": ("board_imu_motion_check.json", [
        "tools/board_imu_motion_check.py", "--data-root", CAL, "--output", "{out}",
        "--figure", "{dir}/board_imu.pdf"]),
    "delay_repeatability": ("delay_repeatability.json", [
        "tools/delay_repeatability.py", "--max-date", "20261031", "--output", "{out}"]),
    "allan": ("allan_static_imu.json", [
        "tools/allan_static_imu.py", "align_imu_static_20260922_115532",
        "align_imu_static_20260922_115758", "--output", "{out}"]),
    "dense_single_anchor": ("reviewer_audit_20261108/dense_single_anchor_recomputed.json", [
        "tools/build_dense_single_anchor_trajectory.py", "--output", "{out}"]),
    "claim_checks": ("revision_claim_checks.json", [
        "-c", "import sys; sys.path.insert(0, '.'); from pathlib import Path; "
              "import tools.verify_revision_claims as m; m.OUT = Path(r'{out}'); m.main()"]),
    "gate_check": ("gate_trajectory_check.json", ["tools/analyze_gate_trajectory.py"]),
    "allan_bound": ("allan_static_bound.json", ["tools/allan_static_bound.py"]),
    "lever_excitation": ("lever_excitation.json", ["tools/analyze_lever_excitation.py"]),
    # Tables V and VI: eval_session_cv.py (frozen) run again into the check
    # folder on the held-out recording and on T1, then summarized from those
    # fresh outputs instead of from datasets/session_cv_matched_*.json.
    "session_cv_matched_cvx": ("session_cv_matched_cvx.json", [
        "tools/eval_session_cv.py", "--cv-tag", "cvx", "--baseline-tag", "cvx",
        "--baselines", "imunet", "ronin_resnet", "tlio_resnet", "--seeds", "0", "1",
        "--output", "{out}"]),
    "session_cv_matched_cvg": ("session_cv_matched_cvg.json", [
        "tools/eval_session_cv.py", "--cv-tag", "cvg", "--baseline-tag", "cvx",
        "--baselines", "imunet", "ronin_resnet", "tlio_resnet", "--seeds", "0", "1",
        "--output", "{out}"]),
    "matched_benchmark": ("matched_benchmark_summary.json", [
        "tools/summarize_matched_benchmark.py", "--cv-dir", "{dir}", "--output", "{out}"]),
}
IN_PLACE = {"gate_check", "allan_bound", "lever_excitation"}
# Fields that name files, list raw samples or record the run, not results.
SKIP_KEYS = {"result", "figure", "output", "data_root", "plan", "generated", "created", "recording_order",
             "note", "twin_prediction", "selfcal_data_root"}


def leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in SKIP_KEYS:
                continue
            yield from leaves(value, f"{prefix}/{key}")
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from leaves(value, f"{prefix}[{i}]")
    else:
        yield prefix, obj


def compare(stored: dict, fresh: dict) -> dict:
    a, b = dict(leaves(stored)), dict(leaves(fresh))
    numbers = exact = 0
    worst = (0.0, "", None, None)
    different_text = []
    for key in sorted(set(a) & set(b)):
        x, y = a[key], b[key]
        if isinstance(x, bool) or isinstance(y, bool) or x is None or y is None \
                or isinstance(x, str) or isinstance(y, str):
            if x != y:
                different_text.append((key, x, y))
            continue
        numbers += 1
        diff = abs(float(x) - float(y))
        exact += diff == 0.0
        # Relative to the size of the number, so 80.22 against 80.23 counts as 1e-4.
        rel = diff / max(abs(float(x)), abs(float(y)), 1.0)
        if rel > worst[0]:
            worst = (rel, key, x, y)
    return {
        "numbers": numbers, "identical": int(exact),
        "largest_relative_difference": worst[0],
        "largest_difference_at": None if not worst[1] else {"key": worst[1], "stored": worst[2], "fresh": worst[3]},
        "other_differences": different_text[:12],
        "only_in_stored": sorted(set(a) - set(b))[:12], "only_in_fresh": sorted(set(b) - set(a))[:12],
    }


def run(name: str) -> dict:
    stored_rel, cmd = JOBS[name]
    stored_path = DATA / stored_rel
    fresh_path = OUT / f"{name}.json"
    args = [c.replace("{out}", str(fresh_path)).replace("{dir}", str(OUT)) for c in cmd]
    start = time.time()
    backup = None
    try:
        if name in IN_PLACE:
            backup = OUT / f"{name}.stored_copy.json"
            shutil.copy2(stored_path, backup)
        proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
        if name in IN_PLACE and proc.returncode == 0:
            shutil.copy2(stored_path, fresh_path)
    finally:
        if backup is not None:
            shutil.copy2(backup, stored_path)
    if proc.returncode != 0:
        return {"job": name, "ok": False, "seconds": round(time.time() - start, 1),
                "error": (proc.stdout + proc.stderr)[-1500:]}
    # The rotation diagnostic names its output after the result file.
    if not fresh_path.is_file():
        alt = OUT / (name.replace("_rotation", "") + "_rotation.json")
        if alt.is_file():
            fresh_path = alt
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
    return {"job": name, "ok": True, "seconds": round(time.time() - start, 1),
            "stored": stored_rel, "fresh": str(fresh_path.relative_to(ROOT)), **compare(stored, fresh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jobs", nargs="*", help="default: all of " + ", ".join(JOBS))
    args = ap.parse_args()
    unknown = [j for j in args.jobs if j not in JOBS]
    if unknown:
        raise SystemExit(f"unknown job(s) {unknown}; choose from {list(JOBS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    report = []
    for name in args.jobs or list(JOBS):
        row = run(name)
        report.append(row)
        if row["ok"]:
            print(f"{name:26s} {row['identical']:5d}/{row['numbers']:<5d} identical, "
                  f"largest relative difference {row['largest_relative_difference']:.2e}"
                  f"{'' if not row['other_differences'] else '  other: ' + str(len(row['other_differences']))}"
                  f"  ({row['seconds']} s)", flush=True)
        else:
            print(f"{name:26s} FAILED\n{row['error']}", flush=True)
    path = OUT / "comparison.json"
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []
    merged = {r["job"]: r for r in previous} | {r["job"]: r for r in report}
    path.write_text(json.dumps(list(merged.values()), ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("wrote", path)


if __name__ == "__main__":
    main()
