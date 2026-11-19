#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Retrain the analyses of the revision that train networks, and compare the
fresh numbers with the stored ones that the manuscript reports.

reproduce_revision_results.py covers the analyses that only apply stored
models; those reproduce exactly. The ones here train from scratch on the GPU,
so a rerun differs slightly from the stored run. This script measures by how
much. Fresh results go to datasets/reproducibility_check/; stored files are
not changed.

    python tools/reproduce_trained_results.py --wait-gpu
    python tools/reproduce_trained_results.py oracle_2x2 latency_context

--wait-gpu starts each job only after the GPU has been idle for five minutes,
and runs one job at a time.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.reproduce_revision_results import leaves  # noqa: E402

DATA = ROOT / "datasets"
OUT = DATA / "reproducibility_check"
REVISION = "reviewer_revision_experiments.json"

SECTION = ("import sys, json, torch; sys.path.insert(0, '.'); from pathlib import Path; "
           "import tools.run_reviewer_revisions as m; "
           "out = m.{fn}(torch.device('cuda' if torch.cuda.is_available() else 'cpu')); "
           "Path(r'{out}').write_text(json.dumps(out, indent=2), encoding='utf-8')")

# name -> (what it reproduces, stored file, section of the stored file or None,
#          fresh file(s), command, numeric leaves to compare)
JOBS = {
    "adapt_repeat": (
        "rig adaptation, balanced folds, seed 0 run a second time",
        None, None, None,
        ["tools/adapt_new_rig_physnet.py", "--balanced", "--seed", "0", "--suffix", "_s0rep"], None),
    "synthetic_factorial": (
        "Table III Group D: matched factorial in the twin, 11 configurations, two seeds",
        "synthetic_factorial.json", "results", "synthetic_factorial.json",
        ["tools/run_synthetic_factorial.py", "--out", "{out}"],
        r"/(test|ood|real_test|real_t1)/err_mm$"),
    "oracle_2x2": (
        "Fig. 6(d): initial velocity x demeaning, two twin corpora, two seeds",
        REVISION, "exp2_oracle_simulation", "oracle_2x2.json",
        ["-c", SECTION.replace("{fn}", "run_oracle_simulation")],
        r"/(open_loop_mm|learned_ens_mm)$"),
    "latency_context": (
        "Section VIII-A: PhysNet retrained with 0, 0.5, 1 and 2 s of context, two seeds",
        REVISION, "exp3_latency_context", "latency_context.json",
        ["-c", SECTION.replace("{fn}", "run_latency_context_sweep")],
        r"/(val_err_mm|test_err_mm)$"),
    "matched_ablation": (
        "Table III Group C: matched factorial on the real recordings, two seeds",
        REVISION, "exp1_matched_ablation", "matched_ablation.json",
        ["-c", SECTION.replace("{fn}", "run_matched_ablation")],
        r"/(val_ens|test_ens)$"),
    "matched_ablation_body_physnet": (
        "Table III Group C, rows PhysNet 8-ch end and dense (first computed with a one-off command)",
        REVISION, "exp1_matched_ablation", "matched_ablation_body_physnet.json",
        ["-c", SECTION.replace("{fn}", "run_body_frame_physnet")],
        r"/(val_ens|test_ens)$"),
    "loso_fold_models": (
        "Tables V and VI: all 80 leave-one-session-out fold models (PhysNet, gated PhysNet, IMUNet, "
        "RoNIN-ResNet, TLIO-ResNet, 8 folds x 2 seeds) trained again and scored on CV, held-out test and T1",
        "matched_benchmark_summary.json", None, "loso_retrained/matched_benchmark_summary.json",
        ["tools/retrain_loso_all.py"],
        r"(/cv_fold_ensemble_mm/(physnet_cvx|physnet_cvg|imunet|ronin_resnet|tlio_resnet)$"
        r"|/(err_mm|fast_mm|ate_single_edge_mm|ate_dense_mm)$)"),
}


# --------------------------------------------------------------------------
# waiting for the GPU
# --------------------------------------------------------------------------
def gpu_utilization() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return float(out[0]) if out else 0.0


def other_jobs(pattern: re.Pattern | None) -> list[str]:
    """Python processes that are not ours and look like compute jobs: more than
    1 GB of memory, or a command line matching --busy-pattern."""
    import psutil
    me = psutil.Process()
    mine = {me.pid} | {p.pid for p in me.children(recursive=True)} | {p.pid for p in me.parents()}
    found = []
    for proc in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            if proc.info["pid"] in mine or "python" not in (proc.info["name"] or "").lower():
                continue
            cmd = " ".join(proc.info["cmdline"] or [])
            heavy = proc.info["memory_info"] is not None and proc.info["memory_info"].rss > 1 << 30
            if heavy or (pattern is not None and pattern.search(cmd)):
                found.append(f"{proc.info['pid']}: {cmd[:90]}")
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return found


def wait_for_gpu(pattern: re.Pattern | None, quiet_minutes: int, max_util: float) -> None:
    quiet, waited = 0, 0
    while quiet < quiet_minutes:
        util, jobs = gpu_utilization(), other_jobs(pattern)
        quiet = quiet + 1 if util <= max_util and not jobs else 0
        if waited % 15 == 0 and quiet < quiet_minutes:
            print(f"[{time.strftime('%H:%M')}] waiting for the GPU: utilization {util:.0f}%, "
                  f"{len(jobs)} other job(s), idle for {quiet} min", flush=True)
        if quiet < quiet_minutes:
            time.sleep(60)
            waited += 1
    print(f"[{time.strftime('%H:%M')}] GPU idle for {quiet_minutes} min", flush=True)


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
def compare(stored, fresh, keep: str) -> dict:
    a = {k: v for k, v in leaves(stored) if re.search(keep, k)}
    b = {k: v for k, v in leaves(fresh) if re.search(keep, k)}
    rows = [{"key": k, "stored": a[k], "fresh": b[k], "difference": round(float(b[k]) - float(a[k]), 3)}
            for k in sorted(set(a) & set(b))]
    diffs = np.abs([r["difference"] for r in rows]) if rows else np.zeros(1)
    worst = max(rows, key=lambda r: abs(r["difference"])) if rows else None
    return {"numbers": len(rows), "median_abs_difference_mm": round(float(np.median(diffs)), 3),
            "max_abs_difference_mm": round(float(diffs.max()), 3), "largest_at": worst,
            "missing_in_fresh": sorted(set(a) - set(b)), "rows": rows}


def compare_adaptation() -> dict:
    """Second run of seed 0 against the first, with seed 1 as the yardstick."""
    import torch
    out = {}
    for fold in ("ab", "ba"):
        runs = {tag: json.loads((DATA / f"stop_eval_20261109_adapted_bal_{fold}{suffix}.json")
                                .read_text(encoding="utf-8"))
                for tag, suffix in (("seed0", ""), ("seed0_repeat", "_s0rep"), ("seed1", "_s1"))}
        numbers = {tag: {"gain_points": round(100 * r["physnet"]["reduction_gain"], 2),
                         "gain_ci95_points": [round(100 * v, 1) for v in r["physnet"]["reduction_gain_ci95"]],
                         "continuous_mm": r["physnet"]["continuous"]["err_mm"],
                         "paused_mm": r["physnet"]["paused"]["err_mm"],
                         "matched_diff_mm": r["physnet"]["matched_diff_mm"],
                         "gain_with_switching_points": round(100 * r["physnet_switch"]["reduction_gain"], 2)}
                   for tag, r in runs.items()}
        weights = {}
        base = DATA / f"physnet_adapt_20261109_bal_{fold}"
        for tag, suffix in (("seed0_repeat", "_s0rep"), ("seed1", "_s1")):
            other = DATA / f"physnet_adapt_20261109_bal_{fold}{suffix}"
            worst, rms = 0.0, []
            for path in sorted(base.glob("physnet_cvg_*.pt")):
                x = torch.load(path, map_location="cpu")["model"]
                y = torch.load(other / path.name, map_location="cpu")["model"]
                d = torch.cat([(x[k].float() - y[k].float()).flatten() for k in x])
                worst = max(worst, float(d.abs().max()))
                rms.append(float(d.pow(2).mean().sqrt()))
            weights[f"seed0_vs_{tag}"] = {"max_abs": worst, "rms_mean_over_16_models": float(np.mean(rms))}
        out[fold] = {"numbers": numbers, "weight_difference": weights}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="*", help="default: all of " + ", ".join(JOBS))
    ap.add_argument("--wait-gpu", action="store_true", help="start each job only when the GPU is idle")
    ap.add_argument("--busy-pattern", default="",
                    help="regex; a running process whose command line matches it counts as a GPU job")
    ap.add_argument("--quiet-minutes", type=int, default=5)
    ap.add_argument("--max-utilization", type=float, default=20.0)
    ap.add_argument("--compare-only", action="store_true", help="compare the fresh files that exist")
    args = ap.parse_args()
    unknown = [j for j in args.jobs if j not in JOBS]
    if unknown:
        raise SystemExit(f"unknown job(s) {unknown}; choose from {list(JOBS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(args.busy_pattern, re.I) if args.busy_pattern else None
    path = OUT / "trained_comparison.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    for name in args.jobs or list(JOBS):
        what, stored_rel, section, fresh_rel, cmd, keep = JOBS[name]
        fresh_path = OUT / fresh_rel if fresh_rel else None
        row = {"reproduces": what}
        if not args.compare_only:
            if args.wait_gpu:
                wait_for_gpu(pattern, args.quiet_minutes, args.max_utilization)
            start = time.time()
            print(f"[{time.strftime('%H:%M')}] {name}: started", flush=True)
            log = (OUT / f"{name}.log").open("w", encoding="utf-8")
            code = subprocess.run([sys.executable, *[c.replace("{out}", str(fresh_path)) for c in cmd]],
                                  cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            log.close()
            row["minutes"] = round((time.time() - start) / 60, 1)
            if code != 0:
                row["error"] = (OUT / f"{name}.log").read_text(encoding="utf-8", errors="replace")[-1200:]
                report[name] = row
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[{time.strftime('%H:%M')}] {name}: FAILED\n{row['error']}", flush=True)
                continue
        if name == "adapt_repeat":
            row["folds"] = compare_adaptation()
            for fold, v in row["folds"].items():
                n = v["numbers"]
                print(f"  fold {fold}: gain seed 0 {n['seed0']['gain_points']}, repeat "
                      f"{n['seed0_repeat']['gain_points']}, seed 1 {n['seed1']['gain_points']} points", flush=True)
        else:
            stored = json.loads((DATA / stored_rel).read_text(encoding="utf-8"))
            if section is not None:
                stored = stored[section]
            fresh = json.loads(fresh_path.read_text(encoding="utf-8"))
            fresh = fresh.get(section, fresh) if name == "synthetic_factorial" else fresh
            where = f"{stored_rel}[{section}]" if section is not None else stored_rel
            row.update(stored=where, fresh=str(fresh_path.relative_to(ROOT)),
                       **compare(stored, fresh, keep))
            print(f"  {row['numbers']} numbers, median |difference| {row['median_abs_difference_mm']} mm, "
                  f"largest {row['max_abs_difference_mm']} mm at {row['largest_at']}", flush=True)
        report[name] = row
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{time.strftime('%H:%M')}] {name}: done ({row.get('minutes', 0)} min)", flush=True)
    print("ALL DONE; wrote", path, flush=True)


if __name__ == "__main__":
    main()
