#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect every cross-method result into one table with its source.

Writes datasets/summary_table.json (value + source path for every cell) and
paper/sections/summary_table.tex (the manuscript table), so the table and the
summary figure are generated from the same numbers.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT_TEX = ROOT / "paper" / "sections" / "summary_table.tex"

COLUMNS = [
    ("syn", "Synth.\\ test", "mm"), ("syn_r2", "", "$R^2$"), ("ood", "OOD", "mm"),
    ("direct", "Real", "direct"), ("val", "Val", "mm"), ("cv", "CV", "mm"),
    ("test", "\\multicolumn{3}{c}{Held-out test}", "mm"), ("test_r2", "", "$R^2$"),
    ("fast", "", "fast"), ("ate", "ATE", "mm"), ("par", "Par.", "M"), ("lat", "Lat.", "ms"),
]
LOWER_BETTER = {"syn", "ood", "direct", "val", "cv", "test", "fast", "ate"}


def _get(path, *keys):
    x = json.loads((DATA / path).read_text(encoding="utf-8"))
    for k in keys:
        x = x[k]
    return x


def cell(path, *keys, scale=1.0):
    return {"value": float(_get(path, *keys)) * scale, "source": f"{path}:" + "/".join(keys)}


def build():
    mb, ext, ood = "method_benchmark_20260924.json", "external_benchmark/test_benchmark.json", \
        "external_benchmark/zero_shot_synthetic_ood.json"
    syn, gate, ngate = "physnet_v1/eval_synonly_181622.json", "session_cv_gate_benchmark.json", \
        "session_cv_benchmark.json"
    traj, lat = "trajectory_comparison/summary.json", "latency_benchmark.json"
    T = ("real_test", "metrics")
    rows = []

    def row(name, fixed_split_test=False, **cells):
        rows.append({"method": name, "fixed_split_test": fixed_split_test, "cells": cells})

    row("Zero motion",
        syn=cell(mb, "datasets", "synthetic_test", "zero_motion", "err_mm"),
        syn_r2=cell(mb, "datasets", "synthetic_test", "zero_motion", "r2"),
        ood=cell(mb, "datasets", "synthetic_ood", "zero_motion", "err_mm"),
        direct=cell(gate, "test", "zero_motion", "err_mm"),
        val=cell(mb, "datasets", "real_validation", "zero_motion", "err_mm"),
        cv=cell(gate, "cv_zero_motion_over_pairs"),
        test=cell(gate, "test", "zero_motion", "err_mm"),
        test_r2=cell(gate, "test", "zero_motion", "r2"),
        fast=cell(gate, "test", "zero_motion", "fast_mm"),
        ate=cell(traj, *T, "zero", "ate_rmse_mm"))
    row("Ridge features", True,
        syn=cell(mb, "datasets", "synthetic_test", "ridge_real_fitted", "err_mm"),
        syn_r2=cell(mb, "datasets", "synthetic_test", "ridge_real_fitted", "r2"),
        ood=cell(mb, "datasets", "synthetic_ood", "ridge_real_fitted", "err_mm"),
        val=cell(mb, "datasets", "real_validation", "ridge_real_fitted", "err_mm"),
        test=cell(mb, "datasets", "real_test", "ridge_real_fitted", "err_mm"),
        test_r2=cell(mb, "datasets", "real_test", "ridge_real_fitted", "r2"),
        fast=cell(mb, "datasets", "real_test", "ridge_real_fitted", "fast_mm"),
        ate=cell(traj, *T, "ridge", "ate_rmse_mm"))
    row("Conv--BiGRU, body frame", True,
        syn=cell(mb, "datasets", "synthetic_test", "synthetic_zero_shot", "err_mm"),
        syn_r2=cell(mb, "datasets", "synthetic_test", "synthetic_zero_shot", "r2"),
        ood=cell(mb, "datasets", "synthetic_ood", "synthetic_zero_shot", "err_mm"),
        direct=cell(mb, "datasets", "real_test", "synthetic_zero_shot", "err_mm"),
        val=cell(mb, "datasets", "real_validation", "synthetic_pretrain_real_finetune", "err_mm"),
        test=cell(mb, "datasets", "real_test", "synthetic_pretrain_real_finetune", "err_mm"),
        test_r2=cell(mb, "datasets", "real_test", "synthetic_pretrain_real_finetune", "r2"),
        fast=cell(mb, "datasets", "real_test", "synthetic_pretrain_real_finetune", "fast_mm"),
        ate=cell(traj, *T, "fine_tuned", "ate_rmse_mm"),
        par={"value": 0.337635, "source": "traj_run_v17/*.pt state_dict (337635 parameters)"})
    row("RoNIN-LSTM (adapted)",
        syn=cell(ext, "ronin_lstm", "zero_shot", "synthetic_test", "err_mm"),
        syn_r2=cell(ext, "ronin_lstm", "zero_shot", "synthetic_test", "r2"),
        direct=cell(ext, "ronin_lstm", "zero_shot", "real_test", "err_mm"),
        par=cell(lat, "models", "ronin_lstm", "parameters", scale=1e-6),
        lat=cell(lat, "models", "ronin_lstm", "ms_per_window"))
    row("RoNIN-ResNet (adapted)", True,
        syn=cell(ext, "ronin_resnet", "zero_shot", "synthetic_test", "err_mm"),
        syn_r2=cell(ext, "ronin_resnet", "zero_shot", "synthetic_test", "r2"),
        ood=cell(ood, "ronin_resnet", "synthetic_ood", "err_mm"),
        direct=cell(ext, "ronin_resnet", "zero_shot", "real_test", "err_mm"),
        val=cell("external_benchmark/ronin_resnet_s0.json", "fine_tuned_real_val", "err_mm"),
        test=cell(ext, "ronin_resnet", "fine_tuned", "real_test", "err_mm"),
        test_r2=cell(ext, "ronin_resnet", "fine_tuned", "real_test", "r2"),
        fast=cell(ext, "ronin_resnet", "fine_tuned", "real_test", "fast_mm"),
        ate=cell(traj, *T, "ronin_fine", "ate_rmse_mm"),
        par=cell(lat, "models", "ronin_resnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "ronin_resnet", "ms_per_window"))
    row("TLIO-ResNet (adapted)", True,
        syn=cell(ext, "tlio_resnet", "zero_shot", "synthetic_test", "err_mm"),
        syn_r2=cell(ext, "tlio_resnet", "zero_shot", "synthetic_test", "r2"),
        ood=cell(ood, "tlio_resnet", "synthetic_ood", "err_mm"),
        direct=cell(ext, "tlio_resnet", "zero_shot", "real_test", "err_mm"),
        val=cell("external_benchmark/tlio_resnet_s0.json", "fine_tuned_real_val", "err_mm"),
        test=cell(ext, "tlio_resnet", "fine_tuned", "real_test", "err_mm"),
        test_r2=cell(ext, "tlio_resnet", "fine_tuned", "real_test", "r2"),
        fast=cell(ext, "tlio_resnet", "fine_tuned", "real_test", "fast_mm"),
        par=cell(lat, "models", "tlio_resnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "tlio_resnet", "ms_per_window"))
    row("IMUNet (adapted)",
        syn=cell(ext, "imunet", "zero_shot", "synthetic_test", "err_mm"),
        syn_r2=cell(ext, "imunet", "zero_shot", "synthetic_test", "r2"),
        ood=cell(ood, "imunet", "synthetic_ood", "err_mm"),
        direct=cell(ext, "imunet", "zero_shot", "real_test", "err_mm"),
        val=cell("external_benchmark/optimized_ensemble.json", "candidate_validation",
                 "imunet_real_ensemble", "err_mm"),
        cv=cell(gate, "cv_calibration", "cv_err_mm_imunet"),
        test=cell(gate, "test", "imunet_cv_ensemble", "err_mm"),
        test_r2=cell(gate, "test", "imunet_cv_ensemble", "r2"),
        fast=cell(gate, "test", "imunet_cv_ensemble", "fast_mm"),
        ate=cell(traj, *T, "cv_imunet", "ate_rmse_mm"),
        par=cell(lat, "models", "imunet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "imunet", "ms_per_window"))
    row("PhysNet (proposed)",
        syn=cell(syn, "synthetic_test", "err_mm"), syn_r2=cell(syn, "synthetic_test", "r2"),
        ood=cell(syn, "synthetic_ood", "err_mm"), direct=cell(syn, "test", "err_mm"),
        val=cell("physnet_v1/eval_wd1.json", "val", "err_mm"),
        cv=cell(ngate, "cv_calibration", "cv_err_mm_physnet"),
        test=cell(ngate, "test", "physnet_cv_ensemble", "err_mm"),
        test_r2=cell(ngate, "test", "physnet_cv_ensemble", "r2"),
        fast=cell(ngate, "test", "physnet_cv_ensemble", "fast_mm"),
        ate=cell(traj, *T, "cv_physnet", "ate_rmse_mm"),
        par=cell(lat, "models", "physnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "physnet", "ms_per_window"))
    row("PhysNet + stillness gate",
        val=cell("physnet_a/eval_ens_gate.json", "val", "err_mm"),
        cv=cell(gate, "cv_calibration", "cv_err_mm_physnet"),
        test=cell(gate, "test", "physnet_cv_ensemble", "err_mm"),
        test_r2=cell(gate, "test", "physnet_cv_ensemble", "r2"),
        fast=cell(gate, "test", "physnet_cv_ensemble", "fast_mm"),
        ate=cell(gate, "test", "trajectory_physnet", "single_edge", "ate_rmse_mm"))
    for label, src in (("no gate", ngate), ("gate", gate)):
        row(f"Equal blend, {label}",
            cv=cell(src, "cv_calibration", "cv_err_mm_equal_blend"),
            test=cell(src, "test", "equal_blend_physnet_imunet", "err_mm"),
            test_r2=cell(src, "test", "equal_blend_physnet_imunet", "r2"),
            fast=cell(src, "test", "equal_blend_physnet_imunet", "fast_mm"),
            ate=cell(src, "test", "trajectory_equal_blend", "single_edge", "ate_rmse_mm"))
        row(f"Calibrated blend, {label}",
            cv=cell(src, "cv_calibration", "cv_err_mm_calibrated"),
            test=cell(src, "test", "cv_calibrated_blend", "err_mm"),
            test_r2=cell(src, "test", "cv_calibrated_blend", "r2"),
            fast=cell(src, "test", "cv_calibrated_blend", "fast_mm"),
            **({"ate": cell(traj, *T, "cv_cv_calibrated_blend", "ate_rmse_mm")}
               if label == "no gate" else {}))
    return rows


def _fmt(key, v):
    if key in ("syn_r2", "test_r2"):
        return f"${v:.3f}$" if v < 0 else f"{v:.3f}"
    if key == "par":
        return f"{v:.2f}"
    if key == "lat":
        return f"{v:.2f}"
    if key in ("direct", "ate"):
        return f"{v:.1f}"
    return f"{v:.2f}"


def to_latex(rows):
    learned = [r for r in rows if r["method"] != "Zero motion"]
    best = {}
    for key, _, _ in COLUMNS:
        vals = [r["cells"][key]["value"] for r in learned if key in r["cells"]]
        if not vals or key in ("par", "lat", "direct"):
            continue
        best[key] = min(vals) if key in LOWER_BETTER else max(vals)
    head1 = " & \\multicolumn{2}{c}{Synth.\\ test} & OOD & Real & Val & CV & \\multicolumn{3}{c}{Held-out test} & ATE & Par. & Lat. \\\\"
    head2 = "Method & mm & $R^2$ & mm & direct & mm & mm & mm & $R^2$ & fast & mm & M & ms \\\\"
    lines = [
        "% Generated by tools/build_summary_table.py; do not edit by hand.",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{All methods under one six-axis, 3-D, 3-s displacement protocol. "
        "Synthetic columns use synthetic-only models; ``real direct'' evaluates them on the "
        "held-out real recording without any real update. ``Val'': fixed-split validation; "
        "``CV'': leave-one-session-out errors of the two-seed fold ensembles with extra-train; "
        "``Held-out test'': CV fold ensembles (16 models) unless marked, with ``fast'' the error "
        "in the fastest third; ATE: pose-graph translation error with one anchor per connected "
        "component (31 components). Bold marks the best learned value per column; on the held-out "
        "test none of the pairwise differences between IMUNet, PhysNet and the blends is "
        "significant (Fig.~\\ref{fig:benchmark}(b)). Latency is per window for a batch of 16 on "
        "one GPU.}",
        "\\label{tab:summary}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular}{l" + "r" * len(COLUMNS) + "}", "\\toprule", head1, head2, "\\midrule",
    ]
    for r in rows:
        out = [r["method"]]
        for key, _, _ in COLUMNS:
            c = r["cells"].get(key)
            if c is None:
                out.append("--")
                continue
            s = _fmt(key, c["value"])
            if key in best and r["method"] != "Zero motion" and abs(c["value"] - best[key]) < 1e-9:
                s = f"\\textbf{{{s}}}"
            if key in ("test", "test_r2", "fast") and r["fixed_split_test"]:
                s += "$^{\\dagger}$" if key == "test" else ""
            if key == "val" and r["method"] == "PhysNet + stillness gate":
                s += "$^{\\S}$"
            out.append(s)
        lines.append(" & ".join(out) + " \\\\")
        if r["method"] in ("TLIO-ResNet (adapted)", "PhysNet + stillness gate"):
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\multicolumn{13}{p{0.97\\textwidth}}{\\footnotesize $^{\\dagger}$Single fixed-split model (fine-tuned from the "
        "twin where applicable). $^{\\S}$Five-seed ensemble trained with the rigidity recordings as "
        "extra data; 29.91~mm without the gate on the same data.}",
        "\\end{tabular}", "\\end{table*}", ""]
    return "\n".join(lines)


def main():
    rows = build()
    (DATA / "summary_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUT_TEX.write_text(to_latex(rows), encoding="utf-8")
    for r in rows:
        print(f"{r['method']:28s}", {k: round(v["value"], 2) for k, v in r["cells"].items()})


if __name__ == "__main__":
    main()
