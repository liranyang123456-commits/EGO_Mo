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
    mb, ext = "method_benchmark_20260924.json", "external_benchmark/test_benchmark.json"
    gate, ngate = "session_cv_gate_benchmark.json", "session_cv_benchmark.json"
    traj, lat = "trajectory_comparison/summary.json", "latency_benchmark.json"
    fac, matched, msum = "synthetic_factorial.json", "session_cv_matched_cvx.json", "matched_benchmark_summary.json"
    T = ("real_test", "metrics")
    rows = []

    def row(name, fixed_split_test=False, original_synthetic=False, **cells):
        rows.append({"method": name, "fixed_split_test": fixed_split_test,
                     "original_synthetic": original_synthetic, "cells": cells})

    def synthetic(config):
        """Matched two-seed synthetic-only ensemble of tools/run_synthetic_factorial.py."""
        return dict(syn=cell(fac, "results", config, "test", "err_mm"),
                    syn_r2=cell(fac, "results", config, "test", "r2"),
                    ood=cell(fac, "results", config, "ood", "err_mm"),
                    direct=cell(fac, "results", config, "real_test", "err_mm"))

    def matched_real(arch):
        """Leave-one-session-out fold ensembles trained like the IMUNet fold models."""
        return dict(cv=cell(msum, "cv_fold_ensemble_mm", arch),
                    test=cell(matched, "test", f"{arch}_cv_ensemble", "err_mm"),
                    test_r2=cell(matched, "test", f"{arch}_cv_ensemble", "r2"),
                    fast=cell(matched, "test", f"{arch}_cv_ensemble", "fast_mm"),
                    ate=cell(matched, "test", f"trajectory_{arch}", "single_edge", "ate_rmse_mm"))

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
        ate=cell(traj, *T, "zero", "ate_rmse_mm"),
        par={"value": 0.0, "source": "no trainable parameters"},
        lat={"value": 0.0, "source": "no computation"})
    row("Ridge features", True,
        syn=cell(mb, "datasets", "synthetic_test", "ridge_real_fitted", "err_mm"),
        syn_r2=cell(mb, "datasets", "synthetic_test", "ridge_real_fitted", "r2"),
        ood=cell(mb, "datasets", "synthetic_ood", "ridge_real_fitted", "err_mm"),
        val=cell(mb, "datasets", "real_validation", "ridge_real_fitted", "err_mm"),
        test=cell(mb, "datasets", "real_test", "ridge_real_fitted", "err_mm"),
        test_r2=cell(mb, "datasets", "real_test", "ridge_real_fitted", "r2"),
        fast=cell(mb, "datasets", "real_test", "ridge_real_fitted", "fast_mm"),
        ate=cell(traj, *T, "ridge", "ate_rmse_mm"))
    row("Conv--BiGRU, body frame", True, True,
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
    row("RoNIN-LSTM (adapted)", False, True,
        syn=cell(ext, "ronin_lstm", "zero_shot", "synthetic_test", "err_mm"),
        syn_r2=cell(ext, "ronin_lstm", "zero_shot", "synthetic_test", "r2"),
        direct=cell(ext, "ronin_lstm", "zero_shot", "real_test", "err_mm"),
        par=cell(lat, "models", "ronin_lstm", "parameters", scale=1e-6),
        lat=cell(lat, "models", "ronin_lstm", "ms_per_window"))
    row("RoNIN-ResNet (adapted)",
        **synthetic("ronin_resnet_body8_end"),
        val=cell("external_benchmark/ronin_resnet_s0.json", "fine_tuned_real_val", "err_mm"),
        **matched_real("ronin_resnet"),
        par=cell(lat, "models", "ronin_resnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "ronin_resnet", "ms_per_window"))
    row("TLIO-ResNet (adapted)",
        **synthetic("tlio_resnet_body8_end"),
        val=cell("external_benchmark/tlio_resnet_s0.json", "fine_tuned_real_val", "err_mm"),
        **matched_real("tlio_resnet"),
        par=cell(lat, "models", "tlio_resnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "tlio_resnet", "ms_per_window"))
    row("IMUNet (adapted)",
        **synthetic("imunet_body8_end"),
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
        **synthetic("physnet_anchor24_dense"),
        val=cell("physnet_v1/eval_wd1.json", "val", "err_mm"),
        cv=cell(ngate, "cv_calibration", "cv_err_mm_physnet"),
        test=cell(ngate, "test", "physnet_cv_ensemble", "err_mm"),
        test_r2=cell(ngate, "test", "physnet_cv_ensemble", "r2"),
        fast=cell(ngate, "test", "physnet_cv_ensemble", "fast_mm"),
        ate=cell(traj, *T, "cv_physnet", "ate_rmse_mm"),
        par=cell(lat, "models", "physnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "physnet", "ms_per_window"))
    row("PhysNet + stillness gate",
        **synthetic("physnet_anchor24_gate"),
        val=cell("physnet_a/eval_ens_gate.json", "val", "err_mm"),
        cv=cell(gate, "cv_calibration", "cv_err_mm_physnet"),
        test=cell(gate, "test", "physnet_cv_ensemble", "err_mm"),
        test_r2=cell(gate, "test", "physnet_cv_ensemble", "r2"),
        fast=cell(gate, "test", "physnet_cv_ensemble", "fast_mm"),
        ate=cell(gate, "test", "trajectory_physnet", "single_edge", "ate_rmse_mm"),
        # Same network as PhysNet; tools/benchmark_latency.py times a gated fold model.
        par=cell(lat, "models", "physnet", "parameters", scale=1e-6),
        lat=cell(lat, "models", "physnet", "ms_per_window"))
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
            # tools/calibrated_blend_ate.py solves the same end-point graph for the gated blend.
            ate=(cell(traj, *T, "cv_cv_calibrated_blend", "ate_rmse_mm") if label == "no gate" else
                 cell("calibrated_blend_ate.json", "held_out", "cvg", "cv_calibrated_blend", "ate_rmse_mm")))
    return rows


def _fmt(key, v):
    if key in ("syn_r2", "test_r2"):
        return f"${v:.3f}$" if v < 0 else f"{v:.3f}"
    if key in ("par", "lat"):
        return "0" if v == 0 else f"{v:.2f}"
    if key in ("direct", "ate"):
        return f"{v:.1f}"
    return f"{v:.2f}"


FAMILIES = ["Zero motion", "RoNIN-ResNet (adapted)", "TLIO-ResNet (adapted)", "IMUNet (adapted)",
            "PhysNet (proposed)", "PhysNet + stillness gate"]
BLENDS = {"Equal blend, no gate": "Equal", "Calibrated blend, no gate": "Calibrated",
          "Equal blend, gate": "Equal, gated PhysNet", "Calibrated blend, gate": "Calibrated, gated PhysNet"}
FIXED = {"Ridge features": "Ridge features", "Conv--BiGRU, body frame": "Conv--BiGRU, body frame"}
BLEND_KEYS = ["cv", "test", "test_r2", "fast", "ate"]
FIXED_KEYS = ["syn", "syn_r2", "ood", "val", "test", "test_r2", "fast", "ate"]
# RoNIN-LSTM stays in datasets/summary_table.json (Fig. 9(a)) but is described in the table note.
OMITTED = {"RoNIN-LSTM (adapted)"}


def to_latex(rows):
    by_name = {r["method"]: r for r in rows}
    assert set(by_name) == set(FAMILIES) | set(BLENDS) | set(FIXED) | OMITTED, "unassigned table row"
    learned = [r for r in rows if r["method"] != "Zero motion"]
    best = {}
    for key, _, _ in COLUMNS:
        vals = [r["cells"][key]["value"] for r in learned if key in r["cells"]]
        # "val" mixes earlier fixed-split trainings that are not matched between rows: no ranking
        if not vals or key in ("par", "lat", "direct", "val"):
            continue
        best[key] = min(vals) if key in LOWER_BETTER else max(vals)

    def cells(r, keys):
        out = []
        for key in keys:
            c = r["cells"][key]          # a missing cell is an error: every panel lists only its own columns
            txt = _fmt(key, c["value"])
            if key in best and r["method"] != "Zero motion" and abs(c["value"] - best[key]) < 1e-9:
                txt = f"\\textbf{{{txt}}}"
            if key == "syn" and r["original_synthetic"]:
                txt += "$^{\\ddagger}$"
            if key == "val" and r["method"] == "PhysNet + stillness gate":
                txt = f"\\rev{{{txt}}}" + "$^{\\S}$"
            out.append(txt)
        return out

    all_keys = [k for k, _, _ in COLUMNS]
    lines = [
        "% Generated by tools/build_summary_table.py; do not edit by hand.",
        "\\begin{table*}[t]", "\\centering",
        "\\caption{Benchmark of All Estimators Under the Common Six-Axis, 3-D, 3-s Displacement Protocol.}",
        "\\label{tab:summary}", "\\footnotesize", "\\setlength{\\tabcolsep}{3.2pt}",
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "r" * len(COLUMNS) + "@{}}", "\\toprule",
        "\\multicolumn{13}{@{}l}{\\rev{\\emph{(a) Estimator families under the matched protocol}}} \\\\",
        " & \\multicolumn{2}{c}{Synth.\\ test} & OOD & Real & Val & CV & \\multicolumn{3}{c}{Held-out test} & ATE & Par. & Lat. \\\\",
        "Method & mm & $R^2$ & mm & direct & mm & mm & mm & $R^2$ & fast & mm & M & ms \\\\",
        "\\midrule",
    ]
    for name in FAMILIES:
        lines.append(" & ".join([name] + cells(by_name[name], all_keys)) + " \\\\")
        if name in ("Zero motion", "TLIO-ResNet (adapted)"):
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular*}", "", "\\vspace{3pt}",
              "\\setlength{\\tabcolsep}{3pt}",
              "\\begin{tabular}[t]{@{}lrrrrr@{}}", "\\toprule",
              "\\multicolumn{6}{@{}l}{\\rev{\\emph{(b) Blends of the PhysNet and IMUNet fold ensembles}}} \\\\",
              " & CV & \\multicolumn{3}{c}{Held-out test} & ATE \\\\",
              "Weights & mm & mm & $R^2$ & fast & mm \\\\", "\\midrule"]
    for name, short in BLENDS.items():
        lines.append(" & ".join([short] + cells(by_name[name], BLEND_KEYS)) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}\\hfill",
              "\\begin{tabular}[t]{@{}lrrrrrrrr@{}}", "\\toprule",
              "\\multicolumn{9}{@{}l}{\\rev{\\emph{(c) Single models of the fixed split}}} \\\\",
              " & \\multicolumn{2}{c}{Synth.\\ test} & OOD & Val & \\multicolumn{3}{c}{Held-out test} & ATE \\\\",
              "Method & mm & $R^2$ & mm & mm & mm & $R^2$ & fast & mm \\\\", "\\midrule"]
    for name, short in FIXED.items():
        lines.append(" & ".join([short] + cells(by_name[name], FIXED_KEYS)) + " \\\\")
    lines += [
        "\\bottomrule", "\\end{tabular}", "",
        "\\vspace{2pt}",
        "\\parbox{\\textwidth}{\\footnotesize \\rev{(a)} \\rev{Synth.\\ columns: matched two-seed "
        "synthetic-only ensembles (Table~\\ref{tab:ablation}, Group~D); ``Real direct'' scores them on the "
        "held-out recording without real updates.} \\rev{``Val'': earlier fixed-split trainings, not part of the "
        "matched protocol (rows differ in seeds, rigidity data, pretraining);} ``CV'' / ``Held-out test'': "
        "two-seed leave-one-session-out fold ensembles (16 models, \\rev{same protocol and data for every family}); "
        "``fast'': fastest third; ATE: 31-component pose-graph error; \\rev{OOD: synthetic out-of-distribution split; Par.: trainable parameters;} Lat.: batched GPU compute per window "
        "($B{=}16$; \\rev{online $B{=}1$: 2000-ms future wait, gridding below 1~ms, 44.3-ms median "
        "16-model inference, $\\approx$2.05~s; Section~\\ref{sec:estimators}}). "
        "\\rev{(b) Blends of the real-data fold ensembles of (a); only the real-data columns apply. "
        "(c) Earlier single models, fixed split only; the adapted RoNIN-LSTM did not beat zero "
        "motion on the synthetic test (58.46~mm) and is omitted.} Bold: best learned value over all panels"
        "\\rev{, Val excluded} "
        "(\\rev{on the held-out test every fold ensemble beats zero motion; only non-gated PhysNet is significantly "
        "worse than RoNIN/TLIO (TLIO: not in a second training)}). \\rev{$^{\\ddagger}$Single seed, original synthetic budget.} "
        "$^{\\S}$Five-seed ensemble with rigidity data (29.91~mm without gate).}",
        "\\end{table*}", ""]
    return "\n".join(lines)


T1 = "traj_20260926_143837"
OUT_T1_TEX = ROOT / "paper" / "sections" / "t1_table.tex"


def t1_table() -> str:
    """Prospective recording T1 and both held-out recordings pooled."""
    s = json.loads((DATA / "matched_benchmark_summary.json").read_text(encoding="utf-8"))
    r = s["test_sessions"][T1]
    pooled = {"cvx": s["test_pooled"]["cvx"], "cvg": s["test_pooled"]["cvg"]}
    blend_ate = json.loads((DATA / "calibrated_blend_ate.json").read_text(encoding="utf-8"))["t1"]
    for tag, key in (("cvx", "equal_blend"), ("cvg", "equal_blend_gate")):
        # the script that fills the calibrated blends must reproduce the stored equal-blend values
        assert abs(blend_ate[tag]["equal_blend"]["ate_rmse_mm"] - r[key]["ate_single_edge_mm"]) < 0.01
    rows = [
        ("Zero motion", r["zero_motion"], r["zero_ate"]["ate_rmse_mm"], pooled["cvx"]["zero_motion"]),
        ("IMUNet", r["imunet"], r["imunet"]["ate_single_edge_mm"], pooled["cvx"]["imunet"]),
        ("RoNIN-ResNet", r["ronin_resnet"], r["ronin_resnet"]["ate_single_edge_mm"], pooled["cvx"]["ronin_resnet"]),
        ("TLIO-ResNet", r["tlio_resnet"], r["tlio_resnet"]["ate_single_edge_mm"], pooled["cvx"]["tlio_resnet"]),
        ("PhysNet", r["physnet"], r["physnet"]["ate_single_edge_mm"], pooled["cvx"]["physnet"]),
        ("PhysNet + gate", r["physnet_gate"], r["physnet_gate"]["ate_single_edge_mm"], pooled["cvg"]["physnet"]),
        ("Equal blend", r["equal_blend"], r["equal_blend"]["ate_single_edge_mm"], pooled["cvx"]["equal_blend"]),
        ("Equal blend, gate", r["equal_blend_gate"], r["equal_blend_gate"]["ate_single_edge_mm"],
         pooled["cvg"]["equal_blend"]),
        ("Calibrated blend", r["calibrated_blend"], blend_ate["cvx"]["cv_calibrated_blend"]["ate_rmse_mm"],
         pooled["cvx"]["cv_calibrated_blend"]),
        ("Calibrated blend, gate", r["calibrated_blend_gate"], blend_ate["cvg"]["cv_calibrated_blend"]["ate_rmse_mm"],
         pooled["cvg"]["cv_calibrated_blend"]),
    ]
    best = {k: min(m[k] for _n, m, _a, _p in rows[1:]) for k in ("err_mm", "fast_mm")}
    best["r2"] = max(m["r2"] for _n, m, _a, _p in rows[1:])
    best["ate"] = min(a for _n, _m, a, _p in rows[1:] if a is not None)
    best["pooled"] = min(p["err_mm"] for _n, _m, _a, p in rows[1:])

    def f(v, key, digits=2):
        if v is None:
            return "--"
        txt = f"{v:.{digits}f}" if v >= 0 or key != "r2" else f"${v:.{digits}f}$"
        return f"\\textbf{{{txt}}}" if key in best and abs(v - best[key]) < 1e-9 else txt

    n_t1, n_pool = r["n"], pooled["cvx"]["n"]
    lines = [
        "% Generated by tools/build_summary_table.py; do not edit by hand.",
        "\\begin{table}[t]", "\\centering",
        "\\caption{\\rev{Prospective Post-Freeze Recording T1 and Pooled Held-Out Results.}}",
        "\\label{tab:t1}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lrrrrr@{}}", "\\toprule",
        " & \\multicolumn{4}{c}{\\rev{T1}} & \\rev{Pooled} \\\\",
        "\\rev{Method} & \\rev{$e_p$ mm} & \\rev{$R^2$} & \\rev{fast mm} & \\rev{ATE mm} & \\rev{$e_p$ mm} \\\\",
        "\\midrule",
    ]
    for name, m, ate, p in rows:
        cells = [f(m["err_mm"], "err_mm"), f(m["r2"], "r2", 3), f(m["fast_mm"], "fast_mm"),
                 f(ate, "ate", 1), f(p["err_mm"], "pooled")]
        if name == "Zero motion":
            cells = [c.replace("\\textbf{", "{") for c in cells]
        lines.append(" & ".join([f"\\rev{{{name}}}"] + [f"\\rev{{{c}}}" for c in cells]) + " \\\\")
        if name in ("Zero motion", "TLIO-ResNet", "PhysNet + gate"):
            lines.append("\\midrule")
    lines += [
        "\\bottomrule",
        "\\multicolumn{6}{@{}p{0.98\\columnwidth}@{}}{\\scriptsize \\rev{T1: 122.4~s, "
        f"{n_t1} windows; Pooled: both held-out recordings ({n_pool} windows). Fold ensembles and metrics as in "
        "Table~\\ref{tab:summary}. "
        "Bold: best learned value.}}",
        "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(lines)


def main():
    rows = build()
    (DATA / "summary_table.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    OUT_TEX.write_text(to_latex(rows), encoding="utf-8")
    OUT_T1_TEX.write_text(t1_table(), encoding="utf-8")
    for r in rows:
        print(f"{r['method']:28s}", {k: round(v["value"], 2) for k, v in r["cells"].items()})


if __name__ == "__main__":
    main()
