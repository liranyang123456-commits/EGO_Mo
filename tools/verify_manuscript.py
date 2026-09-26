#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression checks for the numbers, labels and references in the manuscript.

This script does not recompute models. It verifies that every headline number
reported in the paper still agrees with the immutable evaluation artefacts,
and that LaTeX citations/labels are internally complete.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
PAPER = ROOT / "paper"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(name: str, got: float, expected: float, tol: float = 0.011):
    if not np.isclose(float(got), expected, atol=tol, rtol=0):
        raise AssertionError(f"{name}: source={got}, manuscript={expected}")
    print(f"OK {name:46s} {float(got):8.3f}")


def results():
    cv = load(DATA / "session_cv_gate_benchmark.json")
    syn = load(DATA / "physnet_v1" / "eval_synonly_181622.json")
    ext = load(DATA / "external_benchmark" / "test_benchmark.json")
    ood = load(DATA / "external_benchmark" / "zero_shot_synthetic_ood.json")
    unc = load(DATA / "uncertainty_val.json")
    zupt = load(DATA / "zupt_bound.json")
    traj = load(DATA / "trajectory_comparison" / "summary.json")["real_test"]["metrics"]
    pose = load(DATA / "pure_imu_pose_benchmark.json")["sessions"][
        "traj_20260923_023422"]["cv_calibrated_blend"]

    # Synthetic/direct comparison (Table IV).
    close("PhysNet synthetic test error", syn["synthetic_test"]["err_mm"], 16.69)
    close("PhysNet synthetic test R2", syn["synthetic_test"]["r2"], 0.864)
    close("PhysNet synthetic OOD error", syn["synthetic_ood"]["err_mm"], 34.27)
    close("PhysNet real direct error", syn["test"]["err_mm"], 90.19)
    close("IMUNet synthetic test error",
          ext["imunet"]["zero_shot"]["synthetic_test"]["err_mm"], 21.89)
    close("IMUNet synthetic OOD error",
          ood["imunet"]["synthetic_ood"]["err_mm"], 37.20)
    close("IMUNet real direct error",
          ext["imunet"]["zero_shot"]["real_test"]["err_mm"], 118.78)

    # Real-data comparison (Table V and abstract).
    close("zero-motion test error", cv["test"]["zero_motion"]["err_mm"], 49.67)
    close("IMUNet CV single-model mean", cv["cv_mean_over_pairs"]["imunet"], 26.38)
    close("PhysNet+gate CV single-model mean", cv["cv_mean_over_pairs"]["physnet"], 25.56)
    close("IMUNet sealed-test error",
          cv["test"]["imunet_cv_ensemble"]["err_mm"], 42.93)
    close("PhysNet+gate sealed-test error",
          cv["test"]["physnet_cv_ensemble"]["err_mm"], 43.55)
    close("equal-blend sealed-test error",
          cv["test"]["equal_blend_physnet_imunet"]["err_mm"], 42.21)
    close("equal-blend sealed-test R2",
          cv["test"]["equal_blend_physnet_imunet"]["r2"], 0.276)
    close("calibrated-blend CV error",
          cv["cv_calibration"]["cv_err_mm_calibrated"], 24.51)
    close("calibrated-blend sealed-test error",
          cv["test"]["cv_calibrated_blend"]["err_mm"], 42.29)
    close("calibrated-blend sealed-test R2",
          cv["test"]["cv_calibrated_blend"]["r2"], 0.279)
    close("calibrated-blend fast-third error",
          cv["test"]["cv_calibrated_blend"]["fast_mm"], 72.10)
    # eval_session_cv stores mean(err_b - err_a) under the key "a_minus_b":
    # "physnet_minus_imunet" is IMUNet - PhysNet. The paper reports the
    # PhysNet - IMUNet interval, i.e. the negated, swapped bounds.
    stored = cv["test"]["physnet_minus_imunet_mm_95ci_blocks"]
    close("PhysNet-IMUNet error-difference CI low", -stored[1], -1.84)
    close("PhysNet-IMUNet error-difference CI high", -stored[0], 3.18)
    close("PhysNet improvement over zero CI low",
          cv["test"]["physnet_minus_zero_mm_95ci_blocks"][0], 2.16)
    close("PhysNet improvement over zero CI high",
          cv["test"]["physnet_minus_zero_mm_95ci_blocks"][1], 10.80)
    # CV column of Table V: two-seed fold-ensemble errors.
    close("PhysNet+gate CV fold-ensemble error", cv["cv_calibration"]["cv_err_mm_physnet"], 24.95)
    close("IMUNet CV fold-ensemble error", cv["cv_calibration"]["cv_err_mm_imunet"], 25.77)
    cv_ng = load(DATA / "session_cv_benchmark.json")
    close("PhysNet CV fold-ensemble error", cv_ng["cv_calibration"]["cv_err_mm_physnet"], 25.17)
    physnet_wins = sum(
        fold["physnet"]["err_mm"] < fold["imunet"]["err_mm"]
        for fold in cv["folds"].values()
    )
    if physnet_wins != 6:
        raise AssertionError(f"PhysNet fold wins: source={physnet_wins}, manuscript=6")
    print(f"OK {'PhysNet fold wins':46s} {physnet_wins:8d}")

    # Velocity-proxy tests and shrinkage (Section VIII-B, Fig. 5).
    vp = load(DATA / "velocity_proxy_check.json")["variants"]
    close("window-mean proxy, in-sample %", vp["full"]["in_sample_reduction_percent"], 14.1, 0.06)
    close("window-mean proxy, cross-session %", vp["full"]["cross_session_reduction_percent"], 4.3, 0.06)
    close("context proxy, in-sample %", vp["context"]["in_sample_reduction_percent"], 1.8, 0.06)
    close("context proxy, cross-session %", vp["context"]["cross_session_reduction_percent"], -6.5, 0.06)
    close("oracle |v(ta) dt| mean", vp["oracle_initial"]["mean_proxy_magnitude_mm"], 75.8, 0.06)
    ev = np.load(DATA / "physnet_v1" / "eval_wd1.npz", allow_pickle=True)
    pv, yv = ev["val_pred"] * 1000, ev["val_y"] * 1000
    for k, s in enumerate((0.37, 0.48, 0.20)):
        close(f"prediction slope axis {k}", np.polyfit(yv[:, k], pv[:, k], 1)[0], s, 0.006)
    for k, r in enumerate((-0.88, -0.78, -0.92)):
        close(f"residual-displacement corr axis {k}",
              np.corrcoef(yv[:, k], (pv - yv)[:, k])[0, 1], r, 0.006)
    close("median |pred| / median |ref|", np.median(np.linalg.norm(pv, axis=1)) /
          np.median(np.linalg.norm(yv, axis=1)), 0.44, 0.006)

    # Uncertainty (fixed-split NLL ensemble).
    oos = load(DATA / "uncertainty_oos.json")
    close("uncertainty cross-session 1-sigma", oos["cross_session"]["pooled_coverage_1sigma"], 0.810)
    close("uncertainty cross-session 2-sigma", oos["cross_session"]["pooled_coverage_2sigma"], 0.942)
    close("uncertainty test 1-sigma", oos["test"]["total_rescaled"]["coverage_1sigma"], 0.632)
    close("uncertainty test 2-sigma", oos["test"]["total_rescaled"]["coverage_2sigma"], 0.852)
    close("uncertainty test RMS z", oos["test"]["total_rescaled"]["rms_z"], 1.55, 0.006)

    # Latency (Table IV), same batch and procedure for every model.
    lat = load(DATA / "latency_benchmark.json")["models"]
    for name, ms in (("physnet", 0.16), ("imunet", 0.14), ("tlio_resnet", 0.14),
                     ("ronin_resnet", 0.13), ("ronin_lstm", 0.15)):
        close(f"latency {name}", lat[name]["ms_per_window"], ms, 0.006)
    close("uncertainty validation scale", unc["validation_scale"], 4.559)
    close("uncertainty 1-sigma coverage",
          unc["val"]["total_rescaled"]["coverage_1sigma"], 0.814)
    close("uncertainty 2-sigma coverage",
          unc["val"]["total_rescaled"]["coverage_2sigma"], 0.945)
    # The sub-2-s value is based on one interval; the 2--4-s values are pooled
    # over train/validation/rigidity exactly as plotted.
    close("ZUPT reference <2-s example",
          zupt["ref"]["extra_train (rigid)"]["by_span"]["0.8-2s"]["err_mm"],
          0.70)
    close("ZUPT gyro <2-s example",
          zupt["gyro"]["extra_train (rigid)"]["by_span"]["0.8-2s"]["err_mm"],
          1.20)
    def pooled(attitude, span):
        num = den = 0.0
        for group in ("train", "val", "extra_train (rigid)"):
            entry = zupt[attitude][group].get("by_span", {}).get(span)
            if entry:
                num += entry["err_mm"] * entry["n"]
                den += entry["n"]
        return num / den
    close("ZUPT reference pooled 2--4 s", pooled("ref", "2-4s"), 53.86)
    close("ZUPT gyro pooled 2--4 s", pooled("gyro", "2-4s"), 104.08)

    # Trajectory table/figure.
    close("zero-motion ATE", traj["zero"]["ate_rmse_mm"], 119.24)
    close("Ridge ATE", traj["ridge"]["ate_rmse_mm"], 99.20)
    close("IMUNet CV-ensemble ATE", traj["cv_imunet"]["ate_rmse_mm"], 82.21)
    close("PhysNet CV-ensemble ATE", traj["cv_physnet"]["ate_rmse_mm"], 87.79)
    close("equal-blend ATE", traj["cv_equal_blend"]["ate_rmse_mm"], 81.06)
    close("calibrated-blend ATE",
          traj["cv_cv_calibrated_blend"]["ate_rmse_mm"], 80.70)
    close("pure-IMU orientation RMSE",
          pose["orientation"]["rmse_deg"], 4.5143)
    close("pure-IMU 3-s rotation RPE",
          pose["relative_pose_error"]["3s"]["rotation_rmse_deg"], 3.4796)
    close("pure-IMU calibrated-blend ATE",
          pose["translation"]["ate_rmse_mm"], 80.59)
    close("pure-IMU calibrated-blend 3-s translation RPE",
          pose["relative_pose_error"]["3s"]["translation_rmse_mm"], 50.92)


def text_claims():
    """Numbers whose wording in the manuscript was corrected must stay present,
    and retired claims must not reappear."""
    text = "\n".join(p.read_text(encoding="utf-8")
                     for p in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))])
    required = ["$[-1.8,\\,+3.2]$", "24.95", "25.77", "25.17", "31 connected",
                "14.1\\%", "4.3\\%", "0.37/0.48/0.20", "81.0\\%", "63.2\\%",
                "0.13--0.16~ms", "(-20.9,\\,-4.1,\\,-17.8)", "10--19\\%"]
    retired = ["26.3~mm", "lower inference latency", "reaches that limit",
               "information-limit measurement", "More than\nmore than"]
    for s in required:
        if s not in text:
            raise AssertionError(f"manuscript no longer contains verified value {s!r}")
    for s in retired:
        if s in text:
            raise AssertionError(f"retired claim reappeared: {s!r}")
    print(f"OK text claims: {len(required)} present, {len(retired)} retired absent")


def latex_integrity():
    tex_paths = [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))]
    text = "\n".join(p.read_text(encoding="utf-8") for p in tex_paths)
    labels = set(re.findall(r"\\label\{([^}]+)\}", text))
    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text))
    missing_labels = sorted(refs - labels)
    if missing_labels:
        raise AssertionError(f"undefined LaTeX labels: {missing_labels}")

    cited = set()
    for match in re.findall(r"\\cite\{([^}]+)\}", text):
        cited.update(k.strip() for k in match.split(","))
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    missing_bib = sorted(cited - bib_keys)
    if missing_bib:
        raise AssertionError(f"undefined BibTeX keys: {missing_bib}")

    print(f"OK LaTeX labels: {len(labels)} defined, {len(refs)} referenced")
    print(f"OK BibTeX: {len(cited)} cited keys, all defined")


if __name__ == "__main__":
    results()
    text_claims()
    latex_integrity()
    print("Manuscript regression checks passed.")
