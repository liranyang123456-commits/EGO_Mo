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
from pathlib import Path, PureWindowsPath

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
PAPER = ROOT / "paper"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


# Rig adaptation with three rejected paused takes added (response letter only):
# continuous mm, paused mm, gain, its interval, matched difference, its interval.
PAUSED_HEAVY = {
    "ab": (35.65, 13.62, 22.6, 7.3, 37.5, -15.8, -23.8, -8.2),
    "ba": (24.54, 15.81, 22.9, 11.3, 33.9, -8.6, -12.5, -4.6),
}


def close(name: str, got: float, expected: float, tol: float = 0.011):
    if not np.isclose(float(got), expected, atol=tol, rtol=0):
        raise AssertionError(f"{name}: source={got}, manuscript={expected}")
    print(f"OK {name:46s} {float(got):8.3f}")


def results():
    budget = load(DATA / "label_uncertainty.json")
    for case, bounds in (("nominal", (2.67, 3.05)), ("perturbed", (3.40, 3.81))):
        for value, expected in zip(budget["ranges_of_medians_mm"]["U_" + case], bounds):
            close(f"label budget {case} (mm)", value, expected, 0.005)
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
    # Post-freeze v3 analyses (validation / existing fold predictions only).
    v3 = load(DATA / "physnet_v3" / "results.json")
    slopes = np.array([v3[k]["slope"] for k in ("base", "bias_local", "beta5", "nll_main", "nogate_beta5", "nogate")])
    close("v3 slope x min", slopes[:, 0].min(), 0.42, 0.006)
    close("v3 slope x max", slopes[:, 0].max(), 0.48, 0.006)
    close("v3 slope z min", slopes[:, 2].min(), 0.19, 0.006)
    close("v3 slope z max", slopes[:, 2].max(), 0.29, 0.006)
    sweep = load(DATA / "physnet_v3" / "scale_sweep.json")
    close("v3 gated ATE alpha 1.0", sweep["base"]["1.0"]["ate_mm"], 126.4, 0.06)
    close("v3 gated ATE alpha 1.5", sweep["base"]["1.5"]["ate_mm"], 152.2, 0.06)
    mixj = load(DATA / "physnet_v3" / "mixture.json")
    close("mixture CV error", mixj["cv_mm"]["mix"], 24.63, 0.006)
    close("mixture test error", mixj["test"]["mix"]["err_mm"], 43.89)
    psf = load(DATA / "sim_reference_validation_psf.json")["depths"]["250"]
    close("sim PSF 250 mm RMSE", psf["nominal"]["heldout_rmse_mm"], 0.56, 0.006)
    close("sim PSF 250 mm max |scale| %", max(abs(v) for v in psf["nominal"]["scale_error_percent"].values()), 1.37, 0.006)
    close("sim PSF 250 mm max cross-axis", max(psf["nominal"]["cross_axis_rmse_mm"].values()), 0.64, 0.006)
    close("sim PSF 250 mm MC median RMSE", psf["monte_carlo"]["heldout_rmse_mm_p50_p95"][0], 0.86, 0.006)
    fam = load(DATA / "synthetic_by_family.json")
    periodic = [fam["synthetic_test"][f]["reduction_percent"] for f in ("circle", "depth", "lateral", "mixed")]
    close("synthetic periodic reduction min %", min(periodic), 79, 0.5)
    close("synthetic periodic reduction max %", max(periodic), 87, 0.5)
    close("synthetic random-spline reduction %", fam["synthetic_test"]["random_spline"]["reduction_percent"], 41, 0.5)
    close("synthetic random-spline OOD reduction %", fam["synthetic_ood"]["random_spline"]["reduction_percent"], -16, 0.5)
    cvc = load(DATA / "physnet_v3cv" / "cv_clean.json")
    close("clean-pool CV error", cvc["cv_mm"]["clean"], 26.72, 0.006)
    close("frozen CV error on same six", cvc["cv_mm"]["frozen"], 27.35, 0.006)
    close("clean-pool test error", cvc["test"]["clean"]["err_mm"], 43.38)
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
    close("ZUPT raw gyro pooled 2--4 s", pooled("gyro_raw", "2-4s"), 78.1, 0.06)
    close("ZUPT gyro pooled 8--30 s (m)", pooled("gyro", "8-30s") / 1000, 4.7, 0.05)
    close("ZUPT raw gyro pooled 8--30 s (m)", pooled("gyro_raw", "8-30s") / 1000, 2.0, 0.05)

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


def twin_protocol():
    """Realistic-twin pretraining and the acquisition-protocol study."""
    s = load(DATA / "twin_realistic_summary.json")
    rv = s["real_val_3seed"]
    close("real-only PhysNet 3-seed val", rv["physnet_real_only"]["err_mm"], 29.43)
    close("raw-gyro PhysNet 3-seed val", rv["physnet_bias_none"]["err_mm"], 29.44)
    close("realistic-twin pretrain val", rv["physnet_pre_realistic"]["err_mm"], 29.56)
    close("class-balanced pretrain val", rv["physnet_pre_realistic_bal"]["err_mm"], 29.29)
    close("IMUNet real-only 3-seed val", rv["imunet_real_only"]["err_mm"], 30.63)
    close("IMUNet realistic-twin fine-tuned", rv["imunet_pre_realistic"]["err_mm"], 31.41)
    zs = s["zero_shot_real_val"]
    close("IMUNet direct transfer min", min(zs["imunet_realistic"]["err_mm"]), 37.9, 0.06)
    close("IMUNet direct transfer max", max(zs["imunet_realistic"]["err_mm"]), 38.6, 0.06)
    phys = zs["physnet_pre_realistic"]["err_mm"] + zs["physnet_pre_realistic_bal"]["err_mm"]
    close("PhysNet direct transfer min", min(phys), 37.5, 0.06)
    close("PhysNet direct transfer max", max(phys), 38.8, 0.06)
    table = {"pause_4": (72, 35.3, 34.7, 30.0, 57.5, 33.3),
             "pause_8": (50, 42.2, 39.7, 36.7, 55.7, 43.2),
             "pause_12.5": (43, 43.2, 42.5, 40.3, 55.3, 46.5),
             "pause_20": (31, 46.5, 44.5, 41.8, 54.2, 47.8),
             "pause_inf": (15, 48.6, 44.2, 44.1, 55.5, 55.3)}
    p = s["protocol"]
    for c, row in table.items():
        e = p[c]
        close(f"{c} stop-in-window share", 100 * e["stop_position_share"]["window"], row[0], 0.51)
        close(f"{c} PhysNet session bias", e["physnet_session_bias"]["err_mm"], row[1], 0.051)
        close(f"{c} PhysNet raw gyro", e["physnet_raw_gyro"]["err_mm"], row[2], 0.051)
        close(f"{c} PhysNet raw + switch", e["physnet_raw_gyro_switch"]["err_mm"], row[3], 0.051)
        if len(row) > 4:
            close(f"{c} IMUNet", e["imunet"]["err_mm"], row[4], 0.051)
            close(f"{c} IMUNet + switch", e["imunet_switch"]["err_mm"], row[5], 0.051)
    close("protocol gain 12.5 s vs none (%)",
          100 * (1 - p["pause_12.5"]["physnet_raw_gyro_switch"]["err_mm"]
                 / p["pause_inf"]["physnet_raw_gyro_switch"]["err_mm"]), 9, 0.51)
    close("protocol gain 4 s vs none (%)",
          100 * (1 - p["pause_4"]["physnet_raw_gyro_switch"]["err_mm"]
                 / p["pause_inf"]["physnet_raw_gyro_switch"]["err_mm"]), 32, 0.51)
    close("ideal IMU 4 s", p["ideal_imu"]["pause_4"]["session_bias"], 35.1, 0.051)
    close("ideal IMU no stops", p["ideal_imu"]["pause_inf"]["session_bias"], 46.8, 0.051)
    close("8-s look-back 4 s", p["pause_4"]["physnet_session_bias_8s_context"]["err_mm"], 36.9, 0.051)
    close("label rate 10 fps", p["reference_fps"]["10"], 43.8, 0.051)
    close("label rate 30 fps", p["reference_fps"]["30"], 42.7, 0.051)
    close("switch threshold PhysNet (s)", p["switch_max_span_s"]["physnet"], 2.0)
    close("switch threshold IMUNet (s)", p["switch_max_span_s"]["imunet"], 3.0)
    sw = s["real_windows"]["switch"]
    close("real switched share (%)", 100 * sw["physics_share"], 8.5, 0.051)
    close("real val learned before switch", sw["all"]["learned"], 29.63)
    close("real val after switch", sw["all"]["hybrid"], 29.52)
    cls = s["real_windows"]["by_stop_position"]
    close("real free-window share (%)", 100 * cls["free"]["share"], 65, 0.51)
    close("real context-window share (%)", 100 * cls["context"]["share"], 8, 0.51)
    close("real in-window share (%)", 100 * cls["window"]["share"], 27, 0.51)
    close("real free-window PhysNet", cls["free"]["physnet"], 36.5, 0.051)
    close("real in-window PhysNet", cls["window"]["physnet"], 13.7, 0.051)
    oracle = load(DATA / "stop_anchor_oracle_ideal.json")["synthetic"]["ideal_pause_4"]
    close("ideal stop-anchored, stop in window", oracle["window"]["physics_on_detected"], 4.4, 0.051)


def revision_results():
    """Numbers added in the revision, checked against the files that produce them."""
    fac = load(DATA / "synthetic_factorial.json")["results"]
    close("matched twin: PhysNet 24-ch dense", fac["physnet_anchor24_dense"]["test"]["err_mm"], 16.77)
    close("matched twin: RoNIN-ResNet 8-ch", fac["ronin_resnet_body8_end"]["test"]["err_mm"], 18.10)
    close("matched twin: IMUNet 8-ch", fac["imunet_body8_end"]["test"]["err_mm"], 19.09)
    close("matched twin: TLIO-ResNet 8-ch OOD", fac["tlio_resnet_body8_end"]["ood"]["err_mm"], 33.54)
    close("matched twin: IMUNet 8-ch real direct", fac["imunet_body8_end"]["real_test"]["err_mm"], 82.15)
    m = load(DATA / "matched_benchmark_summary.json")
    cv = m["cv_fold_ensemble_mm"]
    close("matched CV: TLIO-ResNet", cv["tlio_resnet"], 25.00)
    close("matched CV: RoNIN-ResNet", cv["ronin_resnet"], 25.62)
    test, t1 = m["test_sessions"]["traj_20260923_023422"], m["test_sessions"]["traj_20260926_143837"]
    close("matched test: TLIO-ResNet", test["tlio_resnet"]["err_mm"], 42.12)
    close("matched test: RoNIN-ResNet", test["ronin_resnet"]["err_mm"], 42.49)
    close("T1: gated PhysNet", t1["physnet_gate"]["err_mm"], 31.02)
    close("T1: zero motion", t1["zero_motion"]["err_mm"], 40.43)
    close("T1: calibrated blend, gate", t1["calibrated_blend_gate"]["err_mm"], 30.75)
    close("T1: zero-motion ATE", t1["zero_ate"]["ate_rmse_mm"], 128.0, 0.051)
    close("T1: IMUNet ATE", t1["imunet"]["ate_single_edge_mm"], 78.0, 0.051)
    close("T1: gated PhysNet ATE", t1["physnet_gate"]["ate_single_edge_mm"], 162.7, 0.051)
    checks = load(DATA / "revision_claim_checks.json")
    pause = checks["pause_windows"]["matched_contiguous_inside"]
    close("matched pause contrast (mm)", pause["diff_mm"], -3.2, 0.051)
    close("matched pause contrast (%)", pause["diff_pct"], -16.4, 0.51)
    close("switched windows, mean displacement", checks["switched_windows"]["traj_20260923_023422"]
          ["gyro_bias_none"]["mean_disp_mm"], 3.9, 0.051)
    stereo = checks["stereo_check"]
    close("stereo: board centroid median", stereo["board_centroid_3s"]["median_mm"], 0.82)
    close("stereo: camera centre median", stereo["camera_centre_label_3s"]["median_mm"], 8.1, 0.051)
    close("4K: calibrated session median", checks["external_4k_check"]["gtraj_20261002_024309"]
          ["median_diff_mm"], 5.5, 0.051)
    repeated_latency = load(DATA / "reviewer_audit_20261108" / "ensemble_latency_repeated.json")
    close("PhysNet 16-model B=1 latency, repeated median (ms)",
          repeated_latency["median_ms"]["physnet_ensemble16_ms"], 44.3, 0.051)
    close("PhysNet 16-model B=1 latency, repeated low (ms)",
          repeated_latency["range_ms"]["physnet_ensemble16_ms"][0], 43.4, 0.051)
    close("PhysNet 16-model B=1 latency, repeated high (ms)",
          repeated_latency["range_ms"]["physnet_ensemble16_ms"][1], 46.9, 0.051)
    gate = load(DATA / "gate_trajectory_check.json")
    close("gate-active test windows", gate["gate_active"]["windows"], 123)
    close("board-Z ATE, gated", gate["per_axis_ate_mm_xyz"]["cvg"][2], 104.7, 0.051)
    power = load(DATA / "revision_rehearsal" / "stop_power.json")["conditions"]
    close("twin prediction: reduction, pause-free (%)", 100 * power["continuous"]["pooled"]["reduction_switch"], 18.5, 0.051)
    close("twin prediction: reduction, 4-s stops (%)", 100 * power["paused"]["pooled"]["reduction_switch"], 43.6, 0.051)
    close("twin prediction: error, pause-free", power["continuous"]["pooled"]["physnet_switch_mm"], 46.7, 0.051)
    close("twin prediction: error, 4-s stops", power["paused"]["pooled"]["physnet_switch_mm"], 32.0, 0.051)
    delay = load(DATA / "delay_repeatability.json")
    close("delay split-half robust SD (ms)", delay["half_difference_robust_sd_ms"], 10.4, 0.051)
    close("delay, trajectory recordings min (ms)", delay["trajectory_delay_range_ms"][0], 14.0, 0.051)
    close("delay, trajectory recordings max (ms)", delay["trajectory_delay_range_ms"][1], 102.0, 0.051)
    cal_report = load(DATA / "reference_stage_validation.json")
    if cal_report["passed"] or len(cal_report["summary"]["axes"]) != 1:
        raise AssertionError("caliper audit must remain one completed direction, not a passed 3-axis protocol")
    cal = cal_report["summary"]
    close("caliper: board held-out RMSE (mm)", cal["heldout_rmse_mm"], 0.28, 0.006)
    close("caliper: board scale error (%)", cal["axes"]["X"]["scale_error_percent"], -0.40, 0.006)
    close("caliper: board cross-axis RMSE (mm)", cal["axes"]["X"]["cross_axis_rmse_mm"], 0.12, 0.006)
    close("caliper: jaw tilt, IMU (deg)", cal["tilt_check"]["imu_tilt_max_deg"], 0.34, 0.006)
    close("caliper: PnP against jaw IMU tilt (deg RMS)", cal["tilt_check"]["rms_diff_deg"], 0.07, 0.006)
    cam = load(DATA / "reference_caliper_dynamic_camera.json")["overall"]
    board_dynamic = load(DATA / "reference_caliper_dynamic.json")["overall"]
    close("caliper: adjacent actual moves, count", board_dynamic["adjacent_moves"]["n"], 64)
    close("caliper: adjacent board displacement median (mm)",
          board_dynamic["adjacent_moves"]["median_mm"], 0.119, 0.0006)
    close("caliper: adjacent board displacement p95 (mm)",
          board_dynamic["adjacent_moves"]["p95_mm"], 0.309, 0.0006)
    close("caliper: hold-pair displacement count", board_dynamic["hold_pair_displacement"]["n"], 288)
    for step in ("1", "2", "3", "5", "10", "20", "30", "40"):
        if step not in board_dynamic["hold_pair_displacement"]["by_step_mm"]:
            raise AssertionError(f"caliper displacement step {step} mm is missing")
    close("caliper: adjacent camera-label median (mm)",
          cam["adjacent_moves"]["median_mm"], 0.313, 0.0006)
    close("caliper: adjacent camera-label p95 (mm)",
          cam["adjacent_moves"]["p95_mm"], 0.944, 0.0006)
    close("caliper: camera-centre label error median (mm)", cam["label_error"]["median_mm"], 0.49, 0.006)
    close("caliper: camera-centre labels within U_label (%)", 100 * cam["label_error_within_U_label_nominal"], 99.4, 0.051)
    imu = load(DATA / "board_imu_motion_check.json")
    close("board IMU: 1-s per-axis correlation, minimum", imu["rotation_1s"]["axis_corr_min"], 0.991, 0.0006)
    close("board IMU: 3-s relative rotation, median (deg)", imu["windows"]["3 s"]["difference_median_deg"], 0.45, 0.006)
    close("twin prediction: PhysNet without switching, pause-free (%)",
          100 * power["continuous"]["pooled"]["reduction_physnet"], 18.4, 0.051)
    close("twin prediction: PhysNet without switching, 4-s stops (%)",
          100 * power["paused"]["pooled"]["reduction_physnet"], 31.7, 0.051)
    stop = load(DATA / "stop_protocol_experiment.json")
    sw, pn = stop["estimators"]["physnet_switch"], stop["estimators"]["physnet"]
    close("stop experiment: zero motion, continuous (mm)", sw["continuous"]["zero_mm"], 69.1, 0.051)
    close("stop experiment: zero motion, paused (mm)", sw["paused"]["zero_mm"], 18.8, 0.051)
    close("stop experiment: switching, continuous (mm)", sw["continuous"]["err_mm"], 57.3, 0.051)
    close("stop experiment: switching, paused (mm)", sw["paused"]["err_mm"], 25.1, 0.051)
    close("stop experiment: reduction, continuous (%)", 100 * sw["continuous"]["reduction_vs_zero"], 17.0, 0.051)
    close("stop experiment: reduction, paused (%)", 100 * sw["paused"]["reduction_vs_zero"], -33.5, 0.051)
    close("stop experiment: PhysNet without switching, paused (%)", 100 * pn["paused"]["reduction_vs_zero"], -5.6, 0.051)
    close("stop experiment: gain (points)", 100 * sw["reduction_gain"], -50.6, 0.051)
    close("stop experiment: gain CI low", 100 * sw["reduction_gain_ci95"][0], -70.1, 0.051)
    close("stop experiment: gain CI high", 100 * sw["reduction_gain_ci95"][1], -32.6, 0.051)
    close("stop experiment: matched difference (mm)", sw["matched_diff_mm"], 8.4, 0.051)
    close("stop experiment: matched CI low", sw["matched_diff_ci95_mm"][0], 5.2, 0.051)
    close("stop experiment: matched CI high", sw["matched_diff_ci95_mm"][1], 12.3, 0.051)
    close("stop experiment: matched difference, PhysNet (mm)", pn["matched_diff_mm"], 3.0, 0.051)
    close("stop experiment: matched CI low, PhysNet", pn["matched_diff_ci95_mm"][0], 1.6, 0.051)
    close("stop experiment: matched CI high, PhysNet", pn["matched_diff_ci95_mm"][1], 5.1, 0.051)
    per = stop["per_session"]
    paused = [per[n] for n in stop["sessions"]["paused"]]
    close("stop experiment: beats with a stop, P1 (%)", 100 * paused[0]["protocol"]["compliance"], 89.0, 0.5)
    close("stop experiment: beats with a stop, P2 (%)", 100 * paused[1]["protocol"]["compliance"], 84.0, 0.5)
    close("stop experiment: stops/min, min", min(p["stops_per_min"] for p in paused), 13.4, 0.051)
    close("stop experiment: stops/min, max", max(p["stops_per_min"] for p in paused), 15.8, 0.051)
    close("stop experiment: mount to training rig (deg)", per[stop["sessions"]["continuous"][0]]["imu_mount"]
          ["angle_to_nominal_deg"], 25.0, 0.5)
    close("stop experiment: accelerometer at rest (g)", per[stop["sessions"]["continuous"][0]]["acc_rest_norm_g"],
          1.008, 0.0006)
    rot = load(DATA / "stop_protocol_experiment_rotation.json")
    below3, mid, high = rot["bins"][0]["mean_mm"], rot["bins"][1]["mean_mm"], rot["bins"][2]["mean_mm"]
    close("stop rotation < 3 deg: stop-anchored (mm)", below3["stop_anchored"], 11.9, 0.051)
    close("stop rotation < 3 deg: PhysNet (mm)", below3["physnet"], 17.9, 0.051)
    close("stop rotation < 3 deg: zero (mm)", below3["zero"], 15.1, 0.051)
    close("stop rotation 3-8 deg: stop-anchored (mm)", mid["stop_anchored"], 17.6, 0.051)
    close("stop rotation 3-8 deg: PhysNet (mm)", mid["physnet"], 18.2, 0.051)
    close("stop rotation 3-8 deg: zero (mm)", mid["zero"], 16.2, 0.051)
    close("stop rotation 8-15 deg: stop-anchored (mm)", high["stop_anchored"], 45.4, 0.051)
    close("stop rotation 8-15 deg: PhysNet (mm)", high["physnet"], 20.7, 0.051)
    close("stop rotation 8-15 deg: zero (mm)", high["zero"], 21.5, 0.051)
    close("stop rotation: switched windows at or above 8 deg (%)", 100 * rot["share_switched_at_or_above_gate"], 32.0, 0.51)
    close("stop rotation: gated switching, paused (%)", 100 * rot["reduction_vs_zero"]["gated"], 1.2, 0.051)
    tripod = load(DATA / "stop_eval_20261109_batch.json")
    tsw, tpn = tripod["estimators"]["physnet_switch"], tripod["estimators"]["physnet"]
    close("tripod stop: zero motion, continuous (mm)", tpn["continuous"]["zero_mm"], 45.5, 0.051)
    close("tripod stop: zero motion, paused (mm)", tpn["paused"]["zero_mm"], 34.57, 0.051)
    close("tripod stop: PhysNet, continuous (mm)", tpn["continuous"]["err_mm"], 39.41, 0.051)
    close("tripod stop: PhysNet, paused (mm)", tpn["paused"]["err_mm"], 27.42, 0.051)
    close("tripod stop: PhysNet reduction, continuous (%)", 100 * tpn["continuous"]["reduction_vs_zero"], 13.4, 0.051)
    close("tripod stop: PhysNet reduction, paused (%)", 100 * tpn["paused"]["reduction_vs_zero"], 20.7, 0.051)
    close("tripod stop: PhysNet gain (points)", 100 * tpn["reduction_gain"], 7.3, 0.051)
    close("tripod stop: PhysNet matched diff (mm)", tpn["matched_diff_mm"], -3.93, 0.051)
    close("tripod stop: switching, paused (mm)", tsw["paused"]["err_mm"], 42.22, 0.051)
    close("tripod stop: switching gain (points)", 100 * tsw["reduction_gain"], -35.5, 0.051)
    close("tripod stop: switching gain CI low", 100 * tsw["reduction_gain_ci95"][0], -54.5, 0.051)
    close("tripod stop: switching gain CI high", 100 * tsw["reduction_gain_ci95"][1], -19.1, 0.051)
    close("tripod stop: switching matched diff (mm)", tsw["matched_diff_mm"], 12.0, 0.051)
    close("tripod stop: switching matched CI low", tsw["matched_diff_ci95_mm"][0], 6.8, 0.051)
    close("tripod stop: switching matched CI high", tsw["matched_diff_ci95_mm"][1], 17.3, 0.051)
    close("tripod stop: PhysNet gain CI low", 100 * tpn["reduction_gain_ci95"][0], 2.3, 0.051)
    close("tripod stop: PhysNet gain CI high", 100 * tpn["reduction_gain_ci95"][1], 12.4, 0.051)
    close("tripod stop: PhysNet matched CI low", tpn["matched_diff_ci95_mm"][0], -5.9, 0.051)
    close("tripod stop: PhysNet matched CI high", tpn["matched_diff_ci95_mm"][1], -2.1, 0.051)
    # The other pre-specified estimators must be reported with the favourable one.
    tim, tisw, tbl = (tripod["estimators"][k] for k in ("imunet", "imunet_switch", "calibrated_blend"))
    close("tripod stop: IMUNet, continuous (mm)", tim["continuous"]["err_mm"], 36.23, 0.051)
    close("tripod stop: IMUNet, paused (mm)", tim["paused"]["err_mm"], 32.21, 0.051)
    close("tripod stop: IMUNet gain (points)", 100 * tim["reduction_gain"], -13.6, 0.051)
    close("tripod stop: IMUNet gain CI low", 100 * tim["reduction_gain_ci95"][0], -18.3, 0.051)
    close("tripod stop: IMUNet gain CI high", 100 * tim["reduction_gain_ci95"][1], -8.7, 0.051)
    close("tripod stop: IMUNet matched diff (mm)", tim["matched_diff_mm"], 3.7, 0.051)
    close("tripod stop: IMUNet matched CI low", tim["matched_diff_ci95_mm"][0], 2.2, 0.051)
    close("tripod stop: IMUNet matched CI high", tim["matched_diff_ci95_mm"][1], 5.3, 0.051)
    close("tripod stop: IMUNet + switching, paused (mm)", tisw["paused"]["err_mm"], 43.37, 0.051)
    close("tripod stop: IMUNet + switching gain (points)", 100 * tisw["reduction_gain"], -45.8, 0.051)
    close("tripod stop: IMUNet + switching gain CI low", 100 * tisw["reduction_gain_ci95"][0], -66.2, 0.051)
    close("tripod stop: IMUNet + switching gain CI high", 100 * tisw["reduction_gain_ci95"][1], -29.0, 0.051)
    close("tripod stop: IMUNet + switching matched diff (mm)", tisw["matched_diff_mm"], 15.7, 0.051)
    close("tripod stop: IMUNet + switching matched CI low", tisw["matched_diff_ci95_mm"][0], 10.6, 0.051)
    close("tripod stop: IMUNet + switching matched CI high", tisw["matched_diff_ci95_mm"][1], 21.4, 0.051)
    close("tripod stop: blend, continuous (mm)", tbl["continuous"]["err_mm"], 37.63, 0.051)
    close("tripod stop: blend, paused (mm)", tbl["paused"]["err_mm"], 28.41, 0.051)
    close("tripod stop: blend gain (points)", 100 * tbl["reduction_gain"], 0.5, 0.051)
    close("tripod stop: blend gain CI low", 100 * tbl["reduction_gain_ci95"][0], -3.7, 0.051)
    close("tripod stop: blend gain CI high", 100 * tbl["reduction_gain_ci95"][1], 4.8, 0.051)
    close("tripod stop: blend matched diff (mm)", tbl["matched_diff_mm"], -1.4, 0.051)
    close("tripod stop: blend matched CI low", tbl["matched_diff_ci95_mm"][0], -3.1, 0.051)
    close("tripod stop: blend matched CI high", tbl["matched_diff_ci95_mm"][1], 0.3, 0.051)
    tper = tripod["per_session"]
    close("tripod stop: windows, continuous", sum(tper[n]["windows"] for n in tripod["sessions"]["continuous"]), 1903)
    close("tripod stop: windows, paused", sum(tper[n]["windows"] for n in tripod["sessions"]["paused"]), 1842)
    close("tripod stop: total windows", sum(v["windows"] for v in tper.values()), 3745)
    close("handheld stop: total windows", sum(v["windows"] for v in per.values()), 1966)
    stop_sessions = [*stop["sessions"]["continuous"], *stop["sessions"]["paused"],
                     *tripod["sessions"]["continuous"], *tripod["sessions"]["paused"]]
    close("stop batches: total duration (s)",
          sum(load(DATA / s / "capture_summary.json")["seconds"] for s in stop_sessions), 630.5, 0.06)
    close("stop batches: total frames",
          sum(load(DATA / s / "capture_summary.json")["frames"] for s in stop_sessions), 7625)
    close("stop batches: usable frames",
          sum(load(DATA / s / "qc_result.json")["usable"] for s in stop_sessions), 7107)
    close("stop batches: USB IMU samples",
          sum(load(DATA / s / "capture_summary.json")["usb_samples"] for s in stop_sessions), 126568)
    # Gains are differences of the unrounded reductions, as for the physical batches.
    pc, pp = power["continuous"]["pooled"], power["paused"]["pooled"]
    close("twin gain without switching (points)",
          100 * ((1 - pp["physnet_mm"] / pp["zero_mm"]) - (1 - pc["physnet_mm"] / pc["zero_mm"])), 13.3, 0.051)
    close("twin gain with switching (points)",
          100 * ((1 - pp["physnet_switch_mm"] / pp["zero_mm"])
                 - (1 - pc["physnet_switch_mm"] / pc["zero_mm"])), 25.0, 0.051)
    trot = load(DATA / "stop_eval_20261109_batch_rotation.json")
    close("tripod rotation: switched windows at or above 8 deg (%)", 100 * trot["share_switched_at_or_above_gate"], 64.1, 0.051)
    close("tripod rotation: median (deg)", trot["rotation_deg_median"], 11.3, 0.051)
    close("tripod rotation < 3 deg: stop-anchored (mm)", trot["bins"][0]["mean_mm"]["stop_anchored"], 10.0, 0.051)
    close("tripod rotation < 3 deg: PhysNet (mm)", trot["bins"][0]["mean_mm"]["physnet"], 14.8, 0.051)
    close("tripod rotation > 15 deg: stop-anchored (mm)", trot["bins"][3]["mean_mm"]["stop_anchored"], 80.9, 0.051)
    close("tripod rotation > 15 deg: PhysNet (mm)", trot["bins"][3]["mean_mm"]["physnet"], 31.9, 0.051)
    # Rig adaptation, not pre-specified. Balanced folds are the reported result;
    # the paused-heavy folds are the sensitivity case.
    for fold, (c_mm, p_mm, gain, lo, hi, matched, m_lo, m_hi) in (
        ("bal_ab", (35.58, 15.94, 14.6, -0.2, 29.4, -13.0, -21.1, -5.6)),
        ("bal_ba", (25.74, 21.04, 12.6, 2.1, 23.8, -6.3, -10.6, -2.1)),
        ("ab", PAUSED_HEAVY["ab"]),
        ("ba", PAUSED_HEAVY["ba"]),
    ):
        rep = load(DATA / f"stop_eval_20261109_adapted_{fold}.json")
        e = rep["physnet"]
        if set(rep["trained_on"]) & set(rep["tested_on"]["continuous"] + rep["tested_on"]["paused"]):
            raise AssertionError(f"adaptation fold {fold} was scored on a recording it was trained on")
        if rep.get("seed") != 0 or rep.get("n_boot") != 2000:
            raise AssertionError(f"adaptation fold {fold} is not the seed-0 run with 2000 resamples")
        close(f"adapted {fold}: continuous (mm)", e["continuous"]["err_mm"], c_mm, 0.006)
        close(f"adapted {fold}: paused (mm)", e["paused"]["err_mm"], p_mm, 0.006)
        close(f"adapted {fold}: gain (points)", 100 * e["reduction_gain"], gain, 0.051)
        close(f"adapted {fold}: gain CI low", 100 * e["reduction_gain_ci95"][0], lo, 0.051)
        close(f"adapted {fold}: gain CI high", 100 * e["reduction_gain_ci95"][1], hi, 0.051)
        close(f"adapted {fold}: matched diff (mm)", e["matched_diff_mm"], matched, 0.051)
        close(f"adapted {fold}: matched CI low", e["matched_diff_ci95_mm"][0], m_lo, 0.051)
        close(f"adapted {fold}: matched CI high", e["matched_diff_ci95_mm"][1], m_hi, 0.051)
    close("adapted bal_ab: switching gain (points)",
          100 * load(DATA / "stop_eval_20261109_adapted_bal_ab.json")["physnet_switch"]["reduction_gain"], -60.7, 0.051)
    close("adapted bal_ba: switching gain (points)",
          100 * load(DATA / "stop_eval_20261109_adapted_bal_ba.json")["physnet_switch"]["reduction_gain"], -38.5, 0.051)
    # A second seed and the two earlier unseeded runs: the point estimates must stay close.
    close("adapted bal_ab, seed 1: gain (points)",
          100 * load(DATA / "stop_eval_20261109_adapted_bal_ab_s1.json")["physnet"]["reduction_gain"], 14.3, 0.051)
    close("adapted bal_ba, seed 1: gain (points)",
          100 * load(DATA / "stop_eval_20261109_adapted_bal_ba_s1.json")["physnet"]["reduction_gain"], 13.0, 0.051)
    earlier = DATA / "stop_experiment_20261109"
    close("adapted bal_ab, unseeded run 1: gain (points)", 100 * load(
        earlier / "stop_eval_20261109_adapted_bal_ab_unseeded_first_run.json")["physnet"]["reduction_gain"], 15.2, 0.051)
    close("adapted bal_ab, unseeded run 2: gain (points)", 100 * load(
        earlier / "stop_eval_20261109_adapted_bal_ab_unseeded.json")["physnet"]["reduction_gain"], 14.6, 0.051)
    tilt = load(DATA / "stop_anchor_tilt_20261109.json")["pooled"]
    close("tilt check: switched windows", tilt["n"], 1426)
    close("tilt check: published anchor (mm)", tilt["published_anchor"], 44.3, 0.051)
    close("tilt check: tilt-corrected anchor (mm)", tilt["tilt_corrected"], 27.1, 0.051)
    close("tilt check: frozen PhysNet (mm)", tilt["frozen_physnet"], 25.6, 0.051)
    close("tilt check: zero motion (mm)", tilt["zero"], 31.9, 0.051)
    close("tilt check: stop gyro bias (mm)", tilt["stop_gyro_bias"], 46.5, 0.051)
    close("tilt check: > 15 deg, tilt-corrected (mm)", tilt["bins"][3]["tilt_corrected"], 44.5, 0.051)
    # The frozen models must still be the files named in the manifest.
    import hashlib
    manifest = load(DATA / "final_model_manifest.json")
    for row in manifest["models_and_configs"]:
        relative = PureWindowsPath(row["path"]).relative_to(PureWindowsPath("E:/EGO_Mo"))
        local = ROOT.joinpath(*relative.parts)
        digest = hashlib.sha256(local.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise AssertionError(f"frozen file changed: {row['path']}")
    print(f"OK frozen manifest: {len(manifest['models_and_configs'])} files unchanged")
    retrained_results()
    revision = load(DATA / "reviewer_revision_experiments.json")
    oracle = revision["exp2_oracle_simulation"]["pause_inf"]
    for key, expected in (
        ("demeaned_no_v0", 35.95),
        ("demeaned_with_v0", 29.66),
        ("no_demean_no_v0", 34.05),
        ("no_demean_with_v0", 22.65),
    ):
        close(f"oracle 2x2 pause-free: {key} (mm)", oracle[key]["learned_ens_mm"], expected, 0.006)
    # Section X-E: context sweep scored on the windows admissible at c = 2 s, with a strictly
    # causal window (tools/run_context_sweep_common.py). It supersedes exp3_latency_context,
    # whose shortest window kept 50 ms of context and whose contexts were scored on
    # different window sets.
    sweep = load(DATA / "context_sweep_common.json")
    if (sweep["common_windows"]["val"], sweep["common_windows"]["test"]) != (1436, 647):
        raise AssertionError("common window set of the context sweep changed")
    for c, val_rng, test_rng in (("0", (31.1, 31.6), (44.8, 45.8)), ("0.05", (30.9, 31.6), (44.2, 45.0)),
                                 ("0.5", (30.2, 30.5), (43.8, 45.1)), ("1", (29.9, 30.0), (43.9, 44.9)),
                                 ("2", (29.2, 30.1), (44.2, 45.1))):
        for name, key, expected in (("validation", "common_val_err_mm", val_rng), ("held-out", "common_test_err_mm", test_rng)):
            for which, got, want in zip(("low", "high"), sweep["summary"][c][key], expected):
                close(f"context sweep c={c} s: {name} error, {which} (mm)", got, want, 0.051)
    causal = [rep["0"] for rep in sweep["repeats"]]
    if any(r["last_sample_relative_to_tb_test"]["windows_with_sample_at_or_after_tb"] or r["future_wait_ms"] for r in causal):
        raise AssertionError("the c = 0 window uses a sample at or after t_b")
    close("strictly causal window: two-model output delay, largest (ms)",
          max(r["two_model_output_delay_ms"] for r in causal), 4.8, 0.3)
    costs = [rep["0"]["common"]["val_err_mm"] - rep["2"]["common"]["val_err_mm"] for rep in sweep["repeats"]]
    close("strictly causal window: validation cost, smallest (mm)", min(costs), 1.0, 0.051)
    close("strictly causal window: validation cost, largest (mm)", max(costs), 2.4, 0.051)
    gains = [r["common"]["test_gain_over_2s_mm"] for r in causal]
    close("strictly causal window: held-out difference from c = 2 s, lowest (mm)", min(gains), -1.6, 0.051)
    close("strictly causal window: held-out difference from c = 2 s, highest (mm)", max(gains), 0.2, 0.051)
    for r in causal:
        lo, hi = r["common"]["test_gain_over_2s_ci95_mm"]
        if not lo < 0 < hi:
            raise AssertionError("a held-out interval of the causal window excludes zero")
    # Table III(A): 2-, 3- and 4-s context on common validation windows (tools/context_common_windows.py).
    ctx = load(DATA / "context_common_windows.json")["contexts"]
    for label, expected in (("2 s", 33.15), ("3 s", 33.27), ("4 s", 33.11)):
        close(f"context {label} on common validation windows (mm)", ctx[label]["common_err_mm"], expected, 0.006)
    dense = load(DATA / "reviewer_audit_20261108" / "dense_single_anchor_recomputed.json")
    close("single-anchor dense PhysNet ATE (mm)",
          dense["dense_physnet_1anchor"]["ate_rmse_mm"], 81.08, 0.006)
    close("single-anchor dense calibrated-blend ATE (mm)",
          dense["dense_blend_1anchor"]["ate_rmse_mm"], 80.22, 0.006)
    close("single-anchor dense graph components",
          dense["connected_components_with_at_least_3_nodes"], 1)
    if dense["dense_blend_1anchor"]["existing_npz_max_difference_mm"] > 1e-3:
        raise AssertionError("recomputed single-anchor blend no longer matches the plotted NPZ")
    psf = load(DATA / "sim_reference_validation_psf.json")["depths"]["250"]["monte_carlo"]
    for axis, expected in zip("XYZ", (5.34, 4.82, 0.78)):
        close(f"intrinsic Monte Carlo scale spread {axis} (%)",
              psf["scale_error_percent_std"][axis], expected, 0.011)
    allan = load(DATA / "allan_static_imu.json")
    if not all(row["gyro_all_zero"] for row in allan.values()):
        raise AssertionError("long static bench logs no longer have quantized-zero gyro samples")
    # Section VII-A: what the static logs bound (tools/allan_static_bound.py).
    bound = load(DATA / "allan_static_bound.json")
    close("Allan: random-walk bound (1e-5 g/sqrt(s))", 1e5 * bound["random_walk_bound_g_sqrt_s"], 1.3, 0.051)
    close("Allan: bound with +1 sigma (1e-5 g/sqrt(s))",
          1e5 * bound["random_walk_bound_plus_1sigma_g_sqrt_s"], 1.9, 0.051)
    close("Allan: 3-s displacement at the bound (mm)", bound["displacement_3s_rms_mm_per_axis"]["at_bound"], 0.3, 0.051)
    close("Allan: 3-s displacement at the bound +1 sigma (mm)",
          bound["displacement_3s_rms_mm_per_axis"]["at_bound_plus_1sigma"], 0.4, 0.051)
    for (name, s), sigma, slope in zip(bound["sessions"].items(), (48.0, 40.0), (-0.48, -0.50)):
        close(f"Allan: uncertainty at 20 s, {name[-6:]} (%)", 100 * s["relative_sigma"][-1], sigma, 0.51)
        close(f"Allan: largest-axis slope up to 5 s, {name[-6:]}", s["slope_0p1_to_5s"], slope, 0.006)
        if not 2.5 < s["clusters"][-1] < 4.5:
            raise AssertionError("Allan: the logs no longer hold three to four intervals at 20 s")
    gyro = bound["gyroscope"]
    if not (gyro["static_all_zero"] and gyro["moving_all_multiples_of_step"]
            and abs(gyro["moving_smallest_nonzero_dps"] - gyro["step_dps"]) < 1e-3):
        raise AssertionError("gyroscope zeros at rest are no longer explained by the 0.061-deg/s step")
    # Section X-F: excitation of the lever-arm components (tools/analyze_lever_excitation.py).
    lever = load(DATA / "lever_excitation.json")
    for got, expected in zip(lever["lever_mm"]["cvx"]["mean"] + lever["lever_mm"]["cvx"]["sd"]
                             + lever["lever_mm"]["cvg"]["mean"], (-22.6, -5.2, -29.7, 3.8, 1.9, 7.0, -20.9, -4.1, -17.8)):
        close("lever arm of the fold models (mm)", got, expected, 0.051)
    close("lever arm: shift of lambda_z (mm)", lever["shift_gated_minus_nongated_mm"][2], 11.9, 0.051)
    for got, expected in zip(lever["column_rms_norm_xyz"], (0.355, 0.345, 0.151)):
        close("lever arm: RMS norm of a column of R - I", got, expected, 0.0006)
    close("lever arm: displacement change of the shift (mm)", lever["lambda_z_shift_effect_mm"]["vector_rms"], 1.8, 0.051)
    for key, expected in (("held_out", 0.2), ("t1", 0.6)):
        r = lever["nongated_ensemble_with_lambda_z_moved"][key]
        close(f"lever arm: test error change with lambda_z moved, {key} (mm)",
              r["err_lambda_z_moved_mm"] - r["err_stored_mm"], expected, 0.051)
    if lever["windows"]["total"] != 10913:
        raise AssertionError("lever arm: the CV windows changed")


def retrained_results():
    """What the manuscript says about a second training of the analyses that
    train networks (tools/reproduce_trained_results.py)."""
    check = DATA / "reproducibility_check"
    revision = load(DATA / "reviewer_revision_experiments.json")
    # Table III, Group D, in the twin.
    old = load(DATA / "synthetic_factorial.json")["results"]
    new = load(check / "synthetic_factorial.json")["results"]
    if set(old) != set(new):
        raise AssertionError("the retrained twin factorial does not cover the stored configurations")
    for split, expected, tol in (("test", 0.8, 0.051), ("ood", 1.6, 0.051), ("real_test", 15.0, 0.51)):
        close(f"twin factorial retrained: largest change, {split} (mm)",
              max(abs(new[k][split]["err_mm"] - old[k][split]["err_mm"]) for k in old), expected, tol)
    public = [k for k in old if not k.startswith("physnet")]
    for tag, res, expected in (("stored", old, 7.0), ("retrained", new, 9.0)):
        best = min(res[k]["test"]["err_mm"] for k in public)
        if min(res, key=lambda k: res[k]["test"]["err_mm"]) != "physnet_anchor24_dense":
            raise AssertionError(f"PhysNet is not lowest on the synthetic test ({tag})")
        close(f"twin factorial {tag}: PhysNet below the best public configuration (%)",
              100 * (1 - res["physnet_anchor24_dense"]["test"]["err_mm"] / best), expected, 0.51)
    gaps = []
    for res in (old, new):
        for body, anchor in (("imunet_body8_end", "imunet_anchor24_end"),
                             ("ronin_resnet_body8_end", "ronin_resnet_anchor24_end"),
                             ("tlio_resnet_body8_end", "tlio_resnet_anchor24_end"),
                             ("physnet_body8_end", "physnet_anchor24_end")):
            gaps.append(res[body]["real_test"]["err_mm"] - res[anchor]["real_test"]["err_mm"])
        zero = load(DATA / "synthetic_factorial.json")["protocol"]["zero_motion_mm"]["real_test"]
        if min(v["real_test"]["err_mm"] for v in res.values()) <= zero:
            raise AssertionError("a synthetic-only model beats zero motion on the held-out recording")
    close("direct transfer: smallest input effect, both trainings (mm)", min(gaps), 29.0, 0.51)
    close("direct transfer: largest input effect, both trainings (mm)", max(gaps), 111.0, 0.51)
    close("stored OOD, TLIO-ResNet body frame (mm)", old["tlio_resnet_body8_end"]["ood"]["err_mm"], 33.5, 0.051)
    close("stored OOD, RoNIN-ResNet body frame (mm)", old["ronin_resnet_body8_end"]["ood"]["err_mm"], 33.8, 0.051)
    # Fig. 6(d): the 2x2 test.
    o_old, o_new = revision["exp2_oracle_simulation"], load(check / "oracle_2x2.json")
    cells = [(c, k) for c in ("pause_inf", "pause_4") for k in o_old[c] if isinstance(o_old[c][k], dict)]
    close("2x2 test retrained: largest change of a learned error (mm)",
          max(abs(o_new[c][k]["learned_ens_mm"] - o_old[c][k]["learned_ens_mm"]) for c, k in cells), 0.7, 0.051)
    close("2x2 test retrained: largest change of an open-loop error (mm)",
          max(abs(o_new[c][k]["open_loop_mm"] - o_old[c][k]["open_loop_mm"]) for c, k in cells), 0.0, 0.001)
    for res in (o_old, o_new):
        e = {k: res["pause_inf"][k]["learned_ens_mm"] for k in res["pause_inf"] if k != "zero_mm"}
        if not (e["demeaned_no_v0"] - e["demeaned_with_v0"] > e["demeaned_no_v0"] - e["no_demean_no_v0"]
                and e["no_demean_with_v0"] < e["demeaned_with_v0"] < e["no_demean_no_v0"] < e["demeaned_no_v0"]):
            raise AssertionError("the order of the 2x2 test changed")
    # Table III, Group C, on the real recordings.
    c_old = revision["exp1_matched_ablation"]
    c_new = load(check / "matched_ablation.json") | load(check / "matched_ablation_body_physnet.json")
    table = ["conv_bigru_body8_endpoint", "physnet_body_endpoint", "physnet_body_dense",
             "physnet_anchor24_endpoint", "physnet_anchor24_dense", "physnet_anchor24_dense_gate",
             "imunet_body8_endpoint", "imunet_anchor24_endpoint", "ronin_resnet_body8_endpoint",
             "ronin_resnet_anchor24_endpoint", "tlio_resnet_body8_endpoint", "tlio_resnet_anchor24_endpoint"]
    missing = [k for k in table if k not in c_new]
    if missing:
        raise AssertionError(f"Group C rows that the scripts do not regenerate: {missing}")
    close("Group C retrained: largest change of a validation error (mm)",
          max(abs(c_new[k]["val_ens"] - c_old[k]["val_ens"]) for k in table), 1.1, 0.051)
    for tag, res, expected in (("stored", c_old, 0.9), ("retrained", c_new, -0.4)):
        close(f"Group C {tag}: input effect, RoNIN-ResNet (mm)",
              res["ronin_resnet_anchor24_endpoint"]["val_ens"] - res["ronin_resnet_body8_endpoint"]["val_ens"],
              expected, 0.051)
    # Tables V and VI: all 80 leave-one-session-out fold models retrained.
    m_old = load(DATA / "matched_benchmark_summary.json")
    m_new = load(check / "loso_retrained" / "matched_benchmark_summary.json")
    cv_keys = ("physnet_cvx", "physnet_cvg", "imunet", "ronin_resnet", "tlio_resnet")
    close("LOSO retrained: largest change of a CV error (mm)",
          max(abs(m_new["cv_fold_ensemble_mm"][k] - m_old["cv_fold_ensemble_mm"][k]) for k in cv_keys),
          0.27, 0.051)
    methods = (*("imunet", "ronin_resnet", "tlio_resnet", "physnet", "physnet_gate"),
               "equal_blend", "equal_blend_gate", "calibrated_blend", "calibrated_blend_gate")
    ep_diffs, ate_diffs = [], []
    for ses in m_old["test_sessions"]:
        for k in methods:
            a, b = m_old["test_sessions"][ses][k], m_new["test_sessions"][ses][k]
            ep_diffs.append(abs(b["err_mm"] - a["err_mm"]))
            if "ate_single_edge_mm" in a:
                ate_diffs.append(abs(b["ate_single_edge_mm"] - a["ate_single_edge_mm"]))
            if min(m_new["paired_intervals"][ses][k]["gain_over_zero_ci95_mm"]) <= 0:
                raise AssertionError(f"retrained {k} no longer beats zero motion on {ses}")
    close("LOSO retrained: largest change of a test e_p (mm)", max(ep_diffs), 0.39, 0.051)
    close("LOSO retrained: largest change of a test ATE (mm)", max(ate_diffs), 16.4, 0.11)
    # Section X-F: the gated trajectory is worse again in the second training, on both recordings.
    held, t1 = "traj_20260923_023422", "traj_20260926_143837"
    for label, ses, expected in (("held-out", held, (89.7, 103.3)), ("T1", t1, (132.9, 156.8))):
        for key, value in zip(("physnet", "physnet_gate"), expected):
            close(f"LOSO retrained: {key} ATE, {label} (mm)",
                  m_new["test_sessions"][ses][key]["ate_single_edge_mm"], value, 0.051)
        for m in (m_old, m_new):
            r = m["test_sessions"][ses]
            if not (r["physnet_gate"]["err_mm"] < r["physnet"]["err_mm"]
                    and r["physnet_gate"]["ate_single_edge_mm"] > r["physnet"]["ate_single_edge_mm"]):
                raise AssertionError(f"gate trade-off not reproduced on {label}")
    # Section X-F: stillness loss without the gate (tools/train_still_only_folds.py), two trainings.
    abl = load(DATA / "gate_ablation_still_only.json")["results"]
    still_only = [r for r in abl if r["stillness_loss"] and not r["gate"]]
    if len(still_only) != 2:
        raise AssertionError("expected two trainings of the stillness-loss-only configuration")
    for r, (ate_held, ate_t1) in zip(still_only, ((94.8, 129.6), (103.8, 141.7))):
        close(f"stillness loss only, {r['training']}: held-out ATE (mm)", r[held]["ate_mm"], ate_held, 0.051)
        close(f"stillness loss only, {r['training']}: T1 ATE (mm)", r[t1]["ate_mm"], ate_t1, 0.051)
    # Response only: means of the two trainings of each configuration, non-gated / loss only / gated.
    for ses, expected in ((held, (88.8, 99.3, 106.6)), (t1, (132.9, 135.6, 159.8))):
        for (loss, gated), value in zip(((False, False), (True, False), (True, True)), expected):
            runs = [r[ses]["ate_mm"] for r in abl if (r["stillness_loss"], r["gate"]) == (loss, gated)]
            if len(runs) != 2:
                raise AssertionError("expected two trainings of every gate-ablation configuration")
            close(f"gate ablation mean ATE, loss={loss} gate={gated}, {ses[-6:]} (mm)", np.mean(runs), value, 0.051)
    for r in abl:        # the script must reproduce the stored ensembles it is compared with
        if r["tag"] == "cvs":
            continue
        ref = (m_old if r["training"] == "first training" else m_new)["test_sessions"]
        key = "physnet_gate" if r["gate"] else "physnet"
        for ses in (held, t1):
            close(f"gate ablation reproduces {key}, {r['training']}, ATE (mm)",
                  r[ses]["ate_mm"], ref[ses][key]["ate_single_edge_mm"], 0.011)
    # Section X-F: single-anchor dense graph with gyroscope attitude (tools/dense_single_anchor_gyro.py).
    gyro = load(DATA / "dense_single_anchor_gyro.json")
    close("single-anchor dense blend, gyroscope attitude (mm)",
          gyro["gyroscope_orientation_from_first_node"]["dense_blend_ate_mm"], 80.51, 0.006)
    close("single-anchor dense PhysNet, gyroscope attitude (mm)",
          gyro["gyroscope_orientation_from_first_node"]["dense_physnet_ate_mm"], 80.77, 0.006)
    close("single-anchor zero motion (mm)", gyro["zero_motion_ate_mm"], 118.3, 0.051)
    if gyro["nodes"] != 713:
        raise AssertionError("usable frames of the held-out recording changed")
    # Tables V and VI: trajectory error of the calibrated blends (tools/calibrated_blend_ate.py).
    blend = load(DATA / "calibrated_blend_ate.json")
    for label, tag, key, expected in (("held-out, no gate", "cvx", "held_out", 80.7), ("held-out, gate", "cvg", "held_out", 95.2),
                                      ("T1, no gate", "cvx", "t1", 108.2), ("T1, gate", "cvg", "t1", 130.7)):
        close(f"calibrated blend ATE, {label} (mm)", blend[key][tag]["cv_calibrated_blend"]["ate_rmse_mm"], expected, 0.051)
    for key, ses in (("held_out", held), ("t1", t1)):
        for tag, stored in (("cvx", "equal_blend"), ("cvg", "equal_blend_gate")):
            close(f"calibrated-blend script reproduces the stored equal-blend ATE, {key} {tag} (mm)",
                  blend[key][tag]["equal_blend"]["ate_rmse_mm"],
                  m_old["test_sessions"][ses][stored]["ate_single_edge_mm"], 0.011)
    # Section X-F: per-edge board-Z shift of the gated ensemble summed along the pose-graph paths
    # (tools/analyze_gate_trajectory.py, offset_accounting).
    acc = load(DATA / "gate_trajectory_check.json")["offset_accounting"]
    by_start = {tuple(w["start_s"]): w for w in acc["windows_by_start"]}
    build = by_start[(25.0, 60.0)]
    close("gate offset: windows starting 25-60 s", build["windows"], 192)
    close("gate offset: gate-active windows starting 25-60 s", build["gate_active"], 0)
    close("gate offset: largest stillness probability, 25-60 s", build["max_p_still"], 0.06, 0.0051)
    close("gate offset: Z shift per edge, 25-60 s (mm)", build["board_z_gated_minus_nongated_mm"], 4.0, 0.051)
    close("gate offset: share of positive shifts, 25-60 s (%)", 100 * build["share_positive"], 81.0, 0.51)
    close("gate offset: local error, gated, 25-60 s (mm)", build["err_cvg_mm"], 63.8, 0.051)
    close("gate offset: local error, non-gated, 25-60 s (mm)", build["err_cvx_mm"], 66.6, 0.051)
    if sum(by_start[k]["windows"] for k in by_start if k != (15.0, 60.0)) != 647:
        raise AssertionError("gate offset: the start-time intervals do not cover the 647 windows")
    after = next(r for r in acc["offset_at_nodes"] if r["node_time_s"][0] == 60.0 and r["nodes"] == 236)
    close("gate offset: offset after 60 s (mm)", after["offset_mm"], 47.0, 0.051)
    close("gate offset: path edges starting 25-60 s", after["path_edges_starting_25_60"], 12.0, 0.21)
    close("gate offset: contribution of edges starting 25-60 s (mm)", after["from_edges_starting_s"]["25-60"], 54.3, 0.051)
    close("gate offset: path sum equals the offset (mm)", after["path_sum_mm"], after["offset_mm"], 0.051)
    close("gate offset: contributions add up to the offset (mm)",
          sum(after["from_edges_starting_s"].values()), after["offset_mm"], 0.21)
    # Signed Z error of each trajectory, split by where the edges on the path start.
    for tag, before, during, total in (("cvx", 124.9, -70.3, 65.2), ("cvg", 122.8, -16.0, 112.2)):
        part = after[f"z_error_{tag}_from_edges_starting_s"]
        close(f"gate offset: {tag} Z error from edges before 25 s (mm)", part["0-15"] + part["15-25"], before, 0.11)
        close(f"gate offset: {tag} Z error from edges starting 25-60 s (mm)", part["25-60"], during, 0.051)
        close(f"gate offset: {tag} Z error after 60 s (mm)", after[f"z_error_{tag}_mm"], total, 0.051)
        close(f"gate offset: {tag} parts add up to the Z error (mm)", sum(part.values()), total, 0.21)
    close("gate offset: non-gated Z error per edge, 25-60 s (mm)", build["board_z_residual_cvx_mm"], -8.0, 0.051)
    close("gate offset: gated Z error per edge, 25-60 s (mm)", build["board_z_residual_cvg_mm"], -4.0, 0.051)
    shape = acc["graph"]
    if (shape["components_ending_before_22s"], shape["two_largest"]["nodes"], shape["two_largest"]["span_s"]) \
            != (24, [252, 147], [108.1, 66.6]):
        raise AssertionError("gate offset: the description of the 31 components changed")
    both = next(r for r in acc["offset_at_nodes"] if r["node_time_s"] == [15.0, 30.0])
    close("gate offset: common Z error at 15-30 s, non-gated (mm)", both["z_error_cvx_mm"], 101.7, 0.051)
    close("gate offset: common Z error at 15-30 s, gated (mm)", both["z_error_cvg_mm"], 101.7, 0.051)
    graph = acc["graph"]
    if (graph["components"], graph["anchored_in_initial_rest"]["components"], graph["anchored_in_initial_rest"]["nodes"],
            graph["anchored_after_dropout"]["components"], graph["anchored_after_dropout"]["nodes"]) != (31, 29, 598, 2, 80):
        raise AssertionError("gate offset: the pose-graph components changed")
    # Rig adaptation: the same seed a second time.
    for fold in ("ab", "ba"):
        first = load(DATA / f"stop_eval_20261109_adapted_bal_{fold}.json")
        again = load(DATA / f"stop_eval_20261109_adapted_bal_{fold}_s0rep.json")
        for key in ("physnet", "physnet_switch"):
            for (name, a), (_n, b) in zip(_leaves(first[key]), _leaves(again[key])):
                if abs(a - b) > 0.0015:
                    raise AssertionError(f"same-seed repeat differs: fold {fold} {key}{name}: {a} vs {b}")
    print("OK rig adaptation: the same seed reproduces every reported number of both folds")


def _leaves(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _leaves(v, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _leaves(v, f"{prefix}[{i}]")
    else:
        yield prefix, float(obj)


def text_claims():
    """Numbers whose wording in the manuscript was corrected must stay present,
    and retired claims must not reappear."""
    text = "\n".join(p.read_text(encoding="utf-8")
                     for p in [PAPER / "main.tex", *sorted((PAPER / "sections").glob("*.tex"))])
    required = ["24.95", "25.77", "25.17", "31 interleaved tree-shaped components",
                "14.1\\%", "4.3\\%", "0.37/0.48/0.20", "81.0\\%", "63.2\\%",
                "0.13--0.16~ms", "(-20.9,\\,-4.1,\\,-17.8)", "10--19\\%", "24.63", "43.89",
                "16.77", "25.00", "42.12", "31.02", "128.0", "$[-7.1,-0.3]$", "44.3~ms", "8.1~mm",
                "18.5\\%", "43.6\\%", "46.7 against 32.0~mm", "10.4~ms", "14 to 102~ms",
                "0.28~mm", "$-0.40$\\%", "99.4\\%", "0.991", "$-50.6$", "$[-70.1,-32.6]$",
                "raises the error by 33.5\\%", "45.4 against 20.7~mm", "did not reproduce",
                "39.41", "27.42", "35.58 against 15.94~mm", "25.74 against 21.04~mm", "$+7.3$",
                "$+14.6$ ($[-0.2,29.4]$)", "$+12.6$ points ($[2.1,23.8]$", "$+14.3$ and $+13.0$ points",
                "loses 13.6 points", "twin $+13.3$",
                "$-35.5$ points", "15--24~mm/s", "14--102~ms", "$+25.0$-point prediction",
                "$[-54.5,-19.1]$", "$[2.3,12.4]$", "$[-18.3,-8.7]$", "$[-3.7,4.8]$",
                "synthetic-test errors by up to 0.8~mm", "out-of-distribution errors by up to 1.6~mm",
                "changed by up to 15~mm", "29--111-mm effect", "7 and 9\\% below the best public configuration",
                "31.1--31.6", "44.8--45.8", "about 5~ms", "It costs 1.0--2.4~mm on validation",
                "($-1.6$ to $+0.2$~mm)", "33.27 and 33.11~mm against 33.15~mm",
                "103.3 against 89.7~mm and 156.8 against 132.9~mm", "102.3--130.7~mm", "RoNIN-ResNet 112.0~mm",
                "94.8 and 103.8~mm on the held-out recording and 129.6 and 141.7~mm on T1",
                "192 windows that start between 25 and 60~s", "4.0~mm per edge", "leaves 65 and 112~mm",
                "the non-gated by $-8.0$ and the gated by $-4.0$~mm per edge", "hold 399 of the 678 nodes",
                "63.8 against 66.6~mm", "but establishes neither",
                "0.151 against 0.355 and 0.345", "1.8~mm RMS", "0.2 and 0.6~mm",
                "1.9\\times10^{-5}$", "40--48\\%", "0.44~mm RMS per axis",
                "12--23\\%", "19--32\\%",
                "validation errors by up to 1.1~mm", "($+0.9$ to $-0.4$~mm)",
                "33.5 and 33.8 against 34.2~mm", "up to 0.3~mm on CV, 0.4~mm on $e_p$ and 16~mm on ATE",
                "internal three-direction protocol was not completed", "2.05~s",
                "2.67$--$3.05", "3.40$--$3.81", "35.95", "29.66", "22.65", "80.2~mm",
                "64 adjacent actual moves", "0.12~mm median", "p95 0.94~mm", "288 hold pairs",
                "630.5~s", "7625 frames", "7107 usable poses", "1966$^\\dagger$", "3745$^\\dagger$"]
    retired = ["26.3~mm", "lower inference latency", "reaches that limit",
               "information-limit measurement", "More than\nmore than",
               "IMUNet by 24", "31.2\\%", "0.74--3.4", "upward drift", "visual SLAM",
               "Note that dividing", "periodic anchoring", "impractical", "deadband",
               "theestimator", "agreement with the reference clearly", "1-s relative-rotation",
               "about 23 points", "preset moving-speed band",
               "$+25.1$ predicted", "$[-54.3,-19.6]$", "Repeating the whole factorial",
               "TLIO-ResNet with body-frame input is lowest", "TLIO-ResNet is lowest",
               "35.6~ms", "NX8e 4K", "14--90~ms", "about 17~GB", "rather than a replacement for vision",
               "about 6~ms", "about 56~ms", "2.1--2.6~mm below", "validation cost was 0.9 and 1.5~mm",
               "only IMUNet", "12--24\\%", "19--33\\%", "3-s and 4-s context were",
               "mean probability 0.01--0.10", "2--5~mm per edge", "so both the auxiliary loss and the gate contribute",
               "0.144 against 0.362 and 0.356", "0.26~mm RMS along", "without a $+1/2$ segment",
               "parallel interleaved chains", "systematic board-normal shift"]
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
    twin_protocol()
    revision_results()
    text_claims()
    latex_integrity()
    print("Manuscript regression checks passed.")
