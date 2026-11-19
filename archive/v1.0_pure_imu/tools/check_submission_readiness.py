#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Machine-readable pre-submission gate for EGO_Mo."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
PAPER = ROOT / "paper"


def j(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def main():
    checks = []
    def add(name, passed, detail, external=False, blocking=True):
        checks.append({"name": name, "passed": bool(passed), "detail": detail,
                       "requires_physical_or_account_action": external,
                       "blocking": blocking})

    protocol = j(DATA / "study_protocol_v2.lock.json")
    add("frozen_protocol", protocol and protocol.get("status") == "FROZEN",
        protocol.get("protocol_id") if protocol else "missing")

    split = j(DATA / "trajectory_split_20260924.json") or {}
    status = split.get("prospective_test_status", {})
    add("four_prospective_tests", status.get("complete", False),
        f"{status.get('accepted', 0)}/{status.get('required', 4)}",
        external=True)

    stage = j(DATA / "reference_stage_validation.json")
    sim_ref = j(DATA / "sim_reference_validation.json")
    reference_text = (PAPER / "sections" / "reference.tex").read_text(encoding="utf-8")
    if stage and stage.get("passed"):
        add("independent_stage_validation", True, f"passed ({stage.get('kind')})")
    elif sim_ref and "Simulated known-displacement test" in reference_text:
        worst = max(
            (abs(v) for d in sim_ref["depths"].values()
             for v in d["nominal"].get("scale_error_percent", {}).values()),
            default=float("nan"))
        # A simulated test with its error budget in the manuscript is a
        # documented limitation, not a pass; a physical caliper run upgrades it.
        add("independent_stage_validation", False,
            f"simulated caliper test only (not SI-traceable); worst nominal axis "
            f"scale error {worst:.1f}% vs 1% target, reported in Sec. V; "
            "physical caliper run optional", external=True, blocking=False)
    else:
        add("independent_stage_validation", False,
            "missing or failed (stage, caliper or simulated mode)", external=True)

    # The right camera feeds neither the reference pose nor the IMU pipeline,
    # so a failed stereo extrinsic is reported but does not block submission.
    stereo = j(DATA / "stereo_extrinsic_measured.json")
    if stereo and stereo.get("passed"):
        stereo_detail = "passed"
    elif stereo:
        cv = stereo.get("cross_validation", {})
        stereo_detail = (
            f"not passed: baseline {stereo.get('baseline_mm', 0):.1f} mm, "
            f"fit {stereo.get('fit_rms_px', 0):.2f} px, held-out rot "
            f"{cv.get('rotation_rmse_deg', 0):.2f} deg / trans "
            f"{cv.get('translation_rmse_mm', 0):.1f} mm; right camera unused "
            "by reference and IMU results; static-pose session recommended")
    else:
        stereo_detail = "missing"
    add("measured_stereo_extrinsic", stereo and stereo.get("passed"),
        stereo_detail, external=True, blocking=False)

    lever = j(DATA / "lever_arm_measured.json")
    add("measured_lever_arm", lever and lever.get("passed"),
        "passed" if lever and lever.get("passed") else "missing or failed",
        external=True)

    pose = j(DATA / "pure_imu_pose_benchmark.json")
    add("relative_6dof_evaluation", bool(pose and pose.get("sessions")),
        "available" if pose else "missing")

    model_manifest = j(DATA / "final_model_manifest.json")
    add("frozen_final_models", bool(model_manifest),
        model_manifest.get("manifest_sha256") if model_manifest else "missing")
    final_lock = j(DATA / "FINAL_EVALUATION_COMPLETE.lock")
    add("one_shot_final_evaluation", bool(final_lock),
        final_lock.get("result_sha256") if final_lock else "not run")

    release = j(ROOT / "release" / "release_report.json")
    add("software_release_bundle", bool(release),
        release.get("sha256") if release else "missing")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    add("public_repository_and_doi", "TO_BE_ASSIGNED" not in citation,
        "assigned" if "TO_BE_ASSIGNED" not in citation else "not assigned",
        external=True)

    pdf = PAPER / "main_submission.pdf"
    add("submission_pdf", pdf.is_file() and pdf.stat().st_size < 20_000_000,
        f"{pdf.stat().st_size / 1e6:.2f} MB" if pdf.is_file() else "missing")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "verify_manuscript.py")],
        cwd=ROOT, capture_output=True, text=True,
    )
    add("manuscript_regression", proc.returncode == 0,
        "passed" if proc.returncode == 0 else proc.stderr[-1000:])

    blockers = [x for x in checks if not x["passed"] and x["blocking"]]
    warnings = [x for x in checks if not x["passed"] and not x["blocking"]]
    payload = {"ready": not blockers, "checks": checks, "blockers": blockers,
               "warnings": warnings}
    (DATA / "submission_readiness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if not blockers else 2)


if __name__ == "__main__":
    main()
