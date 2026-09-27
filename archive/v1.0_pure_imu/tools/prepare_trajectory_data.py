#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare cleaned chessboard labels and a leakage-free session split.

Existing 2026-09-23 sessions retain their historical split. New sessions use
the split written before capture in session_meta.json. The fixed held-out test
session is never used for fitting or model selection; further predeclared test
recordings (theme T) are appended to the sealed `test` list so the benchmark
can report several independent test sessions.

Rigidity recordings (rigid_*) with a passing rigidity verdict and enough
usable chessboard frames become `extra_train`: rotation-rich, still-phase
training data that never enters validation, test or leave-one-session-out
scoring.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.clean_pose_gt import clean

DATA = ROOT / "datasets"
GT = DATA / "pose_gt"
RAW = DATA / "pose_gt_raw"
FIXED_VAL = "traj_20260923_023241"
FIXED_TEST = "traj_20260923_023422"
EXTRA_MIN_USABLE = 300          # frames with a usable chessboard pose
EXTRA_PREFIXES = ("rigid_",)
PROTOCOL_LOCK = DATA / "study_protocol_v2.lock.json"


def _protocol() -> dict:
    if not PROTOCOL_LOCK.is_file():
        return {}
    return json.loads(PROTOCOL_LOCK.read_text(encoding="utf-8"))


def _summary(name: str) -> dict:
    data = np.load(GT / f"{name}.npz")
    ok = data["ok"] == 1
    still = ok & (data["ble_gyro"] < 8.0) & (data["reproj"] < 1.5)
    return {
        "name": name,
        "frames": int(len(ok)),
        "pnp": int(ok.sum()),
        "pnp_still": int(still.sum()),
        "stereo": int(data["stereo_ok"].sum()),
        "reproj_p50": float(np.median(data["reproj"][ok])) if ok.any() else None,
        "size": [int(v) for v in data["image_size"]],
        "delay_usb_ms": round(float(data["delay_usb"][0]) * 1000.0, 1),
    }


def _export_raw(name: str) -> int:
    data = np.load(GT / f"{name}.npz")
    usable = (
        (data["ok"] == 1)
        & (data["reproj"] < 1.5)
        & (data["ble_gyro"] < 8.0)
    ).astype(np.uint8)
    RAW.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RAW / f"{name}.npz",
        t=data["t"],
        R=data["R"].astype(np.float32),
        p=data["p"].astype(np.float32),
        usable=usable,
        px=data["reproj"].astype(np.float32),
    )
    return int(usable.sum())


def _rigid_verdict(name: str) -> str:
    """Rigidity verdict from the capture QC file, or recomputed if missing."""
    qc = DATA / name / "qc_result.json"
    if qc.is_file():
        try:
            return str(json.loads(qc.read_text(encoding="utf-8")).get("rigidity", ""))
        except json.JSONDecodeError:
            pass
    try:
        from tools.check_rigidity import check
        return str(check(name).get("verdict", ""))
    except Exception:  # noqa: BLE001
        return ""


def main() -> None:
    protocol = _protocol()
    protocol_id = protocol.get("protocol_id")
    audit = json.loads((DATA / "audit_sessions.json").read_text(encoding="utf-8"))
    usable_audit = {
        row["name"]: row
        for row in audit
        if row.get("usable") and row["name"].startswith("traj_")
    }
    # Rigidity recordings are audited with a different gate: they only need
    # a usable IMU stream and a chessboard pose file.
    extra_audit = {
        row["name"]: row
        for row in audit
        if row["name"].startswith(EXTRA_PREFIXES) and (GT / f"{row['name']}.npz").is_file()
        and row.get("usb", {}).get("hz", 0.0) >= 195.0
    }
    old_index_path = GT / "index.json"
    old_index = json.loads(old_index_path.read_text(encoding="utf-8")) if old_index_path.is_file() else []
    old_by_name = {row["name"]: row for row in old_index}

    rows = []
    split = {"train": [], "val": [], "test": [FIXED_TEST], "extra_train": [], "excluded": []}
    quality = {}
    for name, audit_row in sorted({**usable_audit, **extra_audit}.items()):
        path = GT / f"{name}.npz"
        if not path.is_file():
            split["excluded"].append({"name": name, "reason": "missing pose_gt"})
            continue
        # Only newly acquired sessions need the jump cleaner here; historical
        # files have already been cleaned and re-running would obscure counts.
        clean_row = None
        if name not in old_by_name:
            clean_row = clean(path)
        row = _summary(name)
        if clean_row:
            row.update({
                "pnp_clean": clean_row["kept"],
                "pnp_still": clean_row["kept_still"],
                "dropped_jumps": clean_row["dropped_jumps"],
            })
        else:
            for key in ("pnp_clean", "dropped_jumps"):
                if key in old_by_name.get(name, {}):
                    row[key] = old_by_name[name][key]
        raw_count = _export_raw(name)
        row["raw_usable"] = raw_count
        rows.append(row)

        meta_path = DATA / name / "session_meta.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
        if name.startswith(EXTRA_PREFIXES):
            verdict = _rigid_verdict(name)
            if verdict != "rigid":
                assigned = None
                split["excluded"].append({"name": name, "reason": f"rigidity verdict '{verdict}'"})
            elif raw_count < EXTRA_MIN_USABLE:
                assigned = None
                split["excluded"].append({"name": name, "reason": f"only {raw_count} usable frames"})
            else:
                assigned = "extra_train"
        elif name == FIXED_TEST:
            assigned = "test"
        elif name == FIXED_VAL:
            assigned = "val"
        elif meta.get("split") in {"train", "val", "test"}:
            assigned = meta["split"]
            if assigned == "test" and (
                meta.get("theme") != "T" or
                not protocol_id or
                meta.get("protocol_id") != protocol_id or
                not meta.get("prospective_sealed")
            ):
                split["excluded"].append({
                    "name": name,
                    "reason": "new test recording does not match frozen prospective protocol",
                })
                assigned = None
        elif name.startswith("traj_20260923_"):
            assigned = "train"
        else:
            assigned = None
        if assigned is None and not name.startswith(EXTRA_PREFIXES):
            split["excluded"].append({"name": name, "reason": "no predeclared split"})
        elif assigned is not None and name not in split[assigned]:
            # Additional predeclared test recordings stay sealed but are listed
            # so tools/eval_session_cv.py can score every test session once.
            split[assigned].append(name)

        quality[name] = {
            "split": assigned,
            "theme": meta.get("theme"),
            "seconds": audit_row["seconds"],
            "fps": round(audit_row["frames"] / max(audit_row["seconds"], 1e-6), 2),
            "usb_hz": round(audit_row["usb"].get("hz", 0.0), 2),
            "bt_hz": round(audit_row["bt"].get("hz", 0.0), 2),
            "raw_usable": raw_count,
            "usable_fraction": round(raw_count / max(row["frames"], 1), 3),
            "reproj_p50_px": None if row["reproj_p50"] is None else round(row["reproj_p50"], 3),
        }

    # Ensure no session can leak across groups.
    groups = ("train", "val", "test", "extra_train")
    for a in groups:
        for b in groups:
            if a < b and set(split[a]) & set(split[b]):
                raise RuntimeError(f"split overlap: {a}, {b}")
    if FIXED_TEST not in split["test"]:
        raise RuntimeError("fixed test session missing")
    split["test"] = [FIXED_TEST] + sorted(n for n in split["test"] if n != FIXED_TEST)
    if not split["train"] or not split["val"]:
        raise RuntimeError("empty train or validation split")

    GT.mkdir(parents=True, exist_ok=True)
    old_index_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    raw_index = [
        {"name": row["name"], "frames": row["frames"], "usable": row["raw_usable"]}
        for row in rows
    ]
    (RAW / "index.json").write_text(json.dumps(raw_index, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {**split, "quality": quality}
    prospective = [
        n for n in split["test"]
        if n != FIXED_TEST and quality.get(n, {}).get("theme") == "T"
    ]
    payload["protocol_id"] = protocol_id
    payload["prospective_test_status"] = {
        "required": int(protocol.get("prospective_acquisition", {})
                        .get("required_test_sessions", 4)),
        "accepted": len(prospective),
        "sessions": prospective,
        "complete": len(prospective) >= int(
            protocol.get("prospective_acquisition", {})
            .get("required_test_sessions", 4)
        ),
    }
    out = DATA / "trajectory_split_20260924.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
