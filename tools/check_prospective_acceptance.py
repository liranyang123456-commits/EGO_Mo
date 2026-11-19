#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Apply the frozen protocol's acceptance limits to every theme-T recording.

tools/prepare_trajectory_data.py (frozen) only checks theme, split and
protocol ID, so a recording that violates the protocol's sensor or image
limits would still be counted. This gate reads the limits from
datasets/study_protocol_v2.lock.json and, with --apply, moves failing
recordings to datasets/rejected_prospective/ (kept on disk, logged in the
checklist) so the frozen pipeline never sees them. It computes no model
metric.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
REJECTED = DATA / "rejected_prospective"
CHECKLIST = DATA / "prospective_test_checklist.json"


def _log_rows():
    rows = {}
    with (DATA / "collection_log.csv").open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            rows[r["session"]] = r
    return rows


def check(name: str, protocol: dict, log: dict) -> dict:
    acc = protocol["acceptance"]
    sess = DATA / name
    meta = json.loads((sess / "session_meta.json").read_text(encoding="utf-8"))
    summ = json.loads((sess / "capture_summary.json").read_text(encoding="utf-8"))
    req = meta.get("requested", {})
    frames = int(summ.get("frames") or 0)
    left = len(list((sess / "cam0" / "images").glob("*.jpg")))
    row = log.get(name, {})
    usable = float(row["usable_frac"]) if row.get("usable_frac") else None
    reproj = float(row["reproj_p50_px"]) if row.get("reproj_p50_px") else None
    want = protocol["sensor_protocol"]["camera_resolution"]
    tests = {
        "protocol_id": meta.get("protocol_id") == protocol["protocol_id"],
        "sealed": bool(meta.get("prospective_sealed")),
        "resolution": [req.get("width"), req.get("height")] == want,
        "usb_imu_hz": float(summ.get("usb_hz") or 0) >= acc["usb_imu_hz_min"],
        "ble_imu_hz": float(summ.get("bt_hz") or 0) >= acc["ble_imu_hz_min"]
                      and int(summ.get("bt_samples") or 0) > 1000,
        "left_image_fraction": frames > 0 and left / frames >= acc["left_image_fraction_min"],
        "chessboard_usable_fraction": usable is not None
                                      and usable >= acc["chessboard_usable_fraction_min"],
        "reprojection_p50": reproj is not None and reproj <= acc["reprojection_p50_px_max_720p"],
        "frame_drops": int(summ.get("drops") or 0) <= acc["frame_drops_max"],
    }
    return {
        "session": name, "passed": all(tests.values()),
        "failed": [k for k, v in tests.items() if not v],
        "values": {
            "resolution": f"{req.get('width')}x{req.get('height')}",
            "seconds": round(float(summ.get("seconds") or 0), 1),
            "usb_hz": round(float(summ.get("usb_hz") or 0), 1),
            "bt_hz": round(float(summ.get("bt_hz") or 0), 1),
            "bt_samples": int(summ.get("bt_samples") or 0),
            "left_image_fraction": round(left / frames, 3) if frames else None,
            "usable_fraction": usable, "reproj_p50_px": reproj,
            "drops": int(summ.get("drops") or 0),
            "created_at": meta.get("created_at"),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="move failing recordings to datasets/rejected_prospective/")
    args = ap.parse_args()
    protocol = json.loads((DATA / "study_protocol_v2.lock.json").read_text(encoding="utf-8"))
    log = _log_rows()
    results = []
    for sess in sorted(DATA.glob("traj_*")):
        meta_path = sess / "session_meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("theme") == "T" or meta.get("split") == "test" and meta.get("protocol_id"):
            results.append(check(sess.name, protocol, log))
    for r in results:
        print(("PASS " if r["passed"] else "FAIL ") + r["session"],
              "" if r["passed"] else r["failed"], json.dumps(r["values"], ensure_ascii=False))
    days = sorted({(r["values"]["created_at"] or "")[:10] for r in results if r["passed"]})
    print(f"accepted: {sum(r['passed'] for r in results)}; acquisition days: {days}")
    if not args.apply:
        return
    checklist = json.loads(CHECKLIST.read_text(encoding="utf-8"))
    rejected = checklist.setdefault("rejected", [])
    known = {x["session"] for x in rejected}
    REJECTED.mkdir(exist_ok=True)
    for r in results:
        if r["passed"]:
            continue
        shutil.move(str(DATA / r["session"]), str(REJECTED / r["session"]))
        if r["session"] not in known:
            rejected.append({"session": r["session"], "reason": "failed frozen acceptance: "
                             + ", ".join(r["failed"]), "values": r["values"],
                             "kept_on_disk": str(REJECTED / r["session"])})
        else:
            for x in rejected:
                if x["session"] == r["session"]:
                    x["kept_on_disk"] = str(REJECTED / r["session"])
        print("moved", r["session"], "->", REJECTED)
    CHECKLIST.write_text(json.dumps(checklist, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
