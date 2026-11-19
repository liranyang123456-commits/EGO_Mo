#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-run the capture program's post-recording check for recorded sessions.

Same steps as CaptureApp._qc_after_record: chessboard poses with the
session's calibration, rigidity check, qc_result.json (and rigidity.json for
step 9), and a row in datasets/collection_log.csv. Use it when the automatic
check failed after a recording.

    python tools/rerun_session_qc.py rigid_20261108_003840 rigid_20261108_004048
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_capture import camera_config
from ego_capture.app import COLLECTION_LOG, LOG_FIELDS

DATA = ROOT / "datasets"


def rerun(name: str) -> dict:
    from tools.build_pose_gt import process_session
    from tools.check_rigidity import check
    session = DATA / name
    meta = json.loads((session / "session_meta.json").read_text(encoding="utf-8"))
    summary = json.loads((session / "capture_summary.json").read_text(encoding="utf-8"))
    step = meta.get("step", "")
    camera_config.use_for_labels(meta.get("intrinsics_dir"))
    pose = process_session(name)
    rig = check(name)
    frames = max(int(pose.get("frames") or 0), 1)
    still = int(pose.get("pnp_still") or 0)
    frac = still / frames
    p50 = pose.get("reproj_p50")
    seconds = float(summary.get("seconds") or 0.0)
    width, height = pose.get("size") or [0, 0]
    px_limit = 0.4 if width >= 1280 else 1.2
    problems = []
    if step == "traj" and meta.get("theme") != "J" and frac < 0.70:
        problems.append(f"棋盘可用帧 {frac:.0%} < 70%")
    if step == "stage_ref" and frac < 0.80:
        problems.append(f"棋盘可用帧 {frac:.0%} < 80%")
    if p50 is not None and p50 > px_limit:
        problems.append(f"重投影中位数 {p50:.2f} px > {px_limit}")
    if summary.get("drops"):
        problems.append(f"丢帧 {summary['drops']}")
    if step == "rigid" and rig.get("verdict") != "rigid":
        problems.append("IMU 刚性不合格")
        from ego_capture.revision import rigidity_motion_hint
        hint = rigidity_motion_hint(name)
        if hint:
            problems.append(hint)
    row = {"session": name, "time": datetime.now().isoformat(timespec="seconds"),
           "resolution": f"{width}x{height}", "fps": round(frames / seconds, 1) if seconds > 0 else 0.0,
           "theme": meta.get("theme", ""), "split": meta.get("split", ""), "seconds": round(seconds, 1),
           "frames": frames, "board_still": still, "usable_frac": round(frac, 3),
           "reproj_p50_px": None if p50 is None else round(p50, 3), "delay_ms": pose.get("delay_usb_ms"),
           "corr_tilt_x": rig.get("corr_tilt_x", ""), "corr_tilt_y": rig.get("corr_tilt_y", ""),
           "corr_roll": rig.get("corr_roll", ""), "rigidity": rig.get("verdict", ""),
           "verdict": "合格" if not problems else "不合格：" + "；".join(problems)}
    (session / "qc_result.json").write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    if step == "rigid":
        (session / "rigidity.json").write_text(json.dumps(rig, ensure_ascii=False, indent=2), encoding="utf-8")
    new = not os.path.isfile(COLLECTION_LOG)
    with open(COLLECTION_LOG, "a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})
    return {"row": row, "rigidity": rig, "calibration": meta.get("intrinsics_dir")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sessions", nargs="+")
    for name in ap.parse_args().sessions:
        out = rerun(name)
        print(json.dumps({"session": name, "calibration": out["calibration"],
                          **{k: out["row"][k] for k in ("frames", "usable_frac", "reproj_p50_px", "delay_ms",
                                                         "rigidity", "verdict")},
                          "quality": out["rigidity"].get("quality"),
                          "R_camera_imu": out["rigidity"].get("R_camera_imu")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
