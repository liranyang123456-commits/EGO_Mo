#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Everything after the recordings of the revision experiments, in one run.

    python tools/revision_pipeline.py                                   # recordings in datasets/
    python tools/revision_pipeline.py --data-root datasets/revision_sim # virtual capture day

1. Reads the plan that the capture program keeps and lists what was recorded,
   accepted and rejected; an experiment with missing recordings is skipped.
2. Makes sure every accepted recording has its chessboard labels (pose_gt,
   pose_gt_raw); missing ones are rebuilt with the program's own routine.
3. Finds the rigidity check of each recording day and its camera<-IMU rotation.
4. Caliper standard: reference_stage_validation.py and reference_caliper_dynamic.py.
5. Stop experiment: eval_stop_protocol_experiment.py with the frozen models
   and the day's IMU mount.
6. Wording for the manuscript markers: revision_report.py.

Writes revision_pipeline_summary.json and revision_pipeline_summary.md into
the data root. Nothing in the manuscript is changed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import ego_capture.revision as rv
import tools.build_pose_gt as bp
import tools.check_rigidity as cr

DATA = ROOT / "datasets"


def _run(args: list[str]) -> tuple[bool, str]:
    proc = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _caliper_step(static: dict | None, dyn: dict | None) -> dict:
    """Summary of the caliper analysis for revision_pipeline_summary.json/.md."""
    if not static:
        return {"error": "reference_stage_validation.json missing"}
    s = static["summary"]
    step = {
        "passed": static["passed"], "working_distance_mm": s.get("working_distance_mm"),
        "heldout_rmse_mm": round(s["heldout_rmse_mm"], 3), "heldout_max_mm": round(s["heldout_max_mm"], 3),
        "repeatability_p95_mm": round(s["repeatability_p95_mm"], 3),
        "scale_error_percent": {a: round(v["scale_error_percent"], 2) for a, v in s["axes"].items()},
        "cross_axis_rmse_mm": {a: round(v["cross_axis_rmse_mm"], 3) for a, v in s["axes"].items()},
        "label_error": dyn["overall"]["label_error"] if dyn else None,
        "hold_pair_displacement": dyn["overall"].get("hold_pair_displacement") if dyn else None,
        "adjacent_moves": dyn["overall"].get("adjacent_moves") if dyn else None,
        "within_U_label": dyn["overall"]["label_error_within_U_label_nominal"] if dyn else None,
        "off_axis_moving": dyn["overall"]["off_axis_moving"] if dyn else None,
        "directions": len(s["axes"])}
    th = static["thresholds"]
    step["limits_met"] = bool(
        s["heldout_rmse_mm"] <= th["heldout_rmse_mm_max"] and s["heldout_max_mm"] <= th["heldout_max_mm_max"]
        and max(abs(v["scale_error_percent"]) for v in s["axes"].values()) <= th["axis_scale_error_percent_max_abs"]
        and max(v["cross_axis_rmse_mm"] for v in s["axes"].values()) <= th["cross_axis_rmse_mm_max"]
        and s["repeatability_p95_mm"] <= th["repeatability_p95_mm_max"])
    cc = s.get("camera_centre")
    if cc:
        step["camera_centre"] = {
            "passed": static.get("passed_camera_centre"), "heldout_rmse_mm": round(cc["heldout_rmse_mm"], 3),
            "scale_error_percent": {a: round(v["scale_error_percent"], 2) for a, v in cc["axes"].items()}}
        step["tilt_check"] = s.get("tilt_check")
    return step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=DATA)
    ap.add_argument("--rebuild-labels", action="store_true", help="re-solve the chessboard for every recording")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()
    root = args.data_root.resolve()
    rv.DATA_ROOT, rv.PLAN_PATH = root, root / "revision_experiment_plan.json"
    rv.CALIPER_PLAN_CSV = root / "reference_caliper_plan.csv"
    bp.DATA, bp.OUT, cr.DATA = root, root / "pose_gt", root
    sim = root != DATA.resolve()
    out: dict = {"data_root": str(root), "simulated": sim, "steps": {}}
    say = lambda msg: print(msg, flush=True)  # noqa: E731

    # 1. plan
    if not rv.PLAN_PATH.is_file():
        raise SystemExit(f"no plan at {rv.PLAN_PATH}: nothing was recorded with the guided windows")
    plan = rv.load_plan()
    status = {}
    for kind in ("caliper", "stop"):
        entries = plan.get(kind, [])
        done = [e for e in entries if e["status"] == "done" and e.get("session")]
        missing = [e.get("label") or f"{e['axis']}{e['trial']}" for e in entries if e not in done]
        status[kind] = {"planned": len(entries), "done": len(done), "missing": missing,
                        "sessions": [e["session"] for e in done]}
        say(f"[1] {kind}: {len(done)}/{len(entries)} accepted" + (f", missing {missing}" if missing else ""))
    rejected = plan.get("rejected_sessions", [])
    status["rejected_sessions"] = rejected
    out["steps"]["plan"] = status
    if rejected:
        say(f"[1] rejected and re-recorded: {len(rejected)} ({', '.join(r['session'] for r in rejected)})")

    # 2. labels
    rebuilt = []
    for kind in ("caliper", "stop"):
        for name in status[kind]["sessions"]:
            have = (root / "pose_gt" / f"{name}.npz").is_file() and (root / "pose_gt_raw" / f"{name}.npz").is_file()
            if have and not args.rebuild_labels:
                continue
            say(f"[2] labels for {name}")
            rv.build_labels(name, require_board_still=kind == "stop")
            rebuilt.append(name)
    out["steps"]["labels"] = {"rebuilt": rebuilt}

    # 3. rigidity of each recording day
    days = sorted({n.split("_")[1] for n in status["stop"]["sessions"]})
    rig = {}
    for day in days:
        r = rv.todays_rigidity(day)
        if r is None:
            rig[day] = None
            say(f"[3] {day}: no passing rigidity check; the stop analysis cannot use the day's IMU mount")
            continue
        ang = rv.rotation_angle_deg(rv.NOMINAL_R_CI, r["R_camera_imu"])
        rig[day] = {"session": r["session"], "quality": r.get("quality"), "angle_to_nominal_deg": round(ang, 2)}
        say(f"[3] {day}: {r['session']} ({r.get('quality')}), mount {ang:.1f} deg from the frozen one")
    out["steps"]["rigidity"] = rig

    # 4. caliper standard
    final = plan.get("caliper_final")
    if final:
        static, dyn = _load(root / "reference_stage_validation.json"), _load(root / "reference_caliper_dynamic.json")
        out["steps"]["caliper"] = {"final": final, **_caliper_step(static, dyn)}
        say(f"[4] caliper: final result from {final.get('source')} ({final.get('note', '')}); not re-analysed")
    elif status["caliper"]["done"] and not status["caliper"]["missing"]:
        rv.update_caliper_csv(plan)
        ok1, log1 = _run(["tools/reference_stage_validation.py", "analyze", "--mode", "caliper", "--data-root", str(root)])
        ok2, log2 = _run(["tools/reference_caliper_dynamic.py", "--data-root", str(root)])
        _run(["tools/reference_caliper_dynamic.py", "--data-root", str(root), "--quantity", "camera"])
        static, dyn = _load(root / "reference_stage_validation.json"), _load(root / "reference_caliper_dynamic.json")
        if ok1 and static:
            s = static["summary"]
            out["steps"]["caliper"] = _caliper_step(static, dyn if ok2 else None)
            cc = s.get("camera_centre")
            say(f"[4] caliper (board translation): RMSE {s['heldout_rmse_mm']:.2f} mm, scale "
                + ", ".join(f"{a} {v['scale_error_percent']:+.2f}%" for a, v in s["axes"].items())
                + f", limits {'met' if static['passed'] else 'not met'}")
            if cc:
                say(f"[4] caliper (camera centre): RMSE {cc['heldout_rmse_mm']:.2f} mm, limits "
                    f"{'met' if static.get('passed_camera_centre') else 'not met'}"
                    + ("" if not s.get("tilt_check") else
                       f"; jaw tilt up to {s['tilt_check']['imu_tilt_max_deg']:.2f} deg (IMU), "
                       f"PnP vs IMU {s['tilt_check']['rms_diff_deg']:.3f} deg RMS"))
        else:
            out["steps"]["caliper"] = {"error": log1[-600:]}
            say("[4] caliper analysis failed:\n" + log1[-600:])
    else:
        say("[4] caliper: skipped (recordings missing)")

    # 5. stop experiment
    if status["stop"]["done"] and not status["stop"]["missing"]:
        # Each batch keeps its own result file. stop_protocol_experiment.json is
        # the first batch (handheld, 2026-11-08) and is read by the manuscript
        # checks; a later batch in the plan must not replace it.
        stop_out = root / "stop_protocol_experiment.json"
        held = (_load(stop_out) or {}).get("sessions", {})
        if held and set(held.get("continuous", []) + held.get("paused", [])) != set(status["stop"]["sessions"]):
            stop_out = root / f"stop_eval_{'_'.join(days)}_batch.json"
            say(f"[5] stop experiment: stop_protocol_experiment.json holds another batch; "
                f"this batch is written to {stop_out.name}")
        cmd = ["tools/eval_stop_protocol_experiment.py", "--data-root", str(root), "--n-boot", str(args.n_boot),
               "--output", str(stop_out)]
        if any(v is None for v in rig.values()):
            say("[5] stop experiment: a recording day has no rigidity check")
            out["steps"]["stop"] = {"error": "no rigidity check on a recording day"}
        else:
            ok, log = _run(cmd)
            rep = _load(stop_out)
            if ok and rep:
                e = rep["estimators"]["physnet_switch"]
                out["steps"]["stop"] = {
                    "reduction_continuous": e["continuous"]["reduction_vs_zero"],
                    "reduction_paused": e["paused"]["reduction_vs_zero"],
                    "reduction_gain": e["reduction_gain"], "reduction_gain_ci95": e["reduction_gain_ci95"],
                    "matched_diff_mm": e["matched_diff_mm"], "matched_diff_ci95_mm": e["matched_diff_ci95_mm"],
                    "recording_level": rep["recording_level"],
                    "per_session": {n: {"condition": v["condition"], "stops_per_min": v["stops_per_min"],
                                        "compliance": (v.get("protocol") or {}).get("compliance"),
                                        "reduction": v["errors"]["physnet_switch"]["reduction_vs_zero"],
                                        "imu_mount": v["imu_mount"]} for n, v in rep["per_session"].items()}}
                say(f"[5] stop: reduction {e['continuous']['reduction_vs_zero']:+.3f} (C) vs "
                    f"{e['paused']['reduction_vs_zero']:+.3f} (P), gain {e['reduction_gain']:+.3f} "
                    f"{e['reduction_gain_ci95']}")
            else:
                out["steps"]["stop"] = {"error": log[-600:]}
                say("[5] stop analysis failed:\n" + log[-600:])
    else:
        say("[5] stop experiment: skipped (recordings missing)")

    # 6. wording
    ok, log = _run(["tools/revision_report.py", "--results-dir", str(root)])
    out["steps"]["wording"] = str(root / "revision_pending_fill.md") if ok else log[-400:]
    say(f"[6] wording: {out['steps']['wording']}")

    # 7. response figures of the two experiments (previews for a simulated root)
    ok, log = _run(["tools/make_response_figures.py", "--results-dir", str(root), "--only-experiments"])
    out["steps"]["response_figures"] = [l.split("wrote ", 1)[1] for l in log.splitlines() if l.startswith("wrote ")] \
        if ok else log[-400:]
    say(f"[7] response figures: {out['steps']['response_figures']}")

    (root / "revision_pipeline_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                                   default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)),
                                                        encoding="utf-8")
    md = ["# 修稿补充实验：处理结果" + ("（仿真采集，不是实测结果）" if sim else ""), "",
          f"- 数据目录：`{root}`",
          f"- 卡尺：{status['caliper']['done']}/{status['caliper']['planned']} 次往返合格；"
          f"停顿对照：{status['stop']['done']}/{status['stop']['planned']} 段合格；重录 {len(rejected)} 次"]
    for day, r in rig.items():
        md.append(f"- {day} 刚性检查：" + ("缺失" if r is None else f"{r['session']}，IMU 安装与冻结模型相差 {r['angle_to_nominal_deg']}°"))
    c = out["steps"].get("caliper", {})
    if "heldout_rmse_mm" in c:
        md.append(f"- 卡尺位移标准：留出 RMSE {c['heldout_rmse_mm']} mm，最大 {c['heldout_max_mm']} mm，尺度误差 "
                  + "、".join(f"{a} {v:+.2f}%" for a, v in c["scale_error_percent"].items())
                  + f"，串轴 ≤{max(c['cross_axis_rmse_mm'].values()):.2f} mm，重复性 p95 {c['repeatability_p95_mm']} mm，"
                  + ("达到" if c.get("limits_met", c["passed"]) else "未达到") + "预设门限"
                  + (f"（{c['directions']} 个方向）" if c.get("directions", 3) != 3 else ""))
        if isinstance(c.get("label_error"), dict):
            md.append(f"- 标签级误差：中位数 {c['label_error']['median_mm']} mm，p95 {c['label_error']['p95_mm']} mm，"
                      f"{100 * (c['within_U_label'] or 0):.0f}% 落在 U_label 内")
    st = out["steps"].get("stop", {})
    if "reduction_gain" in st:
        lo, hi = st["reduction_gain_ci95"]
        md.append(f"- 停顿对照：相对零运动的误差降幅 连续 {100 * st['reduction_continuous']:.1f}%、停顿 "
                  f"{100 * st['reduction_paused']:.1f}%，差 {100 * st['reduction_gain']:.1f} 个百分点"
                  f"（95% 区间 {100 * lo:.1f}～{100 * hi:.1f}）；位移匹配后差 {st['matched_diff_mm']} mm")
        for n, v in st["per_session"].items():
            comp = "" if v["compliance"] is None else f"，节拍依从 {100 * v['compliance']:.0f}%"
            md.append(f"  - {n}（{v['condition']}）：降幅 {100 * v['reduction']:.1f}%，停顿 {v['stops_per_min']}/分钟{comp}")
    md.append(f"- 论文待补句子：`{out['steps']['wording']}`")
    (root / "revision_pipeline_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    say(f"wrote {root / 'revision_pipeline_summary.md'}")


if __name__ == "__main__":
    main()
