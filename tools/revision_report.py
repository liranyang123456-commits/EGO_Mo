#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wording for the red TO BE COMPLETED markers from the experiment results.

Reads the outputs of the two revision experiments
  datasets/reference_stage_validation.json   (reference_stage_validation.py analyze --mode caliper)
  datasets/reference_caliper_dynamic.json    (reference_caliper_dynamic.py)
  datasets/stop_protocol_experiment.json     (eval_stop_protocol_experiment.py)
and writes datasets/revision_pending_fill.md: every \\pending marker of the
manuscript, response and cover letter with its file and line, the placeholder
and a proposed replacement, plus the two Highlights items. Nothing is written
into the .tex files; every sentence is read and pasted by hand. The wording
follows the result, favourable or not, and the limits and endpoint are the
pre-specified ones.

    python tools/revision_report.py
    python tools/revision_report.py --rehearsal   # twin rehearsal outputs, to check the wording only
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
PAPER = ROOT / "paper"
REH = DATA / "revision_rehearsal"
FILES = ["sections/abstract.tex", "sections/results.tex", "sections/discussion.tex",
         "sections/conclusion.tex", "Response_to_Reviewers_TIM.tex", "Cover_Letter_TIM.tex"]
TWIN_PAPER_PERCENT = 32   # Table IV: one stop every 4 s against pause-free motion


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _pct(x: float, nd: int = 0) -> str:
    v = f"{100 * x:.{nd}f}"
    return f"${v}$" if v.startswith("-") else v


def _ci(lo: float, hi: float, scale: float = 100.0, nd: int = 1) -> str:
    return f"$[{scale * lo:.{nd}f},{scale * hi:.{nd}f}]$"


# --------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------
def caliper_values(static: dict, dynamic: dict) -> dict:
    s, lim = static["summary"], static["thresholds"]
    axes = s["axes"]
    scale = {a: axes[a]["scale_error_percent"] for a in sorted(axes)}
    cross = max(axes[a]["cross_axis_rmse_mm"] for a in axes)
    failed = []
    if s["heldout_rmse_mm"] > lim["heldout_rmse_mm_max"]:
        failed.append("RMSE")
    if s["heldout_max_mm"] > lim["heldout_max_mm_max"]:
        failed.append("maximum-error")
    if max(abs(v) for v in scale.values()) > lim["axis_scale_error_percent_max_abs"]:
        failed.append("scale")
    if cross > lim["cross_axis_rmse_mm_max"]:
        failed.append("cross-axis")
    if s["repeatability_p95_mm"] > lim["repeatability_p95_mm_max"]:
        failed.append("repeatability")
    v = {"rmse": s["heldout_rmse_mm"], "max": s["heldout_max_mm"], "n": s["n_points"],
         "dist": s.get("working_distance_mm"), "scale": scale,
         "smax": max(abs(x) for x in scale.values()), "cross": cross,
         "rep": s["repeatability_p95_mm"], "passed": not failed, "failed": failed}
    if s.get("camera_centre"):
        v["cc_rmse"] = s["camera_centre"]["heldout_rmse_mm"]
    tilt = s.get("tilt_check") or {}
    if "rms_diff_deg" in tilt:
        v.update({"tilt": tilt["imu_tilt_max_deg"], "tilt_rms": tilt["rms_diff_deg"],
                  "tilt_mm": tilt.get("camera_centre_term_mm")})
    if dynamic:
        o = dynamic["overall"]
        v.update({"lmed": o["label_error"]["median_mm"], "lp95": o["label_error"]["p95_mm"],
                  "fU": o["label_error_within_U_label_nominal"],
                  "fUp": o["label_error_within_U_label_perturbed"],
                  "mmed": o["off_axis_moving"].get("median_mm"), "mp95": o["off_axis_moving"].get("p95_mm"),
                  "fast": o.get("off_axis_moving_by_speed", {}).get(">15 mm/s", {})})
        if dynamic.get("camera_overall"):
            oc = dynamic["camera_overall"]
            v.update({"cmed": oc["label_error"]["median_mm"], "cp95": oc["label_error"]["p95_mm"],
                      "cfU": oc["label_error_within_U_label_nominal"]})
    return v


def stop_values(rep: dict, power: dict | None) -> dict:
    e = rep["estimators"]["physnet_switch"]
    per = rep["per_session"]
    cont, paus = rep["sessions"]["continuous"], rep["sessions"]["paused"]
    comp = [per[n]["protocol"]["compliance"] for n in paus
            if per[n].get("protocol") and per[n]["protocol"].get("compliance") is not None]
    v = {"rc": e["continuous"]["reduction_vs_zero"], "rp": e["paused"]["reduction_vs_zero"],
         "ec": e["continuous"]["err_mm"], "ep": e["paused"]["err_mm"],
         "zc": e["continuous"]["zero_mm"], "zp": e["paused"]["zero_mm"],
         "g": e["reduction_gain"], "lo": e["reduction_gain_ci95"][0], "hi": e["reduction_gain_ci95"][1],
         "md": e["matched_diff_mm"], "mref": e["matched_continuous_err_mm"],
         "mlo": e["matched_diff_ci95_mm"][0], "mhi": e["matched_diff_ci95_mm"][1],
         "nc": len(cont), "np": len(paus),
         "spm_c": sum(per[n]["stops_per_min"] for n in cont) / len(cont),
         "spm_p": sum(per[n]["stops_per_min"] for n in paus) / len(paus),
         "compliance": sum(comp) / len(comp) if comp else None,
         "sep": rep.get("recording_level", {}).get("every_paused_above_every_continuous")}
    v["mdp"] = v["md"] / v["mref"] if v["mref"] else float("nan")
    v["excludes_zero"] = v["lo"] > 0 or v["hi"] < 0
    if power:
        c = power["conditions"]
        v["twin_gain"] = c["paused"]["pooled"]["reduction_switch"] - c["continuous"]["pooled"]["reduction_switch"]
        v["twin_rel"] = 1 - c["paused"]["pooled"]["physnet_switch_mm"] / c["continuous"]["pooled"]["physnet_switch_mm"]
    return v


# --------------------------------------------------------------------------
# wording
# --------------------------------------------------------------------------
def caliper_text(c: dict) -> dict[str, str]:
    sc = ", ".join(f"${c['scale'][a]:.1f}$" if c["scale"][a] < 0 else f"{c['scale'][a]:+.1f}" for a in c["scale"])
    dist = f" at the image scale of the recordings ({c['dist']:.0f}~mm with this camera)" if c.get("dist") else ""
    status = ("which meets the pre-specified limits" if c["passed"] else
              "which exceeds the pre-specified " + " and ".join(c["failed"]) +
              (" limit" if len(c["failed"]) == 1 else " limits"))
    where = "in the recorded direction" if len(c["scale"]) == 1 else f"for the {len(c['scale'])} directions"
    pos = (f"the held-out RMSE of the board positions at the {c['n']} holds is {c['rmse']:.2f}~mm "
           f"(maximum {c['max']:.2f}~mm), the scale error without scale fitting is {sc}\\% {where}, "
           f"and the cross-axis RMSE is at most {c['cross']:.2f}~mm{dist}, {status}")
    if "tilt" in c:
        pos += (f"; the jaw tilts by up to {c['tilt']:.2f}$^\\circ$ over the travel, which PnP reproduces to "
                f"{c['tilt_rms']:.2f}$^\\circ$ RMS against the IMU on the jaw and which moves the camera centre in "
                f"board coordinates by up to {c['tilt_mm']:.1f}~mm")
        if "cc_rmse" in c:
            pos += f" (camera-centre coordinates: {c['cc_rmse']:.2f}~mm held-out RMSE)"
    lab = ""
    if "lmed" in c:
        lab = (f"; single-frame board displacements between holds differ from the caliper by {c['lmed']:.2f}~mm "
               f"median (p95 {c['lp95']:.2f}~mm)")
        if "cmed" in c:
            lab += (f", and labels formed from the camera centre, which also carry the jaw tilt, by "
                    f"{c['cmed']:.2f}~mm median (p95 {c['cp95']:.2f}~mm, {_pct(c['cfU'])}\\% within "
                    f"$U_{{\\rm label}}$)")
        else:
            lab += f", {_pct(c['fU'])}\\% of them within $U_{{\\rm label}}$"
        lab += (f"; frames recorded while the board moves lie {c['mmed']:.2f}~mm median "
                f"(p95 {c['mp95']:.2f}~mm) off the caliper axis")
        if c["fast"].get("n"):
            lab += f" ({c['fast']['median_mm']:.2f}~mm above 15~mm/s)"
    short = (f"{c['rmse']:.2f}~mm held-out RMSE and at most {c['smax']:.1f}\\% axis scale error{dist}, "
             + ("within the pre-specified limits" if c["passed"] else
                "outside the pre-specified " + " and ".join(c["failed"]) +
                (" limit" if len(c["failed"]) == 1 else " limits")))
    status_tab = (f"{c['rmse']:.2f}~mm RMSE, {c['smax']:.1f}\\% scale, " +
                  ("limits met" if c["passed"] else "limits not met (" + ", ".join(c["failed"]) + ")"))
    return {"full": pos + lab, "short": short, "table": status_tab}


def stop_text(s: dict) -> dict[str, str]:
    ci = f"95\\% interval {_ci(s['lo'], s['hi'])}"
    gain = s["g"] > 0 and s["excludes_zero"]
    verdict = "" if s["excludes_zero"] else ", an interval that includes zero"
    mci = _ci(s["mlo"], s["mhi"], 1.0) + "~mm"
    if s["md"] < 0:
        matched = (f"at matched displacement the paused recordings have a {abs(s['md']):.1f}-mm "
                   f"({_pct(abs(s['mdp']))}\\%) lower error ({mci})")
    else:
        matched = f"at matched displacement the paused recordings have no lower error ({s['md']:+.1f}~mm, {mci})"
    stops = f"{s['spm_p']:.1f} against {s['spm_c']:.1f} reference-confirmed stops per minute"
    if s.get("compliance") is not None:
        stops += f", with {_pct(s['compliance'])}\\% of the beats followed by an IMU-detected stop"
    twin = f"{TWIN_PAPER_PERCENT}\\%"
    full = (f"a reduction relative to zero motion of {_pct(s['rc'], 1)}\\% for continuous and "
            f"{_pct(s['rp'], 1)}\\% for paused recordings ({stops}), a difference of "
            f"{_pct(s['g'], 1)} points ({ci}){verdict}; {matched}")
    response = full + (f", against the {twin} lower error predicted by the twin" +
                       (f" ({_pct(s['twin_rel'])}\\% when one twin model is scored on both protocols)"
                        if "twin_rel" in s else ""))
    verb = "raised" if s["g"] > 0 else "lowered"
    sentence = (f"a stop every 4~s {verb} the error reduction relative to zero motion from "
                f"{_pct(s['rc'])}\\% to {_pct(s['rp'])}\\% ({_pct(s['g'])} points, {ci}{verdict})")
    abstract = (f"a {_pct(s['g'])}-point larger error reduction with such stops ({ci})" if gain else
                f"no clear gain from such stops ({_pct(s['g'])} points, {ci})")
    if s["md"] < 0:
        discussion = (f"a {_pct(abs(s['mdp']))}\\% lower error at matched displacement ({mci}) and a "
                      f"{_pct(s['g'])}-point difference in reduction relative to zero motion ({ci}), "
                      f"against {twin} in the twin")
    else:
        discussion = (f"no lower error at matched displacement ({s['md']:+.1f}~mm, {mci}) and a "
                      f"{_pct(s['g'])}-point difference in reduction relative to zero motion ({ci}), "
                      f"against {twin} in the twin")
    table = (("confirms the direction" if gain else
              "opposite direction" if s["excludes_zero"] else "does not confirm it") +
             f" ({_pct(s['g'])} points, {_ci(s['lo'], s['hi'], nd=0)})")
    cover = (f"shows a {_pct(s['g'])}-point larger error reduction with a stop every 4~s ({ci})" if gain else
             f"shows no clear gain from a stop every 4~s ({_pct(s['g'])} points, {ci})")
    return {"full": full, "response": response, "sentence": sentence, "abstract": abstract,
            "discussion": discussion, "table": table, "cover": cover}


def proposals(c: dict | None, s: dict | None) -> list[tuple[str, str, str]]:
    """(file, start of the placeholder, replacement)."""
    out = []
    if c:
        ct = caliper_text(c)
        out += [("sections/results.tex", "held-out RMSE and axis scale errors of the pause", ct["full"]),
                ("sections/discussion.tex", "one-sentence summary of the caliper result", ct["short"]),
                ("sections/discussion.tex", "status", ct["table"]),
                ("sections/conclusion.tex", "caliper status", ct["short"]),
                ("Response_to_Reviewers_TIM.tex", "held-out RMSE and axis scale errors; tilt", ct["full"]),
                ("Response_to_Reviewers_TIM.tex", "one-sentence caliper result", ct["short"])]
    if s:
        st = stop_text(s)
        out += [("sections/results.tex", "error reduction relative to zero motion for the continuous", st["full"]),
                ("sections/abstract.tex", "one-sentence result", st["abstract"]),
                ("sections/discussion.tex", "one-sentence summary: physical gain", st["discussion"]),
                ("sections/discussion.tex", "result", st["table"]),
                ("sections/conclusion.tex", "one-sentence result", st["sentence"]),
                ("Response_to_Reviewers_TIM.tex", "reduction relative to zero motion for continuous", st["response"]),
                ("Cover_Letter_TIM.tex", "one-sentence stop result", st["cover"])]
    return out


# --------------------------------------------------------------------------
# markers in the sources
# --------------------------------------------------------------------------
def markers() -> list[dict]:
    found = []
    for rel in FILES:
        text = (PAPER / rel).read_text(encoding="utf-8")
        for m in re.finditer(r"\\pending\{", text):
            depth, i = 1, m.end()
            while depth:
                depth += {"{": 1, "}": -1}.get(text[i], 0)
                i += 1
            found.append({"file": rel, "line": text[:m.start()].count("\n") + 1,
                          "placeholder": text[m.end():i - 1], "start": m.start(), "end": i})
    return found


def abstract_words(replacement: str | None) -> int:
    text = (PAPER / "sections/abstract.tex").read_text(encoding="utf-8")
    body = text[text.find("\\begin{abstract}") + 16:text.find("\\end{abstract}")]
    if replacement is not None:
        body = re.sub(r"\\pending\{[^{}]*\}", lambda _m: replacement, body)
    body = re.sub(r"\\(rev|emph|textbf)\{", "", body)
    body = re.sub(r"\$[^$]*\$", "x", body)
    body = re.sub(r"\\[a-zA-Z]+", "", body).replace("{", " ").replace("}", " ").replace("~", " ")
    return len([w for w in body.split() if re.search(r"[A-Za-z0-9]", w)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true",
                    help="use the twin rehearsal outputs (checks the wording, not a result)")
    ap.add_argument("--results-dir", type=Path, default=DATA,
                    help="folder with the three result files; the wording file is written there")
    args = ap.parse_args()
    lines_note = ""
    if args.rehearsal:
        cal_root = REH / "caliper_150mm" / "events"
        static = _load(cal_root / "reference_stage_validation.json")
        dynamic = _load(cal_root / "reference_caliper_dynamic.json")
        stop = _load(REH / "stop_chain" / "stop_chain.json")
        out = REH / "pending_fill_REHEARSAL.md"
    else:
        static = _load(args.results_dir / "reference_stage_validation.json")
        dynamic = _load(args.results_dir / "reference_caliper_dynamic.json")
        camera_dyn = _load(args.results_dir / "reference_caliper_dynamic_camera.json")
        if dynamic and camera_dyn:
            dynamic["camera_overall"] = camera_dyn["overall"]
        stop = _load(args.results_dir / "stop_protocol_experiment.json")
        out = args.results_dir / "revision_pending_fill.md"
        if args.results_dir.resolve() != DATA.resolve():
            lines_note = "**Simulated recordings (twin): wording test, not a result.**"
    if static and static.get("kind") != "independent_caliper_validation_no_scale_fit":
        static = None
    power = _load(REH / "stop_power.json")
    c = caliper_values(static, dynamic) if static else None
    s = stop_values(stop, power) if stop else None
    props = proposals(c, s)
    lines = ["# Wording for the TO BE COMPLETED markers", ""]
    if args.rehearsal:
        lines += ["**REHEARSAL: twin data. These sentences only test the wording and must not be pasted.**", ""]
    elif lines_note:
        lines += [lines_note, ""]
    lines += [f"- caliper results: {'found' if c else 'missing'}",
              f"- stop experiment results: {'found' if s else 'missing'}", ""]
    if stop:
        root = args.results_dir if not args.rehearsal else REH / "stop_chain"
        rigs = set()
        for name in stop["sessions"]["continuous"] + stop["sessions"]["paused"]:
            meta = Path(name) / "session_meta.json" if Path(name).is_absolute() else root / name / "session_meta.json"
            if meta.is_file():
                rigs.add(json.loads(meta.read_text(encoding="utf-8")).get("rig", "stereo_endoscope_plus_dual_imu"))
        if "monocular_camera_plus_dual_imu" in rigs:
            lines += ["> The stop recordings were made with the monocular camera, not with the endoscope of the "
                      "dataset. Add to experiments.tex (Controlled stop experiment) and to the response, "
                      "Reviewer 1 (b):", "",
                      "```latex",
                      "The recordings use the monocular camera of the caliper standard with the same USB IMU rigidly "
                      "attached; the camera--IMU rotation of the day comes from the rigidity check, and the lever arm "
                      "learned by the frozen models was not adapted, which affects both conditions alike.",
                      "```", ""]
    if s and (s["nc"] != 2 or s["np"] != 2):
        lines += [f"> The design has {s['nc']} continuous and {s['np']} paused recordings: change \"two recordings\" "
                  "in experiments.tex (Controlled stop experiment) and in the response, Reviewer 1 (b).", ""]
    if s:
        lines += [f"> Recording level: every paused recording above every continuous one: {s['sep']}.", ""]
    for mk in markers():
        prop = next((p for p in props if p[0] == mk["file"] and mk["placeholder"].startswith(p[1])), None)
        lines += [f"## {mk['file']}, line {mk['line']}", "", f"Placeholder: `{mk['placeholder'][:140]}`", ""]
        if prop is None:
            lines += ["(no result yet)", ""]
            continue
        lines += ["```latex", prop[2], "```", ""]
        if mk["file"].endswith("abstract.tex"):
            n = abstract_words(prop[2])
            lines += [f"Abstract length after the replacement: {n} words"
                      + (" (over 250: shorten another sentence)" if n > 250 else ""), ""]
    lines += ["## Response tables: cells marked TBC", ""]
    if s:
        per = stop["per_session"]

        def switched(cond):
            names = stop["sessions"][cond]
            w = sum(per[n]["windows"] for n in names)
            return sum(per[n]["switched_pct"] * per[n]["windows"] for n in names) / max(w, 1)
        lines += ["Table R3 (twin prediction vs. physical), physical columns continuous / stop every 4 s:", "",
                  "```latex",
                  f"Zero-motion error (mm) & ... & {s['zc']:.1f} & {s['zp']:.1f} \\\\",
                  f"PhysNet + switching, $e_p$ (mm) & ... & {s['ec']:.1f} & {s['ep']:.1f} \\\\",
                  f"Reduction relative to zero motion & ... & {_pct(s['rc'], 1)}\\% & {_pct(s['rp'], 1)}\\% \\\\",
                  f"Windows switched to stop anchoring & ... & {switched('continuous'):.1f}\\% & {switched('paused'):.1f}\\% \\\\",
                  f"Gain, paused minus continuous & ... & \\multicolumn{{2}}{{c}}{{{_pct(s['g'], 1)} points, {_ci(s['lo'], s['hi'])}}} \\\\",
                  "```", "",
                  "Table R9 (intended uses), row \"Acquisition with stops\": "
                  f"`controlled experiment: {_pct(s['g'], 0)}-point larger reduction, {_ci(s['lo'], s['hi'], nd=0)}` / "
                  f"`{'yes' if s['g'] > 0 and s['excludes_zero'] else 'not shown'}`", ""]
    if c:
        lines += ["Table R6 (checks of the reference), caliper row, Result cell:", "", "```latex",
                  f"RMSE {c['rmse']:.2f}~mm, scale $\\le${c['smax']:.1f}\\%; labels {c.get('lmed', float('nan')):.2f}~mm median, "
                  f"{_pct(c.get('fU', 0))}\\% within $U_{{\\rm label}}$; off-axis in motion {c.get('mmed', float('nan')):.2f}~mm median",
                  "```", ""]
    plain = lambda t: t.replace("~", " ").replace("\\", "").replace("$", "")  # noqa: E731
    lines += ["## Highlights_TIM.md", ""]
    if c:
        lines += [f"- caliper item: {plain(caliper_text(c)['short'])}"]
    if s:
        lines += [f"- stop item: {plain(stop_text(s)['sentence'])}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    sys.exit(main())
