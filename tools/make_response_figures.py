#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures of the response to the reviewers (paper/response_figures/).

  pause_contrast.pdf    R1-C1(a): real windows, error against displacement,
                        with and without a reference-confirmed pause
  stop_capture_scene.pdf R1-C1(b,c): captured frame with detected corners and
                        synchronized 200-Hz IMU / PnP speed streams (C vs. P)
  stop_experiment.pdf   R1-C1(b): first (handheld) physical stop comparison
  stop_experiment_tripod.pdf  R1-C1(c): second (tripod) comparison at matched speed
  caliper_standard.pdf  R1-C3, R2-C3: vernier-caliper displacement standard
  gate_check.pdf        R1-C6, R3-C1: where the gated and non-gated
                        trajectories differ
  allan_static.pdf      R3-C2: accelerometer Allan deviation of the bench logs,
                        its uncertainty and the largest random walk it allows

The two experiment figures are drawn from the results in --results-dir
(datasets/ for the physical recordings). For any other folder, such as the
virtual capture day, they go to paper/response_figures/preview/ and are not
used by the response.

    python tools/make_response_figures.py
    python tools/make_response_figures.py --results-dir datasets/revision_sim
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets"
OUT = ROOT / "paper" / "response_figures"
CACHE = DATA / "response_figure_cache"
from tools.label_uncertainty import budget_ranges, caliper_thresholds

mpl.rcParams.update({
    "font.family": "Times New Roman", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "mathtext.fontset": "stix",
    "axes.unicode_minus": True,
})
C_CONT, C_PAUSE, C_ZERO = "#4c6ef5", "#e8590c", "#868e96"


def _sgn(x: float, nd: int = 1) -> str:
    return f"{x:+.{nd}f}".replace("-", "\u2212")


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _binned(x, y, edges, blocks=None, n_boot=1000, seed=0):
    """Mean of y per bin of x, with a 95% interval from a bootstrap over blocks."""
    idx = np.digitize(x, edges) - 1
    k = len(edges) - 1
    mean = np.array([y[idx == b].mean() if (idx == b).any() else np.nan for b in range(k)])
    count = np.array([(idx == b).sum() for b in range(k)])
    lo = hi = None
    if blocks is not None:
        rng = np.random.default_rng(seed)
        ids = np.unique(blocks)
        members = {b: np.flatnonzero(blocks == b) for b in ids}
        boots = []
        for _ in range(n_boot):
            pick = np.concatenate([members[b] for b in rng.choice(ids, len(ids))])
            xi, yi = idx[pick], y[pick]
            boots.append([yi[xi == b].mean() if (xi == b).any() else np.nan for b in range(k)])
        lo, hi = np.nanquantile(np.array(boots), (0.025, 0.975), axis=0)
    return mean, count, lo, hi


def _panel(ax, letter, title=""):
    ax.set_title(f"({letter}) {title}".rstrip(), loc="left", fontsize=8)


# --------------------------------------------------------------------------
# R1-C1(a): existing recordings
# --------------------------------------------------------------------------
def pause_windows_cache() -> dict:
    path = CACHE / "pause_windows.npz"
    if path.is_file():
        return dict(np.load(path))
    import tools.train_physnet as phys
    import tools.zupt_bound as zb
    from tools.verify_revision_claims import TEST, _pred
    folds = json.loads((DATA / "session_cv_gate_benchmark.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    for ses in list(folds) + [TEST]:
        t, _R, p, usable = phys._load_gt(ses)
        vg, vok = phys.gt_velocity(t, p, usable)
        speed = np.linalg.norm(vg, axis=1) * 1000.0
        segs = zb.still_segments(t, vok, speed, 3.0, 0.4)
        data = phys.build_group([ses], 2.0, 3.0, 1700)
        pred, y = _pred(ses, folds), data["y"]
        dt_med = float(np.median(np.diff(t[usable])))
        t0 = float(data["t_pair"][:, 0].min())
        for k, (ta, tb) in enumerate(data["t_pair"]):
            near = usable[(t[usable] >= ta - 1.5) & (t[usable] <= tb + 1.5) & vok[usable]]
            rows.append((f"{ses}:{int((ta - t0) // 3.0)}", float(np.linalg.norm(y[k]) * 1000),
                         float(np.linalg.norm(pred[k] - y[k]) * 1000),
                         int((speed[near] < 5.0).sum()) * dt_med,
                         any(t[a] >= ta and t[b] <= tb for a, b in segs)))
    out = {"block": np.array([r[0] for r in rows]), "disp_mm": np.array([r[1] for r in rows]),
           "err_mm": np.array([r[2] for r in rows]), "still_s": np.array([r[3] for r in rows]),
           "pause_inside": np.array([r[4] for r in rows])}
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **out)
    return out


def pause_contrast() -> Path:
    w = pause_windows_cache()
    claims = _load(DATA / "revision_claim_checks.json")["pause_windows"]["matched_contiguous_inside"]
    pause, cont = w["pause_inside"].astype(bool), w["still_s"] < 0.1
    edges = np.arange(10.0, 62.5, 5.0)
    centre = 0.5 * (edges[:-1] + edges[1:])
    fig, ax = plt.subplots(figsize=(4.3, 2.55))
    in_range = (w["disp_mm"] >= edges[0]) & (w["disp_mm"] < edges[-1])
    for sel, colour, name in ((cont, C_CONT, "pause-free windows"), (pause, C_PAUSE, "pause inside $(t_a,t_b]$")):
        m, n, lo, hi = _binned(w["disp_mm"][sel], w["err_mm"][sel], edges, w["block"][sel])
        ok = n >= 10
        ax.fill_between(centre[ok], lo[ok], hi[ok], color=colour, alpha=0.18, lw=0)
        ax.plot(centre[ok], m[ok], "o-", color=colour, ms=3.2, lw=1.2,
                label=f"{name} ({int((sel & in_range).sum())} windows)")
        for c, mm, nn in zip(centre[ok], m[ok], n[ok]):
            if colour == C_PAUSE:
                ax.annotate(str(nn), (c, mm), textcoords="offset points", xytext=(0, -9), ha="center",
                            fontsize=6.3, color=colour)
    ax.plot(edges, edges, "--", color=C_ZERO, lw=0.9, label="zero motion (error = displacement)")
    ax.set_xlabel("true 3-s displacement $\\|\\Delta\\mathbf{p}_{ab}\\|$ (mm)")
    ax.set_ylabel("gated PhysNet $e_p$ (mm)")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(loc="upper left", frameon=False)
    ax.text(0.98, 0.04, f"matched difference {_sgn(claims['diff_mm'])} mm ({_sgn(claims['diff_pct'], 0)}%)\n"
            f"95% CI [{_sgn(claims['ci95_mm'][0])}, {_sgn(claims['ci95_mm'][1])}] mm, "
            f"{claims['n_pause_matched']} pause windows in 2.5-mm bins",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    path = OUT / "pause_contrast.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# R1-C1(b): controlled stop experiment
# --------------------------------------------------------------------------
def stop_experiment(results: Path, out: Path, stem: str = "stop_protocol_experiment",
                    filename: str = "stop_experiment.pdf") -> Path | None:
    """stem: result file of one batch. The first (handheld) batch is
    stop_protocol_experiment, the tripod batch stop_eval_20261109_batch."""
    rep, npz = _load(results / f"{stem}.json"), results / f"{stem}.npz"
    if rep is None or not npz.is_file():
        return None
    w = dict(np.load(npz))
    power = _load(DATA / "revision_rehearsal" / "stop_power.json")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(6.6, 2.6), gridspec_kw={"width_ratios": [0.8, 1.1, 1.0]})
    order = rep["sessions"].get("recording_order") or rep["sessions"]["continuous"] + rep["sessions"]["paused"]
    per = rep["per_session"]
    labels, vals, cols = [], [], []
    for n in order:
        v = per[n]
        lab = (v.get("protocol") or {}).get("label") or ("C" if v["condition"] == "continuous" else "P")
        labels.append(lab)
        vals.append(100 * v["errors"]["physnet_switch"]["reduction_vs_zero"])
        cols.append(C_CONT if v["condition"] == "continuous" else C_PAUSE)
    a.bar(range(len(vals)), vals, color=cols, width=0.65)
    ticks = []
    for i, (v, n, lab) in enumerate(zip(vals, order, labels)):
        comp = (per[n].get("protocol") or {}).get("compliance")
        inside = abs(v) >= 8
        a.text(i, (v - 1.0 if v >= 0 else v + 1.0) if inside else (v + 1.0 if v >= 0 else v - 1.0),
               f"{_sgn(v, 0)}%", ha="center", fontsize=6.6, color="white" if inside else "black",
               va=("top" if v >= 0 else "bottom") if inside else ("bottom" if v >= 0 else "top"))
        ticks.append(lab + ("" if comp is None else f"\n{100 * comp:.0f}%"))
    top = max(vals)
    if power:
        for key, colour, name in (("continuous", C_CONT, "twin prediction, continuous"),
                                  ("paused", C_PAUSE, "twin prediction, paused")):
            level = 100 * power["conditions"][key]["pooled"]["reduction_switch"]
            a.axhline(level, color=colour, ls="--", lw=0.9)
            a.text(len(vals) - 0.6, level + 1.0, name, ha="right", va="bottom", fontsize=6.2, color=colour)
            top = max(top, level)
    a.axhline(0, color="black", lw=0.6)
    a.set_ylim(min(0.0, min(vals)) - 6, top + 14)
    a.set_xticks(range(len(vals)), ticks)
    if any(t.count("\n") for t in ticks):
        a.set_xlabel("beats followed by a stop")
    a.set_ylabel("error reduction vs. zero motion (%)")
    e = rep["estimators"]["physnet_switch"]
    _panel(a, "a", f"gain {_sgn(100 * e['reduction_gain'])} points\n95% CI "
                   f"[{_sgn(100 * e['reduction_gain_ci95'][0])}, {_sgn(100 * e['reduction_gain_ci95'][1])}]")
    edges = np.arange(5.0, 85.0, 5.0)
    centre = 0.5 * (edges[:-1] + edges[1:])
    blocks = np.array([f"{s}:{int(t // 3)}" for s, t in zip(w["session"], w["t_a"] - w["t_a"].min())])
    for cond, colour in (("continuous", C_CONT), ("paused", C_PAUSE)):
        sel = w["condition"] == cond
        m, n, lo, hi = _binned(w["disp_mm"][sel], w["err_physnet_switch_mm"][sel], edges, blocks[sel])
        ok = n >= 15
        b.fill_between(centre[ok], lo[ok], hi[ok], color=colour, alpha=0.18, lw=0)
        b.plot(centre[ok], m[ok], "o-", color=colour, ms=3, lw=1.2, label=f"{cond} ({int(sel.sum())} windows)")
    b.plot(edges, edges, "--", color=C_ZERO, lw=0.9, label="zero motion")
    b.set_xlabel("true 3-s displacement (mm)")
    b.set_ylabel("gated PhysNet + switching $e_p$ (mm)")
    b_title = (f"matched difference {_sgn(e['matched_diff_mm'])} mm\n95% CI "
               f"[{_sgn(e['matched_diff_ci95_mm'][0])}, {_sgn(e['matched_diff_ci95_mm'][1])}]")
    b.set_xlim(edges[0], edges[-1])
    b.set_ylim(0, None)
    b.grid(alpha=0.25, lw=0.5)
    b.legend(loc="upper left", frameon=False)
    _panel(b, "b", b_title)

    import tools.stop_rotation_diagnostic as srd
    srd.DATA = results
    paused = w["condition"] == "paused"
    switched = paused & (np.abs(w["err_physnet_switch_mm"] - w["err_physnet_mm"]) > 1e-3)
    rot = np.full(len(paused), np.nan)
    rot[paused] = srd.window_rotation(w["session"][paused], w["t_a"][paused], 3.0, 0.15)
    sel = switched & np.isfinite(rot)
    r_edges = np.array([0.0, 3.0, 8.0, 15.0, 45.0])
    x = np.arange(len(r_edges) - 1)
    for k, (key, colour, name) in enumerate((("err_physnet_switch_mm", C_PAUSE, "stop-anchored"),
                                             ("err_physnet_mm", C_CONT, "gated PhysNet"),
                                             ("err_zero_mm", C_ZERO, "zero motion"))):
        m, n, lo, hi = _binned(rot[sel], w[key][sel], r_edges, blocks[sel])
        c.errorbar(x + (k - 1) * 0.22, m, yerr=[m - lo, hi - m], fmt="o", ms=3.2, color=colour,
                   capsize=1.5, lw=1.0, label=name)
    c.set_xticks(x, [f"{lo:.0f}\u2013{hi:.0f}\u00b0\n{int(k)}" for lo, hi, k in
                     zip(r_edges[:-1], r_edges[1:], np.histogram(rot[sel], r_edges)[0])])
    c.set_xlabel("rotation in window; windows")
    c.set_ylabel("$e_p$ (mm), paused, switched")
    c.set_ylim(0, None)
    c.grid(alpha=0.25, lw=0.5, axis="y")
    c.legend(loc="upper left", frameon=False)
    _panel(c, "c", "switched windows\nby camera rotation")
    fig.tight_layout(w_pad=1.0)
    path = out / filename
    fig.savefig(path)
    plt.close(fig)
    return path


def stop_experiment_tripod(results: Path, out: Path) -> Path | None:
    """R1-C1(c): the second batch, recorded on a tripod at matched speed."""
    return stop_experiment(results, out, "stop_eval_20261109_batch", "stop_experiment_tripod.pdf")


# --------------------------------------------------------------------------
# R1-C3, R2-C3: caliper standard
# --------------------------------------------------------------------------
def _caliper_source(results: Path) -> tuple[Path, Path]:
    """(folder with the caliper sessions and labels, plan CSV) behind the
    caliper result in `results`. The final physical result is the one-direction
    analysis on the self-calibrated labels, named in the capture plan; its
    sessions and plan are not in datasets/ itself."""
    import tools.reference_stage_validation as rsv
    final = (_load(results / "revision_experiment_plan.json") or {}).get("caliper_final")
    if final:
        root = (ROOT / final["source"]).parent
        return root, root / "reference_caliper_plan_onedir.csv"
    return results, results / rsv.CALIPER_PLAN.name


def _caliper_holds(results: Path, key: str = "board_mm"):
    """Held-out along-axis deviation of every hold, as scored by
    reference_stage_validation.py (direction from the other trials of the axis,
    origin on the trial's own first reading, no scale fit). key: 'board_mm'
    (board centre in camera coordinates) or 'median_mm' (camera centre)."""
    import tools.reference_stage_validation as rsv
    rsv.DATA, plan_csv = _caliper_source(results)
    if not plan_csv.is_file():
        return None
    rows = list(csv.DictReader(plan_csv.open(newline="", encoding="utf-8-sig")))
    if not all(r["session"].strip() for r in rows):
        return None
    trials = {}
    for r in rows:
        tr = rsv.caliper_trial(r["session"].strip(), np.array([float(v) for v in r["readings_mm"].split()]), 0.2, 1.5)
        trials.setdefault(r["axis"], []).append((int(r["trial"]), tr["readings"],
                                                 np.array([s[key] for s in tr["segs"]]),
                                                 "fast" in (r.get("notes") or "")))
    out = {}
    for axis, ts in sorted(trials.items()):
        pts = []
        for k, (trial, s, m, fast) in enumerate(ts):
            others = [o for j, o in enumerate(ts) if j != k]
            u, _o, _sc = rsv._line_fit(np.concatenate([o[1] for o in others]),
                                       np.concatenate([o[2] - o[2][np.argmin(o[1])] for o in others]))
            i0 = int(np.argmin(s))
            along = (m - m[i0]) @ u
            for si, ai in zip(s, along):
                pts.append((trial, si - s[i0], ai - (si - s[i0]), float(fast)))
        out[axis] = np.array(pts)
    return out


def caliper_standard(results: Path, out: Path) -> Path | None:
    static = _load(results / "reference_stage_validation.json")
    dyn, npz = _load(results / "reference_caliper_dynamic.json"), results / "reference_caliper_dynamic.npz"
    if static is None or dyn is None or not npz.is_file():
        return None
    board = "camera_centre" in static["summary"]
    holds = _caliper_holds(results, "board_mm" if board else "median_mm")
    if holds is None:
        return None
    d = dict(np.load(npz))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.45), gridspec_kw={"width_ratios": [1.4, 1.0, 0.95]})
    colours = {"X": "#1971c2", "Y": "#2f9e44", "Z": "#c2255c"}
    axes_s = static["summary"]["axes"]
    for axis, pts in holds.items():
        trials = sorted({int(v) for v in pts[:, 0]})
        for k, trial in enumerate(trials):
            sel = pts[:, 0] == trial
            a.plot(pts[sel, 1], pts[sel, 2], "os^Dv<>p"[k % 8], color=colours[axis], ms=3.2,
                   mfc="none" if pts[sel, 3].any() else colours[axis], lw=0,
                   label=f"{axis} {_sgn(axes_s[axis]['scale_error_percent'], 2)}%" if k == 0 else None)
    a.axhline(0, color="black", lw=0.6)
    a.set_xlabel("caliper displacement (mm)")
    a.set_ylabel("PnP minus caliper, along axis (mm)")
    s = static["summary"]
    a_title = f"held-out RMSE {s['heldout_rmse_mm']:.2f} mm, max {s['heldout_max_mm']:.2f} mm"
    lo_y = min(float(np.min(p[:, 2])) for p in holds.values())
    hi_y = max(float(np.max(p[:, 2])) for p in holds.values())
    a.set_ylim(lo_y - 0.1 * (hi_y - lo_y), hi_y + 0.45 * (hi_y - lo_y))
    a.legend(loc="upper center", ncol=3, frameon=False, fontsize=6.4, handletextpad=0.1, columnspacing=0.6,
             title="direction, scale error (one marker per pass, open: fast moves)", title_fontsize=6.2)
    a.grid(alpha=0.25, lw=0.5)
    _panel(a, "a", a_title)
    what = "displacement" if dyn.get("quantity") == "board" else "label"
    err = np.sort(d["label_err_mm"])
    b.plot(err, np.arange(1, len(err) + 1) / len(err), color="black", lw=1.2,
           label="board translation" if what == "displacement" else "label error")
    cam_npz = results / "reference_caliper_dynamic_camera.npz"
    if what == "displacement" and cam_npz.is_file():
        e2 = np.sort(np.load(cam_npz)["label_err_mm"])
        b.plot(e2, np.arange(1, len(e2) + 1) / len(e2), color="#868e96", lw=1.1, ls="--",
               label="camera centre (with jaw tilt)")
        err = np.concatenate([err, e2])
    for u, name, ls in zip(caliper_thresholds(), ("nominal", "perturbed"), ("--", ":")):
        b.axvline(u, color="#e03131", ls=ls, lw=1.0, label=f"$U_{{\\rm label}}$ {name}, {u:.2f} mm")
    b.legend(frameon=False, loc="lower right", fontsize=6.6)
    b.set_xlim(0, max(4.0, float(np.quantile(err, 0.995))))
    b.set_xlabel(f"{what} error, single frames of two holds (mm)")
    b.set_ylabel("cumulative fraction")
    o = dyn["overall"]["label_error"]
    b.grid(alpha=0.25, lw=0.5)
    _panel(b, "b", f"median {o['median_mm']:.2f} mm, p95 {o['p95_mm']:.2f} mm")
    speed, perp = d["motion_speed_mm_s"], d["motion_perp_mm"]
    edges = np.array([0.0, 2.0, 5.0, 10.0, 15.0, 25.0, 40.0, 1e9])
    idx = np.digitize(speed, edges) - 1
    data = [perp[idx == k] for k in range(len(edges) - 1)]
    keep = [k for k, v in enumerate(data) if len(v) >= 5]
    c.boxplot([data[k] for k in keep], positions=range(len(keep)), widths=0.55, showfliers=False,
              medianprops={"color": "black"})
    c.set_xticks(range(len(keep)), [f"{edges[k]:g}-{edges[k + 1]:g}" if edges[k + 1] < 1e8 else f">{edges[k]:g}"
                                    for k in keep], rotation=30)
    c.set_xlabel("board speed (mm/s)")
    c.set_ylabel("distance from caliper axis (mm)")
    c.grid(alpha=0.25, lw=0.5, axis="y")
    _panel(c, "c", "frames recorded while moving")
    fig.tight_layout(w_pad=1.4)
    path = out / "caliper_standard.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def board_imu_motion(results: Path, out: Path) -> Path | None:
    """R1-C3, R2-C3: PnP orientation in motion against the IMU on the board
    (tools/board_imu_motion_check.py)."""
    rep, npz = _load(results / "board_imu_motion_check.json"), results / "board_imu_motion_check.npz"
    if rep is None or not npz.is_file():
        return None
    d = dict(np.load(npz))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.2, 2.4), gridspec_kw={"width_ratios": [1.0, 1.0, 1.2]})
    lim = float(np.quantile(np.r_[d["rot_pnp_deg"], d["rot_gyro_deg"]], 0.995)) * 1.05
    a.plot(d["rot_gyro_deg"], d["rot_pnp_deg"], ".", ms=1.4, color="#495057", alpha=0.45, rasterized=True)
    a.plot([0, lim], [0, lim], color="#e03131", lw=0.8)
    a.set_xlim(0, lim)
    a.set_ylim(0, lim)
    a.set_xlabel("gyroscope, 1-s rotation (deg)")
    a.set_ylabel("PnP, 1-s rotation (deg)")
    r = rep["rotation_1s"]
    _panel(a, "a", f"axis r \u2265 {r['axis_corr_min']:.3f}, scale {r['scale_range'][0]:.2f}\u2013{r['scale_range'][1]:.2f}")
    a.grid(alpha=0.25, lw=0.5)
    edges = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0])
    for L, col in ((1.0, "#2f9e44"), (3.0, "#c2255c")):
        sel = d["window_s"] == L
        path, diff = d["window_path_deg"][sel], d["window_diff_deg"][sel]
        x, y, lo, hi = [], [], [], []
        for k in range(len(edges) - 1):
            m = (path >= edges[k]) & (path < edges[k + 1])
            if m.sum() >= 20:
                x.append(np.median(path[m]))
                y.append(np.median(diff[m]))
                lo.append(np.quantile(diff[m], 0.25))
                hi.append(np.quantile(diff[m], 0.75))
        y = np.array(y)
        b.errorbar(x, y, yerr=[y - np.array(lo), np.array(hi) - y], fmt="o", ms=3, color=col, lw=0.8,
                   capsize=1.5, label=f"{L:g}-s windows")
    dec = rep["decomposition"]
    xs = np.linspace(0.3, edges[-1], 200)
    b.plot(xs, np.sqrt(dec["endpoint_deg"] ** 2 + (dec["per_deg_of_path"] * xs) ** 2), color="black", lw=0.8,
           label=f"{dec['endpoint_deg']:.2f}\u00b0 at the endpoints + {100 * dec['per_deg_of_path']:.1f}% of path")
    b.axhline(dec["budget_relative_deg"], color="#e03131", ls="--", lw=0.8, label=r"budget $\sqrt{2}\,\sigma_\theta$")
    b.set_xscale("log")
    b.set_xlim(0.3, edges[-1])
    b.set_xlabel("rotation integrated by the gyroscope (deg)")
    b.set_ylabel("PnP minus gyroscope (deg)")
    b.legend(frameon=False, fontsize=6.4, loc="upper left")
    b.grid(alpha=0.25, lw=0.5, which="both")
    _panel(b, "b", "relative rotation, median and IQR")
    delays = d["delay_ms"]
    c.plot(np.arange(len(delays)), delays, "o", ms=4, color="#1971c2")
    c.axhline(np.median(delays), color="black", lw=0.6)
    c.set_xticks(np.arange(len(delays)), [f"{k + 1}" for k in range(len(delays))])
    c.set_xlabel("recording")
    c.set_ylabel("camera stamps late by (ms)")
    c.set_ylim(np.median(delays) - 20, np.median(delays) + 12)
    c.grid(alpha=0.25, lw=0.5)
    _panel(c, "c", f"delay per recording, median {np.median(delays):.0f} ms")
    fig.tight_layout(w_pad=1.2)
    path = out / "board_imu_motion.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# R1-C6, R3-C1: stillness gate; R3-C2: Allan deviation
# --------------------------------------------------------------------------
def gate_check() -> Path:
    g = _load(DATA / "gate_trajectory_check.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.6, 2.3), gridspec_kw={"width_ratios": [1.0, 1.35]})
    ax_names = ("X", "Y", "Z (board normal)")
    x = np.arange(3)
    a.bar(x - 0.18, g["per_axis_ate_mm_xyz"]["cvx"], 0.36, color=C_CONT, label="non-gated")
    a.bar(x + 0.18, g["per_axis_ate_mm_xyz"]["cvg"], 0.36, color=C_PAUSE, label="gated")
    for i in range(3):
        for dx, v in ((-0.18, g["per_axis_ate_mm_xyz"]["cvx"][i]), (0.18, g["per_axis_ate_mm_xyz"]["cvg"][i])):
            a.text(i + dx, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=6.3)
    a.set_xticks(x, ax_names)
    a.set_ylabel("trajectory RMSE per axis (mm)")
    a.legend(frameon=False, loc="upper left")
    a.set_ylim(0, 125)
    _panel(a, "a", "trajectory error per axis")
    blocks = g["board_z_residual_by_15s"]
    mid = np.array([0.5 * sum(r["interval_s"]) for r in blocks])
    diff = np.array([r["board_z_residual_cvg_mm"] - r["board_z_residual_cvx_mm"] for r in blocks])
    p = np.array([r["mean_p_still"] for r in blocks])
    b.bar(mid, diff, width=11, color="#adb5bd", edgecolor="#495057", lw=0.6)
    b.axhline(0, color="black", lw=0.6)
    b.set_xlabel("time in the held-out recording (s)")
    b.set_ylabel("$Z$ residual per window,\ngated $-$ non-gated (mm)")
    b2 = b.twinx()
    b2.plot(mid, p, "k.-", lw=0.9, ms=4)
    b2.set_ylabel("mean stillness probability (line)")
    b2.set_ylim(0, 1)
    _panel(b, "b", "15-s blocks of the held-out recording")
    fig.tight_layout(w_pad=1.4)
    path = OUT / "gate_check.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def gate_trajectory_trace() -> Path:
    """R3-C1: board-normal trajectory Z(t), offset between the gated and the
    non-gated trajectory, and stillness-gate probability over the recording."""
    import torch
    import tools.train_physnet as phys
    from tools.build_trajectory_comparison import _components, _solve_positions

    ses = "traj_20260923_023422"
    t_gt, Rg, pg, usable = phys._load_gt(ses)
    z_ng = np.load(DATA / "session_cv_benchmark.npz", allow_pickle=True)
    z_g = np.load(DATA / "session_cv_gate_benchmark.npz", allow_pickle=True)
    pairs = z_ng[f"{ses}__pair"]
    t_pair = z_ng[f"{ses}__t_pair"]
    pred_ng = z_ng[f"{ses}__physnet"].astype(np.float64)
    pred_g = z_g[f"{ses}__physnet"].astype(np.float64)

    node = {int(f): i for i, f in enumerate(usable)}
    edges = np.array([(node[int(a)], node[int(b)]) for a, b in pairs], np.int64)
    traj_ng = _solve_positions(pg[usable], Rg[usable], edges, pred_ng)
    traj_g = _solve_positions(pg[usable], Rg[usable], edges, pred_g)

    t_nodes = t_gt[usable] - t_gt[usable][0]
    z_ref = (pg[usable, 2] - pg[usable][0, 2]) * 1000.0
    z_est_ng = (traj_ng[:, 2] - pg[usable][0, 2]) * 1000.0
    z_est_g = (traj_g[:, 2] - pg[usable][0, 2]) * 1000.0

    # The graph has one anchor per connected component. The components anchored
    # in the initial rest carry the error accumulated since then; the two that
    # start after the board dropout restart from their reference position.
    # Drawn as one line in time order, the two groups would alternate.
    acc = _load(DATA / "gate_trajectory_check.json")
    graph = acc["offset_accounting"]["graph"]
    first = np.zeros(len(usable), bool)
    later = np.zeros(len(usable), bool)
    for nodes in _components(len(usable), edges):
        (first if t_nodes[nodes[0]] < 10.0 else later)[nodes] = True
    if (int(first.sum()), int(later.sum())) != (graph["anchored_in_initial_rest"]["nodes"],
                                                graph["anchored_after_dropout"]["nodes"]):
        raise AssertionError("components differ from datasets/gate_trajectory_check.json")
    rmse = acc["per_axis_ate_mm_xyz"]
    t_re = graph["anchored_after_dropout"]["anchor_s"][0]

    # Compute continuous per-window stillness probability from the 16-model cvg ensemble
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = phys.build_group([ses], 2.0, 3.0, int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ)))
    data_t = phys.to_device(data, device)
    split = _load(DATA / "trajectory_split_20260924.json")
    folds = split["train"] + split["val"]
    models_g = [phys.load_model(DATA / "physnet_cv" / f"physnet_cvg_{s}_s{k}.pt", device)
                for s in folds for k in (0, 1)]
    still = np.mean([m.predict_streams(data_t)["still"] for m in models_g], axis=0)
    p_win = np.zeros(len(data["y"]))
    for k in range(len(p_win)):
        dv = data["dense_valid"][k] > 0
        tau = data["dense_tau"][k][dv]
        sel = (tau >= int(round(2.0 * phys.HZ))) & (tau <= data["tau_b"][k])
        p_win[k] = still[k][dv][sel].mean() if sel.any() else 0.0
    t_win_mid = 0.5 * (t_pair[:, 0] + t_pair[:, 1]) - t_gt[usable][0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.6, 2.95), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1.0], "hspace": 0.28})
    c_ng, c_g = "#0b7285", "#c92a2a"
    for ax in (ax1, ax2):
        ax.axvspan(0, 25, color="#228be6", alpha=0.30, lw=0,
                   label="all gate-active windows start here (0–25 s)" if ax is ax1 else None)
        ax.axvspan(25, 60, color="#fd7e14", alpha=0.26, lw=0,
                   label="offset builds up, gate inactive (25–60 s)" if ax is ax1 else None)
        ax.axvline(25, color="#1864ab", ls="--", lw=0.8, alpha=0.9)
        ax.axvline(60, color="#d9480f", ls="--", lw=0.8, alpha=0.9)
        ax.axvline(t_re, color="#495057", ls=":", lw=0.8)
    ax1.plot(t_nodes, z_ref, color="#212529", lw=1.2, label="Reference $Z(t)$")
    ax1.plot(t_nodes[first], z_est_ng[first], color=c_ng, lw=1.2,
             label=f"Non-gated PhysNet ($Z$ RMSE {rmse['cvx'][2]:.1f} mm)")
    ax1.plot(t_nodes[first], z_est_g[first], color=c_g, lw=1.2, ls="--",
             label=f"Gated PhysNet ($Z$ RMSE {rmse['cvg'][2]:.1f} mm)")
    ax1.plot(t_nodes[later], z_est_ng[later], ".", color=c_ng, ms=2.2, mew=0)
    ax1.plot(t_nodes[later], z_est_g[later], ".", color=c_g, ms=2.2, mew=0)
    ax1.plot([], [], ".", color="#495057", ms=4, mew=0,
             label=f"components re-anchored at {t_re:.1f} s ({int(later.sum())} nodes)")
    ax1.set_ylabel("board-normal $\\Delta Z$ (mm)")
    ax1.set_ylim(-400, 62)
    ax1.set_yticks([-200, -100, 0])
    ax1.grid(alpha=0.25, lw=0.45)
    ax1.legend(loc="lower left", ncol=2, fontsize=6.4, columnspacing=0.9, labelspacing=0.25,
               frameon=True, framealpha=0.8, facecolor="white", edgecolor="none")
    _panel(ax1, "a", "held-out board-normal trajectory Z(t): neither estimate follows the excursion")

    ax2.plot(t_win_mid, p_win, color="#2b8a3e", lw=1.0, label="Window stillness prob. $\\bar{p}_{\\rm still}$")
    ax2.axhline(0.10, color="#2b8a3e", ls="--", lw=0.75, label="Active threshold (0.10)")
    ax2.set_ylabel("gate prob. $\\bar{p}_{\\rm still}$", color="#2b8a3e")
    ax2.set_ylim(-0.2, 1.9)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set_xlabel("time from recording start (s)")
    ax2.grid(alpha=0.25, lw=0.45)

    ax2r = ax2.twinx()
    offset = z_est_g - z_est_ng
    ax2r.plot(t_nodes[first], offset[first], color=c_g, lw=0.9, alpha=0.9,
              label="$Z$ offset, gated $-$ non-gated (mm)")
    ax2r.plot(t_nodes[later], offset[later], ".", color=c_g, ms=2.2, mew=0, label="same, re-anchored components")
    ax2r.axhline(0, color="#868e96", lw=0.5)
    ax2r.set_ylabel("$Z$ offset (mm)", color=c_g)
    ax2r.set_ylim(-14, 95)
    ax2r.set_yticks([0, 20, 40, 60])

    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2r.get_legend_handles_labels()
    ax2r.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=6.3, columnspacing=0.8, labelspacing=0.25,
                frameon=True, framealpha=0.8, facecolor="white", edgecolor="none")
    _panel(ax2, "b", "gate activation and offset between the two trajectories")
    ax2.set_xlim(0, float(t_nodes[-1]))

    path = OUT / "gate_trajectory_trace.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def oracle_2x2_r2() -> Path:
    """R2-C2: Visual comparison of the 2x2 initial-velocity and demeaning test
    across both pause-free and 4-s-stop digital-twin corpora."""
    rev = _load(DATA / "reviewer_revision_experiments.json")["exp2_oracle_simulation"]
    keys = ["demeaned_no_v0", "no_demean_no_v0", "demeaned_with_v0", "no_demean_with_v0"]
    xlabels = ["Demeaned,\nno $v(t_a)$\n(standard)",
               "Exact grav.,\nno $v(t_a)$",
               "Demeaned,\ntrue $v(t_a)$",
               "Exact grav.,\ntrue $v(t_a)$"]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.45), sharey=True)
    x = np.arange(4)
    for ax, (ckey, letter, title) in zip(
        axes,
        (("pause_inf", "a", "pause-free handheld corpus (zero motion 57.4 mm)"),
         ("pause_4", "b", "4-s-stop handheld corpus (zero motion 56.8 mm)"))
    ):
        corp = rev[ckey]
        ol = [corp[k]["open_loop_mm"] for k in keys]
        ln = [corp[k]["learned_ens_mm"] for k in keys]
        sl = [np.mean(corp[k]["slopes"]) for k in keys]
        b1 = ax.bar(x - 0.19, ol, 0.36, color="#ced4da", edgecolor="#495057", lw=0.6,
                    label="Open-loop integration")
        b2 = ax.bar(x + 0.19, ln, 0.36, color=C_CONT, label="Retrained PhysNet (2 seeds)")
        ax.axhline(corp["zero_mm"], color="#e03131", ls="--", lw=0.8, label="Zero-motion baseline")
        for i, (vo, vl, s) in enumerate(zip(ol, ln, sl)):
            ax.text(i - 0.19, vo + 3.0, f"{vo:.1f}", ha="center", va="bottom", fontsize=5.8, color="#495057")
            ax.text(i + 0.19, vl + 3.0, f"{vl:.1f}\n(s={s:.2f})", ha="center", va="bottom",
                    fontsize=5.7, color="#1864ab")
        ax.set_xticks(x, xlabels, fontsize=6.1)
        ax.set_ylim(0, 228)
        ax.grid(axis="y", alpha=0.25, lw=0.45)
        _panel(ax, letter, title)
    axes[0].set_ylabel("3-s displacement error $e_p$ (mm)")
    axes[0].legend(loc="upper right", frameon=False, fontsize=6.2)
    fig.tight_layout(w_pad=1.2)
    path = OUT / "oracle_2x2_r2.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def reference_validation_r2() -> Path:
    """R2-C3: Compact 3-panel summary of propagated label uncertainty U_label,
    vernier-caliper adjacent-move errors, and dynamic moving-board IMU check."""
    cal_b = _load(DATA / "reference_caliper_dynamic.json")["overall"]
    cal_c = _load(DATA / "reference_caliper_dynamic_camera.json")["overall"]
    b_imu = _load(DATA / "board_imu_motion_check.json")

    fig, (a, b, c) = plt.subplots(1, 3, figsize=(7.0, 2.35),
                                  gridspec_kw={"width_ratios": [1.05, 1.05, 1.0]})
    # Panel (a): Propagated 3-s label uncertainty components (validation/test medians)
    terms = ["Pose\n$u_{\\rm end}$", "Timing\n$u_{\\rm time}$", "Scale\n$u_{\\rm cal}$",
             "Expanded\n$U_{\\rm label}$ ($k_c{=}2$)"]
    # Validation--test ranges of the medians as stated in Manuscript Section V-B
    # (bars reach the upper end; the labels give the full range).
    ranges = budget_ranges()
    nom, pert = ([tuple(round(x, 2) for x in ranges[key])
                  for key in ("end", "time", "cal_" + case, "U_" + case)]
                 for case in ("nominal", "perturbed"))
    xt = np.arange(len(terms))
    a.bar(xt - 0.18, [hi for _lo, hi in nom], 0.34, color=C_CONT, label="Nominal intrinsics")
    a.bar(xt + 0.18, [hi for _lo, hi in pert], 0.34, color=C_PAUSE, label="Perturbed intrinsics")

    def _rng(lo, hi):
        return f"{hi:.2f}" if lo == hi else f"{lo:.2f}\u2013\n{hi:.2f}"

    for i, (vn, vp) in enumerate(zip(nom, pert)):
        a.text(i - 0.18, vn[1] + 0.06, _rng(*vn), ha="center", va="bottom", fontsize=5.2, linespacing=0.9)
        a.text(i + 0.18, vp[1] + 0.06, _rng(*vp), ha="center", va="bottom", fontsize=5.2, linespacing=0.9)
    a.set_xticks(xt, terms, fontsize=6.0)
    a.set_ylim(0, 4.9)
    a.set_ylabel("3-s uncertainty (mm)")
    a.legend(loc="upper left", frameon=False, fontsize=6.0)
    a.grid(axis="y", alpha=0.25, lw=0.45)
    _panel(a, "a", "propagated 3-s label budget")

    # Panel (b): Caliper start-to-end move errors vs U_label
    cats = ["Adjacent moves\n(board, n=64)", "Adjacent moves\n(cam label, n=64)", "All hold pairs\n(board, n=288)"]
    meds = [cal_b["adjacent_moves"]["median_mm"],
            cal_c["adjacent_moves"]["median_mm"],
            cal_b["hold_pair_displacement"]["median_mm"]]
    p95s = [cal_b["adjacent_moves"]["p95_mm"],
            cal_c["adjacent_moves"]["p95_mm"],
            cal_b["hold_pair_displacement"]["p95_mm"]]
    xc = np.arange(len(cats))
    b.bar(xc - 0.18, meds, 0.34, color="#2f9e44", label="Median error")
    b.bar(xc + 0.18, p95s, 0.34, color="#74c0fc", edgecolor="#1971c2", lw=0.6, label="95th percentile")
    threshold = caliper_thresholds()[0]
    b.axhline(threshold, color="#e03131", ls="--", lw=0.8, label=f"$U_{{\\rm label}}$ nominal ({threshold:.2f} mm)")
    for i, (vm, vp) in enumerate(zip(meds, p95s)):
        b.text(i - 0.18, vm + 0.06, f"{vm:.2f}", ha="center", va="bottom", fontsize=5.7)
        b.text(i + 0.18, vp + 0.06, f"{vp:.2f}", ha="center", va="bottom", fontsize=5.7)
    b.set_xticks(xc, cats, fontsize=5.8)
    b.set_ylim(0, 4.3)
    b.set_ylabel("displacement error vs. caliper (mm)")
    b.legend(loc="upper left", frameon=False, fontsize=5.9)
    b.grid(axis="y", alpha=0.25, lw=0.45)
    _panel(b, "b", "caliper check (one direction)")

    # Panel (c): Moving-board PnP vs gyroscope orientation error
    w3 = b_imu["windows"]["3 s"]
    rcats = [f"3-s windows,\nboth ends at rest\n(n={w3['rest_to_rest_n']})", f"All 3-s\nwindows\n(n={w3['n']})",
             "Label budget\n$\\sqrt{2}\\,\\sigma_\\theta$"]
    rvals = [w3["rest_to_rest_median_deg"], w3["difference_median_deg"], b_imu["decomposition"]["budget_relative_deg"]]
    rcols = ["#2f9e44", C_CONT, "#e03131"]
    xr = np.arange(len(rcats))
    c.bar(xr, rvals, 0.46, color=rcols, alpha=0.88)
    for i, vr in enumerate(rvals):
        c.text(i, vr + 0.015, f"{vr:.2f}°", ha="center", va="bottom", fontsize=6.1)
    c.set_xticks(xr, rcats, fontsize=6.0)
    c.set_ylim(0, 0.70)
    c.set_ylabel("relative rotation diff. (deg)")
    sc = b_imu["rotation_1s"]["scale_range"]
    c.text(0.04, 0.93, f"1-s per-axis $r \\geq {b_imu['rotation_1s']['axis_corr_min']:.3f}$\n"
                       f"scale {sc[0]:.2f}–{sc[1]:.2f} (n={b_imu['rotation_1s']['test_windows']})",
           transform=c.transAxes, va="top", fontsize=6.0,
           bbox={"boxstyle": "round,pad=0.2", "fc": "#f8f9fa", "ec": "#dee2e6", "lw": 0.5})
    c.grid(axis="y", alpha=0.25, lw=0.45)
    _panel(c, "c", "PnP vs. board IMU in motion")

    fig.tight_layout(w_pad=1.1)
    path = OUT / "reference_validation_r2.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def latency_tradeoff() -> Path:
    """R2-C4: online output delay and accuracy across context lengths.

    Source: datasets/context_sweep_common.json (tools/run_context_sweep_common.py):
    three two-seed trainings per context, every context scored on the windows
    admissible at c = 2 s; c = 0 is strictly causal (last sample before t_b).
    Delays are wait + CPU gridding + twice the single-model GPU time, because the
    retrained ensembles have two models; the frozen 16-model ensemble exists and
    was timed only at c = 2 s.
    """
    d = _load(DATA / "context_sweep_common.json")
    ens16 = _load(DATA / "reviewer_audit_20261108" / "ensemble_latency_repeated.json")["median_ms"]["physnet_ensemble16_ms"]
    cs = [f"{c:g}" for c in d["contexts_s"]]
    first = d["repeats"][0]
    wait = np.array([first[c]["future_wait_ms"] for c in cs])
    comp = np.array([first[c]["cpu_gridding_ms"] + 2.0 * first[c]["gpu_single_model_b1_ms"] for c in cs])
    tot2 = wait + comp
    tot16 = wait[-1] + first[cs[-1]]["cpu_gridding_ms"] + ens16

    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.05, 1.1]})
    x = np.arange(len(cs))
    a.bar(x, wait, 0.55, color="#a5d8ff", edgecolor="#1971c2", lw=0.6, label="wait for samples after $t_b$")
    a.bar(x, comp, 0.55, bottom=wait, color="#2f9e44", label="CPU gridding + GPU (two models)")
    for i, t2 in enumerate(tot2):
        a.text(i, t2 + 45, f"{t2:.1f} ms" if t2 < 100 else f"{t2 / 1000:.3f} s", ha="center", va="bottom", fontsize=5.6)
    a.plot([x[-1] + 0.4], [tot16], marker="D", ms=4, color="#c92a2a", ls="none",
           label=f"frozen 16-model ensemble, measured ({tot16 / 1000:.3f} s)")
    labels = ["0\n(causal)", "0.05", "0.5", "1.0", "2.0\n(selected)"]
    a.set_xticks(x, labels, fontsize=6.2)
    a.set_xlim(-0.55, len(cs) - 0.2)
    a.set_ylim(0, 2750)
    a.set_xlabel("context $c$ on each side (s)")
    a.set_ylabel("output delay after $t_b$ (ms)")
    a.legend(loc="upper left", frameon=False, fontsize=5.8)
    a.grid(axis="y", alpha=0.25, lw=0.45)
    _panel(a, "a", "online single-window output delay")

    val = np.array([[rep[c]["common"]["val_err_mm"] for c in cs] for rep in d["repeats"]])
    tst = np.array([[rep[c]["common"]["test_err_mm"] for c in cs] for rep in d["repeats"]])
    for arr, colour, marker, name in ((val, C_CONT, "o", "Validation"), (tst, "#d9480f", "s", "Held-out test")):
        mid = arr.mean(0)
        b.errorbar(x, mid, yerr=[mid - arr.min(0), arr.max(0) - mid], fmt=marker + "-", color=colour, lw=1.0, ms=3.6,
                   capsize=2.5, label=f"{name} $e_p$: mean and range of 3 trainings")
        for xi, lo, hi in zip(x, arr.min(0), arr.max(0)):
            b.text(xi, hi + 0.55, f"{lo:.1f}\u2013{hi:.1f}", ha="center", va="bottom", fontsize=5.5, color=colour)
    b.set_xticks(x, labels, fontsize=6.2)
    b.set_xlim(-0.5, len(cs) - 0.5)
    b.set_ylim(26.5, 50.5)
    b.set_xlabel("context $c$ on each side (s)")
    b.set_ylabel("3-s displacement error $e_p$ (mm)")
    b.legend(loc="center right", frameon=False, fontsize=5.8, bbox_to_anchor=(1.0, 0.42))
    b.grid(alpha=0.25, lw=0.45)
    n = d["common_windows"]
    _panel(b, "b", f"accuracy on common windows ({n['val']} val, {n['test']} test)")

    fig.tight_layout(w_pad=1.2)
    path = OUT / "latency_tradeoff.pdf"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def allan_static() -> Path:
    """R3-C2: measured accelerometer Allan deviation of the static logs, the
    uncertainty of its last points, and the largest random walk it allows.

    The two red lines are not fitted to the data and are not a trend of it: a
    rate random walk K contributes K sqrt(tau/3), and K is the value at which
    that contribution alone would reach the largest measured deviation at the
    longest averaging time (with and without one sigma of that estimate).
    """
    d = _load(DATA / "allan_static_imu.json")
    bound = _load(DATA / "allan_static_bound.json")
    fig, ax = plt.subplots(figsize=(4.9, 3.0))
    styles = ("-", "--")
    c_axis = ("#1971c2", "#2f9e44", "#c2255c")
    tau_max = 20.0
    for (name, v), ls in zip(d.items(), styles):
        tau = np.array(v["taus_s"])
        adev = np.array(v["acc_adev_g"])
        for k, colour in enumerate(c_axis):
            ax.loglog(tau, adev[:, k], ls, color=colour, lw=1.1,
                      label=f"measured, {'xyz'[k]} axis" if ls == "-" else None)
        # One-sigma uncertainty of the estimate on the largest axis, where it matters.
        s = bound["sessions"][name]
        k = "xyz".index(s["largest_axis"])
        rel = np.array(s["relative_sigma"])
        m = tau >= 5.0
        shift = 0.955 if ls == "-" else 1.045
        ax.errorbar(tau[m] * shift, adev[m, k], yerr=adev[m, k] * rel[m], fmt="none", ecolor=c_axis[k],
                    elinewidth=0.8, capsize=1.8, label="$\\pm1\\sigma$ of the estimate ($y$)" if ls == "-" else None)
    tau_ref = np.array([0.01, tau_max])
    ax.loglog(tau_ref, 3.5e-4 * np.sqrt(0.1 / tau_ref), ":", color="black", lw=0.9, label="white noise, slope $-1/2$")
    tau_k = np.array([0.01, tau_max])
    K0, K1 = bound["random_walk_bound_g_sqrt_s"], bound["random_walk_bound_plus_1sigma_g_sqrt_s"]
    ax.loglog(tau_k, K0 * np.sqrt(tau_k / 3.0), "-.", color="#e03131", lw=1.0,
              label="random-walk scenario, $K\\sqrt{\\tau/3}$")
    ax.loglog(tau_k, K1 * np.sqrt(tau_k / 3.0), "-.", color="#e03131", lw=0.7, alpha=0.6,
              label="same, allowing $+1\\sigma$ at $\\tau=20$ s")
    # Fewer than ten averaging intervals in the longer log.
    few = max(v["seconds"] for v in d.values()) / 10.0
    ax.axvspan(few, tau_max * 1.25, color="#868e96", alpha=0.13, lw=0)
    ax.text(tau_max * 1.2, 1.3e-3, "fewer than 10\naveraging\nintervals", fontsize=6.3, va="top", ha="right",
            color="#495057")
    ax.set_xlim(0.008, tau_max * 1.25)
    ax.set_ylim(6e-6, 1.6e-3)
    ax.set_xlabel("averaging time $\\tau$ (s)")
    ax.set_ylabel("accelerometer Allan deviation ($g$)")
    secs = [f"{v['seconds']:.1f} s" for v in d.values()]
    ax.set_title(f"static bench logs ({' and '.join(secs)}; solid / dashed)", fontsize=7.6)
    ax.grid(alpha=0.25, which="both", lw=0.4)
    ax.legend(frameon=False, ncol=1, loc="lower left", fontsize=6.3, labelspacing=0.22, handlelength=2.2)
    path = OUT / "allan_static.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def stop_capture_scene(results: Path, out: Path) -> Path | None:
    """R1-C1: real camera view with detected chessboard corners and synchronized
    200-Hz IMU / PnP streams comparing continuous and 4-s paused recordings."""
    import cv2
    import tools.build_pose_gt as bp
    import tools.train_physnet as phys
    from ego_capture.sync import load_imu

    cont_name, paus_name = "traj_20261109_024257", "traj_20261109_023932"
    if not ((results / cont_name).is_dir() and (results / paus_name).is_dir()):
        return None
    img_path = results / paus_name / "cam0" / "images" / "000500.jpg"
    if not img_path.is_file():
        imgs = sorted((results / paus_name / "cam0" / "images").glob("*.jpg"))
        if not imgs:
            return None
        img_path = imgs[len(imgs) // 2]

    fig = plt.figure(figsize=(7.2, 2.65))
    gs = fig.add_gridspec(2, 2, width_ratios=[0.92, 1.38], hspace=0.42, wspace=0.28)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1], sharex=ax_c)

    bgr = cv2.imread(str(img_path))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, _pat = bp._find(gray)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ax_img.imshow(rgb)
    if corners is not None:
        pts = corners.reshape(-1, 2)
        ax_img.plot(pts[:, 0], pts[:, 1], "-", color="#40c057", lw=0.45, alpha=0.8)
        ax_img.plot(pts[:, 0], pts[:, 1], "o", color="#2f9e44", ms=1.6, mew=0)
    ax_img.set_xticks([])
    ax_img.set_yticks([])
    ax_img.text(0.03, 0.04, "1280\u00d7720 @ 13.1 fps\nGP050 (11\u00d78, 3 mm)\nreproj. median 0.06 px",
                transform=ax_img.transAxes, fontsize=6.6, color="white",
                bbox={"boxstyle": "round,pad=0.25", "fc": (0, 0, 0, 0.62), "ec": "none"})
    _panel(ax_img, "a", "captured camera frame (tripod P2)")

    win = (8.0, 32.0)
    for ax, ses, letter, title, col in (
        (ax_c, cont_name, "b", "continuous recording C2 (no stops, 200-Hz IMU + PnP)", C_CONT),
        (ax_p, paus_name, "c", "paused recording P2 (metronome beat every 4 s, 1.5-s hold)", C_PAUSE),
    ):
        ev = json.loads((results / ses / "stop_events.json").read_text(encoding="utf-8"))
        t0 = float(ev["t_start"])
        t_imu, usb = load_imu(results / ses / "imu_stream.csv")
        rel_imu = t_imu - t0
        sel_i = (rel_imu >= win[0]) & (rel_imu <= win[1])
        gyro_n = np.linalg.norm(usb[sel_i, 3:6], axis=1)

        t_gt, _R, p_gt, usable = phys._load_gt(ses)
        vg, vok = phys.gt_velocity(t_gt, p_gt, usable)
        spd = np.linalg.norm(vg, axis=1) * 1000.0
        rel_gt = t_gt[usable] - t0
        ok_v = vok[usable] & (rel_gt >= win[0]) & (rel_gt <= win[1])

        for k, st in enumerate(ev.get("imu_stops") or []):
            s0, s1 = st["t_start"] - t0, st["t_end"] - t0
            if s1 >= win[0] and s0 <= win[1]:
                ax.axvspan(max(s0, win[0]), min(s1, win[1]), color="#b2f2bb", alpha=0.65, lw=0,
                           label="IMU-confirmed stop" if k == 0 else None)
        for k, bt in enumerate(ev.get("beat_times") or []):
            br = bt - t0
            if win[0] <= br <= win[1]:
                ax.axvline(br, color="#c92a2a", ls="--", lw=0.75, alpha=0.85,
                           label="4-s metronome beat" if k == 0 else None)

        ax.plot(rel_imu[sel_i], gyro_n, color=col, lw=0.75, alpha=0.9, label="USB $|\\omega|$ (deg/s)")
        ax.plot(rel_gt[ok_v], spd[usable][ok_v], color="#212529", lw=0.95, ls="-",
                label="PnP speed (mm/s)")
        ax.set_xlim(*win)
        ax.set_ylim(0, 72)
        ax.set_ylabel("deg/s or mm/s")
        ax.grid(alpha=0.25, lw=0.45)
        ax.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.3,
                  handlelength=1.4, columnspacing=0.8)
        _panel(ax, letter, title)
    ax_p.set_xlabel("time from recording start (s)")
    path = out / "stop_capture_scene.pdf"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=Path, default=DATA)
    ap.add_argument("--only-experiments", action="store_true", help="draw the two experiment figures only")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    real = args.results_dir.resolve() == DATA.resolve()
    exp_out = OUT if real else OUT / "preview"
    exp_out.mkdir(parents=True, exist_ok=True)
    made = []
    if not args.only_experiments:
        made += [pause_contrast(), gate_check(), gate_trajectory_trace(),
                 oracle_2x2_r2(), reference_validation_r2(), latency_tradeoff(), allan_static()]
    for fn in (stop_capture_scene, stop_experiment, stop_experiment_tripod, caliper_standard, board_imu_motion):
        p = fn(args.results_dir, exp_out)
        print(f"{fn.__name__}: {'not yet (results missing)' if p is None else p}")
        if p is not None:
            made.append(p)
    for p in made:
        print("wrote", p)


if __name__ == "__main__":
    main()
