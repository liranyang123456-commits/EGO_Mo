#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rehearse the two revision experiments in the digital twin before recording.

caliper     Rendered caliper traversals (tools/sim_reference_validation.py
            cache, PSF 1.5 px) are turned into sessions in the format the
            guided capture writes: pose_gt_raw from the same corner detector
            and PnP, caliper_events.json with the hold windows and vernier
            readings (0.02-mm graduation), and the plan CSV. Both analysis
            scripts then run on them unchanged. At 250 mm the hold windows and
            the motion-based segmentation must agree; at 150 mm, the median
            working distance of the real recordings, the segmentation
            previously failed and only the hold windows can score the run.
stop-chain  Four 180-s handheld twin sessions in the order C, P, P, C (no stop,
            or a 1-2 s stop about every 4 s) are written in the raw capture
            format and analysed by tools/eval_stop_protocol_experiment.py with
            the frozen real models. This checks the analysis chain end to end;
            the frozen models are not trained on twin data, so the numbers are
            not a prediction.
stop-power  Twin models trained on the 12.5-s stop protocol (closest to the
            real training recordings) are scored with stop-anchored switching
            on the 4-s and the pause-free test corpora. Pseudo-experiments with
            n recordings of 160 s per condition are drawn from the test
            sequences; each is analysed with the endpoint and the 3-s block
            bootstrap of the physical analysis, which gives the expected gain,
            its spread and the fraction of experiments whose interval excludes
            zero.

Outputs go to datasets/revision_rehearsal/.

    python tools/sim_revision_rehearsal.py caliper
    python tools/sim_revision_rehearsal.py stop-chain
    python tools/sim_revision_rehearsal.py stop-power
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets"
REH = DATA / "revision_rehearsal"
VERNIER_MM = 0.02


# --------------------------------------------------------------------------
# caliper
# --------------------------------------------------------------------------
def _pause_runs(planned, fps=10.0, pause_s=5.0, move_s=1.5):
    from tools.sim_reference_validation import _timeline
    runs, cur = [], None
    for t, _s, is_pause in _timeline(planned, fps, pause_s, move_s):
        if is_pause and cur is None:
            cur = [t, t]
        elif is_pause:
            cur[1] = t
        elif cur is not None:
            runs.append(tuple(cur))
            cur = None
    if cur is not None:
        runs.append(tuple(cur))
    return runs


def _write_caliper_root(depth: int, seed: int, with_events: bool, settle_s: float) -> Path:
    from tools.build_pose_gt import _pose
    cache = DATA / "sim_reference_cache" / f"{depth}_25_10_5_1.5_{seed}_9_psf1.5.pkl"
    with cache.open("rb") as handle:
        trials, K, dist = pickle.load(handle)
    root = REH / f"caliper_{depth}mm" / ("events" if with_events else "segmentation")
    if root.exists():
        shutil.rmtree(root)
    (root / "pose_gt_raw").mkdir(parents=True)
    rng = np.random.default_rng(depth)
    rows = []
    for k, tr in enumerate(trials):
        name = f"caliper_sim{depth}_{tr['axis']}{tr['trial']}"
        planned = [float(v) for v in tr["readings_mm"].split()]
        base = 1.8e9 + 1000.0 * k
        t = base + tr["t"]
        n = len(t)
        R = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
        p = np.zeros((n, 3), np.float32)
        usable = np.zeros(n, np.uint8)
        px = np.zeros(n, np.float32)
        for i, (c, pattern) in enumerate(zip(tr["corners"], tr["patterns"])):
            if c is None:
                continue
            res = _pose(c, pattern, K, dist)
            if res is None:
                continue
            R[i], p[i], usable[i], px[i] = res[0].T, res[1], 1, res[2]
        np.savez_compressed(root / "pose_gt_raw" / f"{name}.npz", t=t, R=R, p=p, usable=usable, px=px)
        if with_events:
            pauses = []
            for idx, (t0, t1) in enumerate(_pause_runs(planned)):
                reading = round(round((planned[idx] + rng.normal(0.0, 0.01)) / VERNIER_MM) * VERNIER_MM, 2)
                pauses.append({"index": idx, "target_mm": planned[idx], "reading_mm": reading,
                               "t_confirm": base + t0 - 0.5, "t_hold_start": base + t0 + settle_s,
                               "t_hold_end": base + t1, "live": {}, "ok": True, "reject_reason": ""})
            events = {"kind": "vernier_caliper_trial", "axis": tr["axis"], "trial": int(tr["trial"]),
                      "targets_mm": planned, "zero_reading_mm": 0.0,
                      "caliper": {"type": "simulated vernier", "resolution_mm": VERNIER_MM},
                      "readings_mm": [q["reading_mm"] for q in pauses],
                      "positions_mm": [q["reading_mm"] for q in pauses],
                      "pauses": pauses, "rejected_holds": []}
            (root / name).mkdir()
            (root / name / "caliper_events.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        rows.append({"axis": tr["axis"], "trial": tr["trial"], "readings_mm": tr["readings_mm"], "session": name})
    with (root / "reference_caliper_plan.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["axis", "trial", "readings_mm", "session"])
        writer.writeheader()
        writer.writerows(rows)
    return root


def _run(cmd) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, *cmd], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    return proc.returncode, (proc.stdout + proc.stderr)[-2000:]


def caliper(_args) -> None:
    out = {"source": "sim_reference_cache, PSF 1.5 px, tilt 25 deg, 10 fps, 5-s pauses, 1.5-s moves",
           "settle_s": 1.0, "depths": {}}
    for depth, seed in ((150, 11), (250, 12)):
        entry = {}
        for with_events in (True, False):
            root = _write_caliper_root(depth, seed, with_events, settle_s=1.0)
            key = "hold_windows" if with_events else "motion_segmentation"
            code, log = _run(["tools/reference_stage_validation.py", "analyze", "--mode", "caliper",
                              "--data-root", str(root)])
            if code != 0:
                entry[key] = {"error": log.strip().splitlines()[-1]}
                continue
            rep = json.loads((root / "reference_stage_validation.json").read_text(encoding="utf-8"))
            s = rep["summary"]
            entry[key] = {
                "passed": rep["passed"], "heldout_rmse_mm": round(s["heldout_rmse_mm"], 3),
                "heldout_max_mm": round(s["heldout_max_mm"], 3),
                "repeatability_p95_mm": round(s["repeatability_p95_mm"], 3),
                "scale_error_percent": {a: round(v["scale_error_percent"], 2) for a, v in s["axes"].items()},
                "cross_axis_rmse_mm": {a: round(v["cross_axis_rmse_mm"], 3) for a, v in s["axes"].items()}}
            if with_events:
                code, log = _run(["tools/reference_caliper_dynamic.py", "--data-root", str(root)])
                if code == 0:
                    dyn = json.loads((root / "reference_caliper_dynamic.json").read_text(encoding="utf-8"))
                    entry["label_level"] = dyn["overall"]
                else:
                    entry["label_level"] = {"error": log.strip().splitlines()[-1]}
        out["depths"][f"{depth}mm"] = entry
        print(depth, json.dumps(entry, ensure_ascii=False)[:600], flush=True)
    path = REH / "caliper_rehearsal.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# stop experiment: chain test with the frozen models
# --------------------------------------------------------------------------
def stop_chain(args) -> None:
    from ego_sim import build_sequence, export_sequence
    from tools.generate_synthetic_corpus import _config
    root = REH / "stop_chain"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    sessions = {"C": [], "P": []}
    for k, cond in enumerate(("C", "P", "P", "C")):
        motion = "handheld_pause" if cond == "P" else "handheld_free"
        cfg = _config(4100 + k, motion, args.duration, False, "opencv", False, (0.0, 0.87, 0.49))
        cfg.pause_interval_s, cfg.pause_duration_s = 4.0, 1.5
        out = export_sequence(build_sequence(cfg), root)
        gt = np.load(out / "ground_truth.npz")
        np.savez(str(out) + ".npz", t=gt["t"], R=gt["R"], p=gt["p"],
                 usable=np.ones(len(gt["t"]), np.uint8), delay_usb=np.zeros(1))
        label = f"{cond}{len(sessions[cond]) + 1}"
        (out / "stop_events.json").write_text(json.dumps(
            {"kind": "stop_protocol_recording", "label": label, "condition": cond, "order": k + 1,
             "note": "twin session for the chain test"}, indent=2), encoding="utf-8")
        sessions[cond].append(str(out))
        print(f"{label}: {out.name}", flush=True)
    output = root / "stop_chain.json"
    code, log = _run(["tools/eval_stop_protocol_experiment.py", "--continuous", *sessions["C"],
                      "--paused", *sessions["P"], "--rigid", "nominal", "--n-boot", str(args.n_boot),
                      "--output", str(output)])
    if code != 0:
        raise SystemExit(log)
    rep = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps({"primary": rep["estimators"]["physnet_switch"],
                      "recording_level": rep["recording_level"]}, ensure_ascii=False, indent=2))


# --------------------------------------------------------------------------
# stop experiment: cross-protocol twin power analysis
# --------------------------------------------------------------------------
def _twin_condition(corpus_name: str, models, v3, sac, device, context, horizon, length, anchor):
    import torch
    from tools.twin_protocol_sweep import corpus
    manifest = json.loads((corpus(corpus_name) / "manifest.json").read_text(encoding="utf-8"))
    paths = [v3._corpus_path(corpus(corpus_name), e["path"]) for e in manifest["imu"]["test"]]
    v3.BIAS_MODE = "none"
    data = v3.build_group(paths, context, horizon, length, cap=160, rng=np.random.default_rng(123))
    dt = v3.to_device(data, device)
    with torch.no_grad():
        pred = np.mean([m.predict(dt) for m in models], axis=0)
    del dt
    torch.cuda.empty_cache()
    est, span = sac.stop_anchor(data, anchor)
    switch = span <= 2.0
    pred_sw = np.where(switch[:, None], est, pred)
    y = data["y"]
    seqs = []
    for name in np.unique(data["session"]):
        sel = data["session"] == name
        ta = data["t_pair"][sel, 0]
        block = ((ta - ta.min()) // 3.0).astype(int)
        e = np.linalg.norm(pred_sw[sel] - y[sel], axis=1)
        z = np.linalg.norm(y[sel], axis=1)
        e_net = np.linalg.norm(pred[sel] - y[sel], axis=1)
        seqs.append({"e": np.bincount(block, e), "z": np.bincount(block, z),
                     "e_net": np.bincount(block, e_net), "n": np.bincount(block)})
    pooled = {"windows": int(len(y)), "switched_pct": round(float(100 * switch.mean()), 1),
              "zero_mm": round(float(np.linalg.norm(y, axis=1).mean() * 1000), 2),
              "physnet_mm": round(float(np.linalg.norm(pred - y, axis=1).mean() * 1000), 2),
              "physnet_switch_mm": round(float(np.linalg.norm(pred_sw - y, axis=1).mean() * 1000), 2)}
    pooled["reduction_switch"] = round(1 - pooled["physnet_switch_mm"] / pooled["zero_mm"], 3)
    pooled["reduction_physnet"] = round(1 - pooled["physnet_mm"] / pooled["zero_mm"], 3)
    return seqs, pooled


def _pseudo(seqs, n_rec, per_rec, rng):
    """Blocks (error sum, |dp| sum) of n_rec pseudo-recordings of per_rec sequences."""
    pick = rng.integers(0, len(seqs), size=(n_rec, per_rec))
    rec_red, e_blocks, z_blocks = [], [], []
    for row in pick:
        e = np.concatenate([seqs[i]["e"] for i in row])
        z = np.concatenate([seqs[i]["z"] for i in row])
        rec_red.append(1.0 - e.sum() / z.sum())
        e_blocks.append(e)
        z_blocks.append(z)
    return np.concatenate(e_blocks), np.concatenate(z_blocks), np.array(rec_red)


def stop_power(args) -> None:
    import torch
    import tools.stop_anchor_check as sac
    import tools.train_physnet_v3 as v3
    import tools.train_seq as api
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    context, horizon = 2.0, 3.0
    length = int(round((1.5 * horizon + 2 * context) * api.HZ))
    anchor = int(round(context * api.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [v3.load_model(DATA / "physnet_proto" / f"physnet_{args.model}_s{s}.pt", device) for s in (0, 1)]
    cond = {}
    for key, corpus_name in (("continuous", "pause_inf"), ("paused", "pause_4")):
        seqs, pooled = _twin_condition(corpus_name, models, v3, sac, device, context, horizon, length, anchor)
        cond[key] = {"corpus": corpus_name, "sequences": len(seqs), "pooled": pooled}
        cond[key]["_seqs"] = seqs
        print(key, pooled, flush=True)
    rng = np.random.default_rng(7)
    per_rec = int(round(args.recording_s / 40.0))
    designs = {}
    for n_rec in (2, 3, 4):
        gains, lows, highs, sep = [], [], [], []
        for _ in range(args.experiments):
            ec, zc, rc = _pseudo(cond["continuous"]["_seqs"], n_rec, per_rec, rng)
            ep, zp, rp = _pseudo(cond["paused"]["_seqs"], n_rec, per_rec, rng)
            gains.append((1 - ep.sum() / zp.sum()) - (1 - ec.sum() / zc.sum()))
            wc = rng.multinomial(len(ec), np.full(len(ec), 1.0 / len(ec)), size=args.n_boot)
            wp = rng.multinomial(len(ep), np.full(len(ep), 1.0 / len(ep)), size=args.n_boot)
            boot = (1 - (wp @ ep) / (wp @ zp)) - (1 - (wc @ ec) / (wc @ zc))
            lo, hi = np.quantile(boot, (0.025, 0.975))
            lows.append(lo)
            highs.append(hi)
            sep.append(rp.min() > rc.max())
        gains, lows, highs = map(np.asarray, (gains, lows, highs))
        designs[f"{n_rec}+{n_rec}"] = {
            "recordings_per_condition": n_rec, "motion_s_per_recording": per_rec * 40,
            "gain_mean": round(float(gains.mean()), 3), "gain_sd": round(float(gains.std()), 3),
            "ci95_halfwidth_mean": round(float(((highs - lows) / 2).mean()), 3),
            "power_ci_excludes_zero": round(float((lows > 0).mean()), 3),
            "every_paused_above_every_continuous": round(float(np.mean(sep)), 3)}
        print(n_rec, designs[f"{n_rec}+{n_rec}"], flush=True)
    for key in cond:
        cond[key].pop("_seqs")
    report = {
        "kind": "twin_cross_protocol_power",
        "model": f"physnet_proto/physnet_{args.model}_s0/s1 (trained on the 12.5-s stop protocol, raw gyro)",
        "endpoint": "reduction relative to zero motion, paused minus continuous, PhysNet + stop-anchored "
                    "switching (S <= 2 s); 3-s block bootstrap as in eval_stop_protocol_experiment.py",
        "conditions": cond, "pseudo_experiments": args.experiments, "n_boot": args.n_boot,
        "designs": designs,
        "caveats": [
            "Twin sequences are 40 s; a 160-s pseudo-recording joins four of them, drawn with replacement.",
            "Real recordings differ more from one another than twin sequences, so the real interval "
            "will be wider than the twin's.",
        ],
    }
    path = args.output or REH / "stop_power.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("caliper")
    p_chain = sub.add_parser("stop-chain")
    p_chain.add_argument("--duration", type=float, default=180.0)
    p_chain.add_argument("--n-boot", type=int, default=500)
    p_pow = sub.add_parser("stop-power")
    p_pow.add_argument("--model", default="pause_12.5_nb")
    p_pow.add_argument("--recording-s", type=float, default=160.0)
    p_pow.add_argument("--experiments", type=int, default=400)
    p_pow.add_argument("--n-boot", type=int, default=400)
    p_pow.add_argument("--output", type=Path, help="default: revision_rehearsal/stop_power.json")
    args = ap.parse_args()
    {"caliper": caliper, "stop-chain": stop_chain, "stop-power": stop_power}[args.cmd](args)


if __name__ == "__main__":
    main()
