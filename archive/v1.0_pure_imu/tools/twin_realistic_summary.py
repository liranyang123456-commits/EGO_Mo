#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Collect the realistic-twin and acquisition-protocol results into one file.

Inputs are the outputs of tools/v3_experiments.py (results_twin_*.json and
metrics_*.json in datasets/physnet_v3), tools/twin_protocol_sweep.py,
tools/stop_anchor_check.py, tools/twin_imunet_synthetic.py,
tools/twin_motion_class_eval.py, tools/train_external_architectures.py
(datasets/external_twin_realistic) and tools/zupt_bound.py.

    python tools/twin_realistic_summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
V3 = DATA / "physnet_v3"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    out = {"real_val_3seed": {}, "zero_shot_real_val": {}, "protocol": {}, "real_windows": {}}
    base3 = load(V3 / "results_twin_base3.json")["base"]
    out["real_val_3seed"]["physnet_real_only"] = {k: base3[k] for k in ("err_mm", "r2", "ate_mm_mean")}
    for name in ("pre_periodic_twin", "pre_realistic", "pre_realistic_bal", "bias_none", "bias_strict"):
        r = load(V3 / f"results_twin_{name}.json")[name]
        out["real_val_3seed"][f"physnet_{name}"] = {k: r[k] for k in ("err_mm", "r2", "ate_mm_mean")}
        if name.startswith("pre_"):
            z = [load(V3 / f"metrics_{name}_s{s}.json")["pretrain"]["zero_shot_real_val"] for s in (0, 1, 2)]
            out["zero_shot_real_val"][f"physnet_{name}"] = {
                "err_mm": [m["err_mm"] for m in z], "r2": [m["r2"] for m in z]}
    ext = [load(DATA / "external_twin_realistic" / f"imunet_s{s}.json") for s in (0, 1, 2)]
    out["zero_shot_real_val"]["imunet_realistic"] = {
        "err_mm": [e["synthetic_zero_shot_real_val"]["err_mm"] for e in ext],
        "r2": [e["synthetic_zero_shot_real_val"]["r2"] for e in ext]}
    cls = load(DATA / "twin_motion_class_eval.json")
    out["real_val_3seed"]["imunet_real_only"] = {"err_mm": cls["all"]["imunet_real"]}
    out["real_val_3seed"]["imunet_pre_realistic"] = {"err_mm": cls["all"]["imunet_fine"]}
    out["real_val_3seed"]["zero_mm"] = cls["all"]["zero_mm"]
    out["real_windows"]["by_stop_position"] = {
        c: {"share": v["share"], "zero_mm": v["zero_mm"], "physnet": v["base"],
            "imunet": v["imunet_real"]} for c, v in cls["by_stop_position"].items()}

    sweep = load(DATA / "twin_protocol_sweep.json")["models"]
    switch = load(DATA / "stop_anchor_check.json")
    imu = load(DATA / "twin_imunet_synthetic.json")
    for c in ("pause_4", "pause_8", "pause_12.5", "pause_20", "pause_inf"):
        row = {}
        for variant, key in (("short", "physnet_session_bias"), ("nb", "physnet_raw_gyro"),
                             ("long", "physnet_session_bias_8s_context")):
            e = sweep[f"{c}_{variant}"][f"test_{c}"]
            row[key] = {"err_mm": e["err_mm"], "r2": e["r2"], "slope": e["slope"]}
            row["zero_mm"] = e["zero_mm"]
            if variant == "short":
                row["stop_position_share"] = {k: v["share"] for k, v in e["by_stop_position"].items()}
        row["physnet_raw_gyro_switch"] = {
            "err_mm": switch["synthetic"][c]["all"]["hybrid"],
            "physics_share": switch["synthetic"][c]["physics_share"]}
        if c in imu["corpora"]:
            a = imu["corpora"][c]["all"]
            row["imunet"] = {"err_mm": a["learned"]}
            row["imunet_switch"] = {"err_mm": a["hybrid"], "physics_share": imu["corpora"][c]["physics_share"]}
        out["protocol"][c] = row
    out["protocol"]["switch_max_span_s"] = {"physnet": switch["chosen_max_span_s"],
                                            "imunet": imu["chosen_max_span_s"]}
    out["protocol"]["ideal_imu"] = {
        c: {"session_bias": sweep[f"ideal_{c}_short"][f"test_ideal_{c}"]["err_mm"],
            "raw": sweep[f"ideal_{c}_nb"][f"test_ideal_{c}"]["err_mm"]} for c in ("pause_4", "pause_inf")}
    out["protocol"]["reference_fps"] = {
        f: sweep[f"fps_{f}_short"]["test_pause_12.5"]["err_mm"] for f in ("10", "30")}
    out["protocol"]["reference_fps"]["15"] = sweep["pause_12.5_short"]["test_pause_12.5"]["err_mm"]
    out["real_windows"]["switch"] = switch["real_val|bias_none"]

    zb = load(DATA / "zupt_bound.json")

    def pooled(att, key):
        num = den = 0.0
        for g in ("train", "val", "extra_train (rigid)"):
            e = zb[att][g].get("by_span", {}).get(key)
            if e:
                num += e["err_mm"] * e["n"]
                den += e["n"]
        return round(num / den, 1) if den else None

    out["zupt_real"] = {att: {k: pooled(att, k) for k in ("0.8-2s", "2-4s", "4-8s", "8-30s")}
                        for att in ("ref", "gyro", "gyro_raw")}
    (DATA / "twin_realistic_summary.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
