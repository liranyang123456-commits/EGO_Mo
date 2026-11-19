#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Controlled physical stop experiment: continuous motion against motion with
a firm stop every few seconds (docs/revision_physical_experiments.md, experiment A).

The frozen models of datasets/final_model_manifest.json are applied without any
update. The rest phases at the start and end of every recording are excluded.
Primary endpoint: the difference between conditions in the error reduction
relative to zero motion, 1 - e_p / mean|dp|, for gated PhysNet with
stop-anchored switching (raw gyroscope rate, S <= 2 s). Because a stopping
operator covers less distance per window, a displacement-matched contrast
(2.5-mm bins weighted by the paused condition) is reported as well. Intervals
are block bootstraps over (recording, 3-s block).

The frozen models assume the camera<-IMU rotation of the nominal mount. New
recordings use the rotation estimated by the same day's rigidity check
(``--rigid auto``): PhysNet and the stop anchor receive it directly, and the
raw stream seen by IMUNet is rotated into the nominal mount.

    python tools/eval_stop_protocol_experiment.py            # sessions from the plan
    python tools/eval_stop_protocol_experiment.py \
        --continuous traj_A traj_B --paused traj_C traj_D
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.stop_anchor_check as sac
import tools.train_physnet as phys
import tools.train_physnet_v3 as v3
import tools.train_seq as api
import tools.zupt_bound as zb
from tools.train_external_architectures import make_model, predict as predict_external

DATA = ROOT / "datasets"
SESSIONS = DATA            # session folders, pose_gt/, pose_gt_raw/ and the plan (--data-root)
CONTEXT, HORIZON = 2.0, 3.0
LENGTH = int(round((1.5 * HORIZON + 2 * CONTEXT) * phys.HZ))
ANCHOR = int(round(CONTEXT * phys.HZ))
MAX_SPAN_S = 2.0


@contextmanager
def imu_mount(R_today):
    """Present a session recorded with camera<-IMU rotation R_today to the
    frozen models, which were trained with api.R_CAMERA_IMU_NEW."""
    if R_today is None:
        yield
        return
    R_nom = api.R_CAMERA_IMU_NEW.copy()
    M = R_nom.astype(np.float64).T @ np.asarray(R_today, np.float64)
    load = api.load_imu

    def load_rotated(path):
        t, usb = load(path)
        usb = np.array(usb, copy=True)
        usb[:, 0:3] = usb[:, 0:3] @ M.T
        usb[:, 3:6] = usb[:, 3:6] @ M.T
        return t, usb

    api.R_CAMERA_IMU_NEW = np.asarray(R_today, np.float32)
    api.load_imu = load_rotated
    try:
        yield
    finally:
        api.R_CAMERA_IMU_NEW = R_nom
        api.load_imu = load


def _events(name: str) -> dict:
    path = (Path(name) if Path(name).is_absolute() else SESSIONS / name) / "stop_events.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def rest_norm(name: str):
    """Accelerometer norm over the initial rest of a recording (samples without
    rotation), or None without stop_events.json."""
    ev = _events(name)
    if "t_start" not in ev:
        return None
    folder = Path(name) if Path(name).is_absolute() else SESSIONS / name
    t, y = api.load_imu(folder / "imu_stream.csv")
    sel = (t >= ev["t_start"]) & (t <= ev["t_motion"]) & (np.linalg.norm(y[:, 3:6], axis=1) < sac.W_MAX_DPS)
    return float(np.median(np.linalg.norm(y[sel, 0:3], axis=1))) if sel.sum() >= 100 else None


@contextmanager
def acc_at_rest(name: str):
    """Scale the accelerometer of a recording so that its norm over the initial
    rest is 1 g, for the stop detector and every model alike; the capture app
    judges stillness around the same norm (stop_events.json, initial_rest)."""
    ref = rest_norm(name)
    if ref is None:
        yield None
        return
    loads = {mod: mod.load_imu for mod in (api, phys, v3)}

    def scaled(load):
        def loader(path):
            t, y = load(path)
            if Path(path).parent.name != Path(name).name:
                return t, y
            y = np.array(y, copy=True)
            y[:, 0:3] /= ref
            return t, y
        return loader

    for mod, load in loads.items():
        mod.load_imu = scaled(load)
    try:
        yield ref
    finally:
        for mod, load in loads.items():
            mod.load_imu = load


def _angle_deg(Ra, Rb) -> float:
    rel = np.asarray(Ra, float).T @ np.asarray(Rb, float)
    return float(np.degrees(np.arccos(np.clip((np.trace(rel) - 1.0) / 2.0, -1.0, 1.0))))


def mount_for(name: str, rigid: str):
    """(R_camera_imu or None for the nominal mount, description)."""
    if rigid == "nominal" or Path(name).is_absolute():
        return None, {"source": "nominal (frozen) mount"}
    if rigid != "auto":
        from tools.check_rigidity import check
        data = check(rigid)
        return np.asarray(data["R_camera_imu"], float), {"source": rigid}
    m = re.search(r"_(\d{8})_\d{6}$", name)
    if not m:
        raise SystemExit(f"{name}: no date in the name; pass --rigid <rigid_session> or --rigid nominal")
    from ego_capture.revision import todays_rigidity
    rig = todays_rigidity(m.group(1))
    if rig is None:
        raise SystemExit(f"{name}: no passing rigidity check (rigid_{m.group(1)}_*) on that day; "
                         "record step 9 or pass --rigid <rigid_session> / --rigid nominal")
    return np.asarray(rig["R_camera_imu"], float), {"source": rig["session"]}


def plan_sessions():
    plan = json.loads((SESSIONS / "revision_experiment_plan.json").read_text(encoding="utf-8"))
    done = sorted((e for e in plan["stop"] if e["status"] == "done" and e.get("session")),
                  key=lambda e: e["order"])
    return ([e["session"] for e in done if e["condition"] == "C"],
            [e["session"] for e in done if e["condition"] == "P"],
            [e["session"] for e in done])


def compliance(name: str):
    path = (Path(name) if Path(name).is_absolute() else SESSIONS / name) / "stop_events.json"
    if not path.is_file():
        return None
    ev = json.loads(path.read_text(encoding="utf-8"))
    return {k: ev.get(k) for k in ("label", "condition", "order", "stops", "stops_per_min", "beats",
                                   "beats_matched", "compliance", "median_moving_speed_mm_s")}


def frozen_models(device):
    manifest = json.loads((DATA / "final_model_manifest.json").read_text(encoding="utf-8"))
    paths = [Path(r["path"]) for r in manifest["models_and_configs"]]
    physnet = [phys.load_model(p, device) for p in paths if p.name.startswith("physnet_cvg_")]
    imunet = []
    for p in paths:
        if p.name.startswith("imunet_cvx_"):
            net = make_model("imunet").to(device)
            net.load_state_dict(torch.load(p, map_location=device))
            net.eval()
            imunet.append(net)
    assert len(physnet) == 16 and len(imunet) == 16, (len(physnet), len(imunet))
    cal = json.loads((DATA / "session_cv_gate_benchmark.json").read_text(encoding="utf-8"))["cv_calibration"]
    return physnet, imunet, float(cal["blend_weight_physnet"]), float(cal["shrink"])


def session_windows(name, physnet, imunet, w, alpha, device, skip_start, skip_end):
    data = phys.build_group([name], CONTEXT, HORIZON, LENGTH)
    y = data["y"]
    data_t = phys.to_device(data, device)
    p_phys = np.mean([m.predict(data_t) for m in physnet], axis=0)
    X, y_ext = api.build(name, CONTEXT, LENGTH)
    assert np.allclose(y_ext, y, atol=1e-6)
    p_imu = np.mean([predict_external(n, X, device) for n in imunet], axis=0)
    v3.BIAS_MODE = "none"
    data_v3 = v3.build_group([name], CONTEXT, HORIZON, LENGTH)
    assert np.allclose(data_v3["y"], y, atol=1e-6)
    est, span = sac.stop_anchor(data_v3, ANCHOR)
    switch = span <= MAX_SPAN_S
    preds = {
        "zero": np.zeros_like(y),
        "physnet": p_phys,
        "imunet": p_imu,
        "calibrated_blend": alpha * (w * p_phys + (1 - w) * p_imu),
        "physnet_switch": np.where(switch[:, None], est, p_phys),
        "imunet_switch": np.where(switch[:, None], est, p_imu),
    }
    t, _R, p, usable = phys._load_gt(name)
    vg, vok = phys.gt_velocity(t, p, usable)
    speed = np.linalg.norm(vg, axis=1) * 1000.0
    segs = zb.still_segments(t, vok, speed, 3.0, 0.4)
    t_pair = data["t_pair"]
    t0, t1 = float(t[usable].min()), float(t[usable].max())
    keep = (t_pair[:, 0] >= t0 + skip_start) & (t_pair[:, 1] <= t1 - skip_end)
    inside = np.array([any(t[a] >= ta and t[b] <= tb for a, b in segs) for ta, tb in t_pair])
    mid = [(t[a], t[b]) for a, b in segs if t[a] >= t0 + skip_start and t[b] <= t1 - skip_end]
    minutes = max((t1 - skip_end) - (t0 + skip_start), 1e-6) / 60.0
    return {
        "y": y[keep], "t_pair": t_pair[keep], "inside": inside[keep], "switch": switch[keep],
        "preds": {k: v[keep] for k, v in preds.items()},
        "stats": {"windows": int(keep.sum()), "stops_per_min": round(len(mid) / minutes, 2),
                  "stop_durations_s": [round(float(b - a), 2) for a, b in mid],
                  "windows_with_stop_inside_pct": round(float(100 * inside[keep].mean()), 1),
                  "switched_pct": round(float(100 * switch[keep].mean()), 1)},
    }


def _pool(rows):
    out = {"y": np.concatenate([r["y"] for r in rows]),
           "block": np.concatenate([[f"{i}:{int((ta - r['t_pair'][:, 0].min()) // 3.0)}"
                                     for ta in r["t_pair"][:, 0]] for i, r in enumerate(rows)])}
    for key in rows[0]["preds"]:
        out[key] = np.concatenate([r["preds"][key] for r in rows])
    return out


def _err(pool, key, idx=None):
    idx = np.arange(len(pool["y"])) if idx is None else idx
    return np.linalg.norm(pool[key][idx] - pool["y"][idx], axis=1)


def _reduction(pool, key, idx=None):
    idx = np.arange(len(pool["y"])) if idx is None else idx
    return 1.0 - _err(pool, key, idx).mean() / np.linalg.norm(pool["y"][idx], axis=1).mean()


def _matched_diff(cont, paus, key, ic=None, ip=None):
    ic = np.arange(len(cont["y"])) if ic is None else ic
    ip = np.arange(len(paus["y"])) if ip is None else ip
    mc = np.linalg.norm(cont["y"][ic], axis=1) * 1000
    mp = np.linalg.norm(paus["y"][ip], axis=1) * 1000
    ec, ep = _err(cont, key, ic) * 1000, _err(paus, key, ip) * 1000
    edges = np.arange(5.0, 102.5, 2.5)
    diff = ref = w = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sc, sp = (mc >= lo) & (mc < hi), (mp >= lo) & (mp < hi)
        if sc.sum() == 0 or sp.sum() == 0:
            continue
        k = float(sp.sum())
        diff += k * (ep[sp].mean() - ec[sc].mean())
        ref += k * ec[sc].mean()
        w += k
    return (diff / w, ref / w) if w else (np.nan, np.nan)


def contrast(cont, paus, key, n_boot, seed=0):
    rng = np.random.default_rng(seed)
    bc, bp = np.unique(cont["block"]), np.unique(paus["block"])
    mem_c = {b: np.flatnonzero(cont["block"] == b) for b in bc}
    mem_p = {b: np.flatnonzero(paus["block"] == b) for b in bp}
    red_c, red_p = _reduction(cont, key), _reduction(paus, key)
    md, mref = _matched_diff(cont, paus, key)
    boots_red, boots_md = [], []
    for _ in range(n_boot):
        ic = np.concatenate([mem_c[b] for b in rng.choice(bc, len(bc))])
        ip = np.concatenate([mem_p[b] for b in rng.choice(bp, len(bp))])
        boots_red.append(_reduction(paus, key, ip) - _reduction(cont, key, ic))
        boots_md.append(_matched_diff(cont, paus, key, ic, ip)[0])
    q = lambda v: [round(float(x), 3) for x in np.nanquantile(v, (0.025, 0.975))]
    return {
        "continuous": {"err_mm": round(float(_err(cont, key).mean() * 1000), 2),
                       "zero_mm": round(float(np.linalg.norm(cont["y"], axis=1).mean() * 1000), 2),
                       "reduction_vs_zero": round(float(red_c), 3)},
        "paused": {"err_mm": round(float(_err(paus, key).mean() * 1000), 2),
                   "zero_mm": round(float(np.linalg.norm(paus["y"], axis=1).mean() * 1000), 2),
                   "reduction_vs_zero": round(float(red_p), 3)},
        "reduction_gain": round(float(red_p - red_c), 3), "reduction_gain_ci95": q(boots_red),
        "matched_diff_mm": round(float(md), 2), "matched_continuous_err_mm": round(float(mref), 2),
        "matched_diff_ci95_mm": q(boots_md),
    }


ESTIMATORS = ("physnet_switch", "physnet", "imunet", "imunet_switch", "calibrated_blend")


def per_recording(row) -> dict:
    zero = float(np.linalg.norm(row["y"], axis=1).mean())
    out = {"zero_mm": round(zero * 1000, 2)}
    for key in ("physnet_switch", "physnet", "imunet"):
        err = float(np.linalg.norm(row["preds"][key] - row["y"], axis=1).mean())
        out[key] = {"err_mm": round(err * 1000, 2), "reduction_vs_zero": round(1.0 - err / zero, 3)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--continuous", nargs="*", default=[])
    ap.add_argument("--paused", nargs="*", default=[])
    ap.add_argument("--rigid", default="auto",
                    help="auto: same-day rigidity check; nominal: frozen mount; or a rigid_* session")
    ap.add_argument("--skip-start", type=float, help="s excluded after the first reference frame "
                                                    "(default: the recording's initial rest + 2 s, else 10 s)")
    ap.add_argument("--skip-end", type=float, help="s excluded before the last reference frame "
                                                  "(default: the recording's final rest + 2 s, else 10 s)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--data-root", type=Path,
                    help="folder with the session folders, pose_gt/, pose_gt_raw/ and the plan "
                         "(rehearsals); the frozen models stay in datasets/")
    ap.add_argument("--allow-seen", action="store_true", help="smoke tests only")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing result file that holds other recordings")
    args = ap.parse_args()
    global SESSIONS
    if args.data_root:
        import ego_capture.revision as rv
        import tools.check_rigidity as cr
        SESSIONS = args.data_root
        for mod in (phys, v3, api, cr):
            mod.DATA = SESSIONS
        rv.DATA_ROOT = SESSIONS
    args.output = args.output or SESSIONS / "stop_protocol_experiment.json"
    order = None
    if not args.continuous and not args.paused:
        args.continuous, args.paused, order = plan_sessions()
        print(f"from revision_experiment_plan.json: order {order}")
    if not args.continuous or not args.paused:
        raise SystemExit("need at least one continuous and one paused recording")
    # Two batches were recorded (handheld 2026-11-08, tripod 2026-11-09). The
    # plan names only the latest, so a run without --output would write the
    # tripod batch over the handheld result that the manuscript reads.
    if args.output.is_file() and not args.force:
        held = json.loads(args.output.read_text(encoding="utf-8")).get("sessions", {})
        old = set(held.get("continuous", []) + held.get("paused", []))
        if old and old != set(args.continuous + args.paused):
            raise SystemExit(
                f"{args.output} holds another batch ({sorted(old)}). Pass --output for this batch "
                "(tripod batch: datasets/stop_eval_20261109_batch.json), or --force to replace it.")
    split = json.loads((DATA / "trajectory_split_20260924.json").read_text(encoding="utf-8"))
    seen = set(split["train"] + split["val"] + split.get("extra_train", []))
    clash = sorted(seen & set(args.continuous + args.paused))
    if clash and not args.allow_seen:
        raise SystemExit(f"sessions used to train the frozen models: {clash}")
    events = {n: compliance(n) for n in args.continuous + args.paused}
    for cond, names in (("C", args.continuous), ("P", args.paused)):
        wrong = [n for n in names if events[n] and events[n]["condition"] != cond]
        if wrong:
            raise SystemExit(f"stop_events.json of {wrong} says condition != {cond}")
    api.LABELS, api.TARGET_S = "pose_gt_raw", HORIZON
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    physnet, imunet, w, alpha = frozen_models(device)
    rows, mounts = {}, {}
    for cond, names in (("continuous", args.continuous), ("paused", args.paused)):
        rows[cond] = []
        for n in names:
            R_today, info = mount_for(n, args.rigid)
            if R_today is not None:
                info["angle_to_nominal_deg"] = round(_angle_deg(api.R_CAMERA_IMU_NEW, R_today), 2)
            mounts[n] = info
            ev = _events(n)
            skip = (args.skip_start if args.skip_start is not None else ev.get("rest_start_s", 8.0) + 2.0,
                    args.skip_end if args.skip_end is not None else ev.get("rest_end_s", 8.0) + 2.0)
            with acc_at_rest(n) as ref, imu_mount(R_today):
                rows[cond].append(session_windows(n, physnet, imunet, w, alpha, device, *skip))
            rows[cond][-1]["stats"].update(acc_rest_norm_g=None if ref is None else round(ref, 4),
                                           skip_s=[round(skip[0], 1), round(skip[1], 1)])
    cont, paus = _pool(rows["continuous"]), _pool(rows["paused"])
    per_session = {}
    for cond, names in (("continuous", args.continuous), ("paused", args.paused)):
        for n, r in zip(names, rows[cond]):
            per_session[n] = {"condition": cond, **r["stats"], "imu_mount": mounts[n],
                              "protocol": events[n], "errors": per_recording(r)}
    red = {n: v["errors"]["physnet_switch"]["reduction_vs_zero"] for n, v in per_session.items()}
    min_p = min(red[n] for n in args.paused)
    max_c = max(red[n] for n in args.continuous)
    report = {
        "sessions": {"continuous": args.continuous, "paused": args.paused, "recording_order": order},
        "per_session": per_session,
        "primary_endpoint": "reduction_gain of physnet_switch",
        "estimators": {k: contrast(cont, paus, k, args.n_boot) for k in ESTIMATORS},
        "recording_level": {
            "physnet_switch_reduction": red,
            "every_paused_above_every_continuous": bool(min_p > max_c),
            "note": "with m continuous and n paused recordings, complete separation has probability "
                    "1/C(m+n, n) under exchangeability (1/6 for 2+2, 1/20 for 3+3, 1/70 for 4+4)",
        },
        "twin_prediction": "Table IV: 4-s stop interval vs pause-free, PhysNet raw rate + switching "
                           "30.0 vs 44.1 mm at 55-57 mm zero motion; cross-protocol rehearsal in "
                           "datasets/revision_rehearsal/stop_power.json",
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    windows = {"condition": [], "session": [], "t_a": [], "disp_mm": [],
               **{f"err_{k}_mm": [] for k in ("zero", "physnet_switch", "physnet", "imunet")}}
    for cond, names in (("continuous", args.continuous), ("paused", args.paused)):
        for n, r in zip(names, rows[cond]):
            windows["condition"] += [cond] * len(r["y"])
            windows["session"] += [Path(str(n)).name] * len(r["y"])
            windows["t_a"].append(r["t_pair"][:, 0])
            windows["disp_mm"].append(np.linalg.norm(r["y"], axis=1) * 1000)
            for k in ("zero", "physnet_switch", "physnet", "imunet"):
                windows[f"err_{k}_mm"].append(np.linalg.norm(r["preds"][k] - r["y"], axis=1) * 1000)
    np.savez_compressed(args.output.with_suffix(".npz"),
                        **{k: np.asarray(v) if k in ("condition", "session") else np.concatenate(v)
                           for k, v in windows.items()})
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
