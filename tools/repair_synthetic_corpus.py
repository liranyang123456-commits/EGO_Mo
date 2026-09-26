#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regenerate failed Blender representatives with visibility-safe parameters."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ego_sim import SimConfig, build_sequence, export_sequence


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus", type=Path)
    args = ap.parse_args()
    manifest_path = args.corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    audit = json.loads((args.corpus / "audit.json").read_text(encoding="utf-8"))
    failed = {
        (row["backend"], row["motion"])
        for row in audit["visual"]
        if not row["passed"]
    }
    repaired = []
    for entry in manifest["blender"]:
        if ("blender", entry["motion"]) not in failed:
            repaired.append(entry)
            continue
        old = entry["config"]
        config = SimConfig(
            duration_s=old["duration_s"],
            camera_fps=2.0,
            imu_hz=200.0,
            width=1920,
            height=1080,
            seed=int(old["seed"]) + 100000,
            motion=entry["motion"],
            translation_mm=min(float(old["translation_mm"]), 12.0),
            depth_mm=135.0,
            depth_change_mm=min(float(old["depth_change_mm"]), 6.0),
            rotation_deg=min(float(old["rotation_deg"]), 4.0),
            speed=min(float(old["speed"]), 0.45),
            light_level=1.0,
            light_flicker=0.05,
            noise_scale=1.0,
            image_noise_std=1.5,
            motion_blur=0.12,
            render_backend="blender",
            render_images=True,
        )
        path = export_sequence(build_sequence(config), args.corpus / "blender")
        repaired.append({
            "name": path.name,
            "path": str(path.relative_to(args.corpus)),
            "motion": entry["motion"],
            "config": asdict(config),
            "replaces": entry["name"],
        })
        print(f"repaired {entry['motion']}: {path.name}", flush=True)
    manifest["blender"] = repaired
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
