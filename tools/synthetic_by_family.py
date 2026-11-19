#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthetic-only PhysNet error per motion family of the synthetic test/OOD sets.

Periodic families let a window estimator infer the initial velocity from the
context; the random-spline family does not, which makes it the closest
synthetic analogue of handheld motion.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"


def main():
    d = json.loads((DATA / "physnet_v1" / "eval_synonly_181622.json").read_text(encoding="utf-8"))
    out = {}
    for split in ("synthetic_test", "synthetic_ood"):
        fam = {}
        for path, m in d[f"{split}_by_session"].items():
            meta = json.loads(Path(path, "session_meta.json").read_text(encoding="utf-8")) \
                if os.path.isfile(os.path.join(path, "session_meta.json")) else {}
            name = meta.get("motion") or meta.get("config", {}).get("motion")
            fam.setdefault(name, []).append((m["err_mm"] * m["n"], m["zero_mm"] * m["n"], m["n"]))
        out[split] = {}
        for name, rows in fam.items():
            r = np.asarray(rows)
            out[split][name] = {"sequences": len(rows), "err_mm": float(r[:, 0].sum() / r[:, 2].sum()),
                                "zero_mm": float(r[:, 1].sum() / r[:, 2].sum()),
                                "reduction_percent": float(100 * (1 - r[:, 0].sum() / r[:, 1].sum()))}
    (DATA / "synthetic_by_family.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
