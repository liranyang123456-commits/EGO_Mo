#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Re-score the 2-, 3- and 4-s context ablations of Table III(A) on common windows.

A pair (t_a, t_b) is admissible only if t_a - c and t_b + c lie inside the
recording, so a longer context drops windows near the start and the end of
each recording. Table III(A) lists each context on its own window set (1436,
1400 and 1370 validation windows). This script uses the stored validation
predictions of those runs (datasets/physnet_v1/pred_{base,ctx3,ctx4}_s{0,1}.npz;
no retraining) and scores all three on the windows admissible at c = 4 s.

    python tools/context_common_windows.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
RUNS = {"2 s": "base", "3 s": "ctx3", "4 s": "ctx4"}


def load(name: str) -> tuple[dict, dict]:
    z = np.load(DATA / "physnet_v1" / f"pred_{name}.npz", allow_pickle=True)
    keys = [(str(s),) + tuple(np.round(y * 1e6).astype(np.int64)) for s, y in zip(z["val_session"], z["val_y"])]
    if len(set(keys)) != len(keys):
        raise AssertionError(f"{name}: validation windows are not uniquely identified by recording and label")
    err = np.linalg.norm(z["val_pred"] - z["val_y"], axis=1) * 1000
    zero = np.linalg.norm(z["val_y"], axis=1) * 1000
    return dict(zip(keys, err)), dict(zip(keys, zero))


def main() -> None:
    runs = {label: [load(f"{name}_s{s}") for s in (0, 1)] for label, name in RUNS.items()}
    common = set(runs["4 s"][0][0])
    for label, seeds in runs.items():
        for err, _zero in seeds:
            if not common <= set(err):
                raise AssertionError(f"context {label}: windows of the 4-s set are missing")
    out = {"common_windows": len(common), "definition": "validation windows admissible at c = 4 s", "contexts": {}}
    for label, seeds in runs.items():
        own = [float(np.mean(list(err.values()))) for err, _z in seeds]
        com = [float(np.mean([err[k] for k in common])) for err, _z in seeds]
        zero = seeds[0][1]
        out["contexts"][label] = {
            "own_windows": len(seeds[0][0]),
            "own_err_mm": round(float(np.mean(own)), 2), "own_zero_mm": round(float(np.mean(list(zero.values()))), 2),
            "common_err_mm": round(float(np.mean(com)), 2),
            "common_err_mm_seeds": [round(v, 2) for v in com],
            "common_zero_mm": round(float(np.mean([zero[k] for k in common])), 2),
        }
    (DATA / "context_common_windows.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
