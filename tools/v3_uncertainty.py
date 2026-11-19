#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-hoc calibration variants for the per-prediction uncertainty.

Uses the fixed-split NLL ensemble. Every variant is fitted on validation only
and scored (a) across the two validation recordings and (b) on the historical
held-out test recording, which these fixed-split models never trained on:
  global     one scale kappa (the manuscript method);
  stratified one scale per tercile of the predicted sigma;
  conformal  split-conformal: the empirical 68.3/95.4% quantiles of |z| on
             validation replace the Gaussian 1 and 2.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import tools.train_seq as api
import tools.train_physnet as phys
from tools.eval_uncertainty import collect

DATA = ROOT / "datasets"
MODELS = ["physnet_a/physnet_nll_s0.pt", "physnet_a/physnet_nll_s1.pt"]
TEST = "traj_20260923_023422"


def _z(part):
    d, p, a, e = part
    return (p - d["y"]) / np.sqrt(np.maximum(a + e, 1e-12)), np.sqrt(a + e)


class Global:
    def fit(self, z, s):
        self.k = float(np.sqrt(np.mean(z ** 2)))
        return self

    def cover(self, z, s, n):
        return float((np.abs(z) <= n * self.k).mean())


class Stratified:
    def fit(self, z, s):
        self.edges = np.quantile(s, [1 / 3, 2 / 3])
        b = np.digitize(s, self.edges)
        self.k = [float(np.sqrt(np.mean(z[b == i] ** 2))) for i in range(3)]
        return self

    def cover(self, z, s, n):
        b = np.digitize(s, self.edges)
        k = np.asarray(self.k)[b]
        return float((np.abs(z) <= n * k).mean())


class Conformal:
    def fit(self, z, s):
        a = np.abs(z).ravel()
        self.q = {1: float(np.quantile(a, 0.683)), 2: float(np.quantile(a, 0.954))}
        return self

    def cover(self, z, s, n):
        return float((np.abs(z) <= self.q[n]).mean())


class Magnitude:
    """sigma_eff = kappa * sigma * (1 + gamma * |d_hat| / median|d_hat|).

    gamma makes the standardized-residual variance equal across terciles of the
    predicted magnitude; kappa is then the RMS standardized residual."""

    def fit(self, z, s, mag):
        self.med = float(np.median(mag))
        b = np.digitize(mag, np.quantile(mag, [1 / 3, 2 / 3]))
        best = None
        for g in np.linspace(0, 3, 61):
            f = (1 + g * mag / self.med)[:, None]
            v = [float(np.mean((z[b == i] / f[b == i]) ** 2)) for i in range(3)]
            score = float(np.std(np.log(v)))
            if best is None or score < best[0]:
                best = (score, g)
        self.g = best[1]
        f = (1 + self.g * mag / self.med)[:, None]
        self.k = float(np.sqrt(np.mean((z / f) ** 2)))
        return self

    def cover(self, z, s, mag, n):
        f = (1 + self.g * mag / self.med)[:, None]
        return float((np.abs(z) <= n * self.k * f).mean())


def main():
    api.LABELS, api.TARGET_S = "pose_gt_raw", 3.0
    length = int(round((1.5 * 3.0 + 2 * 2.0) * phys.HZ))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = [phys.load_model(DATA / m, device) for m in MODELS]
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    raw = {n: collect(models, [n], 2.0, 3.0, length, device) for n in split["val"] + [TEST]}
    parts = {n: _z(raw[n]) for n in raw}
    val = split["val"]
    report = {}
    for name, cls in (("global", Global), ("stratified", Stratified), ("conformal", Conformal)):
        cross = {1: [], 2: []}
        for held in val:
            fit = [n for n in val if n != held]
            zf = np.concatenate([parts[n][0] for n in fit]); sf = np.concatenate([parts[n][1] for n in fit])
            c = cls().fit(zf, sf)
            for n in (1, 2):
                cross[n].append(c.cover(*parts[held], n) if False else c.cover(parts[held][0], parts[held][1], n))
        zv = np.concatenate([parts[n][0] for n in val]); sv = np.concatenate([parts[n][1] for n in val])
        c = cls().fit(zv, sv)
        report[name] = {
            "cross_session_1sigma": float(np.mean(cross[1])), "cross_session_2sigma": float(np.mean(cross[2])),
            "test_1sigma": c.cover(parts[TEST][0], parts[TEST][1], 1),
            "test_2sigma": c.cover(parts[TEST][0], parts[TEST][1], 2),
        }
        print(name, {k: round(v, 3) for k, v in report[name].items()}, flush=True)
    mags = {n: np.linalg.norm(raw[n][1], axis=1) for n in raw}
    cross = {1: [], 2: []}
    for held in val:
        fit = [n for n in val if n != held]
        c = Magnitude().fit(np.concatenate([parts[n][0] for n in fit]),
                            np.concatenate([parts[n][1] for n in fit]),
                            np.concatenate([mags[n] for n in fit]))
        for n in (1, 2):
            cross[n].append(c.cover(parts[held][0], parts[held][1], mags[held], n))
    c = Magnitude().fit(np.concatenate([parts[n][0] for n in val]),
                        np.concatenate([parts[n][1] for n in val]),
                        np.concatenate([mags[n] for n in val]))
    report["magnitude"] = {
        "gamma": c.g, "kappa": c.k,
        "cross_session_1sigma": float(np.mean(cross[1])), "cross_session_2sigma": float(np.mean(cross[2])),
        "test_1sigma": c.cover(parts[TEST][0], parts[TEST][1], mags[TEST], 1),
        "test_2sigma": c.cover(parts[TEST][0], parts[TEST][1], mags[TEST], 2),
    }
    print("magnitude", {k: round(v, 3) for k, v in report["magnitude"].items()}, flush=True)
    (DATA / "v3_uncertainty.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
