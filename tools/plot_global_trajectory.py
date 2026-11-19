import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    t, A, B = [], [], []
    for r in rows:
        if r.get("G_A_tx_mm") and r.get("G_B_tx_mm"):
            t.append(float(r["stamp"]))
            A.append([float(r["G_A_t%s_mm" % ax]) for ax in "xyz"])
            B.append([float(r["G_B_t%s_mm" % ax]) for ax in "xyz"])
    return np.array(t), np.array(A), np.array(B)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for sess, color in (("gtraj_20261002_003129", "tab:blue"),):
    segdir = Path("datasets") / sess / "segments"
    for csv_path in sorted(segdir.glob("segment_*.csv")):
        t, A, B = load(csv_path)
        if len(t) < 5:
            continue
        t = t - t[0]
        axes[0].plot(A[:, 0], A[:, 1], ".", ms=1.5, label="rig A (board)" if csv_path.name.startswith("segment_01") else None)
        axes[0].plot(B[:, 0], B[:, 1], ".", ms=1.5, color="tab:red")
        axes[1].plot(t, np.linalg.norm(A - A[0], axis=1), "-", lw=1)
        axes[1].plot(t, np.linalg.norm(B - B[0], axis=1), "-", lw=1)

axes[0].set_xlabel("x (mm)")
axes[0].set_ylabel("y (mm)")
axes[0].set_title("Global view (4K): board positions, top-down")
axes[0].legend(["rig A board", "GP050 board"], markerscale=8)
axes[0].grid(alpha=0.3)
axes[1].set_xlabel("t (s) within segment")
axes[1].set_ylabel("distance from segment start (mm)")
axes[1].set_title("Motion magnitude over time")
axes[1].legend(["rig A", "rig B"])
axes[1].grid(alpha=0.3)
fig.tight_layout()
out = Path("datasets") / "global_trajectory_check.png"
fig.savefig(out, dpi=130)
print("saved", out)
