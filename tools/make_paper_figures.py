#!/usr/bin/env python3
"""Generate reproducible figures for the TIM manuscript."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "figure.dpi": 180,
    "savefig.bbox": "tight",
})


def _box(ax, x, y, w, h, title, body, face="#eef4fb", edge="#35689a", lw=1.1,
         title_size=8, body_size=7, dashed=False, body_linespacing=1.15):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.006", facecolor=face, edgecolor=edge,
        linewidth=lw, linestyle="--" if dashed else "-", zorder=2))
    if title:
        ax.text(x + w / 2, y + h - 0.055 * h - 0.012, title, ha="center", va="top",
                weight="bold", fontsize=title_size, zorder=3)
    if body:
        # Body sits slightly below the centre when a title occupies the top.
        ax.text(x + w / 2, y + h * (0.42 if title else 0.5), body, ha="center",
                va="center", linespacing=body_linespacing, fontsize=body_size, zorder=3)


def _arrow(ax, start, end, color="#4c5b6a", lw=1.0, style="-|>", dashed=False,
           rad=0.0, ms=13):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle=style, mutation_scale=ms, color=color,
        linewidth=max(lw, 1.15),
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}", zorder=4))


def framework():
    """Single-column system diagram in symbols, read top to bottom in five bands."""
    fig, ax = plt.subplots(figsize=(3.45, 3.60))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xl, xr, wh = 0.012, 0.512, 0.476          # two half-width columns
    full = 0.976

    def head(x, y, h, text, colour="#333333"):
        ax.text(x + 0.018, y + h - 0.009, text, ha="left", va="top", weight="bold",
                fontsize=6.4, color=colour)

    # --- 1 instrument: the three raw signals -------------------------------
    y1, h1 = 0.874, 0.116
    _box(ax, xl, y1, full, h1, "", "", face="#f4f6f9", edge="#7a8798")
    head(xl, y1, h1, "Instrument")
    for x, sym, note in ((0.185, r"$\{{\bf I}_L,{\bf I}_R\}$", "stereo, 10-30 Hz"),
                         (0.500, r"$\tilde{\bf f}_I,\ \tilde{\omega}_I$", "scope IMU, 200 Hz"),
                         (0.815, r"$B,\ \tilde{\omega}_B$", "board witness")):
        ax.text(x, y1 + 0.056, sym, ha="center", va="center", fontsize=7.2)
        ax.text(x, y1 + 0.022, note, ha="center", va="center", fontsize=4.9,
                color="#555555")

    # --- 2 the two measurement paths as operators --------------------------
    y2, h2 = 0.680, 0.150
    _box(ax, xl, y2, wh, h2, "", "", face="#eaf3ea", edge="#3f8a4f")
    head(xl, y2, h2, "Optical reference", "#2f6b3c")
    ax.text(xl + wh / 2, y2 + 0.074, r"${\bf T}_{BC}\leftarrow\mathrm{PnP}({\bf I}_L)$",
            ha="center", va="center", fontsize=6.6)
    ax.text(xl + wh / 2, y2 + 0.034,
            r"$e_{\rm px}\!\leq\!1.5,\ \|\tilde{\omega}_B\|\!<\!8^{\circ}\!/\mathrm{s}$",
            ha="center", va="center", fontsize=5.6)
    _box(ax, xr, y2, wh, h2, "", "", face="#fdf0e8", edge="#c4642f")
    head(xr, y2, h2, "Inertial path", "#a4501f")
    ax.text(xr + wh / 2, y2 + 0.080,
            r"${\bf R}_{a\leftarrow k}\!=\!{\bf R}_{CI}{\bf Q}_a^{\top}"
            r"{\bf Q}_k{\bf R}_{CI}^{\top}$", ha="center", va="center", fontsize=5.8)
    ax.text(xr + wh / 2, y2 + 0.038,
            r"$\ell_k\!=\!{\bf R}_{a\leftarrow k}{\bf R}_{CI}\tilde{\bf f}_{I,k}"
            r"-\hat{\mu}_f$", ha="center", va="center", fontsize=5.8)

    # --- 3 the decomposition ------------------------------------------------
    y3, h3 = 0.474, 0.166
    _box(ax, xl, y3, full, h3, "", "", face="#fffaf0", edge="#b8860b", lw=1.3)
    head(xl, y3, h3, "What each path can supply", "#8a5a00")
    ax.text(0.5, y3 + 0.098,
            r"$\Delta{\bf p}_{ab}\;=\;{\bf v}(t_a)\,\Delta t"
            r"\;+\;\iint{\bf a}\,\mathrm{d}\tau\,\mathrm{d}s$",
            ha="center", va="center", fontsize=8.2)
    for xa, xb in ((0.300, 0.434), (0.494, 0.704)):
        ax.plot([xa, xa, xb, xb],
                [y3 + 0.070, y3 + 0.058, y3 + 0.058, y3 + 0.070],
                color="#b8860b", linewidth=0.7)
    ax.text(0.367, y3 + 0.034, "unobservable", ha="center", va="center", fontsize=5.2,
            color="#8a5a00")
    ax.text(0.367, y3 + 0.013, r"prior or pause", ha="center", va="center",
            fontsize=4.9, color="#8a5a00")
    ax.text(0.599, y3 + 0.034, "measured", ha="center", va="center", fontsize=5.2,
            color="#8a5a00")
    ax.text(0.599, y3 + 0.013,
            r"$\pm{\bf b},\ \pm s,\ \pm\delta\theta\!\times\!{\bf g}$",
            ha="center", va="center", fontsize=4.9, color="#8a5a00")

    # --- 4 twin and estimator ----------------------------------------------
    y4, h4 = 0.268, 0.162
    _box(ax, xl, y4, wh, h4, "", "", face="#f3eefb", edge="#6a4c93")
    head(xl, y4, h4, "Digital twin", "#553a78")
    ax.text(xl + wh / 2, y4 + 0.090,
            r"$\Sigma,\ {\bf b}\!\sim\!\mathrm{RW},\ Q,\ \delta t,\ {\bf g}_B$",
            ha="center", va="center", fontsize=5.6)
    ax.text(xl + wh / 2, y4 + 0.056, "$+$ real-trajectory replay", ha="center",
            va="center", fontsize=5.0)
    ax.text(xl + wh / 2, y4 + 0.022,
            r"$\rightarrow\{\tilde{\bf f},\tilde{\omega}\}_{\rm sim}$",
            ha="center", va="center", fontsize=6.0)
    _box(ax, xr, y4, wh, h4, "", "", face="#eef4fb", edge="#35689a")
    head(xr, y4, h4, "PhysNet", "#2c5170")
    ax.text(xr + wh / 2, y4 + 0.092, r"$\hat{\bf v}_k[1-\sigma(s_k)]$", ha="center",
            va="center", fontsize=6.0)
    ax.text(xr + wh / 2, y4 + 0.056,
            r"$\hat{\bf d}_k\!=\!f_s^{-1}\!\sum\hat{\bf v}"
            r"+({\bf R}\!-\!{\bf I})\lambda$", ha="center", va="center", fontsize=5.6)
    ax.text(xr + wh / 2, y4 + 0.022, r"$\rightarrow\hat{\bf d}(t_b),\ u(\hat{\bf d})$",
            ha="center", va="center", fontsize=5.8)

    # --- 5 audits and evaluation -------------------------------------------
    y5, h5 = 0.068, 0.158
    _box(ax, xl, y5, wh, h5, "", "", face="#f4f6f9", edge="#7a8798")
    head(xl, y5, h5, "Audits")
    for k, line in enumerate((r"$u_A,\ \Sigma_{\rm corner}$",
                              r"$s_p\!=\!\|\delta{\bf p}\|/\|\delta\theta\|$",
                              r"$r_{\rm axis}\!\geq\!0.75$",
                              "ideal/noisy, corners")):
        ax.text(xl + wh / 2, y5 + 0.092 - 0.027 * k, line, ha="center", va="center",
                fontsize=5.0)
    _box(ax, xr, y5, wh, h5, "", "", face="#eef4fb", edge="#35689a")
    head(xr, y5, h5, "Evaluation", "#2c5170")
    for k, line in enumerate((r"$e_p,\ R^2$, ATE",
                              "synthetic / OOD / direct",
                              "8-fold session CV",
                              "held-out test, 95% CI")):
        ax.text(xr + wh / 2, y5 + 0.092 - 0.027 * k, line, ha="center", va="center",
                fontsize=5.0)

    # --- flow ---------------------------------------------------------------
    for x, colour in ((0.24, "#3f8a4f"), (0.76, "#c4642f")):
        _arrow(ax, (x, y1 - 0.004), (x, y2 + h2 + 0.004), color=colour,
               lw=1.35, ms=16)
        _arrow(ax, (x, y2 - 0.004), (x, y3 + h3 + 0.004), color=colour,
               lw=1.35, ms=16)
    _arrow(ax, (0.24, y3 - 0.004), (0.24, y4 + h4 + 0.004),
           color="#6a4c93", lw=1.35, ms=16)
    _arrow(ax, (0.76, y3 - 0.004), (0.76, y4 + h4 + 0.004),
           color="#35689a", lw=1.35, ms=16)
    _arrow(ax, (xl + wh + 0.004, y4 + h4 / 2),
           (xr - 0.004, y4 + h4 / 2), color="#6a4c93", lw=1.35, ms=16)
    _arrow(ax, (0.24, y4 - 0.004), (0.24, y5 + h5 + 0.004),
           color="#7a8798", lw=1.35, ms=16)
    _arrow(ax, (0.76, y4 - 0.004), (0.76, y5 + h5 + 0.004),
           color="#35689a", lw=1.35, ms=16)

    # barrier: nothing flows back from the held-out evaluation
    ax.plot([0.50, 0.50], [y5 - 0.008, y5 + h5 + 0.020], color="#cc3333", linewidth=0.9,
            linestyle=(0, (4, 3)), zorder=5)
    ax.text(0.5, 0.044, "green: optical    orange: inertial    purple: simulation    "
                        "blue: estimation", ha="center", va="top", fontsize=4.9,
            color="#444444")
    ax.text(0.5, 0.015, "red: nothing crosses back from the held-out evaluation",
            ha="center", va="top", fontsize=4.9, color="#cc3333")
    fig.savefig(OUT / "framework.pdf")
    plt.close(fig)


def architecture():
    """Single-column PhysNet diagram, read top to bottom in six bands."""
    fig, ax = plt.subplots(figsize=(3.45, 4.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    xl, full = 0.012, 0.976

    # ---- 1 input window ----------------------------------------------------
    y1, h1 = 0.888, 0.108
    _box(ax, xl, y1, full, h1, "", "", face="#f7f8fa", edge="#7a8798")
    ax.text(0.030, y1 + h1 - 0.008, "Input window", ha="left", va="top", weight="bold",
            fontsize=7.0)
    rng = np.random.default_rng(3)
    tt = np.linspace(0, 1, 400)
    wx0, wxw = 0.115, 0.560
    for k, (colour, amp) in enumerate((("#35689a", 0.011), ("#3f8a4f", 0.009),
                                       ("#c4642f", 0.012))):
        sig = np.convolve(rng.normal(0, 1, 400), np.ones(30) / 30, mode="same")
        sig[150:230] *= 0.10                       # a still phase
        ax.plot(wx0 + wxw * tt, y1 + 0.022 + 0.022 * k
                + amp * sig / (np.abs(sig).max() + 1e-9),
                color=colour, linewidth=0.6, zorder=3)
    ax.add_patch(plt.Rectangle((wx0 + wxw * 0.375, y1 + 0.012), wxw * 0.20, 0.064,
                               facecolor="#999999", alpha=0.18, zorder=1))
    ax.text(0.030, y1 + 0.046, "$\\tilde{\\bf f}_I,\\ \\tilde{\\omega}_I$\n200 Hz",
            ha="left", va="center", fontsize=5.6, linespacing=1.3)
    ax.text(0.700, y1 + 0.046, "$[t_a-2\\,\\mathrm{s},\\,t_b+2\\,\\mathrm{s}]$\n"
                               "grey: a still phase",
            ha="left", va="center", fontsize=5.4, linespacing=1.3)

    # ---- 2 deterministic pre-integration ----------------------------------
    y2, h2 = 0.712, 0.158
    _box(ax, xl, y2, full, h2, "", "", face="#fdf0e8", edge="#c4642f", lw=1.2)
    ax.text(0.030, y2 + h2 - 0.008, "Anchor-frame pre-integration", ha="left", va="top",
            weight="bold", fontsize=7.0)
    lines = [
        r"${\bf R}_{a\leftarrow k}={\bf R}_{CI}{\bf Q}_a^{\top}{\bf Q}_k{\bf R}_{CI}^{\top},"
        r"\quad{\bf f}^{C_a}_k={\bf R}_{a\leftarrow k}{\bf R}_{CI}\tilde{\bf f}_{I,k}$",
        r"$\hat{\mu}_f=\langle{\bf f}^{C_a}_k\rangle_{\rm window},\quad"
        r"\ell_k={\bf f}^{C_a}_k-\hat{\mu}_f$",
        r"$\Delta{\bf v}_k=\Sigma\ell,\quad\Delta{\bf d}_k=\Sigma\Sigma\ell$",
    ]
    for i, line in enumerate(lines):
        ax.text(0.5, y2 + h2 - 0.048 - 0.034 * i, line, ha="center", va="center",
                fontsize=6.0)
    ax.text(0.5, y2 + 0.018, "no learned parameters, 24 channels", ha="center",
            va="center", fontsize=5.8, color="#8a5a00")

    # ---- 3 shared trunk ----------------------------------------------------
    y3, h3 = 0.556, 0.136
    _box(ax, xl, y3, full, h3, "", "", face="#eef4fb", edge="#35689a", lw=1.2)
    ax.text(0.030, y3 + h3 - 0.008, "Shared trunk", ha="left", va="top", weight="bold",
            fontsize=7.0)
    cells = [("Conv 7, stride 2", 0, 0), ("Conv 5, stride 2", 1, 0),
             ("$5\\times$ dilated residual, $d=1..16$", 0, 1), ("BiGRU, 96 units", 1, 1)]
    for text, col, row in cells:
        bx = 0.030 + col * 0.482
        by = y3 + 0.058 - row * 0.038
        ax.add_patch(FancyBboxPatch((bx, by), 0.452, 0.032, boxstyle="round,pad=0.002",
                                    facecolor="#ffffff", edgecolor="#9db6ce",
                                    linewidth=0.7, zorder=3))
        ax.text(bx + 0.226, by + 0.016, text, ha="center", va="center", fontsize=5.4,
                zorder=4)
    ax.text(0.5, y3 + 0.019, "200 to 50 Hz, 0.42 M parameters", ha="center", va="center",
            fontsize=5.8, color="#2c5170")

    # ---- 4 three heads -----------------------------------------------------
    y4, h4 = 0.402, 0.134
    _box(ax, xl, y4, full, h4, "", "", face="#f7f8fa", edge="#7a8798")
    ax.text(0.030, y4 + h4 - 0.008, "Three heads", ha="left", va="top", weight="bold",
            fontsize=7.0)
    heads = [("Velocity", r"$\hat{\bf v}_k$, upsampled to 200 Hz", "#35689a"),
             ("Stillness", r"$\sigma(s_k)$, BCE on reference speed", "#3f8a4f"),
             ("Log-variance", r"$\eta_k\rightarrow\sigma_k$, Gaussian NLL", "#6a4c93")]
    for k, (name, sym, colour) in enumerate(heads):
        by = y4 + h4 - 0.052 - 0.031 * k
        ax.add_patch(FancyBboxPatch((0.030, by - 0.014), 0.916, 0.028,
                                    boxstyle="round,pad=0.002", facecolor="#ffffff",
                                    edgecolor=colour, linewidth=0.8, zorder=3))
        ax.text(0.042, by, name, ha="left", va="center", fontsize=5.6, weight="bold",
                color=colour, zorder=4)
        ax.text(0.300, by, sym, ha="left", va="center", fontsize=5.6, zorder=4)

    # ---- 5 structural kinematics ------------------------------------------
    y5, h5 = 0.236, 0.146
    _box(ax, xl, y5, full, h5, "", "", face="#f3f7fb", edge="#35689a", lw=1.2)
    ax.text(0.030, y5 + h5 - 0.008,
            "Structural kinematics (no free parameters except $\\lambda$)",
            ha="left", va="top", fontsize=6.8, weight="bold")
    ax.text(0.5, y5 + h5 - 0.048,
            r"$\hat{\bf v}_k\leftarrow\hat{\bf v}_k\,[1-\sigma(s_k)]$"
            "    (stillness-conditioned gate)", ha="center", va="center", fontsize=6.2)
    ax.text(0.5, y5 + h5 - 0.090,
            r"$\hat{\bf d}_k=\frac{1}{f_s}\sum_{i=a}^{k}\hat{\bf v}_i"
            r"+({\bf R}_{a\leftarrow k}-{\bf I})\,\lambda,\quad\hat{\bf d}(t_a)\equiv{\bf 0}$",
            ha="center", va="center", fontsize=6.6)
    ax.text(0.5, y5 + 0.017,
            r"$u(\hat{\bf d})=\kappa\sqrt{\sigma_k^2+s^2_{\rm ens}}$,"
            " $\\kappa$ fitted on validation", ha="center", va="center", fontsize=6.0,
            color="#2c5170")

    for k, (ya, yb_, colour) in enumerate((
            (y1, y2 + h2, "#7a8798"), (y2, y3 + h3, "#c4642f"),
            (y3, y4 + h4, "#35689a"), (y4, y5 + h5, "#3f8a4f"))):
        _arrow(ax, (0.5, ya - 0.003), (0.5, yb_ + 0.003), color=colour, lw=0.9)

    # ---- 6 supervised output stream ---------------------------------------
    ax.text(0.5, 0.206, "Output: one displacement stream per window,\n"
                        "supervised at every reference frame inside it",
            ha="center", va="top", fontsize=6.6, weight="bold", linespacing=1.35)
    x0, x1, yb = 0.085, 0.885, 0.058
    ax.plot([x0, x1], [yb, yb], color="#7a8798", linewidth=0.8)
    tt = np.linspace(0, 1, 300)
    curve = yb + 0.098 * (0.55 * 0.5 * (1 - np.cos(1.9 * np.pi * tt)) + 0.35 * tt)
    xx = x0 + (x1 - x0) * tt
    anchor_x = x0 + (x1 - x0) * 0.285
    end_x = x0 + (x1 - x0) * 0.715
    ax.add_patch(plt.Rectangle((anchor_x, yb - 0.008), end_x - anchor_x, 0.135,
                               facecolor="#35689a", alpha=0.07, zorder=1))
    ax.plot(xx, curve, color="#35689a", linewidth=1.3, zorder=3)
    for frac in np.linspace(0.05, 0.95, 14):
        xi = x0 + (x1 - x0) * frac
        yi = np.interp(xi, xx, curve)
        ax.plot([xi, xi], [yb, yi], color="#3f8a4f", linewidth=0.4, alpha=0.55, zorder=2)
        ax.plot([xi], [yi], marker="o", markersize=2.0, color="#3f8a4f", zorder=5)
    for xi, label in ((anchor_x, "$t_a$"), (end_x, "$t_b$")):
        ax.plot([xi, xi], [yb - 0.012, yb + 0.118], color="#cc3333", linewidth=0.8,
                linestyle=(0, (3, 2)), zorder=4)
        ax.text(xi, yb - 0.017, label, ha="center", va="top", fontsize=6.4,
                color="#cc3333")
    ax.text(x1 + 0.005, np.interp(x1, xx, curve), r"$\hat{\bf d}(t)$", ha="left",
            va="center", fontsize=6.6, color="#35689a")
    ax.text(x0 - 0.005, yb, "0", ha="right", va="center", fontsize=6.0, color="#7a8798")
    ax.text(0.5, 0.018, "green: reference displacement at each usable chessboard frame, "
                        "about 145 per window", ha="center", va="top", fontsize=5.4,
            color="#444444")
    _arrow(ax, (0.5, y5 - 0.003), (0.5, 0.212), color="#35689a", lw=0.9)
    fig.savefig(OUT / "architecture.pdf")
    plt.close(fig)


def dataset(session="traj_20260924_143156", stream_session="traj_20260923_020132"):
    """Qualitative view of the corpus: real frames, renders, inertial stream, motion range."""
    import cv2
    import json as _json

    fig = plt.figure(figsize=(3.45, 5.35))
    outer = fig.add_gridspec(2, 1, height_ratios=(1.55, 3.0), hspace=0.32,
                             left=0.155, right=0.975, top=0.955, bottom=0.065)
    top_gs = outer[0].subgridspec(2, 2, hspace=0.28, wspace=0.18)
    plot_gs = outer[1].subgridspec(3, 1, hspace=0.82)

    def board_crop(path, corners=True, size=(220, 150)):
        """Board-centred crop with the detected corners burned into the image."""
        img = cv2.imread(str(path))
        if img is None:
            return np.zeros((size[1], size[0], 3), np.uint8)
        found, pts = cv2.findChessboardCorners(
            cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (11, 8),
            cv2.CALIB_CB_ADAPTIVE_THRESH,
        )
        if found:
            pts = pts.reshape(-1, 2)
            pad = 0.50 * max(np.ptp(pts[:, 0]), np.ptp(pts[:, 1]))
            x0 = int(max(pts[:, 0].min() - pad, 0))
            x1 = int(min(pts[:, 0].max() + pad, img.shape[1]))
            y0 = int(max(pts[:, 1].min() - pad, 0))
            y1 = int(min(pts[:, 1].max() + pad, img.shape[0]))
            if corners:
                for x, y in pts:
                    cv2.circle(img, (int(round(x)), int(round(y))), 2, (80, 255, 80), -1)
            img = img[y0:y1, x0:x1]
        return cv2.resize(img, size, interpolation=cv2.INTER_AREA)

    def montage(paths, detect=True):
        tiles = [board_crop(p, detect) for p in paths[:4]]
        while len(tiles) < 4:
            tiles.append(np.zeros_like(tiles[0]))
        return np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:4])))

    # ---- rows 1-2: four frames per acquisition domain ---------------------
    groups = [
        ("traj_20260923_015507", (120, 500, 900, 1300), "Real, 30 fps"),
        ("traj_20260924_143156", (150, 600, 1050, 1500), "Real, 10 fps"),
        ("rigid_20260924_142902", (80, 260, 460, 680), "Rigidity recording"),
    ]
    for col, (name, indices, label) in enumerate(groups):
        ax = fig.add_subplot(top_gs[col // 2, col % 2])
        paths = [DATA / name / "cam0" / "images" / f"{i:06d}.jpg" for i in indices]
        ax.imshow(cv2.cvtColor(montage(paths), cv2.COLOR_BGR2RGB))
        ax.set_title(f"({chr(97 + col)}) {label}: four frames", fontsize=6.0, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    # Four twin renders from different configurations.
    ax = fig.add_subplot(top_gs[1, 1])
    cand = sorted(DATA.glob("synthetic_blender_smoke/*/cam0/images/*.jpg"))
    if len(cand) < 4:
        cand += sorted(DATA.glob("synthetic_2k_smoke/*/cam0/images/*.jpg"))
    ax.imshow(cv2.cvtColor(montage(cand[:4], detect=False), cv2.COLOR_BGR2RGB))
    ax.set_title("(d) Digital twin: four rendered frames", fontsize=6.0, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])

    # ---- row 2 left: inertial stream with still phases --------------------
    import tools.train_physnet as phys
    ses = phys.Session(stream_session)
    t, Rg, pg, usable = phys._load_gt(stream_session)
    vg, vok = phys.gt_velocity(t, pg, usable)
    speed = np.linalg.norm(vg, axis=1) * 1000.0
    # contiguous still runs, so the shading shows pauses rather than single frames
    sel_all = usable[vok[usable]]
    flag = speed[sel_all] < phys.STILL_MM_S
    runs, k = [], 0
    while k < len(flag):
        if not flag[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(flag) and flag[j + 1] and t[sel_all[j + 1]] - t[sel_all[j]] < 0.35:
            j += 1
        runs.append((float(t[sel_all[k]]), float(t[sel_all[j]])))
        k = j + 1
    span = 16.0
    # a short pause with at least 5 s of reference motion on both sides
    t_lo, t_hi = t[usable[0]], t[usable[-1]]
    mid = [r for r in runs if 0.4 <= r[1] - r[0] <= 2.0
           and r[0] - t_lo > 12.0 and t_hi - r[1] > 6.0]
    centre = 0.5 * sum(mid[len(mid) // 2]) if mid else t_lo + 0.5 * span + 12.0
    t0 = centre - span / 2
    win = (ses.t >= t0) & (ses.t <= t0 + span)
    sel = usable[(t[usable] >= t0) & (t[usable] <= t0 + span)]
    still = sel[(speed[sel] < phys.STILL_MM_S) & vok[sel]]
    shade = [(max(a, t0), min(b, t0 + span)) for a, b in runs
             if b > t0 and a < t0 + span]

    ax = fig.add_subplot(plot_gs[0])
    lin = ses.acc[win] - ses.acc[win].mean(0)          # gravity-free dynamics
    for k, (colour, label) in enumerate((("#c4642f", "$a_x$"), ("#3f8a4f", "$a_y$"),
                                         ("#6a4c93", "$a_z$"))):
        ax.plot(ses.t[win] - t0, lin[:, k], linewidth=0.45, color=colour, label=label)
    ax.plot(ses.t[win] - t0, np.linalg.norm(np.degrees(ses.gyro[win]), axis=1) / 40.0,
            linewidth=0.5, color="#35689a", label=r"$\|\omega\|/40$")
    dt_frame = float(np.median(np.diff(t[usable])))
    for a, b in shade:
        ax.axvspan(a - t0 - dt_frame / 2, b - t0 + dt_frame / 2,
                   color="#777777", alpha=0.30, lw=0, zorder=0)
    ax.set_xlabel("Time (s)", labelpad=1)
    ax.set_ylabel(r"m s$^{-2}$ (demeaned)")
    ax.set_title("(e) Inertial stream at 200 Hz; grey: detected pauses", fontsize=7, pad=3)
    ax.legend(fontsize=5.4, ncol=4, loc="upper center", handlelength=1.0,
              columnspacing=0.7, borderpad=0.25, framealpha=0.85)
    ax.margins(x=0.01)
    ax.grid(alpha=0.2)

    # ---- row 2 right: reference speed with the stillness threshold --------
    ax = fig.add_subplot(plot_gs[1])
    ax.plot(t[sel] - t0, np.maximum(speed[sel], 0.2), linewidth=0.6, color="#2c5170")
    ax.plot(t[still] - t0, np.maximum(speed[still], 0.2), "o", markersize=2.2,
            color="#3f8a4f", label="labelled still")
    ax.axhline(phys.STILL_MM_S, color="#3f8a4f", linestyle="--", linewidth=0.8,
               label=f"{phys.STILL_MM_S:.0f} mm/s")
    ax.axhline(phys.MOVE_MM_S, color="#cc3333", linestyle=":", linewidth=0.8,
               label=f"{phys.MOVE_MM_S:.0f} mm/s")
    ax.set_yscale("log")
    ax.set_xlabel("Time (s)", labelpad=1)
    ax.set_ylabel("Speed (mm/s)")
    ax.set_title("(f) Reference speed and the stillness labels", fontsize=7, pad=3)
    ax.legend(fontsize=5.4, loc="lower left", borderpad=0.25, framealpha=0.85,
              handlelength=1.2)
    ax.margins(x=0.01)
    ax.grid(alpha=0.2, which="both")

    # ---- row 2 far right: measurand distribution --------------------------
    ax = fig.add_subplot(plot_gs[2])
    split = _json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    bins = np.linspace(0, 160, 33)
    for names, label, colour in (
        (split["train"], "train", "#35689a"),
        (split["val"], "val.", "#3f8a4f"),
        (split["test"], "held-out test", "#cc3333"),
        (split.get("extra_train", []), "rigidity", "#c4642f"),
    ):
        mags = []
        for name in names:
            g = phys.build_group([name], 2.0, 3.0, 1700)
            mags.append(np.linalg.norm(g["y"], axis=1) * 1000.0)
        mags = np.concatenate(mags)
        ax.hist(mags, bins=bins, density=True, histtype="step", linewidth=0.9,
                color=colour, label=f"{label}, {mags.mean():.0f} mm")
    ax.set_xlabel("3-s displacement (mm)", labelpad=1)
    ax.set_ylabel("Density")
    ax.set_title("(g) Measurand per split", fontsize=7, pad=3)
    ax.legend(fontsize=5.0, borderpad=0.2, framealpha=0.85, handlelength=1.0,
              labelspacing=0.25)
    ax.grid(alpha=0.2)

    fig.savefig(OUT / "dataset.pdf")
    plt.close(fig)


def uncertainty():
    report = json.loads((DATA / "traj_run_v8" / "gt_uncertainty.json").read_text(encoding="utf-8"))
    val = report["traj_20260923_023241"]
    test = report["traj_20260923_023422"]
    sigmas = [0.1, 0.25, 0.5]
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.95))
    for block, label, marker in ((val, "val.", "o"), (test, "test", "s")):
        pos = [block["monte_carlo"][f"{s:.2f}px"]["raw_pos_mm_med"] for s in sigmas]
        rot = [block["monte_carlo"][f"{s:.2f}px"]["raw_rot_deg_med"] for s in sigmas]
        axes[0].plot(sigmas, pos, marker=marker, markersize=2.8, linewidth=0.8,
                     label=f"{label}: pos.")
        axes[0].plot(sigmas, rot, marker=marker, markersize=2.8, linewidth=0.8,
                     linestyle="--", label=f"{label}: rot.")
    axes[0].set_xlabel("Corner noise (px)", labelpad=1)
    axes[0].set_ylabel("Spread (mm or deg)")
    axes[0].set_title("(a) Monte Carlo", fontsize=6.8)
    axes[0].tick_params(labelsize=5.4)
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=4.8, borderpad=0.2, handlelength=1.2, labelspacing=0.2)
    labels = ["Tilt $x$", "Tilt $y$", "Optical"]
    x = np.arange(3)
    mm = test["rotation_sensitivity"]["mm_per_deg"]
    px = test["rotation_sensitivity"]["px_per_deg"]
    axes[1].bar(x - 0.18, mm, 0.36, label="mm/deg")
    axes[1].bar(x + 0.18, px, 0.36, label="px/deg")
    axes[1].set_xticks(x, labels, fontsize=5.2)
    axes[1].set_title("(b) Tilt sensitivity", fontsize=6.8)
    axes[1].tick_params(axis="y", labelsize=5.4)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=5.0, borderpad=0.2, handlelength=1.0)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "reference_uncertainty.pdf")
    plt.close(fig)


def domain_gap():
    labels = ["p50", "p90", "p99"]
    real_acc = [0.0072, 0.0241, 0.0560]
    initial_acc = [0.0017, 0.0069, 0.0174]
    tuned_acc = [0.0036, 0.0232, 0.0769]
    real_gyro = [4.25, 18.48, 44.96]
    initial_gyro = [14.11, 30.83, 51.14]
    tuned_gyro = [3.98, 11.57, 39.35]
    import cv2
    fig = plt.figure(figsize=(3.45, 3.45))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.25, 1.0), hspace=0.46,
                          wspace=0.42, left=0.145, right=0.975,
                          top=0.945, bottom=0.085)
    # A real screenshot of the implemented workbench. The full control panel,
    # stereo views, 3-D path and all-axis IMU plot remain visible.
    ax_gui = fig.add_subplot(gs[0, :])
    gui_path = OUT / "sim_gui_full.png"
    gui = cv2.imread(str(gui_path))
    if gui is not None:
        gui = cv2.cvtColor(gui, cv2.COLOR_BGR2RGB)
        ax_gui.imshow(gui)
    ax_gui.set_xticks([])
    ax_gui.set_yticks([])
    ax_gui.set_title("(a) Implemented digital-twin workbench: stereo preview, "
                     "3-D motion and 200-Hz IMU", fontsize=6.4, pad=2)

    axes = (fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1]))
    x = np.arange(3)
    for ax, arrays, title, ylabel in (
        (axes[0], (real_acc, initial_acc, tuned_acc), "(b) Acceleration",
         r"$|\|a\|-1g|$ (g)"),
        (axes[1], (real_gyro, initial_gyro, tuned_gyro), "(c) Angular rate",
         r"$\|\omega\|$ (deg/s)"),
    ):
        for offset, values, name in zip((-0.25, 0, 0.25), arrays,
                                        ("real", "initial twin", "tuned twin")):
            ax.bar(x + offset, values, 0.24, label=name)
        ax.set_xticks(x, labels, fontsize=5.4)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=6.8)
        ax.tick_params(axis="y", labelsize=5.4)
        ax.grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=4.8, borderpad=0.2, handlelength=1.0, labelspacing=0.2)
    fig.savefig(OUT / "domain_gap.pdf")
    plt.close(fig)


def _drift_statistics(eval_npz="physnet_v1/eval_wd1.npz"):
    """Residual of the ensemble against the unobservable initial-velocity term,
    and the magnitude of that term as a function of context length."""
    ev = np.load(DATA / eval_npz, allow_pickle=True)
    pred, y, pairs, t_pair, sess = ev["val_pred"], ev["val_y"], ev["val_pair"], \
        ev["val_t_pair"], ev["val_session"]
    err = pred - y
    vm, dt_all = [], []
    for name in np.unique(sess):
        t, Rg, pg, us = _gt(str(name))
        for k in np.flatnonzero(sess == name):
            a, _b = pairs[k]
            ta, tb = t_pair[k]
            lo = us[t[us] >= ta - 2.0]
            hi = us[t[us] <= tb + 2.0]
            if len(lo) and len(hi) and t[hi[-1]] - t[lo[0]] > 4.0:
                v = (pg[hi[-1]] - pg[lo[0]]) / (t[hi[-1]] - t[lo[0]])
                vm.append(Rg[a].T @ v)
            else:
                vm.append(np.full(3, np.nan))
            dt_all.append(tb - ta)
    vm, dt_all = np.asarray(vm), np.asarray(dt_all)
    ok = np.isfinite(vm).all(1)
    drift = vm[ok] * dt_all[ok, None] * 1000
    e = err[ok] * 1000
    X = np.column_stack((drift, np.ones(ok.sum())))
    coef = np.linalg.lstsq(X, e, rcond=None)[0]
    resid = e - X @ coef
    before = float(np.linalg.norm(e, axis=1).mean())
    after = float(np.linalg.norm(resid, axis=1).mean())

    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    lengths = np.array([7, 11, 15, 21, 31, 61])
    rms = []
    for T in lengths:
        vals, c = [], (T - 3) / 2
        for name in split["val"] + split["train"]:
            t, _R, pg, us = _gt(name)
            tu, pu = t[us], pg[us]
            for k in range(0, len(us), 3):
                ta = tu[k]
                lo = np.searchsorted(tu, ta - c)
                hi = np.searchsorted(tu, ta + 3 + c, side="right") - 1
                if lo >= len(tu) or hi < 0 or tu[hi] - tu[lo] < 0.8 * T:
                    continue
                vals.append(np.linalg.norm(pu[hi] - pu[lo]) * 3 / (tu[hi] - tu[lo]) * 1000)
        rms.append(np.sqrt(np.mean(np.square(vals))))
    return drift, e, coef, before, after, lengths, np.asarray(rms)


def observability():
    """Horizon ablation and the information limit in one single-column figure."""
    drift, e, coef, before, after, lengths, rms = _drift_statistics()
    fig = plt.figure(figsize=(3.45, 3.30))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.0), hspace=0.62, wspace=0.42,
                          left=0.155, right=0.975, top=0.925, bottom=0.095)

    ax = fig.add_subplot(gs[0, 0])
    h = np.array([0.5, 1, 2, 3, 4])
    r2 = np.array([0.177, 0.203, 0.265, 0.343, 0.339])
    ax.plot(h, r2, "o-", markersize=2.8, linewidth=0.9, color="#2369a2")
    ax.scatter([3], [0.343], s=34, facecolors="none", edgecolors="#d24b40",
               linewidths=0.9, label="selected")
    ax.set_xlabel("Horizon $h$ (s)", labelpad=1)
    ax.set_ylabel(r"Validation $R^2$")
    ax.set_xticks(h, ["0.5", "1", "2", "3", "4"], fontsize=5.4)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.set_title("(a) Observable horizon", fontsize=6.6)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=5.0, borderpad=0.2, handletextpad=0.3, framealpha=0.85)

    ev = np.load(DATA / "physnet_v1" / "eval_wd1.npz", allow_pickle=True)
    pv, yv = ev["val_pred"] * 1000, ev["val_y"] * 1000
    ax = fig.add_subplot(gs[0, 1])
    slopes = []
    for k, (name, colour) in enumerate((("x", "#2369a2"), ("y", "#45a86b"),
                                        ("z", "#d24b40"))):
        slope = np.polyfit(yv[:, k], pv[:, k], 1)[0]
        slopes.append(float(slope))
        ax.scatter(yv[:, k], pv[:, k], s=1.6, alpha=0.26, color=colour,
                   label=f"{name}: slope {slope:.2f}")
    lim = np.percentile(np.abs(yv), 99)
    ax.plot([-lim, lim], [-lim, lim], color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"Reference $\Delta p$ (mm)", labelpad=1)
    ax.set_ylabel("Prediction (mm)")
    ax.set_title("(b) Predictions are shrunk", fontsize=6.6)
    ax.tick_params(labelsize=5.4)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=4.8, loc="upper left", borderpad=0.2, handletextpad=0.2,
              labelspacing=0.15, markerscale=3.0, framealpha=0.85)

    check = json.loads((DATA / "velocity_proxy_check.json").read_text(encoding="utf-8"))
    var = check["variants"]
    ax = fig.add_subplot(gs[1, :])
    labels = ["window mean\n(incl. target)", "context only\n(excl. target)"]
    ins = [var["full"]["in_sample_reduction_percent"], var["context"]["in_sample_reduction_percent"]]
    crs = [var["full"]["cross_session_reduction_percent"],
           var["context"]["cross_session_reduction_percent"]]
    xpos = np.arange(2)
    b1 = ax.bar(xpos - 0.18, ins, 0.34, color="#9ab7d6", label="fitted and scored on the same pairs")
    b2 = ax.bar(xpos + 0.18, crs, 0.34, color="#2369a2", label="fitted on the other recording")
    ax.bar_label(b1, fmt="%.1f", fontsize=5.0, padding=1)
    ax.bar_label(b2, fmt="%.1f", fontsize=5.0, padding=1)
    ax.axhline(0, color="#333333", linewidth=0.6)
    ax.set_xticks(xpos, labels, fontsize=5.4)
    ax.set_ylabel("Error removed (%)")
    ax.set_title("(c) Affine correction by a reference velocity proxy", fontsize=6.6)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=5.0, borderpad=0.2, handlelength=1.4, labelspacing=0.2,
              framealpha=0.85, loc="upper right")
    fig.savefig(OUT / "observability.pdf")
    plt.close(fig)
    return {"corr": [float(np.corrcoef(drift[:, k], e[:, k])[0, 1]) for k in range(3)],
            "slope": [float(coef[k, k]) for k in range(3)],
            "err_mm_before": before, "err_mm_after": after,
            "drift_rms_mm": dict(zip([int(v) for v in lengths],
                                     [float(v) for v in rms]))}


def benchmark():
    payload = json.loads((DATA / "method_benchmark_20260924.json").read_text(encoding="utf-8"))
    external = json.loads(
        (DATA / "external_benchmark" / "test_benchmark.json").read_text(encoding="utf-8")
    )
    labels = ["Ours", "RoNIN", "TLIO", "IMUNet"]
    synthetic = [
        payload["datasets"]["synthetic_test"]["synthetic_zero_shot"]["err_mm"],
        external["ronin_resnet"]["zero_shot"]["synthetic_test"]["err_mm"],
        external["tlio_resnet"]["zero_shot"]["synthetic_test"]["err_mm"],
        external["imunet"]["zero_shot"]["synthetic_test"]["err_mm"],
    ]
    real = [
        payload["datasets"]["real_test"]["synthetic_zero_shot"]["err_mm"],
        external["ronin_resnet"]["zero_shot"]["real_test"]["err_mm"],
        external["tlio_resnet"]["zero_shot"]["real_test"]["err_mm"],
        external["imunet"]["zero_shot"]["real_test"]["err_mm"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.55))
    for ax, values, title in (
        (axes[0], synthetic, "(a) Direct run: synthetic test"),
        (axes[1], real, "(b) Direct run: held-out real test"),
    ):
        bars = ax.bar(np.arange(len(labels)), values, color=["#e2694f", "#4f9dd9", "#45a86b", "#9b59b6"])
        ax.set_xticks(np.arange(len(labels)), labels, rotation=20, ha="right")
        ax.set_ylabel("Mean 3-s displacement error (mm)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.1f", fontsize=6)
    fig.tight_layout()
    fig.savefig(OUT / "method_benchmark.pdf")
    plt.close(fig)


def trajectories():
    """Six complementary views of the held-out real trajectory."""
    fig, axes = plt.subplots(3, 2, figsize=(3.45, 4.75))
    colors = {
        "ground_truth": "black",
        "ridge": "#e0a43b",
        "cv_imunet": "#9b59b6",
        "cv_physnet": "#e2694f",
        "cv_equal_blend": "#ef4770",
    }
    labels = {
        "ground_truth": "Reference",
        "ridge": "Ridge",
        "cv_imunet": "IMUNet",
        "cv_physnet": "PhysNet",
        "cv_equal_blend": "Blend",
    }
    methods = ("ground_truth", "ridge", "cv_imunet", "cv_physnet", "cv_equal_blend")
    data = np.load(DATA / "trajectory_comparison" / "real_test.npz")
    summary = json.loads(
        (DATA / "trajectory_comparison" / "summary.json").read_text(encoding="utf-8")
    )["real_test"]["metrics"]
    ate_key = {
        "ridge": "ridge",
        "cv_imunet": "cv_imunet",
        "cv_physnet": "cv_physnet",
        "cv_equal_blend": "cv_equal_blend",
    }
    t = data["t"] - data["t"][0]
    truth = data["ground_truth"] * 1000
    errors = {}
    for method in methods:
        if method not in data.files:
            continue
        p = data[method] * 1000
        mask = np.isfinite(p).all(1)
        label = labels[method]
        if method != "ground_truth" and ate_key[method] in summary:
            label += f" ({summary[ate_key[method]]['ate_rmse_mm']:.0f} mm)"
        lw = 1.25 if method == "ground_truth" else 0.62
        axes[0, 0].plot(p[mask, 0], p[mask, 1], color=colors[method],
                        label=label, linewidth=lw)
        axes[0, 1].plot(p[mask, 0], p[mask, 2], color=colors[method], linewidth=lw)
        axes[1, 0].plot(p[mask, 1], p[mask, 2], color=colors[method], linewidth=lw)
        axes[1, 1].plot(t[mask], p[mask, 0], color=colors[method], linewidth=lw)
        axes[2, 0].plot(t[mask], p[mask, 2], color=colors[method], linewidth=lw)
        if method != "ground_truth":
            err = np.linalg.norm(p[mask] - truth[mask], axis=1)
            errors[method] = np.sort(err)
            q = np.linspace(0, 1, len(err), endpoint=False)
            axes[2, 1].plot(errors[method], q, color=colors[method], linewidth=0.72)

    axes[0, 0].set_title("(a) $XY$ trajectory; legend: ATE", fontsize=6.4)
    axes[0, 0].set_xlabel("$X$ (mm)", labelpad=1)
    axes[0, 0].set_ylabel("$Y$ (mm)")
    axes[0, 0].axis("equal")
    axes[0, 0].legend(fontsize=4.2, borderpad=0.18, handlelength=0.9,
                      labelspacing=0.16, framealpha=0.85, loc="upper left")
    axes[0, 1].set_title("(b) $XZ$ trajectory", fontsize=6.4)
    axes[0, 1].set_xlabel("$X$ (mm)", labelpad=1)
    axes[0, 1].set_ylabel("$Z$ (mm)")
    axes[1, 0].set_title("(c) $YZ$ trajectory", fontsize=6.4)
    axes[1, 0].set_xlabel("$Y$ (mm)", labelpad=1)
    axes[1, 0].set_ylabel("$Z$ (mm)")
    axes[1, 1].set_title("(d) $X(t)$", fontsize=6.4)
    axes[1, 1].set_xlabel("Time (s)", labelpad=1)
    axes[1, 1].set_ylabel("mm")
    axes[2, 0].set_title("(e) $Z(t)$", fontsize=6.4)
    axes[2, 0].set_xlabel("Time (s)", labelpad=1)
    axes[2, 0].set_ylabel("mm")
    axes[2, 1].set_title("(f) Position-error CDF", fontsize=6.4)
    axes[2, 1].set_xlabel("Error (mm)", labelpad=1)
    axes[2, 1].set_ylabel("Fraction")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 0]):
        ax.axis("equal")
    for ax in axes.ravel():
        ax.tick_params(labelsize=5.1)
        ax.grid(alpha=0.2)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "trajectory_results.pdf")
    plt.close(fig)


def _gt(name):
    gt = np.load(DATA / "pose_gt_raw" / f"{name}.npz")
    delay = float(np.load(DATA / "pose_gt" / f"{name}.npz")["delay_usb"][0])
    return gt["t"].astype(float) - delay, gt["R"].astype(float), gt["p"].astype(float), np.flatnonzero(gt["usable"] == 1)


def information_limit(eval_npz="physnet_v1/eval_wd1.npz"):
    """(a) residual vs unobservable window-mean velocity; (b) drift term vs window length."""
    ev = np.load(DATA / eval_npz, allow_pickle=True)
    pred, y, pairs, t_pair, sess = ev["val_pred"], ev["val_y"], ev["val_pair"], ev["val_t_pair"], ev["val_session"]
    err = pred - y
    vm, dt_all = [], []
    for name in np.unique(sess):
        t, Rg, pg, us = _gt(str(name))
        sel = np.flatnonzero(sess == name)
        for k in sel:
            a, _b = pairs[k]
            ta, tb = t_pair[k]
            lo = us[t[us] >= ta - 2.0]
            hi = us[t[us] <= tb + 2.0]
            if len(lo) and len(hi) and t[hi[-1]] - t[lo[0]] > 4.0:
                v = (pg[hi[-1]] - pg[lo[0]]) / (t[hi[-1]] - t[lo[0]])
                vm.append(Rg[a].T @ v)
            else:
                vm.append(np.full(3, np.nan))
            dt_all.append(tb - ta)
    vm, dt_all = np.asarray(vm), np.asarray(dt_all)
    ok = np.isfinite(vm).all(1)
    drift = vm[ok] * dt_all[ok, None] * 1000
    e = err[ok] * 1000
    X = np.column_stack((drift, np.ones(ok.sum())))
    coef = np.linalg.lstsq(X, e, rcond=None)[0]
    resid = e - X @ coef
    before = np.linalg.norm(e, axis=1).mean()
    after = np.linalg.norm(resid, axis=1).mean()

    # Drift term as a function of the window length, from reference positions only.
    split = json.loads((DATA / "trajectory_split_paper.json").read_text(encoding="utf-8"))
    lengths = np.array([7, 11, 15, 21, 31, 61])
    rms = []
    for T in lengths:
        vals = []
        c = (T - 3) / 2
        for name in split["val"] + split["train"]:
            t, _R, pg, us = _gt(name)
            tu, pu = t[us], pg[us]
            for k in range(0, len(us), 3):
                ta = tu[k]
                lo = np.searchsorted(tu, ta - c)
                hi = np.searchsorted(tu, ta + 3 + c, side="right") - 1
                if lo >= len(tu) or hi < 0 or tu[hi] - tu[lo] < 0.8 * T:
                    continue
                vals.append(np.linalg.norm(pu[hi] - pu[lo]) * 3 / (tu[hi] - tu[lo]) * 1000)
        rms.append(np.sqrt(np.mean(np.square(vals))))
    rms = np.asarray(rms)

    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.05))
    ax = axes[0]
    axis_names = ("x", "y", "z")
    colors = ("#2369a2", "#45a86b", "#d24b40")
    for k in range(3):
        ax.scatter(drift[:, k], e[:, k], s=2, alpha=0.28, color=colors[k],
                   label=axis_names[k])
    lim = np.percentile(np.abs(drift), 99)
    ax.plot([-lim, lim], [lim, -lim], color="#555555", linewidth=0.8, linestyle="--")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel(r"$\bar v\,\Delta t$ (mm)", labelpad=1)
    ax.set_ylabel("Residual (mm)")
    ax.set_title(f"(a) {before:.1f} $\\rightarrow$ {after:.1f} mm if $\\bar v$ known",
                 fontsize=6.6)
    ax.tick_params(labelsize=5.4)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=5.0, loc="upper right", borderpad=0.2, handletextpad=0.3,
              labelspacing=0.2, markerscale=2.5, framealpha=0.85)
    ax = axes[1]
    ax.plot(lengths, rms, "o-", markersize=2.8, linewidth=0.9, color="#2369a2")
    ax.axhline(before, color="#d24b40", linestyle="--", linewidth=0.9,
               label=f"7-s residual, {before:.0f} mm")
    ax.axhline(after, color="#45a86b", linestyle=":", linewidth=0.9,
               label=f"without the term, {after:.0f} mm")
    ax.set_xscale("log")
    ax.set_xticks(lengths, [str(v) for v in lengths], fontsize=5.4)
    ax.set_xlabel("Context length (s)", labelpad=1)
    ax.set_ylabel(r"$\|\bar v\,\Delta t\|$ RMS (mm)")
    ax.set_title("(b) Shrinks with context, not learned", fontsize=6.6)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=5.0, borderpad=0.2, handlelength=1.2, labelspacing=0.2,
              framealpha=0.85)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "information_limit.pdf")
    plt.close(fig)
    return {"corr": [float(np.corrcoef(drift[:, k], e[:, k])[0, 1]) for k in range(3)],
            "slope": [float(coef[k, k]) for k in range(3)],
            "err_mm_before": float(before), "err_mm_after": float(after),
            "drift_rms_mm": dict(zip([int(v) for v in lengths], [float(v) for v in rms]))}


def session_cv(cv_json="session_cv_benchmark.json"):
    report = json.loads((DATA / cv_json).read_text(encoding="utf-8"))
    folds = report["folds"]
    names = list(folds)
    order = np.argsort([folds[n]["physnet"]["zero_mm"] for n in names])
    names = [names[i] for i in order]
    zero = [folds[n]["physnet"]["zero_mm"] for n in names]
    imu = [folds[n]["imunet"]["err_mm"] for n in names]
    phy = [folds[n]["physnet"]["err_mm"] for n in names]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.45, 2.75))
    ax.barh(y + 0.26, zero, 0.25, color="#b0b0b0", label="zero motion")
    ax.barh(y, imu, 0.25, color="#9b59b6", label="IMUNet (adapted)")
    ax.barh(y - 0.26, phy, 0.25, color="#e2694f", label="PhysNet (proposed)")
    labels = [f"{n[5:13]}-{n[14:18]}" for n in names]
    ax.set_yticks(y, labels, fontsize=5.0)
    ax.set_xlabel("Held-out 3-s error (mm)", labelpad=1)
    ax.tick_params(axis="x", labelsize=5.4)
    ax.set_title("Leave-one-session-out over the eight non-test\n"
                 "recordings, sorted by motion magnitude", fontsize=6.8)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=5.0, loc="lower right", borderpad=0.25, handlelength=1.0,
              labelspacing=0.25, framealpha=0.9)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "session_cv.pdf")
    plt.close(fig)


def benchmark_summary(cv_json="session_cv_gate_benchmark.json"):
    """(a) error reduction relative to zero motion for every method and column,
    (b) paired held-out-test differences with block-bootstrap intervals,
    (c) the per-session cross-validation view. Values come from
    datasets/summary_table.json and datasets/recheck_metrics.json."""
    rows = json.loads((DATA / "summary_table.json").read_text(encoding="utf-8"))
    ci = json.loads((DATA / "recheck_metrics.json").read_text(encoding="utf-8"))
    cv = json.loads((DATA / cv_json).read_text(encoding="utf-8"))
    zero = next(r for r in rows if r["method"] == "Zero motion")["cells"]
    cols = [("syn", "Synth.\ntest"), ("ood", "Synth.\nOOD"), ("direct", "Real\ndirect"),
            ("val", "Val"), ("cv", "CV"), ("test", "Test"), ("fast", "Test\nfast"),
            ("ate", "ATE")]
    names = {"Ridge features": "Ridge", "Conv--BiGRU, body frame": "Conv-BiGRU",
             "RoNIN-LSTM (adapted)": "RoNIN-LSTM", "RoNIN-ResNet (adapted)": "RoNIN-ResNet",
             "TLIO-ResNet (adapted)": "TLIO-ResNet", "IMUNet (adapted)": "IMUNet",
             "PhysNet (proposed)": "PhysNet", "PhysNet + stillness gate": "PhysNet+gate",
             "Equal blend, no gate": "Equal blend", "Calibrated blend, no gate": "Calib. blend",
             "Equal blend, gate": "Equal blend (gate)",
             "Calibrated blend, gate": "Calib. blend (gate)"}
    methods = [r for r in rows if r["method"] in names]
    M = np.full((len(methods), len(cols)), np.nan)
    for i, r in enumerate(methods):
        for j, (k, _l) in enumerate(cols):
            if k in r["cells"]:
                M[i, j] = 100 * (1 - r["cells"][k]["value"] / zero[k]["value"])

    fig = plt.figure(figsize=(7.1, 3.35))
    gs = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.0, 1.05), wspace=0.55,
                          left=0.085, right=0.99, top=0.90, bottom=0.14)
    ax = fig.add_subplot(gs[0, 0])
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-60, vcenter=0, vmax=75)
    im = ax.imshow(np.clip(M, -60, 75), cmap="RdBu", norm=norm, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = M[i, j]
                txt = f"{v:.0f}" if v > -100 else "<-99"
                ax.text(j, i, txt, ha="center", va="center", fontsize=4.9,
                        color="white" if abs(np.clip(v, -60, 75)) > 42 else "#1a1a1a")
            else:
                ax.text(j, i, "·", ha="center", va="center", fontsize=6, color="#999999")
    ax.set_xticks(range(len(cols)), [l for _k, l in cols], fontsize=5.0)
    ax.set_yticks(range(len(methods)), [names[r["method"]] for r in methods], fontsize=5.2)
    for y in (4.5, 7.5):
        ax.axhline(y, color="white", linewidth=1.6)
    ax.tick_params(length=0)
    ax.set_title("(a) Error reduction vs. zero motion (%)", fontsize=6.6)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=4.8)

    ax = fig.add_subplot(gs[0, 1])
    g, ng = ci["session_cv_gate_benchmark"], ci["session_cv_benchmark"]
    def point(src, a, b):
        m = src["metrics"]
        return m[b]["err_mm"] - m[a]["err_mm"] if b != "zero" else \
            m["zero_motion"]["err_mm"] - m[a]["err_mm"]
    items = [
        ("vs. zero motion", None, None, None),
        ("IMUNet", g["ci_mm"]["imunet_vs_zero"], point(g, "imunet", "zero"), "#9b59b6"),
        ("PhysNet", ng["ci_mm"]["physnet_vs_zero"], point(ng, "physnet", "zero"), "#e2694f"),
        ("PhysNet+gate", g["ci_mm"]["physnet_vs_zero"], point(g, "physnet", "zero"), "#c0392b"),
        ("Equal blend (gate)", g["ci_mm"]["equal_blend_vs_zero"], point(g, "equal_blend", "zero"), "#ef4770"),
        ("Calib. blend (gate)", g["ci_mm"]["calibrated_blend_vs_zero"],
         point(g, "cv_calibrated_blend", "zero"), "#b03060"),
        ("vs. IMUNet", None, None, None),
        ("PhysNet", ng["ci_mm"]["physnet_vs_imunet"], point(ng, "physnet", "imunet"), "#e2694f"),
        ("PhysNet+gate", g["ci_mm"]["physnet_vs_imunet"], point(g, "physnet", "imunet"), "#c0392b"),
        ("Equal blend (gate)", g["ci_mm"]["equal_blend_vs_imunet"], point(g, "equal_blend", "imunet"), "#ef4770"),
        ("Calib. blend (gate)", g["ci_mm"]["calibrated_blend_vs_imunet"],
         point(g, "cv_calibrated_blend", "imunet"), "#b03060"),
    ]
    ylab = []
    for y, (lab, interval, pt, colour) in enumerate(items):
        ylab.append(lab)
        if interval is None:
            continue
        sig = interval[0] > 0 or interval[1] < 0
        ax.plot(interval, [y, y], color=colour, linewidth=1.3)
        ax.plot([pt], [y], "o", color=colour, markersize=3.4,
                markerfacecolor=colour if sig else "white")
    ax.axvline(0, color="#333333", linewidth=0.7, linestyle="--")
    ax.set_yticks(range(len(items)), ylab, fontsize=5.2)
    for tick, (lab, interval, _p, _c) in zip(ax.get_yticklabels(), items):
        if interval is None:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel("Error reduction on held-out test (mm)", fontsize=5.6, labelpad=1)
    ax.tick_params(axis="x", labelsize=5.0)
    ax.grid(axis="x", alpha=0.25)
    ax.set_title("(b) Paired differences, 95% CI", fontsize=6.6)

    folds = cv["folds"]
    order = sorted(folds, key=lambda n: folds[n]["physnet"]["zero_mm"])
    y = np.arange(len(order))
    ax = fig.add_subplot(gs[0, 2])
    ax.barh(y + 0.26, [folds[n]["physnet"]["zero_mm"] for n in order], 0.25,
            color="#b0b0b0", label="zero motion")
    ax.barh(y, [folds[n]["imunet"]["err_mm"] for n in order], 0.25,
            color="#9b59b6", label="IMUNet")
    ax.barh(y - 0.26, [folds[n]["physnet"]["err_mm"] for n in order], 0.25,
            color="#c0392b", label="PhysNet+gate")
    ax.set_yticks(y, [f"S{k + 1}" for k in range(len(order))], fontsize=5.2)
    ax.set_xlabel("Held-out 3-s error (mm)", fontsize=5.6, labelpad=1)
    ax.tick_params(axis="x", labelsize=5.0)
    ax.set_title("(c) Leave-one-session-out", fontsize=6.6)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=4.8, loc="lower right", borderpad=0.2, handlelength=1.0,
              labelspacing=0.2, framealpha=0.9)
    fig.savefig(OUT / "estimators.pdf")
    plt.close(fig)


def estimators(cv_json="session_cv_gate_benchmark.json"):
    """Direct-run comparison, held-out-test ensembles and per-session results."""
    external = json.loads((DATA / "external_benchmark" / "test_benchmark.json").read_text(encoding="utf-8"))
    ood = json.loads((DATA / "external_benchmark" / "zero_shot_synthetic_ood.json").read_text(encoding="utf-8"))
    phys_syn = json.loads((DATA / "physnet_v1" / "eval_synonly_181622.json").read_text(encoding="utf-8"))
    cv = json.loads((DATA / cv_json).read_text(encoding="utf-8"))
    labels = ["RoNIN", "TLIO", "IMUNet", "PhysNet"]
    colors = ["#4f9dd9", "#45a86b", "#9b59b6", "#e2694f"]
    syn_test = [external["ronin_resnet"]["zero_shot"]["synthetic_test"]["err_mm"],
                external["tlio_resnet"]["zero_shot"]["synthetic_test"]["err_mm"],
                external["imunet"]["zero_shot"]["synthetic_test"]["err_mm"],
                phys_syn["synthetic_test"]["err_mm"]]
    syn_ood = [ood["ronin_resnet"]["synthetic_ood"]["err_mm"], ood["tlio_resnet"]["synthetic_ood"]["err_mm"],
               ood["imunet"]["synthetic_ood"]["err_mm"], phys_syn["synthetic_ood"]["err_mm"]]
    real_zero = [external["ronin_resnet"]["zero_shot"]["real_test"]["err_mm"],
                 external["tlio_resnet"]["zero_shot"]["real_test"]["err_mm"],
                 external["imunet"]["zero_shot"]["real_test"]["err_mm"],
                 phys_syn["test"]["err_mm"]]
    fig = plt.figure(figsize=(3.45, 4.15))
    gs = fig.add_gridspec(3, 2, height_ratios=(1.0, 1.0, 1.45), hspace=0.80,
                          wspace=0.46, left=0.135, right=0.975, top=0.955,
                          bottom=0.072)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
    for ax, values, title, lab in (
        (axes[0], syn_test, "(a) Synthetic test, direct", labels),
        (axes[1], syn_ood, "(b) Synthetic OOD, direct", labels),
        (axes[2], real_zero, "(c) held-out real test, direct", labels),
    ):
        bars = ax.bar(np.arange(len(lab)), values, color=colors[:len(lab)])
        ax.set_xticks(np.arange(len(lab)), lab, rotation=30, ha="right", fontsize=5.0)
        ax.set_title(title, fontsize=6.4)
        ax.tick_params(axis="y", labelsize=5.2)
        ax.grid(axis="y", alpha=0.25)
        ax.bar_label(bars, fmt="%.0f", fontsize=4.6, padding=1)
        ax.set_ylim(0, max(values) * 1.20)
    axes[0].set_ylabel("mm", fontsize=5.6)
    axes[2].set_ylabel("mm", fontsize=5.6)
    ax = axes[3]
    keys = [("zero_motion", "Zero"), ("imunet_cv_ensemble", "IMUNet"),
            ("physnet_cv_ensemble", "PhysNet"), ("cv_calibrated_blend", "Blend")]
    vals = [cv["test"][k]["err_mm"] for k, _ in keys]
    bars = ax.bar(np.arange(4), vals, color=["#b0b0b0", "#9b59b6", "#e2694f", "#ef4770"])
    ax.set_xticks(np.arange(4), [l for _, l in keys], rotation=30, ha="right",
                  fontsize=5.0)
    ax.set_title("(d) held-out test, CV ensembles", fontsize=6.4)
    ax.tick_params(axis="y", labelsize=5.2)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, fmt="%.0f", fontsize=4.6, padding=1)
    ax.set_ylim(0, max(vals) * 1.20)

    # (e) the per-session view of the cross-validation
    folds = cv["folds"]
    names = sorted(folds, key=lambda n: folds[n]["physnet"]["zero_mm"])
    y = np.arange(len(names))
    ax = fig.add_subplot(gs[2, :])
    ax.barh(y + 0.26, [folds[n]["physnet"]["zero_mm"] for n in names], 0.25,
            color="#b0b0b0", label="zero motion")
    ax.barh(y, [folds[n]["imunet"]["err_mm"] for n in names], 0.25,
            color="#9b59b6", label="IMUNet (adapted)")
    ax.barh(y - 0.26, [folds[n]["physnet"]["err_mm"] for n in names], 0.25,
            color="#e2694f", label="PhysNet (proposed)")
    ax.set_yticks(y, [f"{n[5:13]}-{n[14:18]}" for n in names], fontsize=4.8)
    ax.set_xlabel("Held-out 3-s error (mm)", labelpad=1)
    ax.tick_params(axis="x", labelsize=5.2)
    ax.set_title("(e) Leave-one-session-out, sorted by motion magnitude", fontsize=6.4)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(fontsize=4.8, loc="lower right", borderpad=0.2, handlelength=1.0,
              labelspacing=0.2, framealpha=0.9)
    fig.savefig(OUT / "estimators.pdf")
    plt.close(fig)


def zupt_and_uncertainty(bound_json="zupt_bound.json", unc_json="uncertainty_val.json"):
    """(a) accuracy of zero-velocity-aided integration versus interval length;
    (b) calibration of the predicted displacement uncertainty."""
    bound = json.loads((DATA / bound_json).read_text(encoding="utf-8"))
    unc = json.loads((DATA / unc_json).read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.15))

    ax = axes[0]
    centres = {"0.8-2s": 1.4, "2-4s": 3.0, "4-8s": 6.0, "8-30s": 15.0}
    labels = {"ref": "Reference attitude", "gyro": "Gyroscope, session bias removed",
              "gyro_raw": "Gyroscope, raw rate"}
    for attitude, colour, marker in (("ref", "#2369a2", "o"), ("gyro", "#d24b40", "s"),
                                     ("gyro_raw", "#e39b2d", "^")):
        if attitude not in bound:
            continue
        # Pool the train, validation and rigidity recordings per span bin.
        pool = {}
        for group in ("train", "val", "extra_train (rigid)"):
            for key, entry in bound[attitude][group].get("by_span", {}).items():
                acc = pool.setdefault(key, [0.0, 0])
                acc[0] += entry["err_mm"] * entry["n"]
                acc[1] += entry["n"]
        x = sorted(centres[k] for k in pool)
        keys = sorted(pool, key=lambda k: centres[k])
        y = [pool[k][0] / pool[k][1] for k in keys]
        n = [pool[k][1] for k in keys]
        ax.plot(x, y, marker=marker, linestyle="-", color=colour, markersize=4,
                label=labels[attitude])
        if attitude == "ref":
            for xi, yi, ni in zip(x, y, n):
                ax.annotate(f"n={ni}", (xi, yi), textcoords="offset points", xytext=(3, -9),
                            fontsize=5.5, color="#555555")
    ax.axhline(43.6, color="#45a86b", linestyle="--", linewidth=0.9,
               label="learned 3-s window, 43.6 mm")
    ax.axvspan(2.6, 3.4, color="#999999", alpha=0.18)
    ax.text(3.0, 2500.0, "3-s\nprotocol", ha="center", fontsize=5.6, color="#555555")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([1.4, 3, 6, 15], ["1.4", "3", "6", "15"])
    ax.set_xlabel("Pause interval (s)", labelpad=1)
    ax.set_ylabel("Displacement error (mm)")
    ax.set_title("(a) Zero-velocity-aided bound", fontsize=6.6)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=4.6, loc="lower right", borderpad=0.2, handlelength=1.2, labelspacing=0.2, framealpha=0.85)
    ax.tick_params(labelsize=5.4)

    ax = axes[1]
    keys = [("aleatoric", "Predicted\nvariance"), ("ensemble", "Ensemble\nspread"),
            ("total", "Sum"), ("total_rescaled", f"Sum $\\times$ {unc['validation_scale']:.2f}\n(fitted on val.)")]
    x = np.arange(len(keys))
    one = [unc["val"][k]["coverage_1sigma"] for k, _ in keys]
    two = [unc["val"][k]["coverage_2sigma"] for k, _ in keys]
    ax.bar(x - 0.19, one, 0.36, color="#2369a2", label=r"within $1\sigma$")
    ax.bar(x + 0.19, two, 0.36, color="#7fb3d5", label=r"within $2\sigma$")
    ax.axhline(0.683, color="#d24b40", linestyle="--", linewidth=0.9)
    ax.axhline(0.954, color="#d24b40", linestyle=":", linewidth=0.9)
    ax.text(-0.42, 0.71, "0.683", fontsize=5.4, color="#d24b40")
    ax.text(-0.42, 0.975, "0.954", fontsize=5.4, color="#d24b40")
    ax.set_xticks(x, [label.replace("\n", " ") for _, label in keys], fontsize=5.4,
                  rotation=18, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Interval coverage")
    conf = unc["val"]["total"]["err_mm_most_confident_decile"]
    loose = unc["val"]["total"]["err_mm_least_confident_decile"]
    ax.set_title(f"(b) Coverage; {conf:.0f} vs {loose:.0f} mm by decile", fontsize=6.6)
    ax.tick_params(axis="y", labelsize=5.4)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=5.0, loc="lower right", borderpad=0.2, handlelength=1.0, labelspacing=0.2, framealpha=0.85)
    fig.tight_layout(pad=0.35)
    fig.savefig(OUT / "zupt_uncertainty.pdf")
    plt.close(fig)


def main():
    framework()
    architecture()
    dataset()
    uncertainty()
    zupt_and_uncertainty()
    stats = observability()
    (OUT / "information_limit.json").write_text(json.dumps(stats, indent=2),
                                                encoding="utf-8")
    benchmark_summary()
    domain_gap()
    trajectories()
    print(json.dumps(stats, indent=2))
    print(OUT)


if __name__ == "__main__":
    main()
