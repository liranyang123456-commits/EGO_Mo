#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Launch the EGO_Mo synthetic dataset visual console."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

from ego_sim.app import SimulationWindow


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path)
    ap.add_argument("--tab", choices=("simulation", "trajectory", "benchmark", "dataset"), default="simulation")
    args = ap.parse_args()
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setApplicationName("EGO_Mo Simulator")
    # Force a CJK-capable UI font. The platform fallback selected by the
    # off-screen plugin can otherwise render every Chinese label as a box in
    # reproducibility snapshots.
    app.setFont(QtGui.QFont("Microsoft YaHei", 9))
    window = SimulationWindow()
    window.tabs.setCurrentIndex({
        "simulation": 0,
        "trajectory": 1,
        "benchmark": 2,
        "dataset": 3,
    }[args.tab])
    window.show()
    if args.snapshot:
        def save() -> None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.snapshot))
            app.quit()
        QtCore.QTimer.singleShot(2500, save)
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
