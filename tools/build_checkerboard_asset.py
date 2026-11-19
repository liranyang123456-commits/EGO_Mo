#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the metric GP050 checkerboard OBJ used by the simulator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "sim_assets" / "gp050_board"
SQUARE_MM = 3.0
COLS, ROWS = 12, 9
THICKNESS_MM = 1.0


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    px = 100
    texture = np.empty((ROWS * px, COLS * px, 3), dtype=np.uint8)
    for row in range(ROWS):
        for col in range(COLS):
            value = 238 if (row + col) % 2 == 0 else 16
            texture[row * px:(row + 1) * px, col * px:(col + 1) * px] = value
    Image.fromarray(texture).save(OUT / "checkerboard.png")

    # Origin is the first inner corner, matching tools/build_pose_gt.py.
    x0, x1 = -SQUARE_MM, (COLS - 1) * SQUARE_MM
    y0, y1 = -SQUARE_MM, (ROWS - 1) * SQUARE_MM
    z0, z1 = -THICKNESS_MM / 2, THICKNESS_MM / 2
    vertices = [
        (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
    ]
    lines = ["mtllib gp050_board.mtl", "o GP050_checkerboard"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices]
    lines += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
    lines += [
        "usemtl checker",
        "f 1/1 2/2 3/3 4/4",
        "usemtl side",
        "f 8 7 6 5",
        "f 1 5 6 2",
        "f 2 6 7 3",
        "f 3 7 8 4",
        "f 4 8 5 1",
    ]
    (OUT / "gp050_board.obj").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUT / "gp050_board.mtl").write_text(
        "newmtl checker\nKd 1 1 1\nmap_Kd checkerboard.png\n"
        "newmtl side\nKd 0.25 0.25 0.25\n",
        encoding="utf-8",
    )
    collision = trimesh.creation.box(extents=(x1 - x0, y1 - y0, z1 - z0))
    collision.apply_translation(((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2))
    collision.export(OUT / "gp050_board_collision.stl")
    print(OUT / "gp050_board.obj")


if __name__ == "__main__":
    main()
