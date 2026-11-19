#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Active camera: its locked focus and the intrinsic calibration made with it.

datasets/active_camera.json is written when the focus is locked and when an
intrinsic calibration (step 4) finishes. Every consumer of the intrinsics
takes them from here: the live board detector of the guided windows, the
label pipeline (tools/build_pose_gt.py reads its module constant CALIB at
call time and is pointed at the session's calibration, the file itself is
frozen), and the rigidity check. Without this file the calibration of the
stereo endoscope (2026-09-22) is used, as for all recordings of the paper.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT_DIR / "datasets"
CONFIG = DATA_ROOT / "active_camera.json"
LEGACY_CALIB = "calib_intrinsics_20260922_123120"
# Working distance of the trajectory recordings with the endoscope camera:
# median 150 mm (110-230 mm, 5th-95th percentile) at fx = 1095.3 px (1280 px wide).
LEGACY_FX_1280 = 1095.3
LEGACY_DISTANCE_MM = (110.0, 150.0, 230.0)
FOCUS_TOLERANCE = 3.0   # driver units; the camera of the revision reports 0-1023
# Share of the 6x4 image grid the calibration corners must reach. Beyond the
# reached cells the distortion is extrapolated: with 38% reached, PnP tilted the
# caliper board 0.15-0.3 deg more than the IMU on the board (2026-11-08).
COVERAGE_MIN = 0.8


def autofocus_on(value: Optional[float]) -> bool:
    """DirectShow reports the control mode (1 auto, 2 manual), other backends 1/0."""
    return value is not None and abs(float(value) - 1.0) < 0.01


# Rig of the open camera, set by the capture app: the stereo endoscope keeps
# its own calibration and fixed focus, the monocular camera uses the saved ones.
_MODE: dict[str, Any] = {"stereo": False, "index": None}


def set_mode(stereo: bool, index: Optional[int]) -> None:
    _MODE.update(stereo=bool(stereo), index=index)


def stereo_mode() -> bool:
    return bool(_MODE["stereo"])


def load() -> dict[str, Any]:
    if CONFIG.is_file():
        try:
            return json.loads(CONFIG.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def save(**fields: Any) -> dict[str, Any]:
    cfg = load()
    cfg.update(fields)
    cfg["updated_at"] = datetime.now().isoformat(timespec="seconds")
    CONFIG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    return cfg


def calib_dir(name: Optional[str] = None) -> Path:
    if name:
        return DATA_ROOT / name
    if stereo_mode():
        return DATA_ROOT / LEGACY_CALIB
    return DATA_ROOT / (load().get("calib_dir") or LEGACY_CALIB)


def is_legacy() -> bool:
    return calib_dir().name == LEGACY_CALIB


def intrinsics_json(cam: str = "cam0", name: Optional[str] = None) -> Path:
    return calib_dir(name) / f"camera_calibration_{cam}.json"


def locked_focus() -> Optional[float]:
    if stereo_mode():
        return None
    value = load().get("focus")
    return None if value is None else float(value)


def index_problem() -> Optional[str]:
    """The monocular calibration belongs to one camera; warn if another is open."""
    saved = load().get("camera_index")
    if stereo_mode() or saved is None or _MODE["index"] is None or int(saved) == int(_MODE["index"]):
        return None
    return f"现在的左目是相机 {_MODE['index']}，当前内参是用相机 {saved} 标定的。"


def calibration_focus() -> Optional[float]:
    """Focus position at which the active calibration was made."""
    path = intrinsics_json()
    if not path.is_file():
        return None
    value = (json.loads(path.read_text(encoding="utf-8")).get("focus_state") or {}).get("focus")
    return None if value is None else float(value)


def calibration_coverage(name: Optional[str] = None) -> Optional[float]:
    """Grid coverage of the calibration corners; None for calibrations made
    before it was recorded."""
    path = intrinsics_json(name=name)
    if not path.is_file():
        return None
    value = (json.loads(path.read_text(encoding="utf-8")).get("coverage") or {}).get("fraction")
    return None if value is None else float(value)


def fx_at_width(width: int) -> float:
    path = intrinsics_json()
    if not path.is_file():
        return LEGACY_FX_1280 * width / 1280.0
    cal = json.loads(path.read_text(encoding="utf-8"))
    return float(cal["camera_matrix"][0][0]) * width / float(cal["image_size"][0])


def work_distance_mm(width: int = 1280) -> tuple[float, float, float]:
    """(low, recommended, high) camera-board distance giving the board the same
    size in pixels as in the trajectory recordings."""
    k = fx_at_width(width) / (LEGACY_FX_1280 * width / 1280.0)
    return tuple(round(v * k, 0) for v in LEGACY_DISTANCE_MM)  # type: ignore[return-value]


def ensure_label_files(folder: Path) -> None:
    """tools/build_pose_gt.py (frozen) loads the cam1 intrinsics of every
    session. A monocular calibration has none and its recordings have no cam1
    images, so a marked copy of cam0 satisfies the loader and is never used."""
    cam0, cam1 = folder / "camera_calibration_cam0.json", folder / "camera_calibration_cam1.json"
    if cam0.is_file() and not cam1.is_file():
        data = json.loads(cam0.read_text(encoding="utf-8"))
        data.update(placeholder=True, note="monocular rig without cam1: copy of cam0 for the frozen "
                                           "label loader, never used for pose estimation")
        cam1.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def use_for_labels(name: Optional[str] = None) -> Path:
    """Point the frozen label pipeline at a calibration folder (default: active)."""
    import sys
    if str(ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(ROOT_DIR))
    import tools.build_pose_gt as bp
    bp.CALIB = calib_dir(name)
    ensure_label_files(bp.CALIB)
    return bp.CALIB


def session_calibration(session_dir: Path) -> Optional[str]:
    meta = session_dir / "session_meta.json"
    if not meta.is_file():
        return None
    return json.loads(meta.read_text(encoding="utf-8")).get("intrinsics_dir")


def focus_problem(state: dict[str, Any]) -> Optional[str]:
    """Why the camera's current focus cannot be used with the active calibration."""
    if stereo_mode():
        return None
    problem = index_problem()
    if problem:
        return problem
    af, fv = state.get("autofocus"), state.get("focus")
    if autofocus_on(af):
        return "相机的自动对焦还开着。"
    want = calibration_focus() if not is_legacy() else None
    if want is not None and fv is not None and abs(fv - want) > FOCUS_TOLERANCE:
        return f"相机焦距现在是 {fv:g}，内参标定时是 {want:g}。"
    return None
