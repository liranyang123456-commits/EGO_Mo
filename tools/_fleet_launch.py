"""Pack the decouple code and start the two worker jobs."""

from __future__ import annotations

import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"E:\EGO_Mo")
DISPATCH = Path(r"F:\worldModelEndoscopic\fleet\dispatch.py")
ZIP = ROOT / "datasets" / "ego_mo_decouple.zip"
PY = sys.executable

INCLUDE = [
    "ego_sim",
    "ego_capture/__init__.py",
    "ego_capture/mapping",
    "tools/train_decouple.py",
    "tools/export_decouple_corpus.py",
    "datasets/sim_assets/imu_profile_usb.json",
]


def build_zip() -> None:
    ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in INCLUDE:
            path = ROOT / item
            if path.is_file():
                archive.write(path, item.replace("\\", "/"))
                continue
            for file in path.rglob("*"):
                if file.suffix == ".pyc" or "__pycache__" in file.parts:
                    continue
                if file.is_file():
                    archive.write(file, file.relative_to(ROOT).as_posix())
    print(f"zip {ZIP} {ZIP.stat().st_size}", flush=True)


def dispatch(args: list[str]) -> None:
    result = subprocess.run([PY, str(DISPATCH), *args], text=True, encoding="utf-8", errors="replace")
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    build_zip()
    dispatch(["push", "r9000k", str(ZIP), r"C:/Users/lry/fleet-jobs/ego_mo_decouple.zip"])
    dispatch(["push", "shenzhou", str(ZIP), r"C:/Users/liran/fleet-jobs/ego_mo_decouple.zip"])
    seeds = " ".join(str(s) for s in range(300, 324))
    val = "400 401 402 403"
    r9000 = (
        "Expand-Archive -Force -Path C:\\Users\\lry\\fleet-jobs\\ego_mo_decouple.zip "
        "-DestinationPath C:\\Users\\lry\\fleet-jobs\\ego_mo; "
        "$env:PYTHONPATH='C:\\Users\\lry\\fleet-jobs\\ego_mo'; "
        "$env:KMP_DUPLICATE_LIB_OK='TRUE'; "
        "Set-Location C:\\Users\\lry\\fleet-jobs\\ego_mo; "
        "D:\\anaconda\\envs\\reloc3r\\python.exe -u tools\\train_decouple.py "
        f"--train-seeds {seeds} --val-seeds {val} --duration 12 --epochs 20 "
        "--visual-dropout 0.2 --batch 32 --out datasets\\decouple_abs; "
        "D:\\anaconda\\envs\\reloc3r\\python.exe -u tools\\train_decouple.py "
        f"--train-seeds {seeds} --val-seeds {val} --duration 12 --epochs 20 "
        "--visual-dropout 0.2 --batch 32 --supervise relative --out datasets\\decouple_rel"
    )
    shen = (
        "Expand-Archive -Force -Path C:\\Users\\liran\\fleet-jobs\\ego_mo_decouple.zip "
        "-DestinationPath C:\\Users\\liran\\fleet-jobs\\ego_mo; "
        "$env:PYTHONPATH='C:\\Users\\liran\\fleet-jobs\\ego_mo'; "
        "Set-Location C:\\Users\\liran\\fleet-jobs\\ego_mo; "
        "C:\\Users\\liran\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -u "
        "tools\\export_decouple_corpus.py --seed-start 0 --count 80 --duration 12 "
        "--out datasets\\decouple_corpus"
    )
    dispatch(["run", "r9000k", r9000, "--bg"])
    dispatch(["run", "shenzhou", shen, "--bg"])


if __name__ == "__main__":
    main()
