"""Build the revision ZIP from its reviewed file list and verify every member."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "1.1.0"
FILE_LIST = "release/files-v1.1.0.txt"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def add(archive, name, data):
    info = zipfile.ZipInfo(name, date_time=(2026, 11, 18, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=[VERSION], default=VERSION)
    parser.add_argument("--output", type=Path, default=ROOT / "release" / f"EGO_Mo_v{VERSION}.zip")
    args = parser.parse_args()
    names = (ROOT / FILE_LIST).read_text(encoding="utf-8").splitlines()
    if names != sorted(set(names)):
        raise ValueError("Release file list must be sorted and unique")
    for name in names:
        p = (ROOT / name).resolve()
        if not p.is_relative_to(ROOT) or not p.is_file():
            raise FileNotFoundError(name)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.partial")
    records = []
    with zipfile.ZipFile(temporary, "w") as archive:
        for name in names:
            data = (ROOT / name).read_bytes()
            add(archive, name, data)
            records.append(dict(path=name, bytes=len(data), sha256=digest(data)))
        manifest = dict(version=VERSION, raw_recordings_included=False, files=records)
        add(archive, "MANIFEST.json", json.dumps(manifest, indent=2).encode())
    with zipfile.ZipFile(temporary) as archive:
        for row in records:
            if digest(archive.read(row["path"])) != row["sha256"]:
                raise ValueError(row["path"])
    temporary.replace(output)
    report = dict(archive=output.name, version=VERSION, bytes=output.stat().st_size,
                  sha256=digest(output.read_bytes()), files=len(names) + 1)
    output.with_name("release_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
