import subprocess
from pathlib import Path

opts = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
local = Path(r"E:\EGO_Mo\datasets")
jobs = [
    (
        ["scp", *opts, "-r", r"lry@192.168.1.3:C:/Users/lry/fleet-jobs/ego_mo/datasets/decouple_abs", str(local)],
        "abs",
    ),
    (
        ["scp", *opts, "-r", r"lry@192.168.1.3:C:/Users/lry/fleet-jobs/ego_mo/datasets/decouple_rel", str(local)],
        "rel",
    ),
    (
        ["scp", *opts, "-r", r"liran@192.168.1.5:C:/Users/liran/fleet-jobs/ego_mo/datasets/decouple_corpus", str(local)],
        "corpus",
    ),
]
for argv, name in jobs:
    print("pull", name, flush=True)
    result = subprocess.run(argv, text=True, encoding="utf-8", errors="replace")
    print(name, result.returncode, flush=True)
    if result.stderr:
        print(result.stderr[-500:], flush=True)
