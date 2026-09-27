import subprocess
import sys

DISPATCH = r"F:\worldModelEndoscopic\fleet\dispatch.py"
PROBES = {
    "r9000k": "D:\\anaconda\\envs\\reloc3r\\python.exe -c \"import torch,numpy,scipy,cv2; print(torch.__version__, int(torch.cuda.is_available()))\"",
    "shenzhou": "C:\\Users\\liran\\AppData\\Local\\Programs\\Python\\Python311\\python.exe -c \"import numpy,scipy,cv2; print(1)\"",
}


def main() -> None:
    host = sys.argv[1]
    result = subprocess.run(
        [sys.executable, DISPATCH, "run", host, PROBES[host]],
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    sys.stdout.write(result.stdout or "")
    sys.stderr.write(result.stderr or "")
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
