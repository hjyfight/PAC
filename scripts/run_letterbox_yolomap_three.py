"""Run three letterbox + YOLO official experiments sequentially with logs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_PREFIX = "output_advpatch_retrain_20260622-letterbox_yolomap"
PYTHON = sys.executable


def run_one(index: int) -> None:
    out_dir = ROOT / f"{OUT_PREFIX}-{index}"
    out_dir.mkdir(parents=True, exist_ok=False)
    log_path = out_dir / "run.log"
    cmd = [
        PYTHON,
        "-B",
        str(ROOT / "run_letterbox_yolomap_experiment.py"),
        "--out_dir",
        str(out_dir),
        "--val_batch_size",
        "16",
    ]

    header = f"\n=== RUN {index}: {' '.join(cmd)} ===\n"
    print(header, flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(header)
        log.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()

        code = proc.wait()
        footer = f"\n=== RUN {index} exited with code {code} ===\n"
        print(footer, flush=True)
        log.write(footer)
        log.flush()
        if code != 0:
            raise SystemExit(code)


def main() -> None:
    for index in (1, 2, 3):
        run_one(index)


if __name__ == "__main__":
    main()
