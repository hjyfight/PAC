"""Launch repeated squash/letterbox AdvPatch/CAPGen runs with monitoring.

Outputs stay under goal-workspace by default. Each run uses
run_letterbox_yolomap_experiment.py with YOLOv5 official val.py metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEEDS = [1101, 2202, 3303]


@dataclass
class Task:
    resize_mode: str
    seed: int
    out_dir: Path
    log_path: Path

    @property
    def name(self) -> str:
        return f"{self.resize_mode}_seed{self.seed}"


@dataclass
class Running:
    task: Task
    proc: subprocess.Popen
    started_at: str


def parse_csv_ints(text: str) -> list[int]:
    vals = []
    for part in text.split(','):
        part = part.strip()
        if part:
            vals.append(int(part))
    return vals


def build_tasks(workspace: Path, modes: list[str], seeds: list[int]) -> list[Task]:
    tasks: list[Task] = []
    for mode in modes:
        for seed in seeds:
            out_dir = workspace / mode / f"seed_{seed}"
            tasks.append(Task(
                resize_mode=mode,
                seed=seed,
                out_dir=out_dir,
                log_path=out_dir / "run.log",
            ))
    return tasks


def result_exists(task: Task) -> bool:
    return (task.out_dir / "official_yolo_results.csv").exists()


def launch(task: Task, python_exe: str, val_batch_size: int, force_eval: bool) -> Running:
    task.out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "-u",
        "-B",
        str(ROOT / "run_letterbox_yolomap_experiment.py"),
        "--out_dir",
        str(task.out_dir),
        "--resize_mode",
        task.resize_mode,
        "--seed",
        str(task.seed),
        "--val_batch_size",
        str(val_batch_size),
    ]
    if force_eval:
        cmd.append("--force_eval")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    log = task.log_path.open("a", encoding="utf-8", errors="replace", buffering=1)
    header = f"\n=== {datetime.now().isoformat(timespec='seconds')} START {task.name}: {' '.join(cmd)} ===\n"
    log.write(header)
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    return Running(task=task, proc=proc, started_at=datetime.now().isoformat(timespec="seconds"))


def run_nvidia_smi() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi"],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).stdout
    except Exception as exc:
        return f"nvidia-smi failed: {exc}\n"


def run_python_process_snapshot() -> str:
    ps_cmd = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -like 'python*' } | "
        "Select-Object ProcessId,CommandLine,CreationDate | "
        "ConvertTo-Json -Depth 4"
    )
    try:
        return subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            cwd=str(ROOT),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).stdout
    except Exception as exc:
        return json.dumps({"error": f"python process snapshot failed: {exc}"}, indent=2)

def collect_results(workspace: Path) -> None:
    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "summarize_goal_experiments.py"), "--workspace", str(workspace)],
        cwd=str(ROOT),
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def write_status(status_dir: Path, pending: list[Task], running: list[Running], done: list[Task]) -> None:
    status_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat(timespec="seconds")
    payload = {
        "timestamp": now,
        "pending": [asdict(t) | {"out_dir": str(t.out_dir), "log_path": str(t.log_path)} for t in pending],
        "running": [
            {
                "task": asdict(r.task) | {"out_dir": str(r.task.out_dir), "log_path": str(r.task.log_path)},
                "pid": r.proc.pid,
                "returncode": r.proc.poll(),
                "started_at": r.started_at,
                "has_result_csv": result_exists(r.task),
            }
            for r in running
        ],
        "done": [asdict(t) | {"out_dir": str(t.out_dir), "log_path": str(t.log_path)} for t in done],
    }
    (status_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    csv_path = status_dir / "status_history.csv"
    new_file = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timestamp", "state", "name", "pid", "returncode", "out_dir"])
        if new_file:
            writer.writeheader()
        for t in pending:
            writer.writerow({"timestamp": now, "state": "pending", "name": t.name, "pid": "", "returncode": "", "out_dir": t.out_dir})
        for r in running:
            writer.writerow({"timestamp": now, "state": "running", "name": r.task.name, "pid": r.proc.pid, "returncode": r.proc.poll(), "out_dir": r.task.out_dir})
        for t in done:
            writer.writerow({"timestamp": now, "state": "done", "name": t.name, "pid": "", "returncode": "0", "out_dir": t.out_dir})

    stamp = now.replace(':', '').replace('-', '').replace('T', '_')
    smi_path = status_dir / f"nvidia_smi_{stamp}.log"
    smi_path.write_text(run_nvidia_smi(), encoding="utf-8")
    proc_path = status_dir / f"python_processes_{stamp}.json"
    proc_path.write_text(run_python_process_snapshot(), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=ROOT / "goal-workspace")
    ap.add_argument("--modes", default="letterbox,squash")
    ap.add_argument("--seeds", default=",".join(str(x) for x in DEFAULT_SEEDS))
    ap.add_argument("--max_parallel", type=int, default=3)
    ap.add_argument("--monitor_interval", type=int, default=1200)
    ap.add_argument("--val_batch_size", type=int, default=16)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--monitor_dir", type=Path, default=None,
                    help="Directory for status.json, status_history.csv, and process/GPU snapshots.")
    ap.add_argument("--force_eval", action="store_true")
    args = ap.parse_args()

    workspace = args.workspace.resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    monitor_dir = (args.monitor_dir.resolve()
                   if args.monitor_dir is not None
                   else workspace / "monitor")
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    seeds = parse_csv_ints(args.seeds)
    tasks = build_tasks(workspace, modes, seeds)

    pending = [t for t in tasks if not result_exists(t)]
    done = [t for t in tasks if result_exists(t)]
    running: list[Running] = []
    print(f"workspace={workspace}")
    print(f"pending={len(pending)} done={len(done)} max_parallel={args.max_parallel}", flush=True)

    write_status(monitor_dir, pending, running, done)
    next_monitor = time.time() + max(10, args.monitor_interval)

    while pending or running:
        launched = False
        while pending and len(running) < args.max_parallel:
            task = pending.pop(0)
            print(f"launch {task.name} -> {task.out_dir}", flush=True)
            running.append(launch(task, args.python, args.val_batch_size, args.force_eval))
            launched = True
        if launched:
            write_status(monitor_dir, pending, running, done)

        still_running: list[Running] = []
        for item in running:
            code = item.proc.poll()
            if code is None:
                still_running.append(item)
                continue
            footer = f"\n=== {datetime.now().isoformat(timespec='seconds')} EXIT {item.task.name}: code {code} ===\n"
            with item.task.log_path.open("a", encoding="utf-8", errors="replace") as log:
                log.write(footer)
            if code == 0 and result_exists(item.task):
                done.append(item.task)
            else:
                print(f"failed {item.task.name}, code={code}; see {item.task.log_path}", flush=True)
        running = still_running

        if time.time() >= next_monitor:
            write_status(monitor_dir, pending, running, done)
            collect_results(workspace)
            print(f"monitor {datetime.now().isoformat(timespec='seconds')}: pending={len(pending)} running={len(running)} done={len(done)}", flush=True)
            next_monitor = time.time() + max(10, args.monitor_interval)

        time.sleep(30)

    write_status(monitor_dir, pending, running, done)
    collect_results(workspace)
    print("all queued experiments finished", flush=True)


if __name__ == "__main__":
    main()
