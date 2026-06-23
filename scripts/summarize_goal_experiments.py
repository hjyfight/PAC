"""Summarize goal-workspace YOLO official experiment results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


METHOD_ORDER = ["clean", "advpatch", "p0", "p1", "p2", "r1", "r2", "t1", "t2"]


def infer_resize_mode(result_path: Path) -> str:
    parts = [p.lower() for p in result_path.parts]
    for mode in ("letterbox", "squash"):
        if mode in parts or any(part.startswith(mode + "_") for part in parts):
            return mode
    return "unknown"


def read_results(workspace: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(workspace.rglob("official_yolo_results.csv")):
        rel = csv_path.relative_to(workspace)
        run_name = str(rel.parent).replace("\\", "/")
        resize_mode = infer_resize_mode(csv_path)
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                item = dict(row)
                item["run"] = run_name
                item["resize_mode"] = resize_mode
                item["result_csv"] = str(csv_path)
                rows.append(item)
    return rows


def fnum(row: dict[str, str], key: str) -> float | None:
    try:
        return float(row[key])
    except Exception:
        return None


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path("goal-workspace"))
    args = ap.parse_args()

    workspace = args.workspace.resolve()
    rows = read_results(workspace)
    fields = [
        "run", "resize_mode", "method", "mAP50_percent", "mAP50_95_percent",
        "precision", "recall", "images", "instances", "result_csv",
    ]
    rows_sorted = sorted(
        rows,
        key=lambda r: (r.get("resize_mode", ""), r.get("run", ""), METHOD_ORDER.index(r["method"]) if r.get("method") in METHOD_ORDER else 99),
    )
    write_csv(workspace / "summary_all_results.csv", rows_sorted, fields)

    best_rows: list[dict[str, object]] = []
    for mode in sorted({r.get("resize_mode", "unknown") for r in rows}):
        mode_rows = [r for r in rows if r.get("resize_mode") == mode and r.get("method") != "clean"]
        for method in METHOD_ORDER:
            vals = [r for r in mode_rows if r.get("method") == method and fnum(r, "mAP50_percent") is not None]
            if not vals:
                continue
            best = min(vals, key=lambda r: fnum(r, "mAP50_percent") or 1e9)
            best_rows.append(best)
    write_csv(workspace / "best_by_method.csv", best_rows, fields)

    print(f"found result files: {len(set(r['result_csv'] for r in rows))}")
    print(f"rows: {len(rows)}")
    if best_rows:
        print("best mAP50 by resize/method:")
        for row in best_rows:
            print(f"  {row['resize_mode']:9s} {row['method']:8s} {float(row['mAP50_percent']):6.2f}  {row['run']}")


if __name__ == "__main__":
    main()