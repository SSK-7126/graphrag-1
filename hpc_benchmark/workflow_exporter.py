import csv
import json
from pathlib import Path

def export_workflow_stats(workspace: Path, results: Path):
    stats_path = workspace / "output" / "stats.json"

    if not stats_path.exists():
        raise FileNotFoundError(f"GraphRAG stats not found: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    results.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, metrics in stats.get("workflows", {}).items():
        rows.append({
            "workflow": name,
            "duration_seconds": metrics.get("overall", 0),
            "peak_python_memory_mb": metrics.get("peak_memory_bytes", 0) / (1024 ** 2),
            "memory_delta_mb": metrics.get("memory_delta_bytes", 0) / (1024 ** 2),
            "tracemalloc_overhead_mb": metrics.get("tracemalloc_overhead_bytes", 0) / (1024 ** 2),
        })

    csv_path = results / "workflow_timings.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys() if rows else [
            "workflow", "duration_seconds", "peak_python_memory_mb",
            "memory_delta_mb", "tracemalloc_overhead_mb"
        ])
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "total_runtime_seconds": stats.get("total_runtime", 0),
        "num_documents": stats.get("num_documents", 0),
        "update_documents": stats.get("update_documents", 0),
        "workflow_count": len(rows),
        "workflow_time_sum_seconds": sum(r["duration_seconds"] for r in rows),
        "workflows": rows,
    }

    with (results / "benchmark_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"Workflow CSV: {csv_path}")
    print(f"Summary JSON: {results / 'benchmark_summary.json'}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()

    export_workflow_stats(Path(args.workspace), Path(args.results))
