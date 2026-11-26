import argparse
import subprocess
import sys
import time
from pathlib import Path

from system_monitor import SystemMonitor
from workflow_exporter import export_workflow_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    workspace = Path(args.workspace).resolve()
    results = workspace / "benchmark"

    results.mkdir(parents=True, exist_ok=True)

    command = [
        "uv",
        "--directory", str(repo),
        "run", "--package", "graphrag",
        "graphrag", "index",
        "--root", str(workspace),
    ]

    print("Starting GraphRAG HPC benchmark...")
    print("Command:", subprocess.list2cmdline(command))

    start = time.perf_counter()
    process = subprocess.Popen(command, cwd=repo)

    monitor = SystemMonitor(
        process=process,
        output_file=results / "resource_usage.csv",
        interval=args.interval,
    )

    monitor.start()

    try:
        return_code = process.wait()
    except KeyboardInterrupt:
        print("Interrupted. Stopping GraphRAG...")
        process.terminate()
        process.wait()
        raise
    finally:
        monitor.stop()

    elapsed = time.perf_counter() - start
    print(f"Benchmark command elapsed time: {elapsed:.3f} seconds")

    if return_code != 0:
        raise SystemExit(f"GraphRAG failed with exit code {return_code}")

    export_workflow_stats(workspace, results)
    print("Benchmark completed successfully.")


if __name__ == "__main__":
    main()
