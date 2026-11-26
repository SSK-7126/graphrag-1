# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Multi-mode benchmark comparison harness for GraphRAG HPC evaluation."""

from __future__ import annotations

import csv
import json
import statistics
import time
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from graphrag.config.enums import IndexingMethod

from hpc_parallel.config import ExecutionMode, HPCParallelConfig, ProfileOutputFormat
from hpc_parallel.runner import run_index

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    import pandas as pd
    from graphrag.config.models.graph_rag_config import GraphRagConfig


@dataclass
class ModeBenchmarkResult:
    """Measured execution result for a single ExecutionMode (aggregated across repetitions)."""

    execution_mode: str
    repetitions: int
    mean_wall_clock_seconds: float
    min_wall_clock_seconds: float
    max_wall_clock_seconds: float
    workflows_completed: int
    errors: int
    speedup_vs_baseline: float | None
    speedup_vs_serial: float | None
    peak_memory_rss_mb: float | None
    profile_dir: str
    status: str


async def compare_modes(
    *,
    modes: list[ExecutionMode],
    config_factory: Callable[[Path, ExecutionMode], GraphRagConfig],
    output_dir: Path,
    method: IndexingMethod | str = IndexingMethod.Standard,
    input_documents: pd.DataFrame | None = None,
    skip_validation: bool = True,
    use_cache: bool = False,
    concurrency_limit: int | None = None,
    repetitions: int = 1,
    hpc_overrides: dict[str, Any] | None = None,
) -> list[ModeBenchmarkResult]:
    """Run multiple execution modes over ``repetitions`` and export comparison reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reps = max(1, repetitions)
    raw_runs: dict[str, dict[str, Any]] = {}

    for mode in modes:
        durations: list[float] = []
        last_workflows = 0
        total_errors = 0
        peak_rss_values: list[float] = []
        last_profile_dir = output_dir / mode.value / "profile"

        for rep in range(reps):
            mode_root = output_dir / mode.value / (f"rep_{rep + 1}" if reps > 1 else "")
            profile_dir = mode_root / "profile"
            last_profile_dir = profile_dir
            cfg = config_factory(mode_root, mode)
            hpc_kwargs: dict[str, Any] = {
                "execution_mode": mode,
                "profiling_enabled": True,
                "profile_output_dir": profile_dir,
                "profile_format": ProfileOutputFormat.BOTH,
                "run_id": f"compare-{mode.value}-r{rep + 1}",
                "concurrency_limit": concurrency_limit,
            }
            if hpc_overrides:
                hpc_kwargs.update(hpc_overrides)
            hpc_cfg = HPCParallelConfig(**hpc_kwargs)

            start = time.perf_counter()
            outputs = await run_index(
                config=cfg,
                hpc_config=hpc_cfg,
                method=method,
                use_cache=use_cache,
                skip_validation=skip_validation,
                input_documents=input_documents,
            )
            durations.append(time.perf_counter() - start)
            last_workflows = len(outputs)
            total_errors += sum(1 for o in outputs if o.error is not None)

            prof_json = profile_dir / "function_timings.json"
            if prof_json.exists():
                pdata = json.loads(prof_json.read_text(encoding="utf-8"))
                peak_rss_values.extend(
                    float(s["max_memory_rss_mb"])
                    for s in pdata.get("summaries", [])
                    if s.get("max_memory_rss_mb") is not None
                )

        raw_runs[mode.value] = {
            "durations": durations,
            "mean": statistics.mean(durations),
            "min": min(durations),
            "max": max(durations),
            "workflows_completed": last_workflows,
            "errors": total_errors,
            "peak_rss": max(peak_rss_values) if peak_rss_values else None,
            "profile_dir": str(last_profile_dir),
        }

    baseline_mean = raw_runs.get(ExecutionMode.BASELINE.value, {}).get("mean")
    serial_mean = raw_runs.get(ExecutionMode.SERIAL.value, {}).get("mean")

    results: list[ModeBenchmarkResult] = []
    for mode in modes:
        info = raw_runs[mode.value]
        mean_t = info["mean"]
        results.append(
            ModeBenchmarkResult(
                execution_mode=mode.value,
                repetitions=reps,
                mean_wall_clock_seconds=mean_t,
                min_wall_clock_seconds=info["min"],
                max_wall_clock_seconds=info["max"],
                workflows_completed=info["workflows_completed"],
                errors=info["errors"],
                speedup_vs_baseline=(
                    round(baseline_mean / mean_t, 4)
                    if baseline_mean and mean_t > 0
                    else None
                ),
                speedup_vs_serial=(
                    round(serial_mean / mean_t, 4)
                    if serial_mean and mean_t > 0
                    else None
                ),
                peak_memory_rss_mb=info["peak_rss"],
                profile_dir=info["profile_dir"],
                status="ok" if info["errors"] == 0 else "error",
            )
        )

    csv_path = output_dir / "comparison_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "execution_mode",
                "repetitions",
                "mean_wall_clock_seconds",
                "min_wall_clock_seconds",
                "max_wall_clock_seconds",
                "workflows_completed",
                "errors",
                "speedup_vs_baseline",
                "speedup_vs_serial",
                "peak_memory_rss_mb",
                "profile_dir",
                "status",
            ],
        )
        writer.writeheader()
        for res in results:
            writer.writerow(asdict(res))

    json_path = output_dir / "comparison_summary.json"
    payload: dict[str, Any] = {"results": [asdict(r) for r in results]}
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return results
