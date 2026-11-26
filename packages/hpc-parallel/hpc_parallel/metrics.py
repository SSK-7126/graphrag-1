# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Data models and aggregation helpers for function-level profiling metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FunctionTimingRecord:
    """Single function execution timing and concurrency record.

    Stores strictly non-sensitive operational metadata; never stores prompt
    text, document contents, model outputs, or credentials.
    """

    event_id: str
    run_id: str
    execution_mode: str
    function_name: str
    workflow_name: str | None
    start_timestamp: str
    end_timestamp: str
    wall_clock_seconds: float
    input_count: int
    request_count: int
    batch_count: int
    errors: int
    configured_concurrency: int | None
    concurrency_at_start: int
    peak_concurrency: int
    status: str
    error_type: str | None = None
    platform: str = "asyncio"
    output_count: int = 0
    worker_count: int = 1
    scheduling_overhead_seconds: float = 0.0
    retries: int = 0
    cpu_percent: float | None = None
    memory_rss_mb: float | None = None
    gpu_utilization_percent: float | None = None
    gpu_memory_used_mb: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a plain dictionary."""
        return asdict(self)


@dataclass
class FunctionSummary:
    """Aggregated summary metrics for a single instrumented function."""

    function_name: str
    call_count: int
    total_wall_clock_seconds: float
    mean_wall_clock_seconds: float
    min_wall_clock_seconds: float
    max_wall_clock_seconds: float
    total_input_count: int
    total_output_count: int
    total_request_count: int
    total_batch_count: int
    total_errors: int
    total_retries: int
    total_scheduling_overhead_seconds: float
    max_concurrency_observed: int
    configured_concurrency: int | None
    platforms_used: list[str]
    max_memory_rss_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        """Convert summary to a plain dictionary."""
        return asdict(self)


def summarize_records(
    records: list[FunctionTimingRecord],
) -> list[FunctionSummary]:
    """Compute per-function summary statistics from timing records."""
    grouped: dict[str, list[FunctionTimingRecord]] = defaultdict(list)
    for record in records:
        grouped[record.function_name].append(record)

    summaries: list[FunctionSummary] = []
    for func_name, func_records in sorted(grouped.items()):
        durations = [r.wall_clock_seconds for r in func_records]
        total_time = sum(durations)
        call_count = len(func_records)
        configured_values = [
            r.configured_concurrency
            for r in func_records
            if r.configured_concurrency is not None
        ]
        rss_values = [
            r.memory_rss_mb for r in func_records if r.memory_rss_mb is not None
        ]
        summaries.append(
            FunctionSummary(
                function_name=func_name,
                call_count=call_count,
                total_wall_clock_seconds=total_time,
                mean_wall_clock_seconds=total_time / call_count if call_count else 0.0,
                min_wall_clock_seconds=min(durations) if durations else 0.0,
                max_wall_clock_seconds=max(durations) if durations else 0.0,
                total_input_count=sum(r.input_count for r in func_records),
                total_output_count=sum(r.output_count for r in func_records),
                total_request_count=sum(r.request_count for r in func_records),
                total_batch_count=sum(r.batch_count for r in func_records),
                total_errors=sum(r.errors for r in func_records),
                total_retries=sum(r.retries for r in func_records),
                total_scheduling_overhead_seconds=sum(
                    r.scheduling_overhead_seconds for r in func_records
                ),
                max_concurrency_observed=max(
                    (r.peak_concurrency for r in func_records), default=0
                ),
                configured_concurrency=(
                    max(configured_values) if configured_values else None
                ),
                platforms_used=sorted({r.platform for r in func_records}),
                max_memory_rss_mb=max(rss_values) if rss_values else None,
            )
        )
    return summaries
