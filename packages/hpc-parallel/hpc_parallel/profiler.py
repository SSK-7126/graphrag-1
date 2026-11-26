# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Thread-safe and asyncio-safe function-level profiler."""

from __future__ import annotations

import contextvars
import csv
import json
import threading
import time
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from typing_extensions import Self

from hpc_parallel.config import ExecutionMode, ProfileOutputFormat
from hpc_parallel.metrics import (
    FunctionSummary,
    FunctionTimingRecord,
    summarize_records,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path
    from types import TracebackType

_workflow_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "hpc_parallel_workflow", default=None
)
_active_span_var: contextvars.ContextVar[FunctionSpan | None] = contextvars.ContextVar(
    "hpc_parallel_active_span", default=None
)


class _SubtaskContext:
    """Context manager tracking in-flight subtasks inside a parent span."""

    def __init__(self, span: FunctionSpan) -> None:
        self._span = span

    def __enter__(self) -> Self:
        self._span._enter_subtask()  # noqa: SLF001
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self._span._exit_subtask(exc_type)  # noqa: SLF001


def _sample_process_resources() -> tuple[float | None, float | None]:
    """Sample current process CPU percent and RSS memory (MB) if psutil is available."""
    try:
        import psutil

        proc = psutil.Process()
        cpu_pct = float(proc.cpu_percent(interval=None))
        rss_mb = round(proc.memory_info().rss / (1024**2), 3)
    except Exception:  # noqa: BLE001
        return None, None
    else:
        return cpu_pct, rss_mb


def _sample_gpu_resources() -> tuple[float | None, float | None]:
    """Sample CUDA GPU utilization (%) and memory used (MB) if torch.cuda is active."""
    try:
        import importlib

        torch: Any = importlib.import_module("torch")
        cuda_mod: Any = getattr(torch, "cuda", None)
        if cuda_mod is None or not cuda_mod.is_available():
            return None, None
        mem_mb = round(float(cuda_mod.memory_allocated(0)) / (1024**2), 3)
        util_fn = getattr(cuda_mod, "utilization", None)
        if callable(util_fn):
            raw_util: Any = util_fn(0)
            util_pct: float | None = float(raw_util)
        else:
            util_pct = None
    except Exception:  # noqa: BLE001
        return None, None
    else:
        return util_pct, mem_mb


class FunctionSpan:
    """Active timing and concurrency span for a single function invocation."""

    def __init__(
        self,
        profiler: FunctionProfiler,
        function_name: str,
        *,
        input_count: int = 0,
        request_count: int = 0,
        batch_count: int = 0,
        errors: int = 0,
        configured_concurrency: int | None = None,
        workflow_name: str | None = None,
        platform: str = "asyncio",
        output_count: int = 0,
        worker_count: int = 1,
        scheduling_overhead_seconds: float = 0.0,
        retries: int = 0,
    ) -> None:
        self._profiler = profiler
        self.function_name = function_name
        self.input_count = input_count
        self.request_count = request_count
        self.batch_count = batch_count
        self.errors = errors
        self.configured_concurrency = configured_concurrency
        self.workflow_name = (
            workflow_name if workflow_name is not None else _workflow_var.get()
        )
        self.platform = platform
        self.output_count = output_count
        self.worker_count = worker_count
        self.scheduling_overhead_seconds = scheduling_overhead_seconds
        self.retries = retries
        self._parent_span: FunctionSpan | None = None
        self._span_token: contextvars.Token[FunctionSpan | None] | None = None
        self._lock = threading.Lock()
        self._start_perf = 0.0
        self._start_iso = ""
        self._concurrency_at_start = 0
        self._peak_concurrency = 0
        self._active_subtasks = 0
        self._peak_subtasks = 0
        self._error_type: str | None = None
        self._closed = False

    def __enter__(self) -> Self:
        """Start timing and register active concurrency."""
        self._parent_span = _active_span_var.get()
        if (
            self.configured_concurrency is None
            and self._parent_span is not None
            and self._parent_span.configured_concurrency is not None
        ):
            self.configured_concurrency = self._parent_span.configured_concurrency

        self._span_token = _active_span_var.set(self)
        self._start_iso = datetime.now(timezone.utc).isoformat()
        self._start_perf = time.perf_counter()

        active_now = self._profiler._enter_function(self.function_name)  # noqa: SLF001
        self._concurrency_at_start = active_now
        self._peak_concurrency = active_now
        self._profiler._notify_active_concurrency(  # noqa: SLF001
            self.function_name, active_now
        )
        if self._parent_span is not None:
            self._parent_span.observe_concurrency(active_now)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Stop timing, capture errors, and record metrics."""
        if self._closed:
            return
        self._closed = True

        end_perf = time.perf_counter()
        end_iso = datetime.now(timezone.utc).isoformat()
        elapsed = max(0.0, end_perf - self._start_perf)

        self._profiler._exit_function(self.function_name)  # noqa: SLF001
        if self._span_token is not None:
            _active_span_var.reset(self._span_token)

        if exc_type is not None:
            if self.errors == 0:
                self.errors = 1
            if self._error_type is None:
                self._error_type = exc_type.__name__

        status = "error" if (exc_type is not None or self.errors > 0) else "ok"
        peak = max(
            self._peak_concurrency,
            self._peak_subtasks,
            self._profiler.get_peak_concurrency(self.function_name),
        )
        # Keep per-span peak bounded by the max concurrency observed while this span ran
        # or subtasks spawned inside it.
        with self._lock:
            span_peak = max(self._peak_concurrency, self._peak_subtasks)
        _ = peak  # global peak tracked separately by FunctionProfiler

        cpu_pct, rss_mb = _sample_process_resources()
        gpu_util, gpu_mem = _sample_gpu_resources()

        record = FunctionTimingRecord(
            event_id=uuid.uuid4().hex,
            run_id=self._profiler.run_id,
            execution_mode=self._profiler.execution_mode.value,
            function_name=self.function_name,
            workflow_name=self.workflow_name,
            start_timestamp=self._start_iso,
            end_timestamp=end_iso,
            wall_clock_seconds=elapsed,
            input_count=self.input_count,
            request_count=self.request_count,
            batch_count=self.batch_count,
            errors=self.errors,
            configured_concurrency=self.configured_concurrency,
            concurrency_at_start=self._concurrency_at_start,
            peak_concurrency=span_peak,
            status=status,
            error_type=self._error_type,
            platform=self.platform,
            output_count=self.output_count,
            worker_count=self.worker_count,
            scheduling_overhead_seconds=self.scheduling_overhead_seconds,
            retries=self.retries,
            cpu_percent=cpu_pct,
            memory_rss_mb=rss_mb,
            gpu_utilization_percent=gpu_util,
            gpu_memory_used_mb=gpu_mem,
        )
        self._profiler.record(record)

    def set_counts(
        self,
        *,
        input_count: int | None = None,
        request_count: int | None = None,
        batch_count: int | None = None,
        errors: int | None = None,
        configured_concurrency: int | None = None,
        output_count: int | None = None,
        worker_count: int | None = None,
        scheduling_overhead_seconds: float | None = None,
        retries: int | None = None,
        platform: str | None = None,
    ) -> None:
        """Update numeric counters and platform metadata on the span."""
        with self._lock:
            if input_count is not None:
                self.input_count = input_count
            if request_count is not None:
                self.request_count = request_count
            if batch_count is not None:
                self.batch_count = batch_count
            if errors is not None:
                self.errors = errors
            if configured_concurrency is not None:
                self.configured_concurrency = configured_concurrency
            if output_count is not None:
                self.output_count = output_count
            if worker_count is not None:
                self.worker_count = worker_count
            if scheduling_overhead_seconds is not None:
                self.scheduling_overhead_seconds = scheduling_overhead_seconds
            if retries is not None:
                self.retries = retries
            if platform is not None:
                self.platform = platform

    def increment_requests(self, count: int = 1) -> None:
        """Increment request_count thread-safely."""
        with self._lock:
            self.request_count += count

    def increment_errors(
        self, count: int = 1, *, error_type: str | None = None
    ) -> None:
        """Increment errors count thread-safely without logging sensitive payloads."""
        with self._lock:
            self.errors += count
            if error_type and self._error_type is None:
                self._error_type = error_type

    def observe_concurrency(self, concurrency: int) -> None:
        """Observe concurrency observed during this span's lifetime."""
        with self._lock:
            if concurrency > self._peak_concurrency:
                self._peak_concurrency = concurrency
        if self._parent_span is not None:
            self._parent_span.observe_concurrency(concurrency)

    def subtask(self) -> _SubtaskContext:
        """Return a context manager tracking an in-flight concurrent subtask."""
        return _SubtaskContext(self)

    def _enter_subtask(self) -> None:
        with self._lock:
            self._active_subtasks += 1
            if self._active_subtasks > self._peak_subtasks:
                self._peak_subtasks = self._active_subtasks
            if self._active_subtasks > self._peak_concurrency:
                self._peak_concurrency = self._active_subtasks
            active = self._active_subtasks
        self._profiler._notify_subtask_concurrency(self.function_name, active)  # noqa: SLF001
        if self._parent_span is not None:
            self._parent_span.observe_concurrency(active)

    def _exit_subtask(self, exc_type: type[BaseException] | None) -> None:
        with self._lock:
            self._active_subtasks = max(0, self._active_subtasks - 1)
            if exc_type is not None:
                self.errors += 1
                if self._error_type is None:
                    self._error_type = exc_type.__name__


class FunctionProfiler:
    """Thread-safe collector and exporter for GraphRAG function-level timings."""

    def __init__(
        self,
        *,
        execution_mode: ExecutionMode = ExecutionMode.BASELINE,
        run_id: str | None = None,
    ) -> None:
        self.execution_mode = execution_mode
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.ended_at: str | None = None
        self._lock = threading.Lock()
        self._records: list[FunctionTimingRecord] = []
        self._active_by_func: dict[str, int] = defaultdict(int)
        self._peak_by_func: dict[str, int] = defaultdict(int)
        self._open_spans_by_func: dict[str, set[FunctionSpan]] = defaultdict(set)

    def span(
        self,
        function_name: str,
        *,
        input_count: int = 0,
        request_count: int = 0,
        batch_count: int = 0,
        errors: int = 0,
        configured_concurrency: int | None = None,
        workflow_name: str | None = None,
    ) -> FunctionSpan:
        """Create a timing/concurrency span for a function invocation."""
        span = FunctionSpan(
            self,
            function_name,
            input_count=input_count,
            request_count=request_count,
            batch_count=batch_count,
            errors=errors,
            configured_concurrency=configured_concurrency,
            workflow_name=workflow_name,
        )
        with self._lock:
            self._open_spans_by_func[function_name].add(span)
        return span

    def _enter_function(self, function_name: str) -> int:
        with self._lock:
            self._active_by_func[function_name] += 1
            active = self._active_by_func[function_name]
            if active > self._peak_by_func[function_name]:
                self._peak_by_func[function_name] = active
            return active

    def _notify_active_concurrency(self, function_name: str, active: int) -> None:
        with self._lock:
            open_spans = list(self._open_spans_by_func.get(function_name, ()))
        for open_span in open_spans:
            open_span.observe_concurrency(active)

    def _notify_subtask_concurrency(self, function_name: str, active: int) -> None:
        with self._lock:
            if active > self._peak_by_func[function_name]:
                self._peak_by_func[function_name] = active

    def _exit_function(self, function_name: str) -> None:
        with self._lock:
            self._active_by_func[function_name] = max(
                0, self._active_by_func[function_name] - 1
            )

    def get_peak_concurrency(self, function_name: str) -> int:
        """Return peak concurrency observed for a given function name."""
        with self._lock:
            return self._peak_by_func.get(function_name, 0)

    def record(self, timing_record: FunctionTimingRecord) -> None:
        """Append a completed timing record."""
        with self._lock:
            self._records.append(timing_record)
            # Clean up any closed spans from tracking set
            open_set = self._open_spans_by_func.get(timing_record.function_name)
            if open_set is not None:
                closed = [s for s in open_set if s._closed]  # noqa: SLF001
                for s in closed:
                    open_set.discard(s)

    def record_active_request(self, count: int = 1) -> None:
        """Increment request count on the currently active ContextVar span, if any."""
        active_span = _active_span_var.get()
        if active_span is not None:
            active_span.increment_requests(count)

    @property
    def records(self) -> list[FunctionTimingRecord]:
        """Snapshot copy of all recorded function timing records."""
        with self._lock:
            return list(self._records)

    def summarize(self) -> list[FunctionSummary]:
        """Compute per-function summary metrics."""
        return summarize_records(self.records)

    @contextmanager
    def workflow_context(self, workflow_name: str) -> Generator[None, None, None]:
        """Tag nested spans with the current GraphRAG workflow name."""
        token = _workflow_var.set(workflow_name)
        try:
            yield
        finally:
            _workflow_var.reset(token)

    def export_csv(self, output_dir: Path) -> tuple[Path, Path]:
        """Export individual function timings and per-function summaries to CSV."""
        output_dir.mkdir(parents=True, exist_ok=True)
        records = self.records
        summaries = summarize_records(records)

        timings_path = output_dir / "function_timings.csv"
        record_fields = [
            "event_id",
            "run_id",
            "execution_mode",
            "platform",
            "function_name",
            "workflow_name",
            "start_timestamp",
            "end_timestamp",
            "wall_clock_seconds",
            "scheduling_overhead_seconds",
            "input_count",
            "output_count",
            "request_count",
            "batch_count",
            "errors",
            "retries",
            "worker_count",
            "configured_concurrency",
            "concurrency_at_start",
            "peak_concurrency",
            "cpu_percent",
            "memory_rss_mb",
            "gpu_utilization_percent",
            "gpu_memory_used_mb",
            "status",
            "error_type",
        ]
        with timings_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=record_fields)
            writer.writeheader()
            for rec in records:
                writer.writerow(rec.to_dict())

        summary_path = output_dir / "function_summary.csv"
        summary_fields = [
            "function_name",
            "call_count",
            "total_wall_clock_seconds",
            "mean_wall_clock_seconds",
            "min_wall_clock_seconds",
            "max_wall_clock_seconds",
            "total_input_count",
            "total_output_count",
            "total_request_count",
            "total_batch_count",
            "total_errors",
            "total_retries",
            "total_scheduling_overhead_seconds",
            "max_concurrency_observed",
            "configured_concurrency",
            "platforms_used",
            "max_memory_rss_mb",
        ]
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=summary_fields)
            writer.writeheader()
            for summary in summaries:
                writer.writerow(summary.to_dict())

        return timings_path, summary_path

    def export_json(self, output_dir: Path) -> Path:
        """Export run metadata, summaries, and individual timing records to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        self.ended_at = datetime.now(timezone.utc).isoformat()
        records = self.records
        summaries = summarize_records(records)

        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "execution_mode": self.execution_mode.value,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "total_records": len(records),
            "summaries": [s.to_dict() for s in summaries],
            "records": [r.to_dict() for r in records],
        }
        json_path = output_dir / "function_timings.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return json_path

    def export(
        self,
        output_dir: Path,
        output_format: ProfileOutputFormat = ProfileOutputFormat.BOTH,
    ) -> list[Path]:
        """Export profiling metrics in the requested format(s)."""
        written: list[Path] = []
        if output_format in {ProfileOutputFormat.CSV, ProfileOutputFormat.BOTH}:
            timings_csv, summary_csv = self.export_csv(output_dir)
            written.extend([timings_csv, summary_csv])
        if output_format in {ProfileOutputFormat.JSON, ProfileOutputFormat.BOTH}:
            written.append(self.export_json(output_dir))
        return written
