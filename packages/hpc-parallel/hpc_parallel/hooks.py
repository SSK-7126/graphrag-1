# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Bridge between GraphRAG profiling hooks and hpc_parallel.profiler."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from graphrag.index.run import hpc_hooks

from hpc_parallel.config import (
    ExecutionMode,
    HPCParallelConfig,
    ProfileOutputFormat,
    load_config_from_env,
)
from hpc_parallel.profiler import FunctionProfiler, FunctionSpan

if TYPE_CHECKING:
    from collections.abc import Generator

_state_lock = threading.Lock()
_active_profiler: FunctionProfiler | None = None
_active_config: HPCParallelConfig | None = None


class _ProfilerHookAdapter:
    """Adapter implementing GraphRAG's HPCHookHandler interface."""

    def __init__(
        self,
        profiler: FunctionProfiler,
        config: HPCParallelConfig,
    ) -> None:
        self.profiler = profiler
        self.config = config

    def start_span(
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
        """Start a FunctionSpan on the attached profiler."""
        return self.profiler.span(
            function_name,
            input_count=input_count,
            request_count=request_count,
            batch_count=batch_count,
            errors=errors,
            configured_concurrency=configured_concurrency,
            workflow_name=workflow_name,
        )

    def workflow_context(self, workflow_name: str):
        """Return workflow context manager."""
        return self.profiler.workflow_context(workflow_name)

    def record_active_request(self, count: int = 1) -> None:
        """Record an active request on the current ContextVar span."""
        self.profiler.record_active_request(count)

    def flush(self) -> list[Path]:
        """Export recorded profiling artifacts to configured directory."""
        return self.profiler.export(
            self.config.profile_output_dir,
            self.config.profile_format,
        )


def enable_profiling(
    config: HPCParallelConfig | None = None,
) -> FunctionProfiler:
    """Attach an active FunctionProfiler to GraphRAG hooks."""
    global _active_profiler, _active_config
    cfg = config or HPCParallelConfig(profiling_enabled=True)
    profiler = FunctionProfiler(
        execution_mode=cfg.execution_mode,
        run_id=cfg.run_id,
    )
    adapter = _ProfilerHookAdapter(profiler, cfg)
    with _state_lock:
        _active_profiler = profiler
        _active_config = cfg
        hpc_hooks.set_hook_handler(adapter)
    return profiler


def disable_profiling() -> None:
    """Detach the active profiler and restore zero-overhead GraphRAG defaults."""
    global _active_profiler, _active_config
    with _state_lock:
        _active_profiler = None
        _active_config = None
        hpc_hooks.set_hook_handler(None)


def get_active_profiler() -> FunctionProfiler | None:
    """Return the currently attached FunctionProfiler, if any."""
    with _state_lock:
        return _active_profiler


def is_profiling_enabled() -> bool:
    """Return True when a profiler is attached to GraphRAG hooks."""
    return hpc_hooks.is_profiling_active()


def enable_from_env() -> FunctionProfiler | None:
    """Initialize profiling from environment variables if enabled."""
    cfg = load_config_from_env()
    if not cfg.profiling_enabled:
        return None
    return enable_profiling(cfg)


@contextmanager
def profiling_session(
    *,
    enabled: bool = True,
    output_dir: str | Path = "benchmark/hpc_profile",
    output_format: ProfileOutputFormat | str = ProfileOutputFormat.BOTH,
    execution_mode: ExecutionMode | str = ExecutionMode.BASELINE,
    run_id: str | None = None,
    auto_export: bool = True,
) -> Generator[FunctionProfiler | None, None, None]:
    """Context manager enabling function-level profiling for its scope."""
    if not enabled:
        yield None
        return

    cfg = HPCParallelConfig(
        execution_mode=ExecutionMode(execution_mode),
        profiling_enabled=True,
        profile_output_dir=Path(output_dir),
        profile_format=ProfileOutputFormat(output_format),
        run_id=run_id,
    )
    previous_handler = hpc_hooks.get_hook_handler()
    previous_profiler = get_active_profiler()
    previous_config = _active_config

    profiler = enable_profiling(cfg)
    try:
        yield profiler
    finally:
        try:
            if auto_export:
                profiler.export(cfg.profile_output_dir, cfg.profile_format)
        finally:
            with _state_lock:
                global _active_profiler
                _active_profiler = previous_profiler
                globals()["_active_config"] = previous_config
                hpc_hooks.set_hook_handler(previous_handler)
