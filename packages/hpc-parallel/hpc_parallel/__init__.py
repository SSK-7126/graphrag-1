# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Separate HPC parallelization and profiling package for GraphRAG."""

from __future__ import annotations

from hpc_parallel.benchmark import ModeBenchmarkResult, compare_modes
from hpc_parallel.config import (
    ExecutionMode,
    HPCParallelConfig,
    ProfileOutputFormat,
    load_config_from_env,
)
from hpc_parallel.hooks import (
    disable_profiling,
    enable_from_env,
    enable_profiling,
    get_active_profiler,
    is_profiling_enabled,
    profiling_session,
)
from hpc_parallel.metrics import (
    FunctionSummary,
    FunctionTimingRecord,
    summarize_records,
)
from hpc_parallel.profiler import FunctionProfiler, FunctionSpan
from hpc_parallel.runner import run_index, validate_execution_mode

__all__ = [
    "ExecutionMode",
    "FunctionProfiler",
    "FunctionSpan",
    "FunctionSummary",
    "FunctionTimingRecord",
    "HPCParallelConfig",
    "ModeBenchmarkResult",
    "ProfileOutputFormat",
    "compare_modes",
    "disable_profiling",
    "enable_from_env",
    "enable_profiling",
    "get_active_profiler",
    "is_profiling_enabled",
    "load_config_from_env",
    "profiling_session",
    "run_index",
    "summarize_records",
    "validate_execution_mode",
]
