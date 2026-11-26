# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Configuration and execution mode definitions for hpc_parallel."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExecutionMode(str, Enum):
    """Supported execution modes for GraphRAG HPC evaluation."""

    BASELINE = "baseline"
    """Original GraphRAG execution path (reference baseline)."""

    SERIAL = "serial"
    """Strict single-worker sequential execution."""

    ASYNCIO_OPTIMIZED = "asyncio_optimized"
    """Optimized asyncio execution (bounded queue, barrier-free summarization, pipelined I/O)."""

    CPU_HYBRID = "cpu_hybrid"
    """CPU hybrid execution (asyncio for LLM I/O + ProcessPool/ThreadPool for CPU stages)."""

    MULTI_PLATFORM_HYBRID = "multi_platform_hybrid"
    """Integrated multi-platform hybrid execution (asyncio + Ray + Dask + CUDA)."""

    RAY = "ray"
    """Ray-focused execution (alias/subset for Ray-enabled CPU stages + asyncio)."""

    DASK = "dask"
    """Dask-focused execution (alias/subset for Dask-enabled stages + asyncio)."""

    CUDA = "cuda"
    """CUDA-focused execution (CUDA local tensor ops + asyncio)."""


class ProfileOutputFormat(str, Enum):
    """Output formats supported by the function profiler."""

    CSV = "csv"
    JSON = "json"
    BOTH = "both"


ENV_PROFILE_ENABLED = "GRAPHRAG_HPC_PROFILE"
ENV_EXECUTION_MODE = "GRAPHRAG_HPC_MODE"
ENV_PROFILE_DIR = "GRAPHRAG_HPC_PROFILE_DIR"
ENV_PROFILE_FORMAT = "GRAPHRAG_HPC_PROFILE_FORMAT"
ENV_RUN_ID = "GRAPHRAG_HPC_RUN_ID"
ENV_CONCURRENCY_LIMIT = "GRAPHRAG_HPC_CONCURRENCY"
ENV_DISTRIBUTED_MIN_ITEMS = "GRAPHRAG_HPC_DISTRIBUTED_MIN_ITEMS"
ENV_DATAFRAME_MIN_ROWS = "GRAPHRAG_HPC_DATAFRAME_MIN_ROWS"
ENV_FORCE_DISTRIBUTED = "GRAPHRAG_HPC_FORCE_DISTRIBUTED"

_TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})


@dataclass
class HPCParallelConfig:
    """Configuration for HPC parallelization and function-level profiling."""

    execution_mode: ExecutionMode = ExecutionMode.BASELINE
    """Execution mode to use when running through hpc_parallel."""

    profiling_enabled: bool = False
    """Whether function-level instrumentation hooks record timing metrics."""

    profile_output_dir: Path = field(
        default_factory=lambda: Path("benchmark/hpc_profile")
    )
    """Directory where CSV and/or JSON profiling artifacts are written."""

    profile_format: ProfileOutputFormat = ProfileOutputFormat.BOTH
    """Format for exported profiling artifacts."""

    run_id: str | None = None
    """Optional identifier for the profiling run."""

    concurrency_limit: int | None = None
    """Optional concurrency override for execution modes."""

    distributed_min_items: int = 500
    """Minimum workload item count before Ray distributed dispatch is used."""

    dataframe_min_rows: int = 5000
    """Minimum DataFrame row count before Dask batch processing is used."""

    cpu_pool_min_items: int = 16
    """Minimum workload item count before CPU ProcessPoolExecutor dispatch is used."""

    cpu_workers: int | None = None
    """Shared CPU worker budget across ProcessPool/Ray/Dask to avoid oversubscription."""

    force_distributed: bool = False
    """Bypass distributed workload-size thresholds (for benchmarking/testing)."""

    ray_num_cpus: int | None = None
    """Optional CPU count for local Ray initialization."""

    dask_n_workers: int | None = None
    """Optional worker count for local Dask LocalCluster."""

    cuda_min_elements: int = 4096
    """Minimum vector elements before offloading local post-processing to CUDA."""

    cuda_threads_per_block: int = 256
    """CUDA kernel threads per block (typically 128, 256, or 512; max 1024)."""

    cuda_blocks_per_grid: int | None = None
    """Optional explicit CUDA grid blocks; computed as ceil(n / threads_per_block) when None."""

    cuda_fallback_to_cpu: bool = True
    """If True, fall back to CPU when no CUDA device is available."""

    allow_platform_fallback: bool = True
    """If True, gracefully fall back to local CPU/asyncio when optional platforms are absent."""

    def __post_init__(self) -> None:
        """Normalize enums and paths."""
        if not isinstance(self.execution_mode, ExecutionMode):
            self.execution_mode = ExecutionMode(str(self.execution_mode).lower())
        if not isinstance(self.profile_format, ProfileOutputFormat):
            self.profile_format = ProfileOutputFormat(str(self.profile_format).lower())
        if not isinstance(self.profile_output_dir, Path):
            self.profile_output_dir = Path(self.profile_output_dir)


def parse_bool_env(value: str | None, default: bool = False) -> bool:
    """Parse a boolean environment variable string."""
    if value is None:
        return default
    return value.strip().lower() in _TRUTHY_VALUES


def load_config_from_env() -> HPCParallelConfig:
    """Load HPCParallelConfig from environment variables."""
    profiling_enabled = parse_bool_env(os.environ.get(ENV_PROFILE_ENABLED), False)

    mode_raw = os.environ.get(ENV_EXECUTION_MODE, ExecutionMode.BASELINE.value)
    execution_mode = ExecutionMode(mode_raw.strip().lower())

    dir_raw = os.environ.get(ENV_PROFILE_DIR, "benchmark/hpc_profile")
    profile_output_dir = Path(dir_raw)

    fmt_raw = os.environ.get(ENV_PROFILE_FORMAT, ProfileOutputFormat.BOTH.value)
    profile_format = ProfileOutputFormat(fmt_raw.strip().lower())

    run_id = os.environ.get(ENV_RUN_ID) or None

    concurrency_raw = os.environ.get(ENV_CONCURRENCY_LIMIT)
    concurrency_limit: int | None = None
    if concurrency_raw is not None and concurrency_raw.strip():
        concurrency_limit = int(concurrency_raw.strip())

    min_items_raw = os.environ.get(ENV_DISTRIBUTED_MIN_ITEMS)
    distributed_min_items = (
        int(min_items_raw.strip())
        if min_items_raw is not None and min_items_raw.strip()
        else 500
    )
    df_min_raw = os.environ.get(ENV_DATAFRAME_MIN_ROWS)
    dataframe_min_rows = (
        int(df_min_raw.strip())
        if df_min_raw is not None and df_min_raw.strip()
        else 5000
    )
    force_distributed = parse_bool_env(os.environ.get(ENV_FORCE_DISTRIBUTED), False)

    return HPCParallelConfig(
        execution_mode=execution_mode,
        profiling_enabled=profiling_enabled,
        profile_output_dir=profile_output_dir,
        profile_format=profile_format,
        run_id=run_id,
        concurrency_limit=concurrency_limit,
        distributed_min_items=distributed_min_items,
        dataframe_min_rows=dataframe_min_rows,
        force_distributed=force_distributed,
    )
