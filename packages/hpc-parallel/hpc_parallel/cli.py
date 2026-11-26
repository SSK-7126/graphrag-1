# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CLI entrypoint for hpc_parallel."""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from graphrag.callbacks.console_workflow_callbacks import ConsoleWorkflowCallbacks
from graphrag.config.enums import IndexingMethod

from hpc_parallel.config import (
    ExecutionMode,
    HPCParallelConfig,
    ProfileOutputFormat,
)
from hpc_parallel.runner import run_index


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser for hpc_parallel CLI."""
    parser = argparse.ArgumentParser(
        prog="hpc-graphrag",
        description="Run GraphRAG with HPC execution modes and function-level profiling.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="Build a GraphRAG knowledge graph index with optional profiling.",
    )
    index_parser.add_argument(
        "--root",
        "-r",
        type=Path,
        default=Path.cwd(),
        help="GraphRAG workspace root directory.",
    )
    index_parser.add_argument(
        "--method",
        "-m",
        type=str,
        default=IndexingMethod.Standard.value,
        choices=[m.value for m in IndexingMethod],
        help="GraphRAG indexing method.",
    )
    index_parser.add_argument(
        "--mode",
        type=str,
        default=ExecutionMode.BASELINE.value,
        choices=[m.value for m in ExecutionMode],
        help="HPC execution mode (Phase 1 supports 'baseline').",
    )
    index_parser.add_argument(
        "--profile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable or disable function-level timing instrumentation.",
    )
    index_parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path("benchmark/hpc_profile"),
        help="Directory to write CSV/JSON function timing reports.",
    )
    index_parser.add_argument(
        "--profile-format",
        type=str,
        default=ProfileOutputFormat.BOTH.value,
        choices=[f.value for f in ProfileOutputFormat],
        help="Output format for function timing artifacts.",
    )
    index_parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run identifier for profiling records.",
    )
    index_parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Optional concurrency override for non-baseline modes.",
    )
    index_parser.add_argument(
        "--distributed-min-items",
        type=int,
        default=500,
        help="Minimum item count before Ray distributed dispatch is used.",
    )
    index_parser.add_argument(
        "--dataframe-min-rows",
        type=int,
        default=5000,
        help="Minimum DataFrame row count before Dask batch processing is used.",
    )
    index_parser.add_argument(
        "--cpu-workers",
        type=int,
        default=None,
        help="Shared CPU worker budget across ProcessPool/Ray/Dask.",
    )
    index_parser.add_argument(
        "--force-distributed",
        action="store_true",
        default=False,
        help="Force Ray/Dask execution even below small-workload thresholds.",
    )
    index_parser.add_argument(
        "--cuda-threads-per-block",
        type=int,
        default=256,
        help="CUDA kernel threads per block (1-1024, default 256).",
    )
    index_parser.add_argument(
        "--cuda-blocks",
        type=int,
        default=None,
        help="Optional explicit CUDA grid blocks (auto-computed if omitted).",
    )
    index_parser.add_argument(
        "--no-fallback",
        action="store_true",
        default=False,
        help="Fail instead of falling back when Ray/Dask/CUDA are unavailable.",
    )
    index_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging.",
    )
    index_parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable LLM caching for this run.",
    )
    index_parser.add_argument(
        "--skip-validation",
        action="store_true",
        default=False,
        help="Skip preflight model validation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the hpc_parallel CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        hpc_config = HPCParallelConfig(
            execution_mode=ExecutionMode(args.mode),
            profiling_enabled=args.profile,
            profile_output_dir=args.profile_dir,
            profile_format=ProfileOutputFormat(args.profile_format),
            run_id=args.run_id,
            concurrency_limit=args.concurrency,
            distributed_min_items=args.distributed_min_items,
            dataframe_min_rows=args.dataframe_min_rows,
            cpu_workers=args.cpu_workers,
            force_distributed=args.force_distributed,
            cuda_threads_per_block=args.cuda_threads_per_block,
            cuda_blocks_per_grid=args.cuda_blocks,
            cuda_fallback_to_cpu=not args.no_fallback,
            allow_platform_fallback=not args.no_fallback,
        )
        outputs = asyncio.run(
            run_index(
                root_dir=args.root.resolve(),
                hpc_config=hpc_config,
                method=args.method,
                callbacks=[ConsoleWorkflowCallbacks(verbose=args.verbose)],
                verbose=args.verbose,
                use_cache=not args.no_cache,
                skip_validation=args.skip_validation,
            )
        )
        encountered_errors = any(output.error is not None for output in outputs)
        return 1 if encountered_errors else 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
