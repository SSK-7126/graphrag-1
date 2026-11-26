# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Configurable execution runner for GraphRAG HPC evaluation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import graphrag.api as api
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.config.enums import IndexingMethod
from graphrag.config.load_config import load_config
from graphrag.index.run.run_pipeline import run_pipeline
from graphrag.index.run.utils import create_callback_chain
from graphrag.index.validate_config import validate_config_names
from graphrag.logger.standard_logging import init_loggers
from graphrag_cache.cache_type import CacheType

from hpc_parallel.backends.cuda_backend import ensure_cuda_ready
from hpc_parallel.backends.dask_backend import ensure_dask_available
from hpc_parallel.backends.hybrid_orchestrator import HybridRuntimeManager
from hpc_parallel.backends.pipeline_builder import (
    build_pipeline_for_mode,
    prepare_config_for_mode,
)
from hpc_parallel.backends.ray_backend import ensure_ray_available
from hpc_parallel.config import ExecutionMode, HPCParallelConfig
from hpc_parallel.hooks import profiling_session

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd
    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.typing.pipeline_run_result import PipelineRunResult

logger = logging.getLogger(__name__)


def validate_execution_mode(
    mode: ExecutionMode,
    hpc_config: HPCParallelConfig | None = None,
) -> None:
    """Validate that the requested execution mode and its optional backend are ready."""
    eff_cfg = hpc_config or HPCParallelConfig(execution_mode=mode)
    if mode in {
        ExecutionMode.BASELINE,
        ExecutionMode.SERIAL,
        ExecutionMode.ASYNCIO_OPTIMIZED,
        ExecutionMode.CPU_HYBRID,
    }:
        return
    if mode == ExecutionMode.MULTI_PLATFORM_HYBRID:
        if eff_cfg.allow_platform_fallback:
            return
        ensure_ray_available()
        ensure_dask_available()
        ensure_cuda_ready(eff_cfg)
        return
    if mode == ExecutionMode.RAY:
        ensure_ray_available()
        return
    if mode == ExecutionMode.DASK:
        ensure_dask_available()
        return
    if mode == ExecutionMode.CUDA:
        ensure_cuda_ready(eff_cfg)
        return
    msg = f"Unsupported execution mode: {mode}"
    raise ValueError(msg)


async def run_index(
    *,
    config: GraphRagConfig | None = None,
    root_dir: str | Path | None = None,
    hpc_config: HPCParallelConfig | None = None,
    method: IndexingMethod | str = IndexingMethod.Standard,
    is_update_run: bool = False,
    callbacks: list[WorkflowCallbacks] | None = None,
    additional_context: dict[str, Any] | None = None,
    verbose: bool = False,
    use_cache: bool = True,
    skip_validation: bool = False,
    input_documents: pd.DataFrame | None = None,
) -> list[PipelineRunResult]:
    """Run GraphRAG indexing with configurable execution mode and optional profiling."""
    eff_hpc_config = hpc_config or HPCParallelConfig()
    validate_execution_mode(eff_hpc_config.execution_mode, eff_hpc_config)

    if config is None:
        if root_dir is None:
            msg = "Either 'config' or 'root_dir' must be provided."
            raise ValueError(msg)
        resolved_config = load_config(root_dir=root_dir)
    else:
        resolved_config = config

    if not use_cache:
        resolved_config.cache.type = CacheType.Noop

    if not skip_validation:
        validate_config_names(resolved_config)

    logger.info(
        "Starting HPC GraphRAG index (mode=%s, profiling=%s)",
        eff_hpc_config.execution_mode.value,
        eff_hpc_config.profiling_enabled,
    )

    with profiling_session(
        enabled=eff_hpc_config.profiling_enabled,
        output_dir=eff_hpc_config.profile_output_dir,
        output_format=eff_hpc_config.profile_format,
        execution_mode=eff_hpc_config.execution_mode,
        run_id=eff_hpc_config.run_id,
        auto_export=True,
    ):
        if eff_hpc_config.execution_mode == ExecutionMode.BASELINE:
            return await api.build_index(
                config=resolved_config,
                method=method,
                is_update_run=is_update_run,
                callbacks=callbacks,
                additional_context=additional_context,
                verbose=verbose,
                input_documents=input_documents,
            )

        mode_config = prepare_config_for_mode(resolved_config, eff_hpc_config)
        init_loggers(config=mode_config, verbose=verbose)
        workflow_callbacks = (
            create_callback_chain(callbacks) if callbacks else NoopWorkflowCallbacks()
        )
        method_str = method.value if isinstance(method, IndexingMethod) else method
        eff_method = f"{method_str}-update" if is_update_run else method_str

        async with HybridRuntimeManager(eff_hpc_config) as runtime:
            pipeline = build_pipeline_for_mode(
                mode_config,
                eff_method,
                eff_hpc_config,
                runtime=runtime,
            )

            outputs: list[PipelineRunResult] = []
            workflow_callbacks.pipeline_start(pipeline.names())
            async for output in run_pipeline(
                pipeline,
                mode_config,
                callbacks=workflow_callbacks,
                is_update_run=is_update_run,
                additional_context=additional_context,
                input_documents=input_documents,
            ):
                outputs.append(output)
                if output.error is not None:
                    logger.error("Workflow %s completed with errors", output.workflow)
                    workflow_callbacks.pipeline_error(output.error)
                else:
                    logger.info("Workflow %s completed successfully", output.workflow)
            workflow_callbacks.pipeline_end(outputs)
            return outputs
