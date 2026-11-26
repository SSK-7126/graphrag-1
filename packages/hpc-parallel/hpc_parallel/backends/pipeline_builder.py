# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Mode-aware pipeline construction without mutating global PipelineFactory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from graphrag.config.enums import AsyncType, IndexingMethod
from graphrag.index.typing.pipeline import Pipeline
from graphrag.index.workflows.factory import PipelineFactory

from hpc_parallel.backends.asyncio_ops import (
    run_create_community_reports_optimized,
    run_extract_graph_optimized,
    run_generate_text_embeddings_optimized,
)
from hpc_parallel.backends.hybrid_orchestrator import (
    HybridRuntimeManager,
    make_hybrid_create_base_text_units,
    make_hybrid_extract_graph,
    make_hybrid_extract_graph_nlp,
    make_hybrid_generate_text_embeddings,
)
from hpc_parallel.config import ExecutionMode, HPCParallelConfig

if TYPE_CHECKING:
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.typing.workflow import WorkflowFunction


def prepare_config_for_mode(
    config: GraphRagConfig,
    hpc_config: HPCParallelConfig,
) -> GraphRagConfig:
    """Return a copy of GraphRagConfig tuned for the requested ExecutionMode."""
    mode = hpc_config.execution_mode
    if mode == ExecutionMode.BASELINE:
        return config

    updates: dict[str, object] = {}
    if mode == ExecutionMode.SERIAL:
        updates["concurrent_requests"] = 1
        updates["async_mode"] = AsyncType.AsyncIO
    else:
        updates["async_mode"] = AsyncType.AsyncIO
        if hpc_config.concurrency_limit is not None:
            updates["concurrent_requests"] = max(1, hpc_config.concurrency_limit)

    return config.model_copy(update=updates)


def build_pipeline_for_mode(
    config: GraphRagConfig,
    method: IndexingMethod | str,
    hpc_config: HPCParallelConfig,
    *,
    runtime: HybridRuntimeManager | None = None,
    overrides: dict[str, WorkflowFunction] | None = None,
) -> Pipeline:
    """Build a Pipeline with mode-specific workflow overrides.

    Original ``PipelineFactory.workflows`` is never mutated.
    """
    base_pipeline = PipelineFactory.create_pipeline(config, method)
    mode = hpc_config.execution_mode

    if mode in {ExecutionMode.BASELINE, ExecutionMode.SERIAL}:
        return base_pipeline

    mode_overrides: dict[str, WorkflowFunction] = {
        "extract_graph": run_extract_graph_optimized,
        "create_community_reports": run_create_community_reports_optimized,
        "generate_text_embeddings": run_generate_text_embeddings_optimized,
    }

    if mode in {ExecutionMode.CPU_HYBRID, ExecutionMode.MULTI_PLATFORM_HYBRID}:
        rt = runtime or HybridRuntimeManager(hpc_config)
        mode_overrides["create_base_text_units"] = make_hybrid_create_base_text_units(
            hpc_config, rt
        )
        mode_overrides["extract_graph_nlp"] = make_hybrid_extract_graph_nlp(
            hpc_config, rt
        )
        mode_overrides["extract_graph"] = make_hybrid_extract_graph(hpc_config, rt)
        mode_overrides["generate_text_embeddings"] = (
            make_hybrid_generate_text_embeddings(hpc_config)
        )
    elif mode == ExecutionMode.RAY:
        from hpc_parallel.backends.ray_backend import (
            make_ray_create_base_text_units,
            make_ray_extract_graph_nlp,
        )

        mode_overrides["create_base_text_units"] = make_ray_create_base_text_units(
            hpc_config
        )
        mode_overrides["extract_graph_nlp"] = make_ray_extract_graph_nlp(hpc_config)
    elif mode == ExecutionMode.DASK:
        from hpc_parallel.backends.dask_backend import (
            make_dask_create_base_text_units,
            make_dask_extract_graph_nlp,
        )

        mode_overrides["create_base_text_units"] = make_dask_create_base_text_units(
            hpc_config
        )
        mode_overrides["extract_graph_nlp"] = make_dask_extract_graph_nlp(hpc_config)

    if overrides:
        mode_overrides.update(overrides)

    replaced = [
        (name, mode_overrides.get(name, fn)) for name, fn in base_pipeline.run()
    ]
    return Pipeline(replaced)
