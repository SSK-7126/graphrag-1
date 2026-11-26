# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Dask distributed backend for CPU-bound GraphRAG indexing stages (Phase 3)."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag.index.operations.build_noun_graph.build_noun_graph import _extract_edges
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.workflows.create_base_text_units import (
    run_workflow as run_base_text_units_baseline,
)
from graphrag.index.workflows.extract_graph_nlp import (
    _write_entities,
    _write_relationships,
)
from graphrag.index.workflows.extract_graph_nlp import (
    run_workflow as run_extract_graph_nlp_baseline,
)

from hpc_parallel.backends.ray_backend import (
    _chunk_doc_partition,
    _extract_np_partition,
    should_use_distributed,
)

if TYPE_CHECKING:
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.typing.context import PipelineRunContext

    from hpc_parallel.config import HPCParallelConfig

logger = logging.getLogger(__name__)


def ensure_dask_available() -> Any:
    """Import dask.distributed lazily or raise a clear installation error."""
    try:
        return importlib.import_module("dask.distributed")
    except ImportError as exc:
        msg = (
            "Execution mode 'dask' requires 'dask[distributed]'. "
            "Install it with: pip install 'hpc-parallel[dask]'"
        )
        raise ImportError(msg) from exc


def _get_or_create_dask_client(
    distributed: Any, hpc_config: HPCParallelConfig
) -> tuple[Any, bool]:
    """Return an active Dask Client and whether this call owns its lifecycle."""
    try:
        return distributed.get_client(), False
    except ValueError:
        kwargs: dict[str, Any] = {}
        if hpc_config.dask_n_workers is not None:
            kwargs["n_workers"] = hpc_config.dask_n_workers
        return distributed.Client(**kwargs), True


def make_dask_create_base_text_units(hpc_config: HPCParallelConfig):
    """Create a Dask-enabled ``create_base_text_units`` workflow with small-workload guard."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
        distributed = ensure_dask_available()
        async with context.output_table_provider.open("documents") as docs_table:
            total_rows = await docs_table.length()
            if not should_use_distributed(total_rows, hpc_config):
                logger.info(
                    "Dask create_base_text_units: %d docs < threshold %d; using local execution.",
                    total_rows,
                    hpc_config.distributed_min_items,
                )
                return await run_base_text_units_baseline(config, context)
            docs = [doc async for doc in docs_table]

        client, owns_client = _get_or_create_dask_client(distributed, hpc_config)
        try:
            chunking_dump = config.chunking.model_dump()
            num_parts = max(1, min(len(docs), hpc_config.dask_n_workers or 4))
            part_size = max(1, (len(docs) + num_parts - 1) // num_parts)
            futures = [
                client.submit(
                    _chunk_doc_partition, docs[i : i + part_size], chunking_dump
                )
                for i in range(0, len(docs), part_size)
            ]
            partitions: list[list[dict[str, Any]]] = client.gather(futures)
        finally:
            if owns_client:
                client.close()

        sample_rows: list[dict[str, Any]] = []
        async with context.output_table_provider.open("text_units") as tu_table:
            for part in partitions:
                for row in part:
                    await tu_table.write(row)
                    if len(sample_rows) < 5:
                        sample_rows.append(row)

        return WorkflowFunctionOutput(result=sample_rows)

    return _workflow


def make_dask_extract_graph_nlp(hpc_config: HPCParallelConfig):
    """Create a Dask-enabled ``extract_graph_nlp`` workflow with small-workload guard."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
        distributed = ensure_dask_available()
        async with context.output_table_provider.open(
            "text_units", truncate=False
        ) as tu_table:
            total_rows = await tu_table.length()
            if not should_use_distributed(total_rows, hpc_config):
                logger.info(
                    "Dask extract_graph_nlp: %d text_units < threshold %d; using local execution.",
                    total_rows,
                    hpc_config.distributed_min_items,
                )
                return await run_extract_graph_nlp_baseline(config, context)
            rows = [{"id": r["id"], "text": r["text"]} async for r in tu_table]

        client, owns_client = _get_or_create_dask_client(distributed, hpc_config)
        try:
            analyzer_dump = config.extract_graph_nlp.text_analyzer.model_dump()
            num_parts = max(1, min(len(rows), hpc_config.dask_n_workers or 4))
            part_size = max(1, (len(rows) + num_parts - 1) // num_parts)
            futures = [
                client.submit(
                    _extract_np_partition, rows[i : i + part_size], analyzer_dump
                )
                for i in range(0, len(rows), part_size)
            ]
            partitions: list[list[tuple[str, str]]] = client.gather(futures)
        finally:
            if owns_client:
                client.close()

        title_to_ids: dict[str, list[str]] = {}
        for part in partitions:
            for phrase, tu_id in part:
                title_to_ids.setdefault(phrase, []).append(tu_id)

        nodes_df = pd.DataFrame(
            [
                {"title": t, "frequency": len(ids), "text_unit_ids": ids}
                for t, ids in title_to_ids.items()
            ],
            columns=["title", "frequency", "text_unit_ids"],
        )
        edges_df = _extract_edges(
            title_to_ids,
            nodes_df=nodes_df,
            normalize_edge_weights=config.extract_graph_nlp.normalize_edge_weights,
        )

        async with (
            context.output_table_provider.open("entities") as entities_table,
            context.output_table_provider.open("relationships") as rels_table,
        ):
            entity_samples = await _write_entities(nodes_df, entities_table)
            rel_samples = await _write_relationships(edges_df, rels_table)

        return WorkflowFunctionOutput(
            result={"entities": entity_samples, "relationships": rel_samples}
        )

    return _workflow
