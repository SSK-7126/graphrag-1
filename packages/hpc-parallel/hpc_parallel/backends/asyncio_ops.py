# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Optimized asyncio operations for Phase 2 (no external dependencies)."""

from __future__ import annotations

import asyncio
import inspect
import logging
import traceback
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any, TypeVar, cast

import graphrag.data_model.schemas as schemas
import numpy as np
import pandas as pd
from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.data_model.data_reader import DataReader
from graphrag.data_model.row_transformers import transform_entity_row_for_embedding
from graphrag.index.operations.embed_text.run_embed_text import run_embed_text
from graphrag.index.operations.extract_graph.extract_graph import (
    _merge_entities,
    _merge_relationships,
    _run_extract_graph,
)
from graphrag.index.operations.extract_graph.utils import filter_orphan_relationships
from graphrag.index.operations.finalize_community_reports import (
    finalize_community_reports,
)
from graphrag.index.operations.summarize_communities.explode_communities import (
    explode_communities,
)
from graphrag.index.operations.summarize_communities.graph_context.context_builder import (
    build_level_context,
    build_local_context,
)
from graphrag.index.operations.summarize_communities.graph_context.sort_context import (
    parallel_sort_context_batch,
)
from graphrag.index.operations.summarize_communities.summarize_communities import (
    _generate_report,
    run_extractor,
)
from graphrag.index.operations.summarize_communities.utils import get_levels
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    run_summarize_descriptions,
)
from graphrag.index.run import hpc_hooks
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.derive_from_rows import ParallelizationError
from graphrag.index.workflows.create_community_reports import (
    _prep_claims,
    _prep_edges,
    _prep_nodes,
)
from graphrag.index.workflows.generate_text_embeddings import EMBEDDING_FIELDS
from graphrag.logger.progress import progress_ticker
from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import VectorStore, VectorStoreDocument, create_vector_store

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from graphrag.callbacks.workflow_callbacks import WorkflowCallbacks
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.operations.summarize_communities.typing import CommunityReport
    from graphrag.index.typing.context import PipelineRunContext
    from graphrag_llm.completion import LLMCompletion
    from graphrag_llm.embedding import LLMEmbedding
    from graphrag_llm.tokenizer import Tokenizer
    from graphrag_storage.tables.table import Table
    from graphrag_storage.tables.table_provider import TableProvider

logger = logging.getLogger(__name__)
ItemType = TypeVar("ItemType")


async def derive_from_rows_bounded(
    input: pd.DataFrame,
    transform: Callable[[pd.Series], Awaitable[ItemType]],
    callbacks: WorkflowCallbacks | None = None,
    num_threads: int = 4,
    progress_msg: str = "",
) -> list[ItemType | None]:
    """Bounded worker-pool asyncio row transformation.

    Avoids creating all ``N`` coroutine tasks eagerly and avoids the
    ``asyncio.to_thread`` wrapper overhead of ``derive_from_rows_asyncio_threads``.
    Preserves exact input row ordering and ``ParallelizationError`` semantics.
    """
    callbacks = callbacks or NoopWorkflowCallbacks()
    row_count = len(input)
    concurrency = max(1, num_threads or 4)
    tick = progress_ticker(
        callbacks.progress, num_total=row_count, description=progress_msg
    )

    with hpc_hooks.profile_function(
        "derive_from_rows_asyncio",
        input_count=row_count,
        batch_count=1 if row_count > 0 else 0,
        request_count=row_count,
        configured_concurrency=concurrency,
    ) as span:
        if row_count == 0:
            tick.done()
            return []

        results: list[ItemType | None] = [None] * row_count
        errors: list[tuple[BaseException, str]] = []
        queue: asyncio.Queue[tuple[int, pd.Series] | None] = asyncio.Queue()

        for idx, (_, row) in enumerate(input.iterrows()):
            queue.put_nowait((idx, row))

        num_workers = min(concurrency, row_count)
        for _ in range(num_workers):
            queue.put_nowait(None)

        async def _worker() -> None:
            while True:
                item = await queue.get()
                if item is None:
                    queue.task_done()
                    return
                idx, row = item
                try:
                    with span.subtask():
                        res = transform(row)
                        if inspect.iscoroutine(res) or inspect.isawaitable(res):
                            res = await res
                        results[idx] = cast("ItemType", res)
                except Exception as exc:  # noqa: BLE001
                    errors.append((exc, traceback.format_exc()))
                finally:
                    tick(1)
                    queue.task_done()

        await asyncio.gather(*(_worker() for _ in range(num_workers)))
        tick.done()

        for error, stack in errors:
            logger.error(
                "parallel transformation error", exc_info=error, extra={"stack": stack}
            )

        if errors:
            span.set_counts(errors=len(errors))
            raise ParallelizationError(len(errors), errors[0][1])

        return results


async def summarize_descriptions_concurrent(
    entities_df: pd.DataFrame,
    relationships_df: pd.DataFrame,
    callbacks: WorkflowCallbacks,
    model: LLMCompletion,
    max_summary_length: int,
    max_input_tokens: int,
    prompt: str,
    num_threads: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize entity and relationship descriptions concurrently without a barrier.

    Unlike baseline ``summarize_descriptions``, which awaits all nodes before
    starting edges, this dispatches both entity and relationship tasks onto a
    shared semaphore pool.
    """
    total_inputs = len(entities_df) + len(relationships_df)
    batch_count = (1 if len(entities_df) > 0 else 0) + (
        1 if len(relationships_df) > 0 else 0
    )
    concurrency = max(1, num_threads)
    semaphore = asyncio.Semaphore(concurrency)
    ticker = progress_ticker(
        callbacks.progress,
        total_inputs,
        description="Summarize entity/relationship description progress: ",
    )

    with hpc_hooks.profile_function(
        "summarize_descriptions",
        input_count=total_inputs,
        batch_count=batch_count,
        configured_concurrency=concurrency,
    ) as span:

        async def _run_item(
            item_id: str | tuple[str, str], descriptions: list[str]
        ) -> Any:
            async with semaphore:
                with span.subtask():
                    res = await run_summarize_descriptions(
                        item_id,
                        descriptions,
                        model,
                        max_summary_length,
                        max_input_tokens,
                        prompt,
                    )
                ticker(1)
                return res

        node_coros = [
            _run_item(
                str(row.title),  # type: ignore[attr-defined]
                sorted(set(row.description)),  # type: ignore[attr-defined]
            )
            for row in entities_df.itertuples(index=False)
        ]
        edge_coros = [
            _run_item(
                (str(row.source), str(row.target)),  # type: ignore[attr-defined]
                sorted(set(row.description)),  # type: ignore[attr-defined]
            )
            for row in relationships_df.itertuples(index=False)
        ]

        all_results = await asyncio.gather(*node_coros, *edge_coros)
        n_nodes = len(node_coros)
        node_results = all_results[:n_nodes]
        edge_results = all_results[n_nodes:]

        entity_descriptions = pd.DataFrame([
            {"title": r.id, "description": r.description} for r in node_results
        ])
        relationship_descriptions = pd.DataFrame([
            {
                "source": r.id[0],
                "target": r.id[1],
                "description": r.description,
            }
            for r in edge_results
        ])
        return entity_descriptions, relationship_descriptions


async def embed_text_pipelined(
    input_table: Table,
    callbacks: WorkflowCallbacks,
    model: LLMEmbedding,
    tokenizer: Tokenizer,
    embed_column: str,
    batch_size: int,
    batch_max_tokens: int,
    num_threads: int,
    vector_store: VectorStore,
    id_column: str = "id",
    output_table: Table | None = None,
) -> int:
    """Overlap embedding API batches with vector-store and snapshot table writes."""
    vector_store.create_index()
    flush_size = max(1, batch_size * max(1, num_threads))

    async def _persist_batch(
        ids: list[str],
        vectors: list[list[float] | None],
    ) -> None:
        documents: list[VectorStoreDocument] = []
        skipped = 0
        rows_to_write: list[dict[str, Any]] = []
        for doc_id, doc_vector in zip(ids, vectors, strict=True):
            if doc_vector is None:
                skipped += 1
                continue
            if type(doc_vector) is np.ndarray:
                doc_vector = doc_vector.tolist()
            documents.append(VectorStoreDocument(id=doc_id, vector=doc_vector))
            if output_table is not None:
                rows_to_write.append({"id": doc_id, "embedding": doc_vector})

        if documents:
            await asyncio.to_thread(vector_store.load_documents, documents)
        if skipped > 0:
            logger.warning(
                "Skipped %d rows with None embeddings out of %d",
                skipped,
                len(ids),
            )
        if output_table is not None:
            for row in rows_to_write:
                await output_table.write(row)

    buffer: list[dict[str, Any]] = []
    total_rows = 0
    pending_write: asyncio.Task[None] | None = None

    async for row in input_table:
        text = row.get(embed_column)
        if text is None:
            text = ""
        buffer.append({id_column: row[id_column], embed_column: text})

        if len(buffer) >= flush_size:
            texts = [r[embed_column] for r in buffer]
            ids = [r[id_column] for r in buffer]
            result = await run_embed_text(
                texts,
                callbacks,
                model,
                tokenizer,
                batch_size,
                batch_max_tokens,
                num_threads,
            )
            if pending_write is not None:
                await pending_write
            pending_write = asyncio.create_task(
                _persist_batch(ids, result.embeddings or [])
            )
            total_rows += len(buffer)
            buffer.clear()

    if buffer:
        texts = [r[embed_column] for r in buffer]
        ids = [r[id_column] for r in buffer]
        result = await run_embed_text(
            texts,
            callbacks,
            model,
            tokenizer,
            batch_size,
            batch_max_tokens,
            num_threads,
        )
        if pending_write is not None:
            await pending_write
        pending_write = asyncio.create_task(
            _persist_batch(ids, result.embeddings or [])
        )
        total_rows += len(buffer)

    if pending_write is not None:
        await pending_write

    return total_rows


async def run_extract_graph_optimized(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Optimized asyncio workflow for ``extract_graph``."""
    logger.info("Workflow started (asyncio_optimized): extract_graph")
    reader = DataReader(context.output_table_provider)
    text_units = await reader.text_units()

    extraction_model_config = config.get_completion_model_config(
        config.extract_graph.completion_model_id
    )
    extraction_prompts = config.extract_graph.resolved_prompts()
    extraction_model = create_completion(
        extraction_model_config,
        cache=context.cache.child(config.extract_graph.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    summarization_model_config = config.get_completion_model_config(
        config.summarize_descriptions.completion_model_id
    )
    summarization_prompts = config.summarize_descriptions.resolved_prompts()
    summarization_model = create_completion(
        summarization_model_config,
        cache=context.cache.child(config.summarize_descriptions.model_instance_name),
        cache_key_creator=cache_key_creator,
    )

    async def run_strategy(row: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
        return await _run_extract_graph(
            text=row["text"],
            source_id=row["id"],
            entity_types=config.extract_graph.entity_types,
            model=extraction_model,
            prompt=extraction_prompts.extraction_prompt,
            max_gleanings=config.extract_graph.max_gleanings,
        )

    results = await derive_from_rows_bounded(
        text_units,
        run_strategy,
        context.callbacks,
        num_threads=config.concurrent_requests,
        progress_msg="extract graph progress: ",
    )

    entity_dfs = [r[0] for r in results if r]
    relationship_dfs = [r[1] for r in results if r]
    extracted_entities = _merge_entities(entity_dfs)
    extracted_relationships = filter_orphan_relationships(
        _merge_relationships(relationship_dfs), extracted_entities
    )

    if len(extracted_entities) == 0:
        err = "Graph Extraction failed. No entities detected during extraction."
        logger.error(err)
        raise ValueError(err)
    if len(extracted_relationships) == 0:
        err = "Graph Extraction failed. No relationships detected during extraction."
        logger.error(err)
        raise ValueError(err)

    raw_entities = extracted_entities.copy()
    raw_relationships = extracted_relationships.copy()

    entity_summaries, relationship_summaries = await summarize_descriptions_concurrent(
        entities_df=extracted_entities,
        relationships_df=extracted_relationships,
        callbacks=context.callbacks,
        model=summarization_model,
        max_summary_length=config.summarize_descriptions.max_length,
        max_input_tokens=config.summarize_descriptions.max_input_tokens,
        prompt=summarization_prompts.summarize_prompt,
        num_threads=config.concurrent_requests,
    )

    relationships = extracted_relationships.drop(columns=["description"]).merge(
        relationship_summaries, on=["source", "target"], how="left"
    )
    entities = extracted_entities.drop(columns=["description"]).merge(
        entity_summaries, on="title", how="left"
    )

    await context.output_table_provider.write_dataframe("entities", entities)
    await context.output_table_provider.write_dataframe("relationships", relationships)

    if config.snapshots.raw_graph:
        await context.output_table_provider.write_dataframe(
            "raw_entities", raw_entities
        )
        await context.output_table_provider.write_dataframe(
            "raw_relationships", raw_relationships
        )

    logger.info("Workflow completed (asyncio_optimized): extract_graph")
    return WorkflowFunctionOutput(
        result={"entities": entities, "relationships": relationships}
    )


async def run_create_community_reports_optimized(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Optimized asyncio workflow for ``create_community_reports``."""
    logger.info("Workflow started (asyncio_optimized): create_community_reports")
    reader = DataReader(context.output_table_provider)
    relationships = await reader.relationships()
    entities = await reader.entities()
    communities = await reader.communities()

    claims = None
    if config.extract_claims.enabled and await context.output_table_provider.has(
        "covariates"
    ):
        claims = await reader.covariates()

    model_config = config.get_completion_model_config(
        config.community_reports.completion_model_id
    )
    prompts = config.community_reports.resolved_prompts()
    model = create_completion(
        model_config,
        cache=context.cache.child(config.community_reports.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    tokenizer = model.tokenizer

    nodes = _prep_nodes(explode_communities(communities, entities))
    edges = _prep_edges(relationships)
    prepped_claims = _prep_claims(claims) if claims is not None else None

    local_contexts = build_local_context(
        nodes,
        edges,
        prepped_claims,
        tokenizer,
        context.callbacks,
        config.community_reports.max_input_length,
    )

    reports: list[CommunityReport | None] = []
    tick = progress_ticker(context.callbacks.progress, len(local_contexts))
    community_hierarchy = (
        communities
        .explode("children")
        .rename({"children": "sub_community"}, axis=1)
        .loc[:, ["community", "level", "sub_community"]]
    ).dropna()

    levels = get_levels(nodes)
    for level in levels:
        level_context = build_level_context(
            pd.DataFrame(reports),
            community_hierarchy_df=community_hierarchy,
            local_context_df=local_contexts,
            level=level,
            tokenizer=tokenizer,
            max_context_tokens=config.community_reports.max_input_length,
        )
        if len(level_context) > 1:
            parallel_sort_context_batch(
                level_context,
                tokenizer=tokenizer,
                max_context_tokens=config.community_reports.max_input_length,
                parallel=True,
            )

        async def _run_generate(record: pd.Series) -> CommunityReport | None:
            res = await _generate_report(
                run_extractor,
                community_id=record[schemas.COMMUNITY_ID],
                community_level=record[schemas.COMMUNITY_LEVEL],
                community_context=record[schemas.CONTEXT_STRING],
                model=model,
                extraction_prompt=prompts.graph_prompt,
                max_report_length=config.community_reports.max_length,
            )
            tick()
            return res

        local_reports = await derive_from_rows_bounded(
            level_context,
            _run_generate,
            callbacks=NoopWorkflowCallbacks(),
            num_threads=config.concurrent_requests,
            progress_msg=f"level {level} summarize communities progress: ",
        )
        reports.extend([lr for lr in local_reports if lr is not None])

    output = finalize_community_reports(pd.DataFrame(reports), communities)
    await context.output_table_provider.write_dataframe("community_reports", output)
    logger.info("Workflow completed (asyncio_optimized): create_community_reports")
    return WorkflowFunctionOutput(result=output)


async def run_generate_text_embeddings_optimized(
    config: GraphRagConfig,
    context: PipelineRunContext,
) -> WorkflowFunctionOutput:
    """Optimized asyncio workflow for ``generate_text_embeddings``."""
    logger.info("Workflow started (asyncio_optimized): generate_text_embeddings")
    model_config = config.get_embedding_model_config(
        config.embed_text.embedding_model_id
    )
    model = create_embedding(
        model_config,
        cache=context.cache.child(config.embed_text.model_instance_name),
        cache_key_creator=cache_key_creator,
    )
    tokenizer = model.tokenizer
    table_provider: TableProvider = context.output_table_provider

    for field_name in config.embed_text.names:
        field_config = EMBEDDING_FIELDS[field_name]
        if not await table_provider.has(field_config.table_name):
            logger.warning(
                "Embedding %s is specified but source table '%s' is not in storage. Skipping.",
                field_config.name,
                field_config.table_name,
            )
            continue

        vector_store = create_vector_store(
            config.vector_store,
            config.vector_store.index_schema[field_config.name],
        )
        vector_store.connect()

        async with AsyncExitStack() as stack:
            row_transform = (
                transform_entity_row_for_embedding
                if field_config.row_transform is not None
                else None
            )
            input_table = await stack.enter_async_context(
                table_provider.open(
                    field_config.table_name,
                    truncate=False,
                    transformer=row_transform,
                )
            )
            output_table = None
            if config.snapshots.embeddings:
                output_table = await stack.enter_async_context(
                    table_provider.open(f"embeddings.{field_config.name}")
                )

            count = await embed_text_pipelined(
                input_table=input_table,
                callbacks=context.callbacks,
                model=model,
                tokenizer=tokenizer,
                embed_column=field_config.embed_column,
                batch_size=config.embed_text.batch_size,
                batch_max_tokens=config.embed_text.batch_max_tokens,
                num_threads=config.concurrent_requests,
                vector_store=vector_store,
                output_table=output_table,
            )
        logger.info("Embedded %d rows for %s", count, field_config.name)

    logger.info("Workflow completed (asyncio_optimized): generate_text_embeddings")
    return WorkflowFunctionOutput(result=None)
