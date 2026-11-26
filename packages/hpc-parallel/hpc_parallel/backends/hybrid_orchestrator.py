# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Integrated CPU_HYBRID and MULTI_PLATFORM_HYBRID orchestrator."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING, Any

import pandas as pd
from graphrag.cache.cache_key_creator import cache_key_creator
from graphrag.data_model.data_reader import DataReader
from graphrag.data_model.row_transformers import transform_entity_row_for_embedding
from graphrag.index.operations.build_noun_graph.build_noun_graph import _extract_edges
from graphrag.index.operations.build_noun_graph.np_extractors.factory import (
    create_noun_phrase_extractor,
)
from graphrag.index.operations.embed_text.run_embed_text import (
    TextEmbeddingResult,
    _create_text_batches,
    _execute,
    _prepare_embed_texts,
)
from graphrag.index.operations.extract_graph.extract_graph import (
    _merge_entities,
    _merge_relationships,
    _run_extract_graph,
)
from graphrag.index.operations.extract_graph.utils import filter_orphan_relationships
from graphrag.index.run import hpc_hooks
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.is_null import is_null
from graphrag.index.workflows.create_base_text_units import create_base_text_units
from graphrag.index.workflows.extract_graph_nlp import (
    _write_entities,
    _write_relationships,
    extract_graph_nlp,
)
from graphrag.index.workflows.generate_text_embeddings import EMBEDDING_FIELDS
from graphrag.logger.progress import progress_ticker
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag_llm.completion import create_completion
from graphrag_llm.embedding import create_embedding
from graphrag_vectors import VectorStoreDocument, create_vector_store
from typing_extensions import Self

from hpc_parallel.backends.asyncio_ops import (
    derive_from_rows_bounded,
    summarize_descriptions_concurrent,
)
from hpc_parallel.backends.cuda_backend import (
    detect_cuda_device,
    reconstitute_embeddings_cuda,
)
from hpc_parallel.backends.ray_backend import (
    _chunk_doc_partition,
    _extract_np_partition,
)
from hpc_parallel.config import ExecutionMode, HPCParallelConfig

if TYPE_CHECKING:
    from types import TracebackType

    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.typing.context import PipelineRunContext
    from graphrag_storage.tables.table_provider import TableProvider

logger = logging.getLogger(__name__)


class HybridRuntimeManager:
    """Lazy lifecycle and shared resource manager for hybrid execution.

    Runtimes (ProcessPoolExecutor, Ray, Dask) are initialized only if a stage
    exceeds its workload threshold and requests that platform, and are shut
    down gracefully on pipeline completion or cancellation.
    """

    def __init__(self, hpc_config: HPCParallelConfig) -> None:
        self.hpc_config = hpc_config
        total_cpus = os.cpu_count() or 4
        self.cpu_workers = max(1, min(total_cpus, hpc_config.cpu_workers or total_cpus))
        self._process_pool: ProcessPoolExecutor | None = None
        self._ray_started_by_us = False
        self._dask_client: Any = None
        self._cuda_info = (
            detect_cuda_device()
            if hpc_config.execution_mode == ExecutionMode.MULTI_PLATFORM_HYBRID
            else None
        )

    def get_process_pool(self) -> ProcessPoolExecutor:
        """Lazily create a bounded ProcessPoolExecutor."""
        if self._process_pool is None:
            self._process_pool = ProcessPoolExecutor(max_workers=self.cpu_workers)
        return self._process_pool

    def try_get_ray(self) -> Any | None:
        """Lazily import and initialize Ray if available."""
        try:
            ray = importlib.import_module("ray")
        except ImportError:
            return None
        if not ray.is_initialized():
            num_cpus = min(
                self.cpu_workers,
                self.hpc_config.ray_num_cpus or self.cpu_workers,
            )
            ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
            self._ray_started_by_us = True
        return ray

    def try_get_dask_client(self) -> Any | None:
        """Lazily import and acquire a Dask Client if available."""
        try:
            distributed = importlib.import_module("dask.distributed")
        except ImportError:
            return None
        try:
            return distributed.get_client()
        except ValueError:
            if self._dask_client is None:
                n_workers = min(
                    self.cpu_workers,
                    self.hpc_config.dask_n_workers or self.cpu_workers,
                )
                self._dask_client = distributed.Client(n_workers=n_workers)
            return self._dask_client

    async def __aenter__(self) -> Self:
        """Enter async runtime manager."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Gracefully shut down any lazily initialized runtimes."""
        if self._process_pool is not None:
            self._process_pool.shutdown(wait=True, cancel_futures=exc_type is not None)
            self._process_pool = None
        if self._dask_client is not None:
            try:
                self._dask_client.close()
            finally:
                self._dask_client = None
        if self._ray_started_by_us:
            try:
                ray = importlib.import_module("ray")
                if ray.is_initialized():
                    ray.shutdown()
            except ImportError:
                pass
            finally:
                self._ray_started_by_us = False


def _merge_entities_dask(
    entity_dfs: list[pd.DataFrame], npartitions: int
) -> pd.DataFrame:
    """Merge extracted entity DataFrames using Dask DataFrame when available."""
    dask_df = importlib.import_module("dask.dataframe")
    combined = pd.concat(entity_dfs, ignore_index=True)
    ddf = dask_df.from_pandas(combined, npartitions=max(1, npartitions))
    grouped = (
        ddf
        .groupby(["title", "type"])
        .agg({"description": list, "source_id": list})
        .compute()
        .reset_index()
    )
    grouped["frequency"] = grouped["source_id"].apply(len)
    return grouped.rename(columns={"source_id": "text_unit_ids"})[
        ["title", "type", "description", "text_unit_ids", "frequency"]
    ]


def _merge_relationships_dask(
    relationship_dfs: list[pd.DataFrame], npartitions: int
) -> pd.DataFrame:
    """Merge extracted relationship DataFrames using Dask DataFrame when available."""
    dask_df = importlib.import_module("dask.dataframe")
    combined = pd.concat(relationship_dfs, ignore_index=True)
    ddf = dask_df.from_pandas(combined, npartitions=max(1, npartitions))
    grouped = (
        ddf
        .groupby(["source", "target"])
        .agg({"description": list, "source_id": list, "weight": "sum"})
        .compute()
        .reset_index()
    )
    return grouped.rename(columns={"source_id": "text_unit_ids"})[
        ["source", "target", "description", "text_unit_ids", "weight"]
    ]


def make_hybrid_create_base_text_units(
    hpc_config: HPCParallelConfig,
    runtime: HybridRuntimeManager,
):
    """Build hybrid ``create_base_text_units`` (Ray -> ProcessPool -> local CPU)."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
        tokenizer = get_tokenizer(encoding_model=config.chunking.encoding_model)
        async with (
            context.output_table_provider.open("documents") as docs_table,
            context.output_table_provider.open("text_units") as tu_table,
        ):
            total_rows = await docs_table.length()
            use_ray = (
                hpc_config.execution_mode == ExecutionMode.MULTI_PLATFORM_HYBRID
                and (
                    hpc_config.force_distributed
                    or total_rows >= hpc_config.distributed_min_items
                )
            )
            ray = runtime.try_get_ray() if use_ray else None

            use_pool = ray is None and total_rows >= hpc_config.cpu_pool_min_items

            if ray is None and not use_pool:
                from graphrag_chunking.chunker_factory import create_chunker

                chunker = create_chunker(
                    config.chunking, tokenizer.encode, tokenizer.decode
                )
                with hpc_hooks.profile_function(
                    "hybrid.create_base_text_units",
                    input_count=total_rows,
                    configured_concurrency=1,
                ) as span:
                    span.set_counts(platform="cpu_local", worker_count=1)
                    sample_rows = await create_base_text_units(
                        docs_table,
                        tu_table,
                        total_rows,
                        context.callbacks,
                        tokenizer=tokenizer,
                        chunker=chunker,
                        prepend_metadata=config.chunking.prepend_metadata,
                    )
                    span.set_counts(output_count=len(sample_rows))
                return WorkflowFunctionOutput(result=sample_rows)

            docs = [doc async for doc in docs_table]
            chunking_dump = config.chunking.model_dump()
            n_workers = min(len(docs), runtime.cpu_workers)
            part_size = max(1, (len(docs) + n_workers - 1) // n_workers)
            doc_parts = [
                docs[i : i + part_size] for i in range(0, len(docs), part_size)
            ]

            platform_name = "ray" if ray is not None else "cpu_pool"
            with hpc_hooks.profile_function(
                "hybrid.create_base_text_units",
                input_count=len(docs),
                batch_count=len(doc_parts),
                configured_concurrency=n_workers,
            ) as span:
                t_sched = time.perf_counter()
                if ray is not None:
                    remote_fn = ray.remote(_chunk_doc_partition)
                    futures = [remote_fn.remote(p, chunking_dump) for p in doc_parts]
                    partitions: list[list[dict[str, Any]]] = ray.get(futures)
                else:
                    pool = runtime.get_process_pool()
                    loop = asyncio.get_running_loop()
                    futs = [
                        loop.run_in_executor(
                            pool, _chunk_doc_partition, p, chunking_dump
                        )
                        for p in doc_parts
                    ]
                    partitions = await asyncio.gather(*futs)
                sched_overhead = time.perf_counter() - t_sched

                sample_rows: list[dict[str, Any]] = []
                written = 0
                for part in partitions:
                    for row in part:
                        await tu_table.write(row)
                        written += 1
                        if len(sample_rows) < 5:
                            sample_rows.append(row)

                span.set_counts(
                    platform=platform_name,
                    worker_count=n_workers,
                    output_count=written,
                    scheduling_overhead_seconds=sched_overhead,
                )
            return WorkflowFunctionOutput(result=sample_rows)

    return _workflow


def make_hybrid_extract_graph_nlp(
    hpc_config: HPCParallelConfig,
    runtime: HybridRuntimeManager,
):
    """Build hybrid ``extract_graph_nlp`` (Ray -> ProcessPool -> local CPU)."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
        text_analyzer_cfg = config.extract_graph_nlp.text_analyzer
        text_analyzer = create_noun_phrase_extractor(text_analyzer_cfg)

        async with (
            context.output_table_provider.open(
                "text_units", truncate=False
            ) as tu_table,
            context.output_table_provider.open("entities") as entities_table,
            context.output_table_provider.open("relationships") as rels_table,
        ):
            total_rows = await tu_table.length()
            use_ray = (
                hpc_config.execution_mode == ExecutionMode.MULTI_PLATFORM_HYBRID
                and (
                    hpc_config.force_distributed
                    or total_rows >= hpc_config.distributed_min_items
                )
            )
            ray = runtime.try_get_ray() if use_ray else None
            use_pool = ray is None and total_rows >= hpc_config.cpu_pool_min_items

            if ray is None and not use_pool:
                with hpc_hooks.profile_function(
                    "hybrid.extract_graph_nlp",
                    input_count=total_rows,
                    configured_concurrency=1,
                ) as span:
                    span.set_counts(platform="cpu_local", worker_count=1)
                    result = await extract_graph_nlp(
                        tu_table,
                        context.cache,
                        entities_table=entities_table,
                        relationships_table=rels_table,
                        text_analyzer=text_analyzer,
                        normalize_edge_weights=config.extract_graph_nlp.normalize_edge_weights,
                    )
                return WorkflowFunctionOutput(result=result)

            rows = [{"id": r["id"], "text": r["text"]} async for r in tu_table]
            analyzer_dump = text_analyzer_cfg.model_dump()
            n_workers = min(len(rows), runtime.cpu_workers)
            part_size = max(1, (len(rows) + n_workers - 1) // n_workers)
            row_parts = [
                rows[i : i + part_size] for i in range(0, len(rows), part_size)
            ]

            platform_name = "ray" if ray is not None else "cpu_pool"
            with hpc_hooks.profile_function(
                "hybrid.extract_graph_nlp",
                input_count=len(rows),
                batch_count=len(row_parts),
                configured_concurrency=n_workers,
            ) as span:
                t_sched = time.perf_counter()
                if ray is not None:
                    remote_fn = ray.remote(_extract_np_partition)
                    futures = [remote_fn.remote(p, analyzer_dump) for p in row_parts]
                    partitions: list[list[tuple[str, str]]] = ray.get(futures)
                else:
                    pool = runtime.get_process_pool()
                    loop = asyncio.get_running_loop()
                    futs = [
                        loop.run_in_executor(
                            pool, _extract_np_partition, p, analyzer_dump
                        )
                        for p in row_parts
                    ]
                    partitions = await asyncio.gather(*futs)
                sched_overhead = time.perf_counter() - t_sched

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
                entity_samples = await _write_entities(nodes_df, entities_table)
                rel_samples = await _write_relationships(edges_df, rels_table)
                span.set_counts(
                    platform=platform_name,
                    worker_count=n_workers,
                    output_count=len(nodes_df) + len(edges_df),
                    scheduling_overhead_seconds=sched_overhead,
                )
                return WorkflowFunctionOutput(
                    result={"entities": entity_samples, "relationships": rel_samples}
                )

    return _workflow


def make_hybrid_extract_graph(
    hpc_config: HPCParallelConfig,
    runtime: HybridRuntimeManager,
):
    """Build hybrid ``extract_graph`` (asyncio for LLM calls + Dask/pandas for large merges)."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
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
            cache=context.cache.child(
                config.summarize_descriptions.model_instance_name
            ),
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
        total_ent_rows = sum(len(df) for df in entity_dfs)
        total_rel_rows = sum(len(df) for df in relationship_dfs)

        use_dask = (
            hpc_config.execution_mode == ExecutionMode.MULTI_PLATFORM_HYBRID
            and (
                hpc_config.force_distributed
                or (total_ent_rows + total_rel_rows) >= hpc_config.dataframe_min_rows
            )
        )
        dask_client = runtime.try_get_dask_client() if use_dask else None
        try:
            importlib.import_module("dask.dataframe")
            has_dask_df = dask_client is not None
        except ImportError:
            has_dask_df = False

        if has_dask_df:
            extracted_entities = _merge_entities_dask(entity_dfs, runtime.cpu_workers)
            extracted_relationships = filter_orphan_relationships(
                _merge_relationships_dask(relationship_dfs, runtime.cpu_workers),
                extracted_entities,
            )
        else:
            extracted_entities = _merge_entities(entity_dfs)
            extracted_relationships = filter_orphan_relationships(
                _merge_relationships(relationship_dfs),
                extracted_entities,
            )

        if len(extracted_entities) == 0:
            err = "Graph Extraction failed. No entities detected during extraction."
            raise ValueError(err)
        if len(extracted_relationships) == 0:
            err = (
                "Graph Extraction failed. No relationships detected during extraction."
            )
            raise ValueError(err)

        raw_entities = extracted_entities.copy()
        raw_relationships = extracted_relationships.copy()

        (
            entity_summaries,
            relationship_summaries,
        ) = await summarize_descriptions_concurrent(
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
        await context.output_table_provider.write_dataframe(
            "relationships", relationships
        )
        if config.snapshots.raw_graph:
            await context.output_table_provider.write_dataframe(
                "raw_entities", raw_entities
            )
            await context.output_table_provider.write_dataframe(
                "raw_relationships", raw_relationships
            )
        return WorkflowFunctionOutput(
            result={"entities": entities, "relationships": relationships}
        )

    return _workflow


def make_hybrid_generate_text_embeddings(
    hpc_config: HPCParallelConfig,
):
    """Build hybrid ``generate_text_embeddings`` (asyncio API + CUDA local post-proc + pipelined DB)."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
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
        use_cuda = (
            hpc_config.execution_mode == ExecutionMode.MULTI_PLATFORM_HYBRID
            and detect_cuda_device().available
        )

        for field_name in config.embed_text.names:
            field_config = EMBEDDING_FIELDS[field_name]
            if not await table_provider.has(field_config.table_name):
                continue

            vector_store = create_vector_store(
                config.vector_store,
                config.vector_store.index_schema[field_config.name],
            )
            vector_store.connect()
            vector_store.create_index()

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

                flush_size = max(
                    1, config.embed_text.batch_size * max(1, config.concurrent_requests)
                )
                buffer: list[dict[str, Any]] = []
                pending_write: asyncio.Task[None] | None = None

                async def _run_and_persist(
                    batch_rows: list[dict[str, Any]],
                    field_cfg: Any = field_config,
                    out_tbl: Any = output_table,
                    vs: Any = vector_store,
                ) -> TextEmbeddingResult:
                    texts = [r[field_cfg.embed_column] for r in batch_rows]
                    if is_null(texts):
                        return TextEmbeddingResult(embeddings=None)
                    sem = asyncio.Semaphore(config.concurrent_requests)
                    snippets, sizes = _prepare_embed_texts(
                        texts, tokenizer, config.embed_text.batch_max_tokens
                    )
                    batches = _create_text_batches(
                        snippets,
                        tokenizer,
                        config.embed_text.batch_size,
                        config.embed_text.batch_max_tokens,
                    )
                    ticker = progress_ticker(
                        context.callbacks.progress,
                        len(batches),
                        description="generate embeddings progress: ",
                    )
                    raw_embeddings = await _execute(model, batches, ticker, sem)
                    if use_cuda:
                        reconstituted = reconstitute_embeddings_cuda(
                            raw_embeddings,
                            sizes,
                            min_elements=hpc_config.cuda_min_elements,
                        )
                    else:
                        reconstituted = reconstitute_embeddings_cuda(
                            raw_embeddings,
                            sizes,
                            min_elements=10**12,
                        )

                    docs = [
                        VectorStoreDocument(id=r["id"], vector=vec)
                        for r, vec in zip(batch_rows, reconstituted, strict=True)
                        if vec is not None
                    ]
                    if docs:
                        await asyncio.to_thread(vs.load_documents, docs)
                    if out_tbl is not None:
                        for r, vec in zip(batch_rows, reconstituted, strict=True):
                            if vec is not None:
                                await out_tbl.write({"id": r["id"], "embedding": vec})
                    return TextEmbeddingResult(embeddings=reconstituted)

                async for row in input_table:
                    text = row.get(field_config.embed_column)
                    buffer.append({
                        "id": row["id"],
                        field_config.embed_column: "" if text is None else text,
                    })
                    if len(buffer) >= flush_size:
                        snap = list(buffer)
                        buffer.clear()
                        if pending_write is not None:
                            await pending_write
                        pending_write = asyncio.create_task(
                            _run_and_persist(snap)  # type: ignore[arg-type]
                        )

                if buffer:
                    if pending_write is not None:
                        await pending_write
                    await _run_and_persist(list(buffer))
                elif pending_write is not None:
                    await pending_write

        return WorkflowFunctionOutput(result=None)

    return _workflow
