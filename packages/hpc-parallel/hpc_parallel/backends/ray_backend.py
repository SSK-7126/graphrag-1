# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Ray distributed backend for CPU-bound GraphRAG indexing stages (Phase 3)."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, Any

from graphrag.index.operations.build_noun_graph.build_noun_graph import _extract_edges
from graphrag.index.operations.build_noun_graph.np_extractors.factory import (
    create_noun_phrase_extractor,
)
from graphrag.index.typing.workflow import WorkflowFunctionOutput
from graphrag.index.utils.hashing import gen_sha512_hash
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
from graphrag.tokenizer.get_tokenizer import get_tokenizer
from graphrag_chunking.chunker_factory import create_chunker
from graphrag_chunking.transformers import add_metadata
from graphrag_input import TextDocument

if TYPE_CHECKING:
    import pandas as pd
    from graphrag.config.models.graph_rag_config import GraphRagConfig
    from graphrag.index.typing.context import PipelineRunContext

    from hpc_parallel.config import HPCParallelConfig

logger = logging.getLogger(__name__)


def ensure_ray_available() -> Any:
    """Import Ray lazily or raise a clear installation error."""
    try:
        return importlib.import_module("ray")
    except ImportError as exc:
        msg = (
            "Execution mode 'ray' requires the optional 'ray' package. "
            "Install it with: pip install 'hpc-parallel[ray]'"
        )
        raise ImportError(msg) from exc


def should_use_distributed(
    item_count: int,
    hpc_config: HPCParallelConfig,
) -> bool:
    """Return True if workload size justifies distributed Ray/Dask overhead."""
    if hpc_config.force_distributed:
        return True
    return item_count >= hpc_config.distributed_min_items


def _chunk_doc_partition(
    docs: list[dict[str, Any]],
    chunking_dump: dict[str, Any],
) -> list[dict[str, Any]]:
    """Worker-safe chunking of a partition of document rows."""
    from graphrag_chunking.chunking_config import ChunkingConfig

    chunking_cfg = ChunkingConfig(**chunking_dump)
    tokenizer = get_tokenizer(encoding_model=chunking_cfg.encoding_model)
    chunker = create_chunker(chunking_cfg, tokenizer.encode, tokenizer.decode)
    prepend_metadata = chunking_cfg.prepend_metadata
    out_rows: list[dict[str, Any]] = []

    for doc in docs:
        transformer = None
        if prepend_metadata:
            document = TextDocument(
                id=doc["id"],
                title=doc.get("title", ""),
                text=doc["text"],
                creation_date=doc.get("creation_date", ""),
                raw_data=doc.get("raw_data"),
            )
            metadata = document.collect(prepend_metadata)
            transformer = add_metadata(metadata=metadata, line_delimiter=".\n")

        for chunk in chunker.chunk(doc["text"], transform=transformer):
            if chunk.text is None:
                continue
            row: dict[str, Any] = {
                "id": "",
                "document_id": doc["id"],
                "text": chunk.text,
                "n_tokens": len(tokenizer.encode(chunk.text)),
            }
            row["id"] = gen_sha512_hash(row, ["text"])
            out_rows.append(row)
    return out_rows


def _extract_np_partition(
    rows: list[dict[str, Any]],
    analyzer_dump: dict[str, Any],
) -> list[tuple[str, str]]:
    """Worker-safe noun-phrase extraction for a partition of text units."""
    from graphrag.config.models.extract_graph_nlp_config import TextAnalyzerConfig

    analyzer = create_noun_phrase_extractor(TextAnalyzerConfig(**analyzer_dump))
    pairs: list[tuple[str, str]] = []
    for row in rows:
        tu_id = row["id"]
        pairs.extend((phrase, tu_id) for phrase in analyzer.extract(row["text"]))
    return pairs


def make_ray_create_base_text_units(hpc_config: HPCParallelConfig):
    """Create a Ray-enabled ``create_base_text_units`` workflow with small-workload guard."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
        ray = ensure_ray_available()
        async with context.output_table_provider.open("documents") as docs_table:
            total_rows = await docs_table.length()
            if not should_use_distributed(total_rows, hpc_config):
                logger.info(
                    "Ray create_base_text_units: %d docs < threshold %d; using local execution.",
                    total_rows,
                    hpc_config.distributed_min_items,
                )
                return await run_base_text_units_baseline(config, context)

            docs = [doc async for doc in docs_table]

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
            if hpc_config.ray_num_cpus is not None:
                init_kwargs["num_cpus"] = hpc_config.ray_num_cpus
            ray.init(**init_kwargs)

        chunking_dump = config.chunking.model_dump()
        remote_fn = ray.remote(_chunk_doc_partition)
        num_parts = max(1, min(len(docs), hpc_config.ray_num_cpus or 4))
        part_size = max(1, (len(docs) + num_parts - 1) // num_parts)
        futures = [
            remote_fn.remote(docs[i : i + part_size], chunking_dump)
            for i in range(0, len(docs), part_size)
        ]
        partitions: list[list[dict[str, Any]]] = ray.get(futures)

        sample_rows: list[dict[str, Any]] = []
        async with context.output_table_provider.open("text_units") as tu_table:
            for part in partitions:
                for row in part:
                    await tu_table.write(row)
                    if len(sample_rows) < 5:
                        sample_rows.append(row)

        return WorkflowFunctionOutput(result=sample_rows)

    return _workflow


def make_ray_extract_graph_nlp(hpc_config: HPCParallelConfig):
    """Create a Ray-enabled ``extract_graph_nlp`` workflow with small-workload guard."""

    async def _workflow(
        config: GraphRagConfig,
        context: PipelineRunContext,
    ) -> WorkflowFunctionOutput:
        import pandas as pd

        ray = ensure_ray_available()
        async with context.output_table_provider.open(
            "text_units", truncate=False
        ) as tu_table:
            total_rows = await tu_table.length()
            if not should_use_distributed(total_rows, hpc_config):
                logger.info(
                    "Ray extract_graph_nlp: %d text_units < threshold %d; using local execution.",
                    total_rows,
                    hpc_config.distributed_min_items,
                )
                return await run_extract_graph_nlp_baseline(config, context)
            rows = [{"id": r["id"], "text": r["text"]} async for r in tu_table]

        if not ray.is_initialized():
            init_kwargs: dict[str, Any] = {"ignore_reinit_error": True}
            if hpc_config.ray_num_cpus is not None:
                init_kwargs["num_cpus"] = hpc_config.ray_num_cpus
            ray.init(**init_kwargs)

        analyzer_dump = config.extract_graph_nlp.text_analyzer.model_dump()
        remote_fn = ray.remote(_extract_np_partition)
        num_parts = max(1, min(len(rows), hpc_config.ray_num_cpus or 4))
        part_size = max(1, (len(rows) + num_parts - 1) // num_parts)
        futures = [
            remote_fn.remote(rows[i : i + part_size], analyzer_dump)
            for i in range(0, len(rows), part_size)
        ]
        partitions: list[list[tuple[str, str]]] = ray.get(futures)

        title_to_ids: dict[str, list[str]] = {}
        for part in partitions:
            for phrase, tu_id in part:
                title_to_ids.setdefault(phrase, []).append(tu_id)

        nodes_df: pd.DataFrame = pd.DataFrame(
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
