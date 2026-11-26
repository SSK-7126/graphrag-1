# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Validation tests for Phase 2 (serial, asyncio_optimized), Phase 3 (Ray, Dask), and Phase 4 (CUDA)."""

from __future__ import annotations

import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.index.operations.embed_text.run_embed_text import _reconstitute_embeddings
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportResponse,
    FindingModel,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.index.run.utils import create_run_context
from graphrag.index.utils.derive_from_rows import ParallelizationError
from hpc_parallel import (
    ExecutionMode,
    HPCParallelConfig,
    compare_modes,
    profiling_session,
    run_index,
)
from hpc_parallel.backends.asyncio_ops import (
    derive_from_rows_bounded,
    embed_text_pipelined,
    summarize_descriptions_concurrent,
)
from hpc_parallel.backends.cuda_backend import (
    compute_cuda_launch_config,
    detect_cuda_device,
    ensure_cuda_ready,
    reconstitute_embeddings_cuda,
)
from hpc_parallel.backends.dask_backend import make_dask_create_base_text_units
from hpc_parallel.backends.ray_backend import (
    make_ray_create_base_text_units,
    should_use_distributed,
)
from pandas.testing import assert_frame_equal

from tests.unit.config.utils import get_default_graphrag_config

if TYPE_CHECKING:
    from pathlib import Path


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text.split()))))

    def decode(self, tokens: list[int]) -> str:
        return " ".join("tok" for _ in tokens)

    def num_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


class _OverlapMockCompletion:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.tokenizer = _DummyTokenizer()
        self.active = 0
        self.peak = 0

    async def completion_async(self, **_kwargs: Any) -> Any:
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            await asyncio.sleep(self.delay)
            return SimpleNamespace(content="merged summary", formatted_response=None)
        finally:
            self.active -= 1


async def test_derive_from_rows_bounded_order_and_errors(tmp_path: Path) -> None:
    """Verify bounded worker queue preserves row order and error counts."""
    df = pd.DataFrame({"x": [10, 20, 30, 40, 50]})

    async def _transform(row: pd.Series) -> int:
        await asyncio.sleep(0.01 * (5 - int(row["x"]) // 10))
        return int(row["x"]) + 1

    with profiling_session(output_dir=tmp_path) as profiler:
        assert profiler is not None
        res = await derive_from_rows_bounded(
            df, _transform, NoopWorkflowCallbacks(), num_threads=3
        )
    assert res == [11, 21, 31, 41, 51]
    assert profiler.records[0].peak_concurrency <= 3
    assert profiler.records[0].peak_concurrency >= 2

    async def _fail_one(row: pd.Series) -> int:
        await asyncio.sleep(0.005)
        if int(row["x"]) == 30:
            msg = "boom"
            raise ValueError(msg)
        return int(row["x"])

    with pytest.raises(ParallelizationError) as exc_info:
        await derive_from_rows_bounded(
            df, _fail_one, NoopWorkflowCallbacks(), num_threads=2
        )
    assert exc_info.value.num_errors == 1


async def test_summarize_descriptions_concurrent_equivalence() -> None:
    """Verify concurrent node+edge summarization matches baseline output."""
    entities = pd.DataFrame([
        {"title": "N1", "description": ["a", "b"]},
        {"title": "N2", "description": ["single"]},
    ])
    rels = pd.DataFrame([
        {"source": "N1", "target": "N2", "description": ["r1", "r2"]},
    ])
    model_base = _OverlapMockCompletion(delay=0.01)
    model_opt = _OverlapMockCompletion(delay=0.01)

    ent_base, rel_base = await summarize_descriptions(
        entities,
        rels,
        NoopWorkflowCallbacks(),
        model_base,  # type: ignore[arg-type]
        100,
        1000,
        "p {entity_name} {description_list} {max_length}",
        4,
    )
    ent_opt, rel_opt = await summarize_descriptions_concurrent(
        entities,
        rels,
        NoopWorkflowCallbacks(),
        model_opt,  # type: ignore[arg-type]
        100,
        1000,
        "p {entity_name} {description_list} {max_length}",
        4,
    )
    assert_frame_equal(ent_base, ent_opt)
    assert_frame_equal(rel_base, rel_opt)
    assert model_opt.peak >= 2


async def test_embed_text_pipelined_writes_all_rows() -> None:
    """Verify pipelined embedding overlaps API & persistence without losing rows."""

    class _FakeTable:
        def __init__(self, rows: list[dict[str, Any]]) -> None:
            self._rows = rows
            self.written: list[dict[str, Any]] = []

        def __aiter__(self):
            async def _gen():
                await asyncio.sleep(0)
                for r in self._rows:
                    yield r

            return _gen()

        async def write(self, row: dict[str, Any]) -> None:
            self.written.append(row)

    class _FakeVectorStore:
        def __init__(self) -> None:
            self.created = False
            self.loaded: list[Any] = []

        def create_index(self) -> None:
            self.created = True

        def load_documents(self, docs: list[Any]) -> None:
            self.loaded.extend(docs)

    class _FakeEmbeddingModel:
        async def embedding_async(self, *, input: list[str], **_kwargs: Any) -> Any:
            await asyncio.sleep(0.01)
            return SimpleNamespace(embeddings=[[1.0, 0.0] for _ in input])

    in_table = _FakeTable([{"id": f"id-{i}", "text": f"text {i}"} for i in range(5)])
    out_table = _FakeTable([])
    vs = _FakeVectorStore()

    count = await embed_text_pipelined(
        input_table=in_table,  # type: ignore[arg-type]
        callbacks=NoopWorkflowCallbacks(),
        model=_FakeEmbeddingModel(),  # type: ignore[arg-type]
        tokenizer=_DummyTokenizer(),  # type: ignore[arg-type]
        embed_column="text",
        batch_size=2,
        batch_max_tokens=100,
        num_threads=1,
        vector_store=vs,  # type: ignore[arg-type]
        output_table=out_table,  # type: ignore[arg-type]
    )
    assert count == 5
    assert vs.created
    assert [d.id for d in vs.loaded] == [f"id-{i}" for i in range(5)]
    assert [r["id"] for r in out_table.written] == [f"id-{i}" for i in range(5)]


async def test_compare_modes_serial_baseline_asyncio_optimized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run serial, baseline, and asyncio_optimized end-to-end and verify output equivalence."""
    monkeypatch.setattr(
        "graphrag.index.workflows.create_base_text_units.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "hpc_parallel.backends.hybrid_orchestrator.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )
    mock_entity_response = (
        '("entity"<|>ORG_ONE<|>ORGANIZATION<|>Org One is a research lab)\n'
        "##\n"
        '("entity"<|>PERSON_ONE<|>PERSON<|>Person One leads Org One)\n'
        "##\n"
        '("relationship"<|>PERSON_ONE<|>ORG_ONE<|>Person One leads Org One<|>1.0)'
    )
    mock_report = CommunityReportResponse(
        title="Lab Community",
        summary="Summary of lab community",
        rating=4.0,
        rating_explanation="Clear",
        findings=[
            FindingModel(summary="Leadership", explanation="Person One leads Org One")
        ],
    ).model_dump_json()

    def _cfg_factory(mode_root: Path, _mode: ExecutionMode) -> Any:
        cfg = get_default_graphrag_config()
        cfg.async_mode = "asyncio"  # type: ignore[assignment]
        cfg.output_storage.base_dir = str(mode_root / "output")
        assert cfg.cache.storage is not None
        cfg.cache.storage.base_dir = str(mode_root / "cache")
        cfg.reporting.base_dir = str(mode_root / "logs")
        cfg.vector_store.db_uri = str(mode_root / "output" / "lancedb")
        cfg.workflows = [
            "create_base_text_units",
            "create_final_documents",
            "extract_graph",
            "finalize_graph",
            "create_communities",
            "create_final_text_units",
            "create_community_reports",
        ]
        cfg.completion_models["default_completion_model"].type = "mock"
        cfg.completion_models["default_completion_model"].mock_responses = [
            mock_entity_response
        ]
        cfg.completion_models["report_mock"] = cfg.completion_models[
            "default_completion_model"
        ].model_copy(update={"mock_responses": [mock_report]})
        cfg.community_reports.completion_model_id = "report_mock"
        return cfg

    input_docs = pd.DataFrame([
        {
            "id": "doc-1",
            "title": "Doc 1",
            "text": "Person One leads Org One.",
            "creation_date": "2026-01-01",
            "metadata": {"source": "compare-test"},
        }
    ])

    results = await compare_modes(
        modes=[
            ExecutionMode.SERIAL,
            ExecutionMode.BASELINE,
            ExecutionMode.ASYNCIO_OPTIMIZED,
            ExecutionMode.CPU_HYBRID,
            ExecutionMode.MULTI_PLATFORM_HYBRID,
        ],
        config_factory=_cfg_factory,
        output_dir=tmp_path / "cmp",
        input_documents=input_docs,
        concurrency_limit=4,
        repetitions=2,
    )
    assert len(results) == 5
    assert all(r.errors == 0 for r in results)
    assert (tmp_path / "cmp" / "comparison_summary.csv").exists()
    assert (tmp_path / "cmp" / "comparison_summary.json").exists()

    serial_profile = json.loads(
        (
            tmp_path / "cmp" / "serial" / "rep_1" / "profile" / "function_timings.json"
        ).read_text(encoding="utf-8")
    )
    assert all(s["max_concurrency_observed"] <= 1 for s in serial_profile["summaries"])

    for table in ("documents", "text_units", "entities", "relationships"):
        dfs = [
            pd.read_parquet(
                tmp_path / "cmp" / m.value / "rep_1" / "output" / f"{table}.parquet"
            )
            for m in (
                ExecutionMode.SERIAL,
                ExecutionMode.BASELINE,
                ExecutionMode.ASYNCIO_OPTIMIZED,
                ExecutionMode.CPU_HYBRID,
                ExecutionMode.MULTI_PLATFORM_HYBRID,
            )
        ]
        for other in dfs[1:]:
            assert list(dfs[0].columns) == list(other.columns)
            assert len(dfs[0]) == len(other)


async def test_ray_and_dask_threshold_gating_and_mocked_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Ray/Dask skip small workloads and dispatch large/forced workloads."""
    monkeypatch.setattr(
        "graphrag.index.workflows.create_base_text_units.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "hpc_parallel.backends.hybrid_orchestrator.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "hpc_parallel.backends.ray_backend.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )

    assert not should_use_distributed(10, HPCParallelConfig(distributed_min_items=500))
    assert should_use_distributed(500, HPCParallelConfig(distributed_min_items=500))
    assert should_use_distributed(
        1, HPCParallelConfig(distributed_min_items=500, force_distributed=True)
    )

    # Fake ray module
    ray_calls: list[int] = []
    fake_ray = ModuleType("ray")
    fake_ray.is_initialized = lambda: True  # type: ignore[attr-defined]

    def _remote(fn: Any) -> Any:
        class _RemoteWrapper:
            def remote(self, *args: Any) -> Any:
                ray_calls.append(1)
                return fn(*args)

        return _RemoteWrapper()

    fake_ray.remote = _remote  # type: ignore[attr-defined]
    fake_ray.get = lambda futures: list(futures)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ray", fake_ray)

    # Fake dask.distributed module
    dask_calls: list[int] = []
    fake_dask = ModuleType("dask")
    fake_dist = ModuleType("dask.distributed")

    class _FakeClient:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def submit(self, fn: Any, *args: Any) -> Any:
            dask_calls.append(1)
            return fn(*args)

        def gather(self, futures: list[Any]) -> list[Any]:
            return list(futures)

        def close(self) -> None:
            pass

    fake_dist.get_client = lambda: (_ for _ in ()).throw(ValueError("no client"))  # type: ignore[attr-defined]
    fake_dist.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dask", fake_dask)
    monkeypatch.setitem(sys.modules, "dask.distributed", fake_dist)

    docs_df = pd.DataFrame([
        {
            "id": f"d{i}",
            "title": f"T{i}",
            "text": f"hello world {i}",
            "creation_date": "2026-01-01",
        }
        for i in range(4)
    ])
    cfg = get_default_graphrag_config()

    # 1. Below threshold -> local execution, 0 remote calls
    ctx_small = create_run_context()
    await ctx_small.output_table_provider.write_dataframe("documents", docs_df)
    await make_ray_create_base_text_units(
        HPCParallelConfig(execution_mode=ExecutionMode.RAY, distributed_min_items=500)
    )(cfg, ctx_small)
    assert len(ray_calls) == 0

    # 2. Forced distributed -> Ray & Dask dispatch partitions & match local output
    ctx_ray = create_run_context()
    await ctx_ray.output_table_provider.write_dataframe("documents", docs_df)
    await make_ray_create_base_text_units(
        HPCParallelConfig(
            execution_mode=ExecutionMode.RAY,
            force_distributed=True,
            ray_num_cpus=2,
        )
    )(cfg, ctx_ray)
    assert len(ray_calls) == 2

    ctx_dask = create_run_context()
    await ctx_dask.output_table_provider.write_dataframe("documents", docs_df)
    await make_dask_create_base_text_units(
        HPCParallelConfig(
            execution_mode=ExecutionMode.DASK,
            force_distributed=True,
            dask_n_workers=2,
        )
    )(cfg, ctx_dask)
    assert len(dask_calls) == 2

    df_small = await ctx_small.output_table_provider.read_dataframe("text_units")
    df_ray = await ctx_ray.output_table_provider.read_dataframe("text_units")
    df_dask = await ctx_dask.output_table_provider.read_dataframe("text_units")
    assert_frame_equal(df_small, df_ray)
    assert_frame_equal(df_small, df_dask)


async def test_cuda_backend_detection_fallback_and_reconstitution(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Verify CUDA device probing, CPU fallback, and embedding reconstitution equivalence."""
    info = detect_cuda_device()
    assert info.available is False

    with pytest.raises(RuntimeError):
        ensure_cuda_ready(
            HPCParallelConfig(
                execution_mode=ExecutionMode.CUDA, cuda_fallback_to_cpu=False
            )
        )

    fallback_info = ensure_cuda_ready(
        HPCParallelConfig(execution_mode=ExecutionMode.CUDA, cuda_fallback_to_cpu=True)
    )
    assert fallback_info.available is False

    # Verify CPU fallback works end-to-end via run_index
    monkeypatch.setattr(
        "graphrag.index.workflows.create_base_text_units.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "hpc_parallel.backends.hybrid_orchestrator.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )
    cfg = get_default_graphrag_config()
    cfg.output_storage.base_dir = str(tmp_path / "out")
    assert cfg.cache.storage is not None
    cfg.cache.storage.base_dir = str(tmp_path / "cache")
    cfg.reporting.base_dir = str(tmp_path / "logs")
    cfg.vector_store.db_uri = str(tmp_path / "out" / "lancedb")
    cfg.workflows = ["create_base_text_units", "create_final_documents"]
    docs = pd.DataFrame([
        {
            "id": "doc-1",
            "title": "D1",
            "text": "Sample text",
            "creation_date": "2026-01-01",
            "metadata": {"source": "cuda-fallback"},
        }
    ])
    outputs = await run_index(
        config=cfg,
        hpc_config=HPCParallelConfig(
            execution_mode=ExecutionMode.CUDA,
            cuda_fallback_to_cpu=True,
            profiling_enabled=False,
        ),
        skip_validation=True,
        input_documents=docs,
    )
    assert all(o.error is None for o in outputs)

    # Verify explicit CUDA launch geometry calculation
    assert compute_cuda_launch_config(3072, threads_per_block=256) == (12, 256)
    assert compute_cuda_launch_config(1536, threads_per_block=128) == (12, 128)
    assert compute_cuda_launch_config(
        3072, threads_per_block=256, blocks_per_grid=16
    ) == (
        16,
        256,
    )
    with pytest.raises(ValueError, match="threads_per_block"):
        compute_cuda_launch_config(3072, threads_per_block=0)
    with pytest.raises(ValueError, match="threads_per_block"):
        compute_cuda_launch_config(3072, threads_per_block=2048)

    # Verify reconstitute_embeddings_cuda matches baseline numerically
    raw = [[3.0, 4.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]
    sizes = [0, 1, 2, 1]
    expected = _reconstitute_embeddings(raw, sizes)
    actual_cpu = reconstitute_embeddings_cuda(raw, sizes, min_elements=999999)
    assert expected[0] is None
    assert actual_cpu[0] is None
    for exp_v, act_v in zip(expected[1:], actual_cpu[1:], strict=True):
        assert exp_v is not None
        assert act_v is not None
        assert np.allclose(exp_v, act_v)
