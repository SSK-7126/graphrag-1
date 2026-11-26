# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Unit and equivalence tests for hpc_parallel Phase 1 instrumentation."""

from __future__ import annotations

import asyncio
import csv
import json
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest
from graphrag.callbacks.noop_workflow_callbacks import NoopWorkflowCallbacks
from graphrag.index.operations.embed_text.run_embed_text import run_embed_text
from graphrag.index.operations.extract_graph.graph_extractor import GraphExtractor
from graphrag.index.operations.summarize_communities.community_reports_extractor import (
    CommunityReportResponse,
    CommunityReportsExtractor,
    FindingModel,
)
from graphrag.index.operations.summarize_descriptions.summarize_descriptions import (
    summarize_descriptions,
)
from graphrag.index.run import hpc_hooks
from graphrag.index.utils.derive_from_rows import (
    ParallelizationError,
    derive_from_rows_asyncio,
)
from graphrag.index.workflows.create_community_reports import (
    run_workflow as run_community_reports_workflow,
)
from graphrag.index.workflows.extract_graph import (
    run_workflow as run_extract_graph_workflow,
)
from hpc_parallel import (
    ExecutionMode,
    HPCParallelConfig,
    ProfileOutputFormat,
    disable_profiling,
    enable_profiling,
    is_profiling_enabled,
    load_config_from_env,
    profiling_session,
    run_index,
    validate_execution_mode,
)
from pandas.testing import assert_frame_equal

from tests.unit.config.utils import get_default_graphrag_config
from tests.verbs.util import create_test_context

if TYPE_CHECKING:
    from pathlib import Path

SENSITIVE_PROMPT_SENTINEL = "SECRET_PROMPT_SENTINEL_DO_NOT_LOG"
SENSITIVE_KEY_SENTINEL = "sk-secret-api-key-sentinel-9999"


class _DummyTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(range(max(1, len(text.split()))))

    def decode(self, tokens: list[int]) -> str:
        return " ".join("tok" for _ in tokens)

    def num_tokens(self, text: str) -> int:
        return max(1, len(text.split()))


class _SlowMockEmbedding:
    def __init__(self, delay: float = 0.02) -> None:
        self.delay = delay
        self.api_key = SENSITIVE_KEY_SENTINEL

    async def embedding_async(self, *, input: list[str], **_kwargs: Any) -> Any:
        await asyncio.sleep(self.delay)
        return SimpleNamespace(embeddings=[[0.1, 0.2, 0.3, 0.4] for _ in input])


class _SlowMockCompletion:
    def __init__(
        self,
        *,
        content: str = "summary",
        formatted_response: Any = None,
        delay: float = 0.02,
        fail: bool = False,
    ) -> None:
        self.content = content
        self.formatted_response = formatted_response
        self.delay = delay
        self.fail = fail
        self.tokenizer = _DummyTokenizer()
        self.api_key = SENSITIVE_KEY_SENTINEL

    async def completion_async(self, **_kwargs: Any) -> Any:
        await asyncio.sleep(self.delay)
        if self.fail:
            msg = f"Failure carrying {SENSITIVE_KEY_SENTINEL} and {SENSITIVE_PROMPT_SENTINEL}"
            raise RuntimeError(msg)
        return SimpleNamespace(
            content=self.content,
            formatted_response=self.formatted_response,
        )


def test_profiling_enable_disable_and_env_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify profiling defaults to off and toggles cleanly."""
    disable_profiling()
    assert not is_profiling_enabled()

    with profiling_session(enabled=False, output_dir=tmp_path) as disabled_prof:
        assert disabled_prof is None
        assert not is_profiling_enabled()
    assert not list(tmp_path.iterdir())

    profiler = enable_profiling(
        HPCParallelConfig(
            execution_mode=ExecutionMode.BASELINE,
            profiling_enabled=True,
            profile_output_dir=tmp_path,
        )
    )
    assert profiler is not None
    assert is_profiling_enabled()
    disable_profiling()
    assert not is_profiling_enabled()

    monkeypatch.setenv("GRAPHRAG_HPC_PROFILE", "true")
    monkeypatch.setenv("GRAPHRAG_HPC_MODE", "baseline")
    monkeypatch.setenv("GRAPHRAG_HPC_PROFILE_DIR", str(tmp_path / "env_out"))
    monkeypatch.setenv("GRAPHRAG_HPC_PROFILE_FORMAT", "json")
    monkeypatch.setenv("GRAPHRAG_HPC_RUN_ID", "env-run-1")
    monkeypatch.setenv("GRAPHRAG_HPC_CONCURRENCY", "12")
    cfg = load_config_from_env()
    assert cfg.profiling_enabled is True
    assert cfg.execution_mode == ExecutionMode.BASELINE
    assert cfg.profile_output_dir == tmp_path / "env_out"
    assert cfg.profile_format == ProfileOutputFormat.JSON
    assert cfg.run_id == "env-run-1"
    assert cfg.concurrency_limit == 12


def test_execution_mode_validation() -> None:
    """Core and hybrid modes validate immediately; strict Ray/Dask/CUDA check backends."""
    for mode in (
        ExecutionMode.BASELINE,
        ExecutionMode.SERIAL,
        ExecutionMode.ASYNCIO_OPTIMIZED,
        ExecutionMode.CPU_HYBRID,
        ExecutionMode.MULTI_PLATFORM_HYBRID,
    ):
        validate_execution_mode(mode)

    for dist_mode in (ExecutionMode.RAY, ExecutionMode.DASK):
        with pytest.raises(ImportError):
            validate_execution_mode(dist_mode)

    with pytest.raises(RuntimeError):
        validate_execution_mode(
            ExecutionMode.CUDA,
            HPCParallelConfig(
                execution_mode=ExecutionMode.CUDA, cuda_fallback_to_cpu=False
            ),
        )


async def test_instrumented_run_embed_text_and_execute(tmp_path: Path) -> None:
    """Verify run_embed_text and _execute record timing, batches, requests, and concurrency."""
    inputs = [f"document text {i} {SENSITIVE_PROMPT_SENTINEL}" for i in range(6)]
    model = _SlowMockEmbedding(delay=0.03)
    tokenizer = _DummyTokenizer()

    with profiling_session(
        output_dir=tmp_path,
        output_format=ProfileOutputFormat.BOTH,
        run_id="embed-test",
    ) as profiler:
        assert profiler is not None
        res = await run_embed_text(
            input=inputs,
            callbacks=NoopWorkflowCallbacks(),
            model=model,  # type: ignore[arg-type]
            tokenizer=tokenizer,  # type: ignore[arg-type]
            batch_size=2,
            batch_max_tokens=100,
            num_threads=3,
        )
        assert res.embeddings is not None
        assert len(res.embeddings) == 6

    records = profiler.records
    by_func = {r.function_name: r for r in records}
    assert "run_embed_text" in by_func
    assert "_execute" in by_func

    run_rec = by_func["run_embed_text"]
    exec_rec = by_func["_execute"]
    assert run_rec.input_count == 6
    assert run_rec.batch_count == 3
    assert run_rec.request_count == 3
    assert run_rec.configured_concurrency == 3
    assert run_rec.wall_clock_seconds > 0
    assert run_rec.errors == 0

    assert exec_rec.input_count == 6
    assert exec_rec.batch_count == 3
    assert exec_rec.request_count == 3
    assert exec_rec.peak_concurrency >= 2


async def test_instrumented_extractors_derive_and_summarize(tmp_path: Path) -> None:
    """Verify GraphExtractor, CommunityReportsExtractor, derive_from_rows_asyncio, summarize_descriptions."""
    report_obj = CommunityReportResponse(
        title="Test Report",
        summary="Test Summary",
        findings=[FindingModel(summary="F1", explanation="E1")],
        rating=5.0,
        rating_explanation="Good",
    )
    ok_completion = _SlowMockCompletion(
        content="ok",
        formatted_response=report_obj,
        delay=0.02,
    )
    failing_completion = _SlowMockCompletion(fail=True, delay=0.01)

    with profiling_session(
        output_dir=tmp_path,
        output_format=ProfileOutputFormat.BOTH,
        run_id="ops-test",
    ) as profiler:
        assert profiler is not None

        # 1. GraphExtractor._process_document with max_gleanings=1 (2 requests per doc)
        extractor = GraphExtractor(
            model=ok_completion,  # type: ignore[arg-type]
            prompt=f"{SENSITIVE_PROMPT_SENTINEL} {{input_text}} {{entity_types}}",
            max_gleanings=1,
        )
        await asyncio.gather(
            extractor._process_document("alpha", ["person"]),  # noqa: SLF001
            extractor._process_document("beta", ["person"]),  # noqa: SLF001
            extractor._process_document("gamma", ["person"]),  # noqa: SLF001
        )

        # 2. CommunityReportsExtractor.__call__ (2 ok concurrent + 1 caught error)
        rep_ok = CommunityReportsExtractor(
            model=ok_completion,  # type: ignore[arg-type]
            extraction_prompt=f"{SENSITIVE_PROMPT_SENTINEL} {{input_text}} {{max_report_length}}",
            max_report_length=100,
        )
        rep_err = CommunityReportsExtractor(
            model=failing_completion,  # type: ignore[arg-type]
            extraction_prompt=f"{SENSITIVE_PROMPT_SENTINEL} {{input_text}} {{max_report_length}}",
            max_report_length=100,
        )
        await asyncio.gather(rep_ok("ctx1"), rep_ok("ctx2"), rep_err("ctx3"))

        # 3. derive_from_rows_asyncio (success + error cases)
        df = pd.DataFrame({"val": [1, 2, 3, 4]})

        async def _double(row: pd.Series) -> int:
            await asyncio.sleep(0.02)
            return int(row["val"]) * 2

        out = await derive_from_rows_asyncio(
            df, _double, NoopWorkflowCallbacks(), num_threads=2
        )
        assert out == [2, 4, 6, 8]

        async def _fail_two(row: pd.Series) -> int:
            await asyncio.sleep(0.01)
            if int(row["val"]) in {2, 4}:
                msg = f"bad row {SENSITIVE_KEY_SENTINEL}"
                raise ValueError(msg)
            return int(row["val"])

        with pytest.raises(ParallelizationError):
            await derive_from_rows_asyncio(
                df, _fail_two, NoopWorkflowCallbacks(), num_threads=2
            )

        # 4. summarize_descriptions (1 single-desc node skipped, 1 multi-desc node + 1 multi-desc edge)
        entities_df = pd.DataFrame([
            {"title": "A", "description": ["only one"]},
            {"title": "B", "description": ["desc one", "desc two"]},
        ])
        relationships_df = pd.DataFrame([
            {"source": "A", "target": "B", "description": ["r1", "r2"]},
        ])
        ent_sum, rel_sum = await summarize_descriptions(
            entities_df=entities_df,
            relationships_df=relationships_df,
            callbacks=NoopWorkflowCallbacks(),
            model=ok_completion,  # type: ignore[arg-type]
            max_summary_length=100,
            max_input_tokens=1000,
            prompt=f"{SENSITIVE_PROMPT_SENTINEL} {{entity_name}} {{description_list}} {{max_length}}",
            num_threads=2,
        )
        assert len(ent_sum) == 2
        assert len(rel_sum) == 1

    # Verify exported CSV & JSON and ensure zero sensitive leaks
    timings_csv = tmp_path / "function_timings.csv"
    summary_csv = tmp_path / "function_summary.csv"
    timings_json = tmp_path / "function_timings.json"
    assert timings_csv.exists()
    assert summary_csv.exists()
    assert timings_json.exists()

    for artifact in (timings_csv, summary_csv, timings_json):
        text = artifact.read_text(encoding="utf-8")
        assert SENSITIVE_PROMPT_SENTINEL not in text
        assert SENSITIVE_KEY_SENTINEL not in text

    payload = json.loads(timings_json.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "baseline"
    assert payload["total_records"] == len(profiler.records)

    with timings_csv.open("r", encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == len(profiler.records)

    # Check function-specific assertions
    doc_recs = [
        r
        for r in profiler.records
        if r.function_name == "GraphExtractor._process_document"
    ]
    assert len(doc_recs) == 3
    assert all(r.request_count == 2 for r in doc_recs)
    assert max(r.peak_concurrency for r in doc_recs) >= 2

    cr_recs = [
        r
        for r in profiler.records
        if r.function_name == "CommunityReportsExtractor.__call__"
    ]
    assert len(cr_recs) == 3
    assert sum(r.errors for r in cr_recs) == 1
    assert any(r.error_type == "RuntimeError" for r in cr_recs)
    assert max(r.peak_concurrency for r in cr_recs) >= 2

    derive_recs = [
        r for r in profiler.records if r.function_name == "derive_from_rows_asyncio"
    ]
    assert len(derive_recs) == 2
    assert derive_recs[0].input_count == 4
    assert derive_recs[0].errors == 0
    assert derive_recs[0].peak_concurrency >= 2
    assert derive_recs[1].errors == 2

    sum_recs = [
        r for r in profiler.records if r.function_name == "summarize_descriptions"
    ]
    assert len(sum_recs) == 1
    assert sum_recs[0].input_count == 3
    assert sum_recs[0].batch_count == 2
    assert sum_recs[0].request_count == 2  # single-description node skipped LLM
    assert sum_recs[0].configured_concurrency == 2


async def test_output_equivalence_baseline_vs_instrumented(tmp_path: Path) -> None:
    """Verify GraphRAG outputs are identical with profiling disabled vs enabled."""
    mock_entity_response = (
        '("entity"<|>COMPANY_A<|>COMPANY<|>Company_A is a test company)\n'
        "##\n"
        '("entity"<|>COMPANY_B<|>COMPANY<|>Company_B owns Company_A)\n'
        "##\n"
        '("relationship"<|>COMPANY_A<|>COMPANY_B<|>Company_A is owned by Company_B<|>2)'
    )
    mock_report = CommunityReportResponse(
        title="Report Title",
        summary="Report Summary",
        rating=3.0,
        rating_explanation="Solid",
        findings=[FindingModel(summary="Finding 1", explanation="Explanation 1")],
    ).model_dump_json()

    async def _run_workflows(
        use_profiling: bool, out_dir: Path
    ) -> dict[str, pd.DataFrame]:
        ctx_graph = await create_test_context(storage=["text_units"])
        cfg_graph = get_default_graphrag_config()
        cfg_graph.async_mode = "asyncio"  # type: ignore[assignment]
        cfg_graph.completion_models["default_completion_model"].type = "mock"
        cfg_graph.completion_models["default_completion_model"].mock_responses = [
            mock_entity_response
        ]

        ctx_reports = await create_test_context(
            storage=["covariates", "relationships", "entities", "communities"]
        )
        cfg_reports = get_default_graphrag_config()
        cfg_reports.async_mode = "asyncio"  # type: ignore[assignment]
        cfg_reports.completion_models["default_completion_model"].type = "mock"
        cfg_reports.completion_models["default_completion_model"].mock_responses = [
            mock_report
        ]

        with profiling_session(enabled=use_profiling, output_dir=out_dir):
            await run_extract_graph_workflow(cfg_graph, ctx_graph)
            await run_community_reports_workflow(cfg_reports, ctx_reports)

        return {
            "entities": await ctx_graph.output_table_provider.read_dataframe(
                "entities"
            ),
            "relationships": await ctx_graph.output_table_provider.read_dataframe(
                "relationships"
            ),
            "community_reports": await ctx_reports.output_table_provider.read_dataframe(
                "community_reports"
            ),
        }

    baseline_outputs = await _run_workflows(False, tmp_path / "disabled")
    assert not (tmp_path / "disabled").exists()

    profiled_outputs = await _run_workflows(True, tmp_path / "enabled")
    assert (tmp_path / "enabled" / "function_timings.json").exists()

    for table_name in ("entities", "relationships", "community_reports"):
        base_df = baseline_outputs[table_name].drop(columns=["id"], errors="ignore")
        prof_df = profiled_outputs[table_name].drop(columns=["id"], errors="ignore")
        assert_frame_equal(base_df, prof_df)


async def test_run_index_end_to_end_with_mock_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify hpc_parallel.run_index runs standard GraphRAG indexing and preserves equivalence."""
    monkeypatch.setattr(
        "graphrag.index.workflows.create_base_text_units.get_tokenizer",
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

    def _build_cfg(out_root: Path) -> Any:
        cfg = get_default_graphrag_config()
        cfg.async_mode = "asyncio"  # type: ignore[assignment]
        cfg.output_storage.base_dir = str(out_root / "output")
        assert cfg.cache.storage is not None
        cfg.cache.storage.base_dir = str(out_root / "cache")
        cfg.reporting.base_dir = str(out_root / "logs")
        cfg.vector_store.db_uri = str(out_root / "output" / "lancedb")
        cfg.embed_text.names = []  # keep test focused on non-lancedb pipeline + run_embed_text tested separately
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
        report_model_cfg = cfg.completion_models["default_completion_model"].model_copy(
            update={"mock_responses": [mock_report]}
        )
        cfg.completion_models["report_mock_model"] = report_model_cfg
        cfg.community_reports.completion_model_id = "report_mock_model"
        return cfg

    input_docs = pd.DataFrame([
        {
            "id": "doc-1",
            "title": "Doc 1",
            "text": "Person One leads Org One.",
            "creation_date": "2026-01-01",
            "metadata": {"source": "unit-test"},
        }
    ])

    # Run via hpc_parallel with profiling disabled vs enabled
    res_unprofiled = await run_index(
        config=_build_cfg(tmp_path / "run_base"),
        hpc_config=HPCParallelConfig(
            execution_mode=ExecutionMode.BASELINE,
            profiling_enabled=False,
            profile_output_dir=tmp_path / "prof_off",
        ),
        skip_validation=True,
        input_documents=input_docs,
    )
    assert all(r.error is None for r in res_unprofiled)
    assert not (tmp_path / "prof_off").exists()

    res_profiled = await run_index(
        config=_build_cfg(tmp_path / "run_prof"),
        hpc_config=HPCParallelConfig(
            execution_mode=ExecutionMode.BASELINE,
            profiling_enabled=True,
            profile_output_dir=tmp_path / "prof_on",
        ),
        skip_validation=True,
        input_documents=input_docs,
    )
    assert all(r.error is None for r in res_profiled)
    assert (tmp_path / "prof_on" / "function_timings.json").exists()
    assert (tmp_path / "prof_on" / "function_timings.csv").exists()

    for table in (
        "documents",
        "text_units",
        "entities",
        "relationships",
        "community_reports",
    ):
        df_base = pd.read_parquet(tmp_path / "run_base" / "output" / f"{table}.parquet")
        df_prof = pd.read_parquet(tmp_path / "run_prof" / "output" / f"{table}.parquet")
        assert list(df_base.columns) == list(df_prof.columns)
        assert len(df_base) == len(df_prof)

    # Confirm workflow_name tagging worked in exported JSON
    exported = json.loads(
        (tmp_path / "prof_on" / "function_timings.json").read_text(encoding="utf-8")
    )
    funcs = {rec["function_name"] for rec in exported["records"]}
    assert "derive_from_rows_asyncio" in funcs
    assert "GraphExtractor._process_document" in funcs
    assert "summarize_descriptions" in funcs
    assert "CommunityReportsExtractor.__call__" in funcs
    assert any(rec["workflow_name"] == "extract_graph" for rec in exported["records"])
    assert not hpc_hooks.is_profiling_active()


async def test_env_var_profiling_auto_flushes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify GRAPHRAG_HPC_PROFILE=1 instruments standard GraphRAG API and flushes on exit."""
    from graphrag import api

    monkeypatch.setattr(
        "graphrag.index.workflows.create_base_text_units.get_tokenizer",
        lambda **_kwargs: _DummyTokenizer(),
    )

    monkeypatch.setenv("GRAPHRAG_HPC_PROFILE", "1")
    monkeypatch.setenv("GRAPHRAG_HPC_PROFILE_DIR", str(tmp_path / "env_prof"))
    monkeypatch.setenv("GRAPHRAG_HPC_PROFILE_FORMAT", "both")
    hpc_hooks.set_hook_handler(None)
    hpc_hooks._env_checked = False  # noqa: SLF001

    mock_entity_response = (
        '("entity"<|>ORG_ONE<|>ORGANIZATION<|>Org One is a research lab)\n'
        "##\n"
        '("entity"<|>PERSON_ONE<|>PERSON<|>Person One leads Org One)\n'
        "##\n"
        '("relationship"<|>PERSON_ONE<|>ORG_ONE<|>Person One leads Org One<|>1.0)'
    )
    cfg = get_default_graphrag_config()
    cfg.async_mode = "asyncio"  # type: ignore[assignment]
    cfg.output_storage.base_dir = str(tmp_path / "output")
    assert cfg.cache.storage is not None
    cfg.cache.storage.base_dir = str(tmp_path / "cache")
    cfg.reporting.base_dir = str(tmp_path / "logs")
    cfg.vector_store.db_uri = str(tmp_path / "output" / "lancedb")
    cfg.workflows = ["create_base_text_units", "extract_graph"]
    cfg.completion_models["default_completion_model"].type = "mock"
    cfg.completion_models["default_completion_model"].mock_responses = [
        mock_entity_response
    ]

    input_docs = pd.DataFrame([
        {
            "id": "doc-1",
            "title": "Doc 1",
            "text": "Person One leads Org One.",
            "creation_date": "2026-01-01",
            "metadata": {"source": "env-test"},
        }
    ])

    outputs = await api.build_index(
        config=cfg,
        input_documents=input_docs,
    )
    assert all(r.error is None for r in outputs)
    assert (tmp_path / "env_prof" / "function_timings.csv").exists()
    assert (tmp_path / "env_prof" / "function_timings.json").exists()
    assert hpc_hooks.get_hook_handler() is None
