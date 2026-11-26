# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Zero-overhead profiling hook interface for optional hpc_parallel instrumentation."""

from __future__ import annotations

import os
import threading
from contextlib import AbstractContextManager, nullcontext
from typing import TYPE_CHECKING, Protocol

from typing_extensions import Self

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType


class SubtaskHandle(Protocol):
    """Protocol for an active subtask concurrency context."""

    def __enter__(self) -> Self:
        """Enter subtask context."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit subtask context."""
        ...


class SpanHandle(Protocol):
    """Protocol for a function timing span."""

    def __enter__(self) -> Self:
        """Start span."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """End span."""
        ...

    def set_counts(
        self,
        *,
        input_count: int | None = None,
        request_count: int | None = None,
        batch_count: int | None = None,
        errors: int | None = None,
        configured_concurrency: int | None = None,
        output_count: int | None = None,
        worker_count: int | None = None,
        scheduling_overhead_seconds: float | None = None,
        retries: int | None = None,
        platform: str | None = None,
    ) -> None:
        """Update numeric counters on the span."""
        ...

    def increment_requests(self, count: int = 1) -> None:
        """Increment request counter."""
        ...

    def increment_errors(
        self, count: int = 1, *, error_type: str | None = None
    ) -> None:
        """Increment error counter."""
        ...

    def subtask(self) -> SubtaskHandle:
        """Create a subtask concurrency tracker."""
        ...


class HPCHookHandler(Protocol):
    """Protocol implemented by hpc_parallel's profiling adapter."""

    def start_span(
        self,
        function_name: str,
        *,
        input_count: int = 0,
        request_count: int = 0,
        batch_count: int = 0,
        errors: int = 0,
        configured_concurrency: int | None = None,
        workflow_name: str | None = None,
    ) -> SpanHandle:
        """Create and return a timing span."""
        ...

    def workflow_context(self, workflow_name: str) -> AbstractContextManager[None]:
        """Return context manager tagging nested spans with workflow_name."""
        ...

    def record_active_request(self, count: int = 1) -> None:
        """Increment request count on the current active span."""
        ...

    def flush(self) -> list[Path]:
        """Export recorded profiling artifacts."""
        ...


class _NoopSubtask:
    """No-op subtask context manager used when profiling is disabled."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None


_NOOP_SUBTASK = _NoopSubtask()


class _NoopSpan:
    """No-op span used when profiling is disabled."""

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        return None

    def set_counts(
        self,
        *,
        input_count: int | None = None,
        request_count: int | None = None,
        batch_count: int | None = None,
        errors: int | None = None,
        configured_concurrency: int | None = None,
        output_count: int | None = None,
        worker_count: int | None = None,
        scheduling_overhead_seconds: float | None = None,
        retries: int | None = None,
        platform: str | None = None,
    ) -> None:
        return None

    def increment_requests(self, count: int = 1) -> None:
        return None

    def increment_errors(
        self, count: int = 1, *, error_type: str | None = None
    ) -> None:
        return None

    def subtask(self) -> _NoopSubtask:
        return _NOOP_SUBTASK


_NOOP_SPAN = _NoopSpan()

_hook_lock = threading.RLock()
_hook_handler: HPCHookHandler | None = None
_env_checked = False
_env_auto_started = False


def set_hook_handler(handler: HPCHookHandler | None) -> None:
    """Register or clear the active HPC profiling hook handler."""
    global _hook_handler, _env_auto_started
    with _hook_lock:
        _hook_handler = handler
        if handler is None:
            _env_auto_started = False


def get_hook_handler() -> HPCHookHandler | None:
    """Return the currently registered hook handler, if any."""
    with _hook_lock:
        return _hook_handler


def _maybe_init_from_env() -> None:
    global _env_checked, _env_auto_started
    env_flag = os.environ.get("GRAPHRAG_HPC_PROFILE", "").strip().lower()
    if env_flag not in {"1", "true", "yes", "on"}:
        _env_checked = True
        return

    with _hook_lock:
        if _hook_handler is not None or _env_checked:
            return
        _env_checked = True
        try:
            from hpc_parallel.hooks import enable_from_env

            profiler = enable_from_env()
            if profiler is not None:
                _env_auto_started = True
        except ImportError:
            return


def is_profiling_active() -> bool:
    """Return True when HPC profiling is currently enabled."""
    if _hook_handler is not None:
        return True
    if not _env_checked:
        _maybe_init_from_env()
    return _hook_handler is not None


def profile_function(
    function_name: str,
    *,
    input_count: int = 0,
    request_count: int = 0,
    batch_count: int = 0,
    errors: int = 0,
    configured_concurrency: int | None = None,
    workflow_name: str | None = None,
) -> SpanHandle:
    """Create a timing/concurrency span if profiling is active, else return no-op."""
    if not is_profiling_active():
        return _NOOP_SPAN
    handler = _hook_handler
    if handler is None:
        return _NOOP_SPAN
    return handler.start_span(
        function_name,
        input_count=input_count,
        request_count=request_count,
        batch_count=batch_count,
        errors=errors,
        configured_concurrency=configured_concurrency,
        workflow_name=workflow_name,
    )


def workflow_context(workflow_name: str) -> AbstractContextManager[None]:
    """Tag nested spans with workflow_name when profiling is active."""
    if not is_profiling_active():
        return nullcontext()
    handler = _hook_handler
    if handler is None:
        return nullcontext()
    return handler.workflow_context(workflow_name)


def record_active_request(count: int = 1) -> None:
    """Increment request_count on the active span when profiling is active."""
    handler = _hook_handler
    if handler is not None:
        handler.record_active_request(count)


def flush_env_profiling() -> None:
    """Flush profiling outputs when profiling was auto-initialized from env vars."""
    global _env_auto_started, _env_checked, _hook_handler
    with _hook_lock:
        if not _env_auto_started or _hook_handler is None:
            return
        handler = _hook_handler
        _hook_handler = None
        _env_auto_started = False
        _env_checked = False
    handler.flush()
