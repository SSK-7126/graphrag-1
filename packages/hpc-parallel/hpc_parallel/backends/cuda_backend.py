# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CUDA local compute backend with explicit blocks and threads-per-block configuration.

Remote API-based LLM/embedding calls are network-bound and remain on
``asyncio_optimized``; CUDA is applied only to local tensor/vector
operations when Numba CUDA or PyTorch + a CUDA GPU are available and
payload size justifies host-to-device transfer.
"""

from __future__ import annotations

import importlib
import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hpc_parallel.config import HPCParallelConfig

logger = logging.getLogger(__name__)

DEFAULT_CUDA_THREADS_PER_BLOCK = 256
MAX_CUDA_THREADS_PER_BLOCK = 1024


@dataclass(frozen=True)
class CUDADeviceInfo:
    """Result of probing local CUDA availability."""

    available: bool
    device_name: str | None = None
    backend: str | None = None
    reason: str | None = None


def compute_cuda_launch_config(
    num_elements: int,
    threads_per_block: int = DEFAULT_CUDA_THREADS_PER_BLOCK,
    blocks_per_grid: int | None = None,
) -> tuple[int, int]:
    """Compute CUDA ``(blocks_per_grid, threads_per_block)`` launch geometry."""
    if threads_per_block <= 0 or threads_per_block > MAX_CUDA_THREADS_PER_BLOCK:
        msg = (
            f"threads_per_block must be in [1, {MAX_CUDA_THREADS_PER_BLOCK}], "
            f"got {threads_per_block}."
        )
        raise ValueError(msg)
    if blocks_per_grid is not None:
        if blocks_per_grid <= 0:
            msg = f"blocks_per_grid must be positive, got {blocks_per_grid}."
            raise ValueError(msg)
        return blocks_per_grid, threads_per_block
    computed_blocks = max(1, math.ceil(max(1, num_elements) / threads_per_block))
    return computed_blocks, threads_per_block


def detect_cuda_device() -> CUDADeviceInfo:
    """Probe whether Numba CUDA or PyTorch with an active CUDA GPU is available."""
    try:
        numba_cuda: Any = importlib.import_module("numba.cuda")
        if numba_cuda.is_available():
            dev = numba_cuda.get_current_device()
            name = getattr(dev, "name", b"cuda:0")
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="replace")
            return CUDADeviceInfo(
                available=True, device_name=str(name), backend="numba"
            )
    except ImportError:
        logger.debug("numba.cuda not installed; checking PyTorch CUDA.")
    except Exception as exc:  # noqa: BLE001
        logger.debug("numba.cuda probe failed: %s", type(exc).__name__)

    try:
        torch: Any = importlib.import_module("torch")
    except ImportError:
        return CUDADeviceInfo(
            available=False,
            reason="Neither Numba CUDA nor PyTorch ('torch') is installed.",
        )

    cuda_mod: Any = getattr(torch, "cuda", None)
    if cuda_mod is None or not cuda_mod.is_available():
        return CUDADeviceInfo(
            available=False,
            reason="PyTorch is installed, but no CUDA-capable GPU is available.",
        )

    try:
        device_name = str(cuda_mod.get_device_name(0))
    except Exception:  # noqa: BLE001
        device_name = "cuda:0"
    return CUDADeviceInfo(available=True, device_name=device_name, backend="torch")


def ensure_cuda_ready(hpc_config: HPCParallelConfig) -> CUDADeviceInfo:
    """Validate CUDA availability or fall back to CPU if configured."""
    info = detect_cuda_device()
    if info.available:
        logger.info(
            "CUDA mode active on %s (backend=%s, threads_per_block=%d, blocks_per_grid=%s).",
            info.device_name,
            info.backend,
            hpc_config.cuda_threads_per_block,
            hpc_config.cuda_blocks_per_grid or "auto",
        )
        return info

    if hpc_config.cuda_fallback_to_cpu:
        logger.warning(
            "CUDA unavailable (%s); falling back to CPU asyncio_optimized.",
            info.reason,
        )
        return info

    msg = (
        f"Execution mode 'cuda' cannot start: {info.reason} "
        "Install Numba/PyTorch with CUDA support or set cuda_fallback_to_cpu=True."
    )
    raise RuntimeError(msg)


def _reconstitute_with_numba_kernel(
    raw_embeddings: list[list[float]],
    sizes: list[int],
    *,
    blocks_per_grid: int,
    threads_per_block: int,
) -> list[list[float] | None]:
    """Execute explicit Numba CUDA kernel with ``[blocks_per_grid, threads_per_block]``."""
    numba_cuda: Any = importlib.import_module("numba.cuda")
    raw_arr = np.asarray(raw_embeddings, dtype=np.float64)
    dim = raw_arr.shape[1]

    @numba_cuda.jit
    def _avg_rows_kernel(
        src: Any,
        start_idx: int,
        count: int,
        out_vec: Any,
    ) -> None:
        col = numba_cuda.grid(1)
        if col < out_vec.shape[0]:
            total = 0.0
            for r in range(start_idx, start_idx + count):
                total += src[r, col]
            out_vec[col] = total / count

    d_src = numba_cuda.to_device(raw_arr)
    d_out = numba_cuda.device_array(dim, dtype=np.float64)

    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
        elif size == 1:
            embeddings.append(raw_embeddings[cursor])
            cursor += 1
        else:
            _avg_rows_kernel[blocks_per_grid, threads_per_block](
                d_src, cursor, size, d_out
            )
            avg_host = d_out.copy_to_host()
            norm = float(np.linalg.norm(avg_host))
            embeddings.append((avg_host / norm).tolist())
            cursor += size
    return embeddings


def reconstitute_embeddings_cuda(
    raw_embeddings: list[list[float]],
    sizes: list[int],
    *,
    min_elements: int = 4096,
    threads_per_block: int = DEFAULT_CUDA_THREADS_PER_BLOCK,
    blocks_per_grid: int | None = None,
) -> list[list[float] | None]:
    """Reconstitute multi-snippet embeddings using CUDA when beneficial, else NumPy.

    When Numba CUDA is available, launches the averaging kernel explicitly with
    ``[blocks_per_grid, threads_per_block]`` (default ``threads_per_block=256``,
    ``blocks_per_grid=ceil(dim / threads_per_block)``).
    """
    if not raw_embeddings:
        return [None if s == 0 else [] for s in sizes]

    dim = len(raw_embeddings[0]) if raw_embeddings else 0
    total_elements = len(raw_embeddings) * dim
    blocks, threads = compute_cuda_launch_config(
        dim, threads_per_block=threads_per_block, blocks_per_grid=blocks_per_grid
    )
    info = detect_cuda_device()

    if not info.available or total_elements < min_elements:
        return _reconstitute_cpu(raw_embeddings, sizes)

    if info.backend == "numba":
        return _reconstitute_with_numba_kernel(
            raw_embeddings,
            sizes,
            blocks_per_grid=blocks,
            threads_per_block=threads,
        )

    torch: Any = importlib.import_module("torch")
    tensor = torch.tensor(raw_embeddings, dtype=torch.float64, device="cuda")

    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
        elif size == 1:
            embeddings.append(raw_embeddings[cursor])
            cursor += 1
        else:
            chunk = tensor[cursor : cursor + size]
            avg = chunk.mean(dim=0)
            norm = torch.linalg.norm(avg)
            normalized = (avg / norm).detach().cpu().tolist()
            embeddings.append(normalized)
            cursor += size
    return embeddings


def _reconstitute_cpu(
    raw_embeddings: list[list[float]],
    sizes: list[int],
) -> list[list[float] | None]:
    embeddings: list[list[float] | None] = []
    cursor = 0
    for size in sizes:
        if size == 0:
            embeddings.append(None)
        elif size == 1:
            embeddings.append(raw_embeddings[cursor])
            cursor += 1
        else:
            chunk = raw_embeddings[cursor : cursor + size]
            average = np.average(chunk, axis=0)
            normalized = average / np.linalg.norm(average)
            embeddings.append(normalized.tolist())
            cursor += size
    return embeddings
