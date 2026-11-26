# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""CUDA local compute backend for Phase 4.

Remote API-based LLM/embedding calls are network-bound and remain on
``asyncio_optimized``; CUDA is applied only to local tensor/vector
operations when PyTorch + a CUDA GPU are available and payload size
justifies host-to-device transfer.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from hpc_parallel.config import HPCParallelConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CUDADeviceInfo:
    """Result of probing local CUDA availability."""

    available: bool
    device_name: str | None = None
    reason: str | None = None


def detect_cuda_device() -> CUDADeviceInfo:
    """Probe whether PyTorch with an active CUDA GPU is available."""
    try:
        torch = importlib.import_module("torch")
    except ImportError:
        return CUDADeviceInfo(
            available=False,
            reason="PyTorch ('torch') is not installed.",
        )

    cuda_mod = getattr(torch, "cuda", None)
    if cuda_mod is None or not cuda_mod.is_available():
        return CUDADeviceInfo(
            available=False,
            reason="PyTorch is installed, but no CUDA-capable GPU is available.",
        )

    device_name = None
    try:
        device_name = str(cuda_mod.get_device_name(0))
    except Exception:  # noqa: BLE001
        device_name = "cuda:0"
    return CUDADeviceInfo(available=True, device_name=device_name)


def ensure_cuda_ready(hpc_config: HPCParallelConfig) -> CUDADeviceInfo:
    """Validate CUDA availability or fall back to CPU if configured."""
    info = detect_cuda_device()
    if info.available:
        logger.info(
            "CUDA mode active on %s (remote LLM API calls remain on asyncio_optimized).",
            info.device_name,
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
        "Install PyTorch with CUDA support or set cuda_fallback_to_cpu=True."
    )
    raise RuntimeError(msg)


def reconstitute_embeddings_cuda(
    raw_embeddings: list[list[float]],
    sizes: list[int],
    *,
    min_elements: int = 4096,
) -> list[list[float] | None]:
    """Reconstitute multi-snippet embeddings using CUDA when beneficial, else NumPy.

    Matches ``graphrag.index.operations.embed_text.run_embed_text._reconstitute_embeddings``
    numerically while offloading large multi-snippet L2 normalization batches to GPU.
    """
    if not raw_embeddings:
        return [None if s == 0 else [] for s in sizes]

    dim = len(raw_embeddings[0]) if raw_embeddings else 0
    total_elements = len(raw_embeddings) * dim
    info = detect_cuda_device()

    if not info.available or total_elements < min_elements:
        return _reconstitute_cpu(raw_embeddings, sizes)

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

