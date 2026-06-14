"""
Chunk Limiter — caps the ranked chunk list passed to the LLM / agents.
Preserves the original ordering (rerank score) and returns the slice
plus counts, so callers can log both original and limited sizes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

logger = logging.getLogger("acadia-log-iq")


# Default cap — conservative. Keeps token usage predictable without
# starving the LLM of context. Override via limit_chunks(..., max_chunks=N).
DEFAULT_MAX_CHUNKS: int = 8


@dataclass
class ChunkLimitResult:
    limited: List[Tuple[str, str, Dict[str, Any], float]]
    original_count: int = 0
    limited_count: int = 0
    applied: bool = False
    max_chunks: int = DEFAULT_MAX_CHUNKS


def limit_chunks(
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    *,
    max_chunks: int = DEFAULT_MAX_CHUNKS,
) -> ChunkLimitResult:
    """
    Truncate ranked_chunks to the top `max_chunks` while preserving order.

    Safe on empty or None input; never raises. If the list is already
    within the limit, `applied` is False and the original list is returned.
    """
    chunks = list(ranked_chunks or [])
    original_count = len(chunks)
    cap = max(1, int(max_chunks) if max_chunks else DEFAULT_MAX_CHUNKS)

    if original_count <= cap:
        return ChunkLimitResult(
            limited=chunks,
            original_count=original_count,
            limited_count=original_count,
            applied=False,
            max_chunks=cap,
        )

    limited = chunks[:cap]
    logger.info(
        "Chunk limiter applied: %d -> %d (cap=%d)",
        original_count, len(limited), cap,
    )
    return ChunkLimitResult(
        limited=limited,
        original_count=original_count,
        limited_count=len(limited),
        applied=True,
        max_chunks=cap,
    )
