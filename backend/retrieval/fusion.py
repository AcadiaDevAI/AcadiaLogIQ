"""
Weighted Result Fusion — merges candidates from vector, BM25, and keyword channels.
Uses Reciprocal Rank Fusion (RRF) with per-channel weights that adapt
based on the query classifier's strategy recommendation.
Only deduplicates by chunk ID so no results are lost.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings
from backend.retrieval.query_classifier import QueryIntent

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Standard result shape used throughout Phase 3
# ---------------------------------------------------------------------------
# Each result is a tuple: (chunk_id, text, metadata_dict, fusion_score)
FusedResult = Tuple[str, str, Dict[str, Any], float]


# ---------------------------------------------------------------------------
# Weight adjustment based on query strategy
# ---------------------------------------------------------------------------
def _adjusted_weights(intent: QueryIntent) -> Tuple[float, float, float]:
    """
    Shift channel weights based on the classifier's strategy.

    - keyword strategy  → boost keyword & BM25, reduce vector
    - semantic strategy → boost vector, reduce keyword
    - mixed strategy    → use config defaults as-is

    Weights are always normalized to sum to 1.0.
    """
    v_w = settings.VECTOR_WEIGHT
    b_w = settings.BM25_WEIGHT
    k_w = settings.KEYWORD_WEIGHT

    if intent.strategy == "keyword":
        # Shift weight from vector to keyword + BM25
        shift = 0.15
        v_w = max(0.10, v_w - shift)
        k_w += shift * 0.6
        b_w += shift * 0.4
    elif intent.strategy == "semantic":
        # Shift weight from keyword to vector
        shift = 0.10
        k_w = max(0.05, k_w - shift)
        v_w += shift

    # Normalize so they sum to 1.0
    total = v_w + b_w + k_w
    return v_w / total, b_w / total, k_w / total


# ---------------------------------------------------------------------------
# RRF fusion
# ---------------------------------------------------------------------------
def fuse_results(
    *,
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Tuple[str, str, Dict, float]],
    keyword_results: List[Dict[str, Any]],
    metadata_results: Optional[List[Dict[str, Any]]] = None,
    intent: QueryIntent,
    max_results: int = 30,
) -> List[FusedResult]:
    """
    Merge results from all search channels using Reciprocal Rank Fusion.

    How it works:
    1. Assign each channel's results a rank (1-based, by their native score)
    2. Compute RRF score per chunk: weight / (RRF_K + rank) for each channel
    3. Sum RRF scores across channels for chunks that appear in multiple
    4. Sort by total fused score descending
    5. Return deduplicated results up to max_results

    Exact-match terms (from keyword channel) get a configurable boost.
    """
    rrf_k = settings.RRF_K
    v_weight, b_weight, k_weight = _adjusted_weights(intent)

    # Accumulators: chunk_id → {score, text, metadata, channels}
    scores: Dict[str, float] = {}
    data: Dict[str, Dict[str, Any]] = {}  # chunk_id → {text, metadata}
    channels_hit: Dict[str, int] = {}     # chunk_id → count of channels

    # --- Vector channel (results from pgvector_search) ---
    for rank_idx, hit in enumerate(vector_results, start=1):
        cid = hit["id"]
        rrf_score = v_weight / (rrf_k + rank_idx)
        scores[cid] = scores.get(cid, 0.0) + rrf_score
        channels_hit[cid] = channels_hit.get(cid, 0) + 1
        if cid not in data:
            data[cid] = {"text": hit["text"], "metadata": hit["metadata"]}

    # --- BM25 channel (results from in-memory BM25 index) ---
    for rank_idx, (cid, text_val, meta, _bm25_score) in enumerate(bm25_results, start=1):
        rrf_score = b_weight / (rrf_k + rank_idx)
        scores[cid] = scores.get(cid, 0.0) + rrf_score
        channels_hit[cid] = channels_hit.get(cid, 0) + 1
        if cid not in data:
            data[cid] = {"text": text_val, "metadata": meta}

    # --- Keyword / full-text channel ---
    for rank_idx, hit in enumerate(keyword_results, start=1):
        cid = hit["id"]
        rrf_score = k_weight / (rrf_k + rank_idx)

        # Boost exact-match results (they found the literal term in the text)
        if hit.get("search_type") == "ilike_fallback":
            rrf_score *= settings.EXACT_TERM_BOOST

        scores[cid] = scores.get(cid, 0.0) + rrf_score
        channels_hit[cid] = channels_hit.get(cid, 0) + 1
        if cid not in data:
            data[cid] = {"text": hit["text"], "metadata": hit["metadata"]}

    # --- Metadata filter channel (optional, small boost for metadata matches) ---
    if metadata_results:
        meta_weight = 0.05  # small supplemental weight
        for rank_idx, hit in enumerate(metadata_results, start=1):
            cid = hit["id"]
            rrf_score = meta_weight / (rrf_k + rank_idx)
            scores[cid] = scores.get(cid, 0.0) + rrf_score
            channels_hit[cid] = channels_hit.get(cid, 0) + 1
            if cid not in data:
                data[cid] = {"text": hit["text"], "metadata": hit["metadata"]}

    # --- Multi-channel bonus ---
    # Chunks found by 2+ channels get a 15% boost (agreement signal)
    for cid, count in channels_hit.items():
        if count >= 2:
            scores[cid] *= 1.15
        if count >= 3:
            scores[cid] *= 1.10  # additional boost for 3+ channel agreement

    # --- Sort and return ---
    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)

    results: List[FusedResult] = []
    for cid in sorted_ids[:max_results]:
        d = data[cid]
        results.append((cid, d["text"], d["metadata"], scores[cid]))

    logger.debug(
        "Fusion complete: vector=%d bm25=%d keyword=%d metadata=%d → fused=%d "
        "(weights: v=%.2f b=%.2f k=%.2f, strategy=%s)",
        len(vector_results), len(bm25_results), len(keyword_results),
        len(metadata_results or []), len(results),
        v_weight, b_weight, k_weight, intent.strategy,
    )

    return results
