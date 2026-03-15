"""
Retrieval Orchestrator — single entry point for all Phase 3 retrieval.
Coordinates: query classification → parallel search channels → fusion → reranking.
Called by the /ask endpoint in api.py. Returns ranked chunks ready for context assembly.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.config import settings
from backend.retrieval.query_classifier import QueryIntent, classify_query
from backend.retrieval.keyword_search import fulltext_search, metadata_filter_search
from backend.retrieval.fusion import fuse_results, FusedResult
from backend.retrieval.reranker import BaseReranker, create_reranker

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Result container returned by the orchestrator
# ---------------------------------------------------------------------------
@dataclass
class RetrievalResult:
    """
    Complete retrieval result returned to the caller.

    Fields:
        ranked   — final reranked chunks: list of (chunk_id, text, metadata, score)
        intent   — query classification details
        stats    — timing and diagnostic counters
    """
    ranked: List[FusedResult] = field(default_factory=list)
    intent: Optional[QueryIntent] = None
    stats: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Module-level reranker instance (lazy-initialized)
# ---------------------------------------------------------------------------
_reranker: Optional[BaseReranker] = None


def _get_reranker(generate_fn: Callable) -> BaseReranker:
    """Lazy-init the reranker singleton so it's created once at first query."""
    global _reranker
    if _reranker is None:
        _reranker = create_reranker(generate_fn)
    return _reranker


# ---------------------------------------------------------------------------
# Main orchestrator function
# ---------------------------------------------------------------------------
def retrieve(
    *,
    query: str,
    query_embedding: List[float],
    owner_id: str,
    allowed_file_ids: Set[str],
    file_type: str = "kb",
    generate_fn: Callable = None,
    embed_fn: Callable = None,
    bm25_search_fn: Callable = None,
    vector_search_fn: Callable = None,
) -> RetrievalResult:
    """
    Main retrieval entry point. Runs the full Phase 3 pipeline:

    1. Classify the query (keyword / semantic / mixed)
    2. Run search channels in parallel:
       - Vector search (pgvector cosine similarity)
       - BM25 search (in-memory term frequency index)
       - Keyword search (PostgreSQL full-text + ILIKE fallback)
       - Metadata filter (optional, vendor/product/domain matching)
    3. Fuse results using strategy-aware Reciprocal Rank Fusion
    4. Rerank top candidates using the configured reranker
    5. Return final ranked results with diagnostics

    Args:
        query           — user's question text
        query_embedding — precomputed embedding vector for the query
        owner_id        — document owner for access filtering
        allowed_file_ids — set of active file IDs to search within
        file_type       — document type filter (default "kb")
        generate_fn     — LLM generation function for the reranker
        bm25_search_fn  — BM25 search callable (bm25.search)
        vector_search_fn — pgvector search callable (pgvector_search)

    Returns:
        RetrievalResult with ranked chunks, intent info, and timing stats
    """
    t_start = time.perf_counter()
    result = RetrievalResult()

    # =====================================================================
    # Step 1: Classify the query to determine search strategy
    # =====================================================================
    if settings.ENABLE_QUERY_CLASSIFICATION:
        intent = classify_query(query)
    else:
        intent = QueryIntent(strategy="mixed", reason="classification disabled")
    result.intent = intent

    t_classify = time.perf_counter()

    # =====================================================================
    # Step 2: Run search channels in parallel
    # =====================================================================
    # We use a ThreadPoolExecutor to run all channels concurrently.
    # Each channel returns its results independently.

    vector_results: List[Dict[str, Any]] = []
    bm25_results: List[Tuple[str, str, Dict, float]] = []
    keyword_results: List[Dict[str, Any]] = []
    metadata_results: List[Dict[str, Any]] = []

    allowed_ids_list = list(allowed_file_ids) if allowed_file_ids else None

    def _run_vector():
        """Channel 1: pgvector cosine similarity search."""
        if vector_search_fn is None:
            return []
        try:
            hits = vector_search_fn(
                query_embedding=query_embedding,
                n_results=settings.VECTOR_CANDIDATES,
                allowed_file_ids=allowed_ids_list,
            )
            # Filter by owner and allowed files
            filtered = []
            for hit in hits:
                meta = hit.get("metadata", {})
                if meta.get("owner_id", "anonymous") != owner_id:
                    continue
                if allowed_file_ids and meta.get("file_id") not in allowed_file_ids:
                    continue
                filtered.append(hit)
            return filtered
        except Exception as exc:
            logger.warning("Vector search channel failed: %s", exc)
            return []

    def _run_bm25():
        """Channel 2: in-memory BM25 term frequency search."""
        if bm25_search_fn is None:
            return []
        try:
            raw_hits = bm25_search_fn(
                query,
                n_results=settings.BM25_CANDIDATES,
                file_type=file_type,
            )
            # Filter by owner and allowed files
            filtered = []
            for doc_id, text_val, meta, score in raw_hits:
                if meta.get("owner_id", "anonymous") != owner_id:
                    continue
                if allowed_file_ids and meta.get("file_id") not in allowed_file_ids:
                    continue
                filtered.append((doc_id, text_val, meta, score))
            return filtered
        except Exception as exc:
            logger.warning("BM25 search channel failed: %s", exc)
            return []

    def _run_keyword():
        """Channel 3: PostgreSQL full-text search + ILIKE fallback."""
        try:
            return fulltext_search(
                terms=intent.extracted_terms,
                n_results=settings.KEYWORD_CANDIDATES,
                allowed_file_ids=allowed_file_ids,
                owner_id=owner_id,
            )
        except Exception as exc:
            logger.warning("Keyword search channel failed: %s", exc)
            return []

    def _run_metadata():
        """Channel 4: metadata filter search (vendor/product/domain)."""
        if not settings.ENABLE_METADATA_FILTER or not intent.metadata_hints:
            return []
        try:
            return metadata_filter_search(
                metadata_hints=intent.metadata_hints,
                n_results=settings.METADATA_FILTER_CANDIDATES,
                allowed_file_ids=allowed_file_ids,
                owner_id=owner_id,
            )
        except Exception as exc:
            logger.warning("Metadata filter channel failed: %s", exc)
            return []

    # --- Execute channels concurrently ---
    # Strategy-aware: skip channels that won't contribute much
    tasks = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Always run vector (it's the backbone)
        if intent.strategy != "keyword":
            tasks["vector"] = executor.submit(_run_vector)
        else:
            # Even for keyword queries, run vector with reduced priority
            tasks["vector"] = executor.submit(_run_vector)

        # Always run BM25 (fast, in-memory)
        tasks["bm25"] = executor.submit(_run_bm25)

        # Always run keyword search (Phase 3 addition)
        tasks["keyword"] = executor.submit(_run_keyword)

        # Run metadata filter if we have hints
        if intent.metadata_hints:
            tasks["metadata"] = executor.submit(_run_metadata)

        # Collect results
        for name, future in tasks.items():
            try:
                res = future.result(timeout=15)  # 15s timeout per channel
                if name == "vector":
                    vector_results = res
                elif name == "bm25":
                    bm25_results = res
                elif name == "keyword":
                    keyword_results = res
                elif name == "metadata":
                    metadata_results = res
            except Exception as exc:
                logger.warning("Search channel '%s' timed out or failed: %s", name, exc)

    t_search = time.perf_counter()

    # =====================================================================
    # Step 3: Fuse results from all channels
    # =====================================================================
    fused = fuse_results(
        vector_results=vector_results,
        bm25_results=bm25_results,
        keyword_results=keyword_results,
        metadata_results=metadata_results if metadata_results else None,
        intent=intent,
        max_results=settings.RERANK_CANDIDATES,
    )

    t_fuse = time.perf_counter()

    # =====================================================================
    # Step 4: Rerank the fused candidates
    # =====================================================================
    if fused and generate_fn:
        reranker = _get_reranker(generate_fn)
        ranked = reranker.rerank(query, fused, top_k=settings.RERANK_TOP_K)
    else:
        ranked = fused[: settings.RERANK_TOP_K]

    t_rerank = time.perf_counter()

    result.ranked = ranked

    # =====================================================================
    # Step 5: Collect diagnostics
    # =====================================================================
    result.stats = {
        "search_mode": f"hybrid_phase3 (strategy={intent.strategy})",
        "query_strategy": intent.strategy,
        "query_keyword_score": round(intent.keyword_score, 3),
        "query_semantic_score": round(intent.semantic_score, 3),
        "vector_candidates": len(vector_results),
        "bm25_candidates": len(bm25_results),
        "keyword_candidates": len(keyword_results),
        "metadata_candidates": len(metadata_results),
        "fused_total": len(fused),
        "reranked_total": len(ranked),
        "timing_classify_ms": int((t_classify - t_start) * 1000),
        "timing_search_ms": int((t_search - t_classify) * 1000),
        "timing_fuse_ms": int((t_fuse - t_search) * 1000),
        "timing_rerank_ms": int((t_rerank - t_fuse) * 1000),
        "timing_total_ms": int((t_rerank - t_start) * 1000),
    }

    logger.info(
        "Retrieval complete: strategy=%s vector=%d bm25=%d kw=%d meta=%d → fused=%d → reranked=%d (%dms)",
        intent.strategy,
        len(vector_results), len(bm25_results),
        len(keyword_results), len(metadata_results),
        len(fused), len(ranked),
        result.stats["timing_total_ms"],
    )

    return result
