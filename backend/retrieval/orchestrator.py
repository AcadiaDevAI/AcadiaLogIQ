"""
Retrieval Orchestrator — single entry point for all Phase 3 retrieval.
Coordinates: query classification → parallel search channels → fusion → reranking.
Called by the /ask endpoint in api.py. Returns ranked chunks ready for context assembly.
"""

from __future__ import annotations

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from backend.config import settings
from backend.retrieval.query_classifier import QueryIntent, classify_query
from backend.retrieval.keyword_search import (
    fulltext_search,
    metadata_filter_search,
    identifier_exact_search,
    ticket_id_exact_search,  # legacy alias — kept for import stability
)
from backend.retrieval.fusion import fuse_results, FusedResult
from backend.retrieval.reranker import BaseReranker, create_reranker

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Goal 2 — Schema-agnostic identifier extractor
# ---------------------------------------------------------------------------
# Regexes and canonical formats live in settings.IDENTIFIER_PATTERNS /
# IDENTIFIER_CANONICAL_FORMAT so new schemas can be added with no code
# changes. Patterns are evaluated in declaration order; earlier matches
# own their span so 'INC-10005' cannot be re-tagged as an issue_key.


def _extract_identifiers(query: str) -> List[Tuple[str, str]]:
    """
    Extract all (canonical_id, id_type) pairs from `query`.

    Returns a list of (canonical_id, id_type) tuples where id_type matches
    the keys in settings.IDENTIFIER_PATTERNS (and by contract matches
    chunks.metadata_json->>'id_type' written by the ingestion schema).

    Handles hyphen-stripped queries (query_expansion normalizes 'INC-10001'
    → 'INC 10001') via the pattern's optional [-\\s]? separator and always
    returns the canonical hyphenated form for SQL comparison.
    """
    if not query:
        return []
    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    covered: List[Tuple[int, int]] = []

    for id_type, pattern in settings.IDENTIFIER_PATTERNS.items():
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error as exc:
            logger.warning(
                "[retrieve] IDENTIFIER_PATTERNS[%s] invalid regex (%s) — skipping",
                id_type, exc,
            )
            continue
        fmt = settings.IDENTIFIER_CANONICAL_FORMAT.get(id_type, "{0}")
        for m in regex.finditer(query):
            span = m.span()
            # Skip if this span overlaps a previously matched identifier.
            if any(s <= span[0] < e or s < span[1] <= e for s, e in covered):
                continue
            raw = m.group(1) if m.groups() else m.group(0)
            try:
                canonical = fmt.format(raw)
            except Exception:
                canonical = raw
            key = (canonical.upper(), id_type)
            if key in seen:
                continue
            seen.add(key)
            out.append((canonical, id_type))
            covered.append(span)
    return out


def _extract_ticket_ids(query: str) -> List[str]:
    """Back-compat shim: return only ticket_number canonical IDs."""
    return [cid for cid, itype in _extract_identifiers(query) if itype == "ticket_number"]


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

    logger.info("[retrieve] entry — query=%r", query[:200])

    # =====================================================================
    # Step 0 (Goal 2): Identifier exact-match short-circuit
    # =====================================================================
    # When the query names one or more identifiers matching any pattern in
    # settings.IDENTIFIER_PATTERNS, answer from an exact JSONB lookup on
    # primary_id and skip the four parallel channels entirely. Deterministic,
    # ~1ms, schema-agnostic (tickets, Jira issue keys, KB articles, ...).
    requested_identifiers = _extract_identifiers(query)
    logger.info("[retrieve] extracted identifiers=%s", requested_identifiers)
    # Legacy alias for any downstream reader still expecting ticket IDs.
    requested_ticket_ids = [
        cid for cid, itype in requested_identifiers if itype == "ticket_number"
    ]
    if requested_identifiers:
        logger.info("[retrieve] short-circuit firing via identifier_exact_search")
        try:
            exact_rows = identifier_exact_search(
                identifiers=requested_identifiers,
                allowed_file_ids=allowed_file_ids,
                n_results=max(settings.RERANK_TOP_K, 20),
            )
        except Exception as exc:
            # Fail safe: if the exact lookup crashes we fall through to
            # the generic pipeline rather than surfacing an error.
            logger.warning("Identifier exact lookup raised (%s) — falling through", exc)
            exact_rows = None

        if exact_rows is not None:
            # Diagnostics for the identifier-exact short-circuit. Exposes
            # whether the rendered chunk actually contains the expected
            # labeled sections — if it doesn't, the schema's ingest() has a
            # shape-specific bug we need to chase.
            top_meta = (exact_rows[0].get("metadata", {}) or {}) if exact_rows else {}
            top_primary = (
                (top_meta.get("metadata_json") or {}).get("primary_id")
                or top_meta.get("incident_number")
            ) if exact_rows else None
            logger.info(
                "[identifier_exact] ids=%s rows_returned=%d top_chunk_primary_id=%s",
                requested_identifiers, len(exact_rows), top_primary,
            )
            if exact_rows:
                top_content = (
                    exact_rows[0].get("text")
                    or exact_rows[0].get("content")
                    or exact_rows[0].get("contextualized_content")
                    or ""
                )
                expected_labels = [
                    "ROOT CAUSE:",
                    "RESOLUTION DETAIL:",
                    "ITIL 5-WHY ROOT CAUSE:",
                    "SOP EXECUTION STEPS:",
                    "QA AUDITOR GAPS:",
                ]
                sections_present = [lbl for lbl in expected_labels if lbl in top_content]
                logger.info(
                    "[identifier_exact] top_chunk_length=%d sections_present=%s",
                    len(top_content), sections_present,
                )

            if exact_rows:
                ranked: List[FusedResult] = []
                for row in exact_rows:
                    meta = row.get("metadata", {}) or {}
                    ranked.append(
                        (
                            row["id"],
                            row["text"],
                            meta,
                            float(row.get("rank") or 1.0),
                        )
                    )
                result.intent = QueryIntent(
                    strategy="keyword",
                    reason=f"identifier exact match: {requested_identifiers}",
                )
                result.ranked = ranked[: settings.RERANK_TOP_K]
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)
                result.stats = {
                    "search_mode": "identifier_exact",
                    "requested_identifiers": requested_identifiers,
                    # Legacy alias — consumers that still key off
                    # requested_ticket_ids / search_mode=="ticket_id_exact"
                    # keep working.
                    "requested_ticket_ids": requested_ticket_ids,
                    "matched_count": len(ranked),
                    "timing_total_ms": elapsed_ms,
                }
                logger.info(
                    "Retrieval short-circuit: identifier_exact for %s → %d chunks (%dms)",
                    requested_identifiers, len(ranked), elapsed_ms,
                )
                return result
            else:
                # Zero rows = identifier(s) genuinely not indexed. Do NOT
                # fall through — return a clean signal so the caller can
                # say "not found" rather than hallucinating.
                result.intent = QueryIntent(
                    strategy="keyword",
                    reason=f"identifier not found: {requested_identifiers}",
                )
                result.ranked = []
                elapsed_ms = int((time.perf_counter() - t_start) * 1000)
                first_canonical = (
                    requested_identifiers[0][0] if requested_identifiers else None
                )
                result.stats = {
                    "search_mode": "identifier_not_found",
                    "requested_identifiers": requested_identifiers,
                    "requested_ticket_ids": requested_ticket_ids,
                    "ticket_id": first_canonical,
                    "matched_count": 0,
                    "timing_total_ms": elapsed_ms,
                }
                logger.info(
                    "Retrieval short-circuit: identifier_not_found for %s (%dms)",
                    requested_identifiers, elapsed_ms,
                )
                return result

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

    # def _run_vector():
    #     """Channel 1: pgvector cosine similarity search."""
    #     if vector_search_fn is None:
    #         return []
    #     try:
    #         hits = vector_search_fn(
    #             query_embedding=query_embedding,
    #             n_results=settings.VECTOR_CANDIDATES,
    #             allowed_file_ids=allowed_ids_list,
    #         )
    #         # Filter by owner and allowed files
    #         filtered = []
    #         for hit in hits:
    #             meta = hit.get("metadata", {})
    #             if meta.get("owner_id", "anonymous") != owner_id:
    #                 continue
    #             if allowed_file_ids and meta.get("file_id") not in allowed_file_ids:
    #                 continue
    #             filtered.append(hit)
    #         return filtered
    #     except Exception as exc:
    #         logger.warning("Vector search channel failed: %s", exc)
    #         return []

    # def _run_bm25():
    #     """Channel 2: in-memory BM25 term frequency search."""
    #     if bm25_search_fn is None:
    #         return []
    #     try:
    #         raw_hits = bm25_search_fn(
    #             query,
    #             n_results=settings.BM25_CANDIDATES,
    #             file_type=file_type,
    #         )
    #         # Filter by owner and allowed files
    #         filtered = []
    #         for doc_id, text_val, meta, score in raw_hits:
    #             if meta.get("owner_id", "anonymous") != owner_id:
    #                 continue
    #             if allowed_file_ids and meta.get("file_id") not in allowed_file_ids:
    #                 continue
    #             filtered.append((doc_id, text_val, meta, score))
    #         return filtered
    #     except Exception as exc:
    #         logger.warning("BM25 search channel failed: %s", exc)
    #         return []

    # def _run_keyword():
    #     """Channel 3: PostgreSQL full-text search + ILIKE fallback."""
    #     try:
    #         return fulltext_search(
    #             terms=intent.extracted_terms,
    #             n_results=settings.KEYWORD_CANDIDATES,
    #             allowed_file_ids=allowed_file_ids,
    #             owner_id=owner_id,
    #         )
    #     except Exception as exc:
    #         logger.warning("Keyword search channel failed: %s", exc)
    #         return []

    # def _run_metadata():
    #     """Channel 4: metadata filter search (vendor/product/domain)."""
    #     if not settings.ENABLE_METADATA_FILTER or not intent.metadata_hints:
    #         return []
    #     try:
    #         return metadata_filter_search(
    #             metadata_hints=intent.metadata_hints,
    #             n_results=settings.METADATA_FILTER_CANDIDATES,
    #             allowed_file_ids=allowed_file_ids,
    #             owner_id=owner_id,
    #         )
    #     except Exception as exc:
    #         logger.warning("Metadata filter channel failed: %s", exc)
    #         return []

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
            # Files are shared — only filter by allowed_file_ids, not owner_id
            filtered = []
            for hit in hits:
                meta = hit.get("metadata", {})
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
            # Files are shared — only filter by allowed_file_ids, not owner_id
            filtered = []
            for doc_id, text_val, meta, score in raw_hits:
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
                owner_id=None,  # ← shared files, no owner filter
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
                owner_id=None,  # ← shared files, no owner filter
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
    # Brief 4 / Opt 4: skip the Mistral reranker when the fused list is
    # smaller than RERANK_MIN_CHUNKS — nothing meaningful to re-order, and
    # the Bedrock call adds ~2s of pure overhead.
    _skip_rerank_tiny = (
        bool(getattr(settings, "SKIP_RERANK_ON_TINY_RESULTS_ENABLED", False))
        and len(fused) < int(getattr(settings, "RERANK_MIN_CHUNKS", 3))
    )
    if _skip_rerank_tiny:
        logger.info(
            "[rerank] skipped — only %d chunks retrieved (threshold=%d)",
            len(fused), int(settings.RERANK_MIN_CHUNKS),
        )
        ranked = fused[: settings.RERANK_TOP_K]
    elif fused and generate_fn:
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
