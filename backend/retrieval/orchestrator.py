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
# Sprint 2.8 — regex-only aggregation detector used to raise the reranker
# cap for list-style aggregation queries. Deterministic, no LLM cost, safe
# to call on every retrieve().
from backend.retrieval.metadata_sql import detect_aggregation_intent

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
# Sprint 2.5 — Hotfix identifier extraction (flag-gated)
# ---------------------------------------------------------------------------
# Broader patterns than settings.IDENTIFIER_PATTERNS so new customer formats
# (INC-NEBULA-772, INC_546, raw 13-digit IDs) match without config changes.
# Also consults learned_vocabulary so customer-specific tokens are preserved
# even when they don't match any regex.
_HOTFIX_IDENTIFIER_PATTERNS: List[Tuple[str, str]] = [
    # Letter prefix + hyphen/underscore + alphanumeric segments:
    # INC-10001, INC-NEBULA-772, INC_546, CHG-2024-001
    (r"\b([A-Z]{2,})[-_]([A-Z0-9]+(?:[-_][A-Z0-9]+)*)\b", "identifier"),
    # Long numeric IDs (>=10 digits) — raw ticket numbers
    (r"\b(\d{10,})\b", "numeric_id"),
    # Standard ticket-ish: letters + digits with optional space/hyphen
    (r"\b([A-Z]{2,6})[- ]?(\d{3,})\b", "ticket_like"),
]

_HOTFIX_TOKEN_STRIP_RE = re.compile(r"^[\W_]+|[\W_]+$")


def _extract_identifiers_hotfix(query: str) -> List[Tuple[str, str]]:
    """Hotfix identifier extractor — broader patterns + vocabulary lookup.

    Returns list of (canonical_id, id_type) tuples. Falls back to
    learned_vocabulary for tokens classified as 'identifier' that don't
    match any regex. Result is compatible with the downstream
    identifier_exact_search signature.

    v2 Bug #1: when IDENTIFIER_VOCAB_TYPE_CHECK is on, consult the learned
    vocabulary before classifying a regex-matched candidate as an
    identifier. If the vocabulary says the token is a 'field_name' or
    'enum_value', drop it — so Resolution_Quality_Score (a JSON key) is
    not mistakenly treated as a ticket number.
    """
    if not query:
        return []

    # Vocab lookup helper — single DB hit per token. Import at function
    # scope so the module stays importable when the DB is unreachable.
    _vocab_type_check = bool(getattr(settings, "IDENTIFIER_VOCAB_TYPE_CHECK", False))
    try:
        from backend.services.vocabulary_learner import (
            get_token_type,
            is_identifier_type,
        )
    except Exception:
        def get_token_type(_tok: str) -> str:  # type: ignore
            return "unknown"

        def is_identifier_type(_tok: str) -> bool:  # type: ignore
            return False

    out: List[Tuple[str, str]] = []
    seen: Set[Tuple[str, str]] = set()
    covered: List[Tuple[int, int]] = []

    # 1. Regex-based extraction (upper-cased for case-insensitive match).
    upper_query = query.upper()
    for pattern, id_type in _HOTFIX_IDENTIFIER_PATTERNS:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            logger.warning("[hotfix_ids] invalid pattern %r: %s", pattern, exc)
            continue
        for m in regex.finditer(upper_query):
            span = m.span()
            if any(s <= span[0] < e or s < span[1] <= e for s, e in covered):
                continue
            canonical = m.group(0)
            key = (canonical, id_type)
            if key in seen:
                continue
            # v2 Bug #1: vocabulary-based type filter. Compare using the
            # original-cased span so field names stored as 'Resolution_Quality_Score'
            # match their learned form. Fall back to the upper form when the
            # case-preserved lookup misses.
            if _vocab_type_check:
                cased = query[span[0]:span[1]]
                learned = get_token_type(cased)
                if learned == "unknown":
                    learned = get_token_type(canonical)
                if learned in ("field_name", "enum_value"):
                    logger.info(
                        "[id_extract] skipping %s=%s (token=%s)",
                        learned, canonical, cased,
                    )
                    covered.append(span)  # claim span so other patterns don't re-tag
                    continue
            seen.add(key)
            out.append((canonical, id_type))
            covered.append(span)

    # 2. Vocabulary-backed fallback for tokens that weren't regex-captured.
    for raw_tok in query.split():
        core = _HOTFIX_TOKEN_STRIP_RE.sub("", raw_tok)
        if not core or len(core) < 3:
            continue
        canon = core.upper()
        if any(canon == c for c, _ in out):
            continue
        try:
            if is_identifier_type(canon):
                out.append((canon, "vocab_identifier"))
        except Exception:
            continue

    return out


# ---------------------------------------------------------------------------
# v2 Bug #3 + #5 — Multi-column identifier lookup with suffix stripping
# ---------------------------------------------------------------------------
# Real-world data stores the same ticket under several metadata_json keys
# (incident_number, vector_id, external_incident_id, ticket_number) and
# vector pipelines often append suffixes like _SEMANTIC_UNIT / _CHUNK so
# INC-ALPHA-001_SEMANTIC_UNIT never matches the raw INC-ALPHA-001 primary_id.
# The helper below broadens the exact lookup to all identifier-bearing JSON
# columns and retries with common suffixes stripped. keyword_search.py is in
# the must-stay-untouched list, so the wider lookup lives here.

_IDENTIFIER_SUFFIXES: Tuple[str, ...] = (
    "_SEMANTIC_UNIT",
    "_CHUNK",
    "_EMBEDDING",
    "_DOC",
    "_ROOT",
)

_IDENTIFIER_METADATA_COLUMNS: Tuple[str, ...] = (
    "primary_id",
    "incident_number",
    "vector_id",
    "external_incident_id",
    "ticket_number",
)


def _strip_identifier_suffixes(token: str) -> str:
    """Strip known vector-pipeline suffixes (case-insensitive). Returns the
    input unchanged when nothing matches."""
    if not token:
        return token
    upper = token.upper()
    for suf in _IDENTIFIER_SUFFIXES:
        if upper.endswith(suf):
            return token[: len(token) - len(suf)]
    return token


def _expand_identifier_candidates(
    identifiers: List[Tuple[str, str]],
) -> List[str]:
    """Return the union of raw + suffix-stripped identifier strings
    (upper-cased, deduped) for multi-column SQL lookup."""
    out: List[str] = []
    seen: Set[str] = set()
    for cid, _itype in identifiers or []:
        if not cid:
            continue
        raw = str(cid).strip().upper()
        if raw and raw not in seen:
            seen.add(raw)
            out.append(raw)
        if getattr(settings, "IDENTIFIER_SUFFIX_STRIP_ENABLED", False):
            stripped = _strip_identifier_suffixes(raw).upper()
            if stripped and stripped not in seen:
                seen.add(stripped)
                out.append(stripped)
    return out


def _multi_column_identifier_search(
    identifiers: List[Tuple[str, str]],
    allowed_file_ids: Optional[Set[str]],
    n_results: int = 20,
) -> List[Dict[str, Any]]:
    """Exact-match lookup across multiple metadata_json identifier columns.

    Covers primary_id, incident_number, vector_id, external_incident_id,
    ticket_number. Retries with common suffixes stripped so variants such
    as INC-ALPHA-001_SEMANTIC_UNIT resolve to INC-ALPHA-001. Returns rows
    in the same dict shape identifier_exact_search uses so the caller can
    plug them directly into the ranked list.
    """
    candidates = _expand_identifier_candidates(identifiers)
    if not candidates:
        return []

    try:
        from backend.db.connection import SessionLocal
        from sqlalchemy import bindparam, text as _sql_text
    except Exception as exc:
        logger.warning("[multi_col_lookup] import failed: %s", exc)
        return []

    col_exprs = ", ".join(
        f"c.metadata_json->>'{col}'" for col in _IDENTIFIER_METADATA_COLUMNS
    )
    sql = f"""
        SELECT
            c.id,
            c.content,
            c.contextualized_content,
            c.summary,
            c.section_heading,
            c.chunk_type,
            c.labels_json,
            c.metadata_json,
            d.id::text       AS file_id,
            d.owner_id,
            d.name           AS source,
            d.file_type
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE UPPER(COALESCE({col_exprs})) = ANY(:identifiers)
          AND d.status = 'active'
          AND dv.is_active = TRUE
          AND d.current_version_id = dv.id
    """

    params: Dict[str, Any] = {
        "identifiers": candidates,
        "limit": int(n_results),
    }
    if allowed_file_ids:
        sql += " AND d.id IN :allowed_ids"
        params["allowed_ids"] = tuple(allowed_file_ids)
    sql += " LIMIT :limit"

    hits: List[Dict[str, Any]] = []
    try:
        with SessionLocal() as db:
            stmt = _sql_text(sql)
            if allowed_file_ids:
                stmt = stmt.bindparams(bindparam("allowed_ids", expanding=True))
            rows = db.execute(stmt, params).mappings().all()
        for row in rows:
            hits.append({
                "id": row["id"],
                "text": row["contextualized_content"] or row["content"],
                "rank": 1.0,
                "search_type": "identifier_multi_column",
                "metadata": {
                    "file_id": row["file_id"],
                    "owner_id": row["owner_id"],
                    "source": row["source"],
                    "file_type": row["file_type"],
                    "section_heading": row["section_heading"],
                    "chunk_type": row["chunk_type"],
                    "summary": row["summary"],
                    "labels_json": row["labels_json"] or {},
                    "metadata_json": row["metadata_json"] or {},
                },
            })
    except Exception as exc:
        logger.warning("[multi_col_lookup] query failed: %s", exc)
        return []

    logger.info(
        "[multi_col_lookup] candidates=%s recovered=%d rows",
        candidates, len(hits),
    )
    return hits


def _fallback_like_scan(
    tokens: List[str],
    allowed_file_ids: Optional[Set[str]],
    n_results: int = 20,
) -> List[Dict[str, Any]]:
    """Last-ditch LIKE scan over chunks.metadata_json for identifiers the
    primary_id lookup missed (e.g. identifier stored in a nested field).

    Returns rows in the same shape as identifier_exact_search so the caller
    can plug them straight into the ranked list.
    """
    if not tokens:
        return []
    try:
        from backend.db.connection import engine
        from sqlalchemy import text as _sql_text
    except Exception as exc:
        logger.warning("[fallback_scan] engine import failed: %s", exc)
        return []

    params: Dict[str, Any] = {"limit": n_results}
    like_clauses: List[str] = []
    for i, tok in enumerate(tokens):
        key = f"tok{i}"
        params[key] = f"%{tok}%"
        like_clauses.append(f"metadata_json::text ILIKE :{key}")

    where_clauses = ["(" + " OR ".join(like_clauses) + ")"]
    if allowed_file_ids:
        params["document_ids"] = list(allowed_file_ids)
        where_clauses.append("document_id = ANY(:document_ids)")

    sql = (
        "SELECT id, content, metadata_json, document_id "
        "FROM chunks WHERE " + " AND ".join(where_clauses) +
        " LIMIT :limit"
    )
    try:
        with engine.connect() as conn:
            rows = conn.execute(_sql_text(sql), params).mappings().fetchall()
    except Exception as exc:
        logger.warning("[fallback_scan] query failed: %s", exc)
        return []

    out: List[Dict[str, Any]] = []
    for r in rows:
        meta = dict(r.get("metadata_json") or {})
        meta["document_id"] = r.get("document_id")
        out.append({
            "id": r["id"],
            "text": r.get("content") or "",
            "metadata": {"metadata_json": meta, **meta},
            "rank": 0.75,
        })
    logger.info(
        "[fallback_scan] tokens=%s recovered=%d rows", tokens, len(out),
    )
    return out


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
    raw_query: Optional[str] = None,
    doc_kinds: Optional[List[str]] = None,   # Sprint 3-PREP-A
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
    # Sprint 3-PREP-A — mode→corpus filter.
    # =====================================================================
    # When the caller has requested one or more doc_kinds, narrow
    # allowed_file_ids to documents matching those kinds BEFORE any
    # channel runs. This reuses the existing access-filter plumbing —
    # every search channel (vector, BM25, keyword, metadata, identifier
    # exact) already honors allowed_file_ids, so the filter propagates
    # with zero channel-level changes.
    if bool(doc_kinds):
        _candidates_before = len(allowed_file_ids or set())
        try:
            from backend.db.connection import SessionLocal
            from sqlalchemy import bindparam, text as _kind_sql_text
            stmt = _kind_sql_text(
                """
                SELECT d.id::text AS id
                FROM documents d
                WHERE d.status = 'active'
                  AND d.doc_kind IN :kinds
                """
            ).bindparams(bindparam("kinds", expanding=True))
            with SessionLocal() as db:
                rows = db.execute(stmt, {"kinds": list(doc_kinds)}).mappings().all()
            kind_matched_ids = {r["id"] for r in rows}
            if allowed_file_ids:
                allowed_file_ids = set(allowed_file_ids) & kind_matched_ids
            else:
                allowed_file_ids = kind_matched_ids
        except Exception as exc:
            # Fail-open: if the filter query blows up, skip it rather than
            # breaking retrieval. Logged so downstream sprints can see it.
            logger.warning("[doc_kind_filter] filter query failed (%s) — bypassing", exc)
        _candidates_after = len(allowed_file_ids or set())
        logger.info(
            "[doc_kind_filter] kinds=%s candidates_before=%d candidates_after=%d",
            list(doc_kinds),
            _candidates_before,
            _candidates_after,
        )

    # =====================================================================
    # Step 0 (Goal 2): Identifier exact-match short-circuit
    # =====================================================================
    # When the query names one or more identifiers matching any pattern in
    # settings.IDENTIFIER_PATTERNS, answer from an exact JSONB lookup on
    # primary_id and skip the four parallel channels entirely. Deterministic,
    # ~1ms, schema-agnostic (tickets, Jira issue keys, KB articles, ...).
    # Prefer the raw, pre-normalization query when the caller supplied one
    # — required so dash/underscore identifiers (INC-NEBULA-772, INC_546)
    # are visible to the regex + vocab extractor. expand_query normalizes
    # dashes to spaces, which corrupts letter-prefix identifier matching.
    extract_source = raw_query if raw_query else query
    logger.info(
        "[retrieve] extraction source=%s len=%d",
        "raw" if raw_query else "normalized",
        len(extract_source or ""),
    )
    requested_identifiers = _extract_identifiers_hotfix(extract_source)
    # Union with the legacy extractor so canonical ticket_number matches
    # (needed by identifier_exact_search) are not lost. Legacy extractor
    # also runs on the raw source when available.
    legacy_ids = _extract_identifiers(extract_source)
    seen_keys = {(c.upper(), t) for c, t in requested_identifiers}
    for cid, itype in legacy_ids:
        if (cid.upper(), itype) not in seen_keys:
            requested_identifiers.append((cid, itype))
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
                # Zero rows = identifier(s) genuinely not indexed by primary_id.
                # v2 Bug #3 + #5: retry against all identifier-bearing JSON
                # columns (incident_number, vector_id, external_incident_id,
                # ticket_number) with common vector-pipeline suffixes stripped
                # before falling back to the LIKE scan.
                recovered: List[Dict[str, Any]] = []
                multi_col_used = False
                if getattr(settings, "IDENTIFIER_MULTI_COLUMN_LOOKUP", False):
                    try:
                        recovered = _multi_column_identifier_search(
                            identifiers=requested_identifiers,
                            allowed_file_ids=allowed_file_ids,
                            n_results=max(settings.RERANK_TOP_K, 20),
                        )
                        multi_col_used = True
                    except Exception as exc:
                        logger.warning("[multi_col_lookup] raised: %s", exc)
                        recovered = []

                if not recovered:
                    try:
                        recovered = _fallback_like_scan(
                            tokens=[cid for cid, _ in requested_identifiers],
                            allowed_file_ids=allowed_file_ids,
                            n_results=max(settings.RERANK_TOP_K, 20),
                        )
                    except Exception as exc:
                        logger.warning("[fallback_scan] raised: %s", exc)
                        recovered = []

                if recovered:
                    ranked: List[FusedResult] = []
                    for row in recovered:
                        meta = row.get("metadata", {}) or {}
                        ranked.append(
                            (
                                row["id"],
                                row["text"],
                                meta,
                                float(row.get("rank") or 0.75),
                            )
                        )
                    mode_label = (
                        "identifier_multi_column"
                        if multi_col_used and any(
                            r.get("search_type") == "identifier_multi_column"
                            for r in recovered
                        )
                        else "identifier_fallback_like"
                    )
                    result.intent = QueryIntent(
                        strategy="keyword",
                        reason=f"{mode_label}: {requested_identifiers}",
                    )
                    result.ranked = ranked[: settings.RERANK_TOP_K]
                    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
                    result.stats = {
                        "search_mode": mode_label,
                        "requested_identifiers": requested_identifiers,
                        "requested_ticket_ids": requested_ticket_ids,
                        "matched_count": len(ranked),
                        "timing_total_ms": elapsed_ms,
                    }
                    logger.info(
                        "Retrieval short-circuit: %s for %s → %d chunks (%dms)",
                        mode_label, requested_identifiers, len(ranked), elapsed_ms,
                    )
                    return result

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
    # Sprint 2.8 — aggregation queries need higher recall. List-style
    # queries must surface all matching tickets, not just the top-10
    # "answer this one question"-optimal chunks. Detect aggregation
    # intent locally via the regex-only fast-path (no LLM cost) and
    # raise the reranker cap for those queries only.
    _rerank_cap = settings.RERANK_TOP_K
    try:
        _agg_intent_probe = detect_aggregation_intent(raw_query or query)
    except Exception as exc:
        logger.warning("[rerank] aggregation probe failed: %s", exc)
        _agg_intent_probe = None
    if _agg_intent_probe is not None:
        _rerank_cap = int(getattr(settings, "RERANK_TOP_K_AGGREGATION", 40))
        logger.info(
            "[rerank] aggregation intent detected (op=%s) → cap raised to %d",
            _agg_intent_probe.operation, _rerank_cap,
        )

    if _skip_rerank_tiny:
        logger.info(
            "[rerank] skipped — only %d chunks retrieved (threshold=%d)",
            len(fused), int(settings.RERANK_MIN_CHUNKS),
        )
        ranked = fused[: _rerank_cap]
    elif fused and generate_fn:
        reranker = _get_reranker(generate_fn)
        ranked = reranker.rerank(query, fused, top_k=_rerank_cap)
    else:
        ranked = fused[: _rerank_cap]

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


# ---------------------------------------------------------------------------
# Sprint 4 — Fingerprint-First Expert Copilot
# ---------------------------------------------------------------------------
# Exact-match lookup on the gold-ticket Fingerprints array (JSONB). This
# is intentionally NOT a hybrid retrieval: fingerprints are
# system-generated error codes (e.g., BGP-5-ADJCHANGE) — exact match
# is the only signal that matters. The `?` operator on a JSONB array
# uses the narrow GIN index created in migration 035.
#
# Flag-off returns None immediately. No SQL is executed.
# ---------------------------------------------------------------------------
def retrieve_by_fingerprint(
    fingerprint: str,
    *,
    min_quality_score: Optional[int] = None,
    limit: int = 1,
    return_chunk_id: bool = False,
) -> Optional[Any]:
    """Return the single best gold-ticket metadata_json for an exact
    fingerprint match, or None if no match / invalid format.

    Sprint 4 contract:
      - Validates against settings.FINGERPRINT_REGEX. API layer also
        validates before calling, but the defensive check here lets
        tests call this directly.
      - Orders by Resolution_Quality_Score DESC, then created_at DESC,
        so the "best" historical record is returned when multiple
        tickets share the same fingerprint.
      - Returns a plain dict (not the SQLA row proxy) so callers can
        pass it directly into run_composer as a JSON finding.

    Sprint 5 extension (additive, fully backward-compatible):
      - When return_chunk_id=True, returns a (metadata_json, chunk_id)
        tuple on hit and (None, None) on miss. The chunk_id is the
        cache key for the per-chunk Expert Copilot answer cache added
        by migration 037. Existing callers (Sprint 4 + tests) do NOT
        pass this kwarg and still receive the original Optional[Dict]
        return.
    """
    fp = (fingerprint or "").strip()
    if not re.match(getattr(settings, "FINGERPRINT_REGEX", r"^$"), fp):
        return (None, None) if return_chunk_id else None

    if min_quality_score is None:
        min_quality_score = int(
            getattr(settings, "FINGERPRINT_MIN_QUALITY_SCORE", 3)
        )

    try:
        from backend.db.connection import engine
        from sqlalchemy import text
    except Exception as exc:
        logger.warning("[fingerprint_lookup] engine import failed: %s", exc)
        return (None, None) if return_chunk_id else None

    # Correction note (see SPRINT_4 correction log): the spec queries the
    # `documents` table, but gold-ticket JSON is stored one-row-per-ticket
    # on `chunks.metadata_json` (see _ingest_gold_ticket_json). The `?`
    # operator on the narrow Fingerprints sub-path uses the
    # idx_chunks_fingerprints GIN index from migration 035.
    sql = text(
        """
        SELECT
            id,
            document_id,
            metadata_json,
            created_at,
            COALESCE(
              (metadata_json->'Metadata'->>'Resolution_Quality_Score')::int,
              (metadata_json->>'resolution_quality_score')::int,
              0
            ) AS qscore
        FROM chunks
        WHERE metadata_json -> 'Metadata' -> 'Fingerprints' ? :fp
          AND COALESCE(
            (metadata_json->'Metadata'->>'Resolution_Quality_Score')::int,
            (metadata_json->>'resolution_quality_score')::int,
            0
          ) >= :min_score
        ORDER BY qscore DESC NULLS LAST, created_at DESC
        LIMIT :lim
        """
    )

    try:
        with engine.connect() as conn:
            row = conn.execute(
                sql,
                {"fp": fp, "min_score": int(min_quality_score), "lim": int(limit)},
            ).mappings().first()
    except Exception as exc:
        logger.warning("[fingerprint_lookup] query failed fp=%s: %s", fp, exc)
        return (None, None) if return_chunk_id else None

    if not row:
        logger.info("[fingerprint_lookup] miss fp=%s", fp)
        return (None, None) if return_chunk_id else None

    metadata_json = row["metadata_json"]
    # psycopg3 returns dict; older drivers may return str. Normalize.
    if isinstance(metadata_json, str):
        try:
            import json as _json
            metadata_json = _json.loads(metadata_json)
        except Exception:
            metadata_json = {}
    elif not isinstance(metadata_json, dict):
        metadata_json = dict(metadata_json) if metadata_json else {}

    logger.info(
        "[fingerprint_lookup] hit fp=%s chunk_id=%s doc_id=%s qscore=%d",
        fp, row["id"], row["document_id"], int(row["qscore"] or 0),
    )
    if return_chunk_id:
        return metadata_json, row["id"]
    return metadata_json
