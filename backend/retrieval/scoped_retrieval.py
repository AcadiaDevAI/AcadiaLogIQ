"""Single-incident scoped retrieval — Sprint 12.2.

Purpose
-------
When a chat session is scoped to one Incident_Number (set by the
Stage 0 per-bullet "Ask in Chat" handoff via
chat_sessions.scope_incident_id, migration 042), every /ask in that
session must answer strictly from THAT one ticket's chunks — no
PDFs, no other tickets, no general knowledge.

Why this is a distinct module (not part of `orchestrator.retrieve`)
-------------------------------------------------------------------
The hybrid orchestrator is built for the opposite problem: rank
across the whole authorized corpus using vector + BM25 + full-text
+ metadata channels, fuse, then rerank. For a single-ticket scope
that architecture is wrong:

  * The four channels each return a globally-ranked top-N. A
    generic question like "What was the root cause?" matches
    semantically loud chunks across many tickets; the chunks of the
    *one* scoped ticket can easily not survive into the fused
    candidate pool.
  * Filtering scope post-fusion (the original Sprint 12.1 attempt)
    drops everything that is not the scoped ticket — and if those
    chunks were never candidates to begin with, the result is empty.
  * For a universe of 10–30 chunks (one ticket's worth), running
    BM25 / fusion / Mistral rerank is wasted compute. Plain pgvector
    cosine on a tiny in-scope set is sufficient.

The right design: filter at the source. Issue ONE SQL that does
both the per-incident WHERE and the cosine rank in a single
round-trip via pgvector's `<=>` operator. The result is a
deterministic top-K of chunks all guaranteed to belong to the
scoped ticket.

Isolation guarantees
--------------------
This module:

  * Does NOT import from `orchestrator` at module top-level (only
    inside the function body, to lift `RetrievalResult` /
    `QueryIntent` for shape compatibility) — keeps imports acyclic.
  * Does NOT modify any orchestrator behavior. The orchestrator
    stays unaware of scope. Search-in-KB calls into it as before.
  * Returns the same `RetrievalResult` shape downstream `/ask`
    already consumes, so the answer composition / citation /
    persistence stages are unchanged.

Public API
----------
``retrieve_within_incident(*, scope_incident_id, query_embedding,
allowed_file_ids=None, top_k=12) -> RetrievalResult``

Returns empty `ranked` when the ticket is unknown to the corpus
(legitimate case — caller's polite-no-answer path then fires).
Never raises — DB / parsing errors degrade to empty `ranked` with
a flagged `stats["fetch_failed"]` so callers can distinguish "not
found" from "look-up failed."
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Set

from sqlalchemy import text as _sql_text

from backend.db.connection import engine


logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────
# A typical ingested ticket yields 10–30 chunks (one per gold-schema
# section: Executive_Sharable_RCA, ITIL_5_Why, Operational_SOP, ...).
# We surface up to 50 so the LLM sees the *entire* ticket regardless
# of cosine ranking — the universe is one ticket, prompt budget is
# not the bottleneck (Claude Haiku has 200K tokens; 50 chunks ≈ a
# few thousand). This also makes `has_sufficient_document_support`
# downstream much more likely to find query keywords inside the
# scoped chunks, which prevents `/ask` from falling through to the
# unscoped conversational-fallback path.
DEFAULT_SCOPED_TOP_K = 50


# ─────────────────────────────────────────────────────────────
# SQL templates
# ─────────────────────────────────────────────────────────────
# We split into BASE + optional FILE-IDS clause + TAIL so we can
# branch on whether `allowed_file_ids` is provided without ever
# binding an empty list to PostgreSQL (which is invalid for
# `= ANY(:file_ids)`).
#
# The metadata filter mirrors `keyword_search.identifier_exact_search`'s
# established pattern — primary_id with incident_number as the
# legacy fallback. The general GIN index on `metadata_json`
# (migration 002) supports `->>'primary_id'` lookups efficiently.
#
# `<=>` is pgvector's cosine-distance operator (0 = identical,
# 2 = opposite). We rank ASC by distance and convert to a
# higher-is-better similarity score on the Python side so the
# returned tuple's score field matches the convention used by
# `fusion.FusedResult`.
_SQL_BASE = """
SELECT
    c.id::text                              AS id,
    COALESCE(c.contextualized_content,
             c.content)                     AS text,
    c.metadata_json                         AS metadata_json,
    c.section_heading                       AS section_heading,
    c.chunk_type                            AS chunk_type,
    c.summary                               AS summary,
    c.labels_json                           AS labels_json,
    d.id::text                              AS file_id,
    d.owner_id                              AS owner_id,
    d.name                                  AS source,
    d.file_type                             AS file_type,
    (e.embedding <=> CAST(:query_embedding AS vector)) AS distance,
    (e.embedding IS NULL)                   AS missing_embedding
FROM chunks c
LEFT JOIN embeddings e
  ON e.chunk_id = c.id
JOIN documents d
  ON d.id = c.document_id
JOIN document_versions dv
  ON dv.id = c.document_version_id
WHERE
    UPPER(COALESCE(c.metadata_json->>'primary_id',
                   c.metadata_json->>'incident_number'))
        = :scope_id_upper
    AND d.status = 'active'
    AND dv.is_active = TRUE
    AND d.current_version_id = dv.id
"""

_SQL_FILE_IDS_CLAUSE = "    AND d.id::text = ANY(:file_ids)\n"

# LEFT JOIN to embeddings means a chunk with no embedding row still
# comes back — its `distance` is NULL. That's the right behaviour
# for a scoped chat: when the universe is one ticket, having the
# content (even unranked) is strictly better than returning empty.
# We sort embedded chunks first (by cosine distance ASC) and
# unembedded chunks after, so any ranking signal we have is used,
# but we never silently drop the ticket because of an
# embedding-pipeline gap. Mirrors how identifier_exact_search
# returns chunks WITHOUT joining the embeddings table at all.
_SQL_TAIL = """
ORDER BY
    CASE WHEN e.embedding IS NULL THEN 1 ELSE 0 END ASC,
    distance ASC NULLS LAST
LIMIT :top_k
"""


# ─────────────────────────────────────────────────────────────
# Small private helpers
# ─────────────────────────────────────────────────────────────
def _vector_literal(values: List[float]) -> str:
    """Convert a Python float list into a pgvector text literal.

    Kept local rather than imported from `backend.vector_store` so
    this module has no dependency on the larger persistence layer
    (it only needs the SQLAlchemy engine + the `text` SQL builder).
    """
    return "[" + ",".join(f"{float(v):.8f}" for v in values) + "]"


def _wrap_metadata(row: Any) -> Dict[str, Any]:
    """Build the metadata dict in the exact shape the rest of
    `/ask` expects (matches `vector_store.pgvector_search` lines
    ~1186-1196). Keeps `metadata_json` nested as a sub-dict so
    citation / sources logic downstream finds primary_id where
    it already looks for it."""
    return {
        "file_id": row.get("file_id"),
        "owner_id": row.get("owner_id"),
        "source": row.get("source"),
        "file_type": row.get("file_type"),
        "section_heading": row.get("section_heading"),
        "chunk_type": row.get("chunk_type"),
        "summary": row.get("summary"),
        "labels_json": row.get("labels_json") or {},
        "metadata_json": row.get("metadata_json") or {},
    }


def _empty_result(
    scope_id: Optional[str],
    elapsed_ms: int,
    *,
    fetch_failed: bool = False,
    no_embedding: bool = False,
):
    """Build an empty RetrievalResult with diagnostic stats.
    Imports `RetrievalResult` lazily — see module docstring."""
    from backend.retrieval.orchestrator import RetrievalResult
    from backend.retrieval.query_classifier import QueryIntent

    res = RetrievalResult()
    res.intent = QueryIntent(
        strategy="scoped_incident",
        reason=(
            f"chat scoped to {scope_id}"
            if scope_id
            else "chat scope missing"
        ),
    )
    res.ranked = []
    res.stats = {
        "search_mode": "scoped_incident",
        "scope_incident_id": scope_id,
        "matched_count": 0,
        "timing_total_ms": elapsed_ms,
    }
    if fetch_failed:
        res.stats["fetch_failed"] = True
    if no_embedding:
        res.stats["no_embedding"] = True
    return res


# ─────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────
def retrieve_within_incident(
    *,
    scope_incident_id: str,
    query_embedding: List[float],
    allowed_file_ids: Optional[Set[str]] = None,
    top_k: int = DEFAULT_SCOPED_TOP_K,
):
    """Fetch and cosine-rank chunks belonging to a single Incident_Number.

    Args:
        scope_incident_id: The Incident_Number this chat is scoped
            to (e.g. ``"INC-PHOENIX-402"``). Compared
            case-insensitively against ``metadata_json.primary_id``
            with ``metadata_json.incident_number`` as a legacy
            fallback (matches `identifier_exact_search`'s pattern).
        query_embedding: Pre-computed query embedding (Amazon Titan
            v2, 1024 dims). Used by pgvector's ``<=>`` operator
            inside the same SQL that applies the scope filter.
        allowed_file_ids: Optional set of authorized document IDs.
            When provided, narrows the result to chunks owned by
            those documents only — preserves the project's existing
            file-level access-control story. ``None`` = no extra
            constraint beyond the incident filter.
        top_k: Number of top-ranked chunks to return. Defaults to
            :data:`DEFAULT_SCOPED_TOP_K`.

    Returns:
        ``RetrievalResult`` with:

        * ``ranked``: list of ``(chunk_id, text, metadata, score)``
          tuples — same shape `fusion.FusedResult` uses, so
          downstream `/ask` code (answer composition, citations,
          chat persistence) needs no awareness of the path split.
          Score is similarity (``1 - cosine_distance``); higher is
          more relevant.
        * ``intent``: a :class:`QueryIntent` flagged with
          ``strategy="scoped_incident"`` for telemetry.
        * ``stats``: diagnostic counters including
          ``search_mode="scoped_incident"``, ``scope_incident_id``,
          ``matched_count``, ``top_score``, ``timing_total_ms``,
          plus optional ``fetch_failed`` / ``no_embedding`` flags
          when the call degraded gracefully.

    Never raises. DB / parsing failures are logged at WARNING and
    return an empty `ranked` with the diagnostic flags set, so the
    caller's polite-no-answer path can fire naturally.
    """
    t_start = time.perf_counter()
    sid = (scope_incident_id or "").strip()

    if not sid:
        logger.warning("[scoped_retrieval] called without scope_incident_id")
        return _empty_result(
            None, int((time.perf_counter() - t_start) * 1000),
        )

    if not query_embedding:
        logger.warning(
            "[scoped_retrieval] called without query_embedding scope=%s", sid,
        )
        return _empty_result(
            sid, int((time.perf_counter() - t_start) * 1000),
            no_embedding=True,
        )

    params: Dict[str, Any] = {
        "scope_id_upper": sid.upper(),
        "query_embedding": _vector_literal(query_embedding),
        "top_k": int(top_k),
    }
    if allowed_file_ids:
        sql = _SQL_BASE + _SQL_FILE_IDS_CLAUSE + _SQL_TAIL
        params["file_ids"] = [str(fid) for fid in allowed_file_ids]
    else:
        sql = _SQL_BASE + _SQL_TAIL

    try:
        with engine.connect() as conn:
            rows = conn.execute(_sql_text(sql), params).mappings().all()
    except Exception as exc:
        logger.warning(
            "[scoped_retrieval] DB fetch failed scope=%s err=%s", sid, exc,
        )
        return _empty_result(
            sid, int((time.perf_counter() - t_start) * 1000),
            fetch_failed=True,
        )

    # Lazy import here too — same reason as in _empty_result.
    from backend.retrieval.orchestrator import RetrievalResult
    from backend.retrieval.query_classifier import QueryIntent

    ranked = []
    missing_embeddings = 0
    for row in rows:
        raw_dist = row.get("distance")
        is_missing = bool(row.get("missing_embedding"))
        if is_missing or raw_dist is None:
            # No embedding row → cannot cosine-rank this chunk. Surface
            # it anyway with a neutral similarity (0.5) so the LLM
            # still gets the ticket content as context. The SQL ORDER
            # BY already ranks embedded chunks ahead, so this score
            # only matters when ALL of the ticket's chunks are
            # un-embedded.
            similarity = 0.5
            missing_embeddings += 1
        else:
            # pgvector cosine distance ∈ [0, 2]. For Titan-normalized
            # vectors the practical range is [0, ~1]; similarity =
            # 1 - distance is higher-is-better and consistent with
            # FusedResult / reranker output. Clamped at 0 so consumers
            # that interpret score as a 0-1 confidence don't see
            # negatives in the rare opposite-vector case.
            similarity = max(0.0, 1.0 - float(raw_dist))
        ranked.append((
            str(row["id"]),
            str(row.get("text") or ""),
            _wrap_metadata(row),
            similarity,
        ))

    elapsed_ms = int((time.perf_counter() - t_start) * 1000)
    res = RetrievalResult()
    res.intent = QueryIntent(
        strategy="scoped_incident",
        reason=f"chat scoped to {sid}",
    )
    res.ranked = ranked
    res.stats = {
        "search_mode": "scoped_incident",
        "scope_incident_id": sid,
        "matched_count": len(ranked),
        "top_score": ranked[0][3] if ranked else 0.0,
        "missing_embeddings": missing_embeddings,
        "timing_total_ms": elapsed_ms,
    }
    logger.info(
        "[scoped_retrieval] scope=%s returned=%d "
        "(missing_embeddings=%d) top_score=%.3f (%dms)",
        sid, len(ranked), missing_embeddings,
        ranked[0][3] if ranked else 0.0, elapsed_ms,
    )
    return res
