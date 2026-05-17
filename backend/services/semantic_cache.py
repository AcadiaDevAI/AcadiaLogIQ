"""
Semantic Answer Cache (Brief 5 / Part 1).

Cross-user, document-scoped cache keyed on (query_embedding,
file_ids_fingerprint). Sits behind the exact-string `AnswerCache` and
catches paraphrased queries ("What caused INC-10015?" vs "Root cause of
INC-10015?") so the organization pays once for answers asked many ways.

Layered validation:
  Layer 1 (entry)     — only validated, high-confidence answers may enter
  Layer 2 (retrieval) — identifier/negation/source-availability/staleness
  Layer 3 (feedback)  — user dislike deletes the row immediately

All ops are fail-safe: any exception is logged and treated as miss.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import text

from backend.config import settings
from backend.db.connection import SessionLocal

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class SemanticCacheHit:
    cache_id: str
    cached_query: str
    answer_text: str
    answer_metadata: Dict[str, Any]
    similarity: float
    created_at: datetime
    hit_count: int


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_NEGATION_TOKENS = {
    "not", "no", "never", "except", "without",
    "excluding", "didn't", "doesn't", "don't", "none",
}


def _fingerprint_file_ids(file_ids: Iterable[str]) -> str:
    sorted_ids = sorted(str(fid) for fid in (file_ids or []))
    joined = ",".join(sorted_ids)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def _extract_negation_tokens(query: str) -> List[str]:
    words = re.findall(r"\b[\w']+\b", (query or "").lower())
    return sorted({w for w in words if w in _NEGATION_TOKENS})


def _similarity_threshold() -> float:
    """Hotfix: tightened cosine threshold reduces near-miss cache hits
    on paraphrased-but-different queries."""
    return float(getattr(
        settings, "HOTFIX_SEMANTIC_CACHE_THRESHOLD", 0.985,
    ))


_NUMERIC_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _numeric_tokens(query: str) -> List[str]:
    """Hotfix: numbers that semantically partition meaning (top-5 vs top-10,
    P1 vs P3, >30 days vs >90 days). Treat as a strict equality gate so
    'top 5 tickets' doesn't reuse the answer for 'top 10 tickets'.
    """
    if not query:
        return []
    return sorted(set(_NUMERIC_RE.findall(query)))


def _extract_cache_identifiers(query: str) -> List[str]:
    from backend.retrieval.orchestrator import _extract_identifiers
    return sorted({canonical for canonical, _id_type in _extract_identifiers(query or "")})


def _embed_query(query: str) -> Optional[List[float]]:
    """Titan V2 embedding for the query. Reuses api.safe_embed."""
    try:
        from backend.api import safe_embed
        return safe_embed(query)
    except Exception as exc:
        logger.warning("[semantic_cache] embed failed: %s", exc)
        return None


def _emb_to_pg(vec: List[float]) -> str:
    """Format a python list as a pgvector literal string: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{float(v):.7f}" for v in vec) + "]"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def lookup(query: str, active_file_ids: Iterable[str]) -> Optional[SemanticCacheHit]:
    """
    Return a SemanticCacheHit if a valid semantic match exists, else None.
    Runs all four Layer 2 validations (identifier, negation, source, TTL).
    Increments hit_count and updates last_hit_at on a hit.
    """
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None

    try:
        fingerprint = _fingerprint_file_ids(active_file_ids)
        query_emb = _embed_query(query)
        if not query_emb:
            return None
        incoming_identifiers = _extract_cache_identifiers(query)
        incoming_negations = _extract_negation_tokens(query)

        emb_literal = _emb_to_pg(query_emb)

        sql = text("""
            SELECT id, query_text, answer_text, answer_metadata_json,
                   source_file_ids, identifiers_extracted, negation_tokens,
                   created_at, hit_count,
                   1 - (query_embedding <=> CAST(:emb AS vector)) AS similarity
            FROM semantic_answer_cache
            WHERE file_ids_fingerprint = :fp
              AND created_at > now() - make_interval(days => :ttl_days)
            ORDER BY query_embedding <=> CAST(:emb AS vector)
            LIMIT 1
        """)

        with SessionLocal() as db:
            row = db.execute(sql, {
                "emb": emb_literal,
                "fp": fingerprint,
                "ttl_days": int(settings.SEMANTIC_CACHE_TTL_DAYS),
            }).mappings().first()

        if not row:
            return None

        similarity = float(row["similarity"])
        active_threshold = _similarity_threshold()
        if similarity < active_threshold:
            if settings.SEMANTIC_CACHE_LOG_HITS:
                logger.info(
                    "[semantic_cache] below threshold: sim=%.3f < %.3f query=%r",
                    similarity, active_threshold, (query or "")[:80],
                )
            return None

        # Layer 2 check 1: identifier match
        cached_ids = sorted(row["identifiers_extracted"] or [])
        if cached_ids != incoming_identifiers:
            logger.info(
                "[semantic_cache] identifier mismatch: cached=%s incoming=%s — reject",
                cached_ids, incoming_identifiers,
            )
            return None

        # Layer 2 check 2: negation match
        cached_negs = sorted(row["negation_tokens"] or [])
        if cached_negs != incoming_negations:
            logger.info(
                "[semantic_cache] negation mismatch: cached=%s incoming=%s — reject",
                cached_negs, incoming_negations,
            )
            return None

        # Hotfix: numeric-token equality. 'top 5' and 'top 10' share almost
        # all their embedding mass but are different questions — reject when
        # the number set differs.
        incoming_nums = _numeric_tokens(query)
        cached_nums = _numeric_tokens(row["query_text"] or "")
        if cached_nums != incoming_nums:
            logger.info(
                "[semantic_cache] numeric mismatch: cached=%s incoming=%s — reject",
                cached_nums, incoming_nums,
            )
            return None

        # Layer 2 check 3: source file still accessible
        cached_sources = set(row["source_file_ids"] or [])
        current_files = set(str(f) for f in (active_file_ids or []))
        if cached_sources and not cached_sources.issubset(current_files):
            logger.info(
                "[semantic_cache] source files no longer accessible: missing=%s — reject",
                cached_sources - current_files,
            )
            return None

        # Passed all checks — bump hit counter
        new_hit_count = int(row["hit_count"]) + 1
        with SessionLocal() as db:
            db.execute(
                text("""
                    UPDATE semantic_answer_cache
                    SET hit_count = hit_count + 1, last_hit_at = now()
                    WHERE id = :id
                """),
                {"id": row["id"]},
            )
            db.commit()

        logger.info(
            "[semantic_cache] HIT sim=%.3f cached_query=%r identifiers=%s hit_count=%d",
            similarity, (row["query_text"] or "")[:80], cached_ids, new_hit_count,
        )

        metadata = row["answer_metadata_json"] or {}
        if isinstance(metadata, str):
            try:
                metadata = json.loads(metadata)
            except Exception:
                metadata = {}

        return SemanticCacheHit(
            cache_id=str(row["id"]),
            cached_query=row["query_text"],
            answer_text=row["answer_text"],
            answer_metadata=metadata,
            similarity=similarity,
            created_at=row["created_at"],
            hit_count=new_hit_count,
        )

    except Exception as exc:
        logger.warning("[semantic_cache] lookup failed (%s) — treating as miss", exc)
        return None


def put(
    *,
    query: str,
    answer_text: str,
    answer_metadata: Dict[str, Any],
    source_file_ids: Iterable[str],
    active_file_ids: Iterable[str],
    confidence: float,
    grounding_score: float,
    grounding_passed: bool,
    fabrications: int,
    model_used: str,
    origin_owner_id: str,
) -> Optional[str]:
    """
    Insert a new cache row if the answer meets Layer 1 gating criteria.
    Returns the cache row UUID on success, None if the answer was rejected.
    """
    if not settings.SEMANTIC_CACHE_ENABLED:
        return None

    try:
        # Layer 1 — only cache validated, high-confidence, grounded answers
        if confidence < settings.SEMANTIC_CACHE_MIN_CONFIDENCE_TO_CACHE:
            logger.info(
                "[semantic_cache] not caching: confidence=%.2f < %.2f",
                confidence, settings.SEMANTIC_CACHE_MIN_CONFIDENCE_TO_CACHE,
            )
            return None
        if not grounding_passed:
            logger.info("[semantic_cache] not caching: grounding_failed")
            return None
        if fabrications > 0:
            logger.info("[semantic_cache] not caching: fabrications=%d", fabrications)
            return None
        if not (answer_text or "").strip():
            logger.info("[semantic_cache] not caching: empty answer")
            return None
        if model_used in {
            "trivial_lookup",
            "identifier_not_found",
            "ticket_id_not_found",
            "canned_fallback",
            "semantic_cache",
            "guardrail_block",
        }:
            return None

        fingerprint = _fingerprint_file_ids(active_file_ids)
        query_emb = _embed_query(query)
        if not query_emb:
            return None
        identifiers = _extract_cache_identifiers(query)
        negations = _extract_negation_tokens(query)
        emb_literal = _emb_to_pg(query_emb)

        sql = text("""
            INSERT INTO semantic_answer_cache
                (query_text, query_embedding, answer_text, answer_metadata_json,
                 file_ids_fingerprint, source_file_ids, identifiers_extracted,
                 negation_tokens, confidence, grounding_score, model_used,
                 origin_owner_id)
            VALUES
                (:q, CAST(:emb AS vector), :a, CAST(:meta AS jsonb),
                 :fp, :src, :ids, :negs, :conf, :gnd, :model, :owner)
            RETURNING id
        """)

        with SessionLocal() as db:
            row = db.execute(sql, {
                "q": query,
                "emb": emb_literal,
                "a": answer_text,
                "meta": json.dumps(answer_metadata or {}, default=str),
                "fp": fingerprint,
                "src": [str(f) for f in (source_file_ids or [])],
                "ids": identifiers,
                "negs": negations,
                "conf": float(confidence),
                "gnd": float(grounding_score),
                "model": model_used or "unknown",
                "owner": str(origin_owner_id or ""),
            }).mappings().first()
            db.commit()

        cache_id = str(row["id"])
        logger.info(
            "[semantic_cache] stored: id=%s query=%r identifiers=%s",
            cache_id, (query or "")[:80], identifiers,
        )

        _evict_if_over_capacity()
        return cache_id

    except Exception as exc:
        logger.warning("[semantic_cache] put failed (%s) — skipping", exc)
        return None


def invalidate(cache_id: str, reason: str = "user_feedback") -> bool:
    """Delete a specific cache row. Returns True if a row was removed."""
    if not cache_id:
        return False
    try:
        with SessionLocal() as db:
            result = db.execute(
                text("DELETE FROM semantic_answer_cache WHERE id = CAST(:id AS uuid)"),
                {"id": str(cache_id)},
            )
            db.commit()
            deleted = (result.rowcount or 0) > 0
        if deleted:
            logger.info("[semantic_cache] invalidated id=%s reason=%s", cache_id, reason)
        return deleted
    except Exception as exc:
        logger.warning("[semantic_cache] invalidate failed (%s)", exc)
        return False


def invalidate_by_source_file(file_id: str) -> int:
    """Bulk delete cache rows whose source_file_ids includes `file_id`."""
    if not file_id:
        return 0
    try:
        with SessionLocal() as db:
            result = db.execute(
                text(
                    "DELETE FROM semantic_answer_cache "
                    "WHERE :fid = ANY(source_file_ids)"
                ),
                {"fid": str(file_id)},
            )
            db.commit()
            deleted = int(result.rowcount or 0)
        if deleted:
            logger.info(
                "[semantic_cache] invalidated %d rows referencing file_id=%s",
                deleted, file_id,
            )
        return deleted
    except Exception as exc:
        logger.warning("[semantic_cache] invalidate_by_source_file failed (%s)", exc)
        return 0


def prune_expired() -> int:
    """Delete rows older than SEMANTIC_CACHE_TTL_DAYS."""
    try:
        with SessionLocal() as db:
            result = db.execute(
                text(
                    "DELETE FROM semantic_answer_cache "
                    "WHERE created_at < now() - make_interval(days => :d)"
                ),
                {"d": int(settings.SEMANTIC_CACHE_TTL_DAYS)},
            )
            db.commit()
            deleted = int(result.rowcount or 0)
        if deleted:
            logger.info("[semantic_cache] pruned %d expired rows", deleted)
        return deleted
    except Exception as exc:
        logger.warning("[semantic_cache] prune_expired failed (%s)", exc)
        return 0


def _evict_if_over_capacity() -> int:
    """Hard cap enforcement — oldest rows go first."""
    max_entries = int(settings.SEMANTIC_CACHE_MAX_ENTRIES)
    if max_entries <= 0:
        return 0
    try:
        with SessionLocal() as db:
            total = db.execute(
                text("SELECT COUNT(*) FROM semantic_answer_cache")
            ).scalar() or 0
            if total <= max_entries:
                return 0
            to_evict = int(total) - max_entries
            result = db.execute(
                text("""
                    DELETE FROM semantic_answer_cache
                    WHERE id IN (
                        SELECT id FROM semantic_answer_cache
                        ORDER BY created_at ASC
                        LIMIT :n
                    )
                """),
                {"n": to_evict},
            )
            db.commit()
            deleted = int(result.rowcount or 0)
        if deleted:
            logger.info("[semantic_cache] evicted %d oldest rows (cap=%d)", deleted, max_entries)
        return deleted
    except Exception as exc:
        logger.warning("[semantic_cache] eviction failed (%s)", exc)
        return 0


# ---------------------------------------------------------------------------
# Cross-Cutting Analytical Router — cache helpers
#
# Append-only additions for the analytical routing layer. Existing cache
# lookup/put/invalidate functions are untouched; these helpers produce a
# corpus-versioned cache key for analytical queries so results invalidate
# automatically when new tickets are uploaded (MAX(updated_at) changes).
# ---------------------------------------------------------------------------


def get_analytical_cache_key(query: str, corpus_version: str) -> str:
    """
    Generate cache key for analytical queries.
    Includes corpus_version so cache invalidates when new tickets uploaded.
    """
    key_input = f"analytical|{(query or '').lower().strip()}|{corpus_version}"
    return hashlib.sha256(key_input.encode("utf-8")).hexdigest()[:16]


def get_corpus_version() -> str:
    """
    Get current corpus version — hash of MAX(updated_at) across
    document_metadata. Changes when new documents are uploaded, which
    automatically invalidates analytical cache entries keyed on this value.

    Fail-safe: on any DB error returns the sentinel "unknown" so callers
    can still produce a cache key (it just won't roll over on upload).
    """
    try:
        with SessionLocal() as db:
            row = db.execute(
                text("SELECT MAX(updated_at) AS max_updated FROM document_metadata")
            ).mappings().first()
            max_updated = row["max_updated"] if row else None
            if max_updated is None:
                return "empty"
            return hashlib.sha256(
                str(max_updated).encode("utf-8")
            ).hexdigest()[:12]
    except Exception as exc:
        logger.warning("[semantic_cache] corpus_version lookup failed (%s)", exc)
        return "unknown"
