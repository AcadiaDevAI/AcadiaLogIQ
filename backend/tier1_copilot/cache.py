"""Signature-keyed Tier-1 answer cache.

Backed by the tier1_answer_cache table created by migration 038. Cache
key is the SHA-1 produced in `normalizer.normalize_alert`. TTL defaults
to settings.TIER1_CACHE_TTL_DAYS (7d).

All helpers are exception-safe: any DB failure logs and returns None /
False so the handler degrades cleanly to the uncached path.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


def get_cached_answer(signature_hash: str) -> Optional[Dict[str, Any]]:
    """Return the cached answer dict for a signature_hash, or None.

    Also returns None if the row exists but is past its expires_at
    timestamp — stale reads are treated as misses (no in-line DELETE;
    a background sweep can harvest later).

    Sprint 10.3 — return value now includes `top_5_match_ids`. Empty
    list when the row was cached before migration 041 ran (the column
    has DEFAULT '{}' so reads never NULL out)."""
    if not signature_hash:
        return None
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT answer_json, confidence, matched_chunk_id,
                           alert_signature, top_5_match_ids, expires_at
                    FROM tier1_answer_cache
                    WHERE signature_hash = :h
                    """
                ),
                {"h": signature_hash},
            ).mappings().first()
    except Exception as exc:
        logger.warning("[tier1_copilot] cache get failed: %s", exc)
        return None

    if not row:
        return None

    expires = row["expires_at"]
    if expires is not None:
        now = datetime.now(timezone.utc)
        try:
            exp_dt = expires if expires.tzinfo else expires.replace(tzinfo=timezone.utc)
        except Exception:
            exp_dt = now
        if exp_dt < now:
            return None

    answer = row["answer_json"]
    if isinstance(answer, str):
        try:
            answer = json.loads(answer)
        except Exception:
            return None
    return {
        "answer_json": answer,
        "confidence": row["confidence"],
        "matched_chunk_id": row["matched_chunk_id"],
        "alert_signature": row["alert_signature"],
        # Sprint 10.3 — list() cast at the boundary per spec §8 type
        # discipline. psycopg returns TEXT[] as a list; the cast is
        # defensive against future driver changes that might return
        # tuples or numpy arrays.
        "top_5_match_ids": list(row.get("top_5_match_ids") or []),
    }


def set_cached_answer(
    *,
    signature_hash: str,
    alert_signature: str,
    answer: Dict[str, Any],
    confidence: str,
    matched_chunk_id: Optional[str],
    top_5_match_ids: Optional[list] = None,
    ttl_days: Optional[int] = None,
) -> bool:
    """Upsert a cache row. Returns True on success, False on error.

    Sprint 10.3 — `top_5_match_ids` is now persisted alongside the
    answer payload so cache hits can repopulate the new session's
    cohort. Without this, /tier1/journey/{sid}/initial sees an empty
    cohort on every cache hit and renders blank panels (the structural
    bug 10.3 closes per spec §3).
    """
    if not signature_hash or not answer:
        return False
    ttl = int(ttl_days if ttl_days is not None else settings.TIER1_CACHE_TTL_DAYS)
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO tier1_answer_cache (
                        signature_hash, alert_signature, answer_json,
                        matched_chunk_id, confidence, top_5_match_ids,
                        created_at, expires_at
                    )
                    VALUES (
                        :h, :sig, CAST(:a AS JSONB),
                        :mcid, :conf, :top5,
                        NOW(), NOW() + (:ttl || ' days')::interval
                    )
                    ON CONFLICT (signature_hash)
                    DO UPDATE SET
                        answer_json = EXCLUDED.answer_json,
                        matched_chunk_id = EXCLUDED.matched_chunk_id,
                        confidence = EXCLUDED.confidence,
                        top_5_match_ids = EXCLUDED.top_5_match_ids,
                        created_at = NOW(),
                        expires_at = NOW() + (:ttl || ' days')::interval
                    """
                ),
                {
                    "h": signature_hash,
                    "sig": alert_signature,
                    "a": json.dumps(answer, ensure_ascii=False),
                    "mcid": matched_chunk_id,
                    "conf": confidence,
                    # Sprint 10.3 — list() cast at the boundary per spec
                    # §8 type discipline. SQLAlchemy + psycopg3 + TEXT[]
                    # is finicky; tuples and numpy arrays fail silently.
                    "top5": list(top_5_match_ids or []),
                    "ttl": ttl,
                },
            )
        return True
    except Exception as exc:
        logger.warning("[tier1_copilot] cache set failed: %s", exc)
        return False


def invalidate_by_chunk(chunk_id: str) -> int:
    """Delete every cache row whose matched_chunk_id points at this chunk.

    Returns the number of rows deleted. Useful when a ticket is
    re-uploaded — the answer it produced is no longer authoritative.
    """
    if not chunk_id:
        return 0
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            result = conn.execute(
                text(
                    "DELETE FROM tier1_answer_cache WHERE matched_chunk_id = :c"
                ),
                {"c": chunk_id},
            )
            return int(getattr(result, "rowcount", 0) or 0)
    except Exception as exc:
        logger.warning("[tier1_copilot] cache invalidate failed: %s", exc)
        return 0


def cache_size() -> int:
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT COUNT(*) FROM tier1_answer_cache")
            ).first()
        return int(row[0]) if row else 0
    except Exception:
        return 0
