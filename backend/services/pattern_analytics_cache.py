"""
Pattern analytics cache — stores computed topic-level statistics.

TTL-based cache: pattern stats for a topic expire after PATTERN_ANALYTICS_CACHE_TTL_HOURS
(default 24 hours). Dramatically reduces cost when multiple users ask about the
same topic patterns. All access is wrapped in try/except; a cache failure never
propagates to callers.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text

from backend.config import settings
from backend.db.connection import engine

logger = logging.getLogger("acadia-log-iq")


def _compute_topic_key(topic: str) -> str:
    """Normalize a topic string into a stable cache key (sha256 prefix)."""
    normalized = (topic or "").strip().lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def get_cached_pattern(
    organization_id: str,
    topic: str,
) -> Optional[Dict[str, Any]]:
    """Return cached pattern_data if a fresh (non-expired) entry exists."""
    if not getattr(settings, "PATTERN_ANALYTICS_ENABLED", False):
        return None

    topic_key = _compute_topic_key(topic)

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT pattern_data, confidence_score, matched_ticket_count, computed_at
                    FROM pattern_analytics_cache
                    WHERE organization_id = :oid
                      AND topic_key = :tkey
                      AND expires_at > NOW()
                    LIMIT 1
                    """
                ),
                {"oid": organization_id, "tkey": topic_key},
            ).mappings().first()
    except Exception as exc:
        logger.warning("[pattern_cache] get failed: %s", exc)
        return None

    if not row:
        return None

    data = row["pattern_data"]
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except Exception:
            data = None
    if not data:
        return None

    logger.info(
        "[pattern_cache] HIT topic=%r computed_at=%s matched=%d confidence=%.2f",
        (topic or "")[:40],
        row["computed_at"],
        int(row["matched_ticket_count"] or 0),
        float(row["confidence_score"] or 0.0),
    )
    return data


def store_pattern(
    organization_id: str,
    topic: str,
    pattern_data: Dict[str, Any],
) -> None:
    """Upsert computed pattern stats into the cache with a TTL."""
    if not getattr(settings, "PATTERN_ANALYTICS_ENABLED", False):
        return

    topic_key = _compute_topic_key(topic)
    ttl_hours = getattr(settings, "PATTERN_ANALYTICS_CACHE_TTL_HOURS", 24)
    expires_at = datetime.now(timezone.utc) + timedelta(hours=ttl_hours)

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO pattern_analytics_cache
                        (organization_id, topic_key, pattern_data,
                         confidence_score, matched_ticket_count, expires_at)
                    VALUES (:oid, :tkey, CAST(:data AS JSONB),
                            :conf, :matched, :expires)
                    ON CONFLICT (organization_id, topic_key)
                    DO UPDATE SET
                        pattern_data = EXCLUDED.pattern_data,
                        confidence_score = EXCLUDED.confidence_score,
                        matched_ticket_count = EXCLUDED.matched_ticket_count,
                        computed_at = NOW(),
                        expires_at = EXCLUDED.expires_at
                    """
                ),
                {
                    "oid": organization_id,
                    "tkey": topic_key,
                    "data": json.dumps(pattern_data),
                    "conf": float(pattern_data.get("confidence_score", 0.0) or 0.0),
                    "matched": int(pattern_data.get("total_count", 0) or 0),
                    "expires": expires_at,
                },
            )
    except Exception as exc:
        logger.warning("[pattern_cache] store failed: %s", exc)
        return

    logger.info(
        "[pattern_cache] stored topic=%r matched=%d confidence=%.2f ttl=%dh",
        (topic or "")[:40],
        int(pattern_data.get("total_count", 0) or 0),
        float(pattern_data.get("confidence_score", 0.0) or 0.0),
        ttl_hours,
    )


def invalidate_cache(organization_id: str, topic: Optional[str] = None) -> int:
    """Delete cache rows for an org (all topics when topic is None)."""
    try:
        with engine.begin() as conn:
            if topic:
                result = conn.execute(
                    text(
                        """
                        DELETE FROM pattern_analytics_cache
                        WHERE organization_id = :oid AND topic_key = :tkey
                        """
                    ),
                    {"oid": organization_id, "tkey": _compute_topic_key(topic)},
                )
            else:
                result = conn.execute(
                    text(
                        "DELETE FROM pattern_analytics_cache WHERE organization_id = :oid"
                    ),
                    {"oid": organization_id},
                )
            return int(result.rowcount or 0)
    except Exception as exc:
        logger.warning("[pattern_cache] invalidate failed: %s", exc)
        return 0
