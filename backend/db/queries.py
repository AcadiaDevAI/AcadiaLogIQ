"""
Focused ad-hoc query helpers that don't belong in a full repository layer.

Currently provides `load_similar_tickets_for_topic` for the Layer 3 Pattern
Analytics service. Uses the SQLAlchemy engine from `backend.db.connection` and
the project's existing schema (chunks.metadata_json JSONB column written by
the ingestion pipeline).

Failure policy: every helper must return a benign empty result on any database
error; callers rely on this to keep the request path alive when analytics
extras fail.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from sqlalchemy import text

from backend.config import settings
from backend.db.connection import engine

logger = logging.getLogger("acadia-log-iq")


def _parse_opened_date(raw: Any) -> Any:
    """Best-effort date parsing; returns a datetime or None."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    raw_str = str(raw).strip()
    if not raw_str:
        return None
    for parser in (
        lambda s: datetime.fromisoformat(s),
        lambda s: datetime.fromisoformat(s.replace("Z", "+00:00")),
        lambda s: datetime.strptime(s, "%Y-%m-%d"),
        lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"),
    ):
        try:
            return parser(raw_str)
        except Exception:
            continue
    return None


def _coerce_bool(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("true", "t", "yes", "y", "1")


def _coerce_int(val: Any) -> int:
    if val is None:
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        try:
            return int(float(val))
        except (TypeError, ValueError):
            return 0


def load_similar_tickets_for_topic(
    topic: str,
    organization_id: str,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Load historical tickets matching a topic with permissive OR-matching.

    Primary (flag-gated) path widens the search surface: splits the topic
    into individual terms and matches ANY term across chunk content,
    summary, resolution_text, and component — using tsvector OR-query as
    the main signal and ILIKE as a plural/variant fallback. This is the
    fix for the "insufficient data" false-negatives when enough tickets
    are present but only match a single term of a compound topic like
    "failure_router".

    Flag `PATTERN_ANALYTICS_SQL_OR_MATCHING_ENABLED` toggles this path.
    On False, or if the OR-matching query raises, execution falls back
    to `_load_similar_tickets_legacy` (preserved verbatim below).
    """
    if not getattr(settings, "PATTERN_ANALYTICS_SQL_OR_MATCHING_ENABLED", True):
        return _load_similar_tickets_legacy(topic, organization_id, max_results)

    terms = [
        t for t in (topic or "").replace("_", " ").split()
        if t and len(t) >= 3
    ]
    if not terms:
        logger.info("[pattern_analytics] no searchable terms from topic=%r", topic)
        return []
    terms = terms[:5]

    ts_or_query = " | ".join(terms)
    ilike_patterns = [f"%{term}%" for term in terms]

    sql = """
        WITH matches AS (
            SELECT DISTINCT ON (d.id)
                d.id::text                                AS document_id,
                c.metadata_json->>'primary_id'            AS primary_id,
                c.metadata_json->>'incident_number'       AS incident_number,
                c.metadata_json->>'opened_date'           AS opened_date,
                c.metadata_json->>'resolution_text'       AS resolution_text,
                c.metadata_json->>'resolution'            AS resolution_alt,
                c.metadata_json->>'summary'               AS summary,
                c.metadata_json->>'sla_met'               AS sla_met,
                c.metadata_json->>'sla_target_met'        AS sla_target_met,
                c.metadata_json->>'quality_score'         AS quality_score,
                c.metadata_json->'resolution_groups'      AS resolution_groups
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE d.owner_id = :oid
              AND d.status = 'active'
              AND c.metadata_json IS NOT NULL
              AND COALESCE(
                  c.metadata_json->>'primary_id',
                  c.metadata_json->>'incident_number'
              ) IS NOT NULL
              AND (
                  to_tsvector(
                      'english',
                      COALESCE(c.content, '') || ' ' ||
                      COALESCE(c.metadata_json->>'summary', '') || ' ' ||
                      COALESCE(c.metadata_json->>'resolution_text', '')
                  ) @@ to_tsquery('english', :ts_or_query)
                  OR EXISTS (
                      SELECT 1 FROM unnest(CAST(:ilike_patterns AS TEXT[])) AS pat
                      WHERE
                           COALESCE(c.content, '')                           ILIKE pat
                        OR COALESCE(c.metadata_json->>'summary', '')         ILIKE pat
                        OR COALESCE(c.metadata_json->>'resolution_text', '') ILIKE pat
                        OR COALESCE(c.metadata_json->>'component', '')       ILIKE pat
                  )
              )
        )
        SELECT * FROM matches
        LIMIT :limit
    """

    params: Dict[str, Any] = {
        "oid": organization_id,
        "ts_or_query": ts_or_query,
        "ilike_patterns": ilike_patterns,
        "limit": int(max_results),
    }

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
    except Exception as exc:
        logger.warning(
            "[pattern_analytics] SQL OR-matching failed (topic=%r): %s — falling back to legacy",
            topic, exc,
        )
        return _load_similar_tickets_legacy(topic, organization_id, max_results)

    tickets: List[Dict[str, Any]] = []
    for row in rows:
        ticket_id = row.get("primary_id") or row.get("incident_number")
        if not ticket_id:
            continue
        tickets.append({
            "ticket_id": str(ticket_id).strip(),
            "opened_date": _parse_opened_date(row.get("opened_date")),
            "resolution_text": (
                row.get("resolution_text") or row.get("resolution_alt") or ""
            ),
            "summary": row.get("summary") or "",
            "sla_met": _coerce_bool(row.get("sla_met") or row.get("sla_target_met")),
            "quality_score": _coerce_int(row.get("quality_score")),
            "resolution_groups": row.get("resolution_groups") or [],
        })

    logger.info(
        "[pattern_analytics] SQL OR-match: topic=%r terms=%s found=%d tickets",
        topic, terms, len(tickets),
    )
    return tickets


def _load_similar_tickets_legacy(
    topic: str,
    organization_id: str,
    max_results: int = 50,
) -> List[Dict[str, Any]]:
    """
    Legacy ILIKE-OR-on-chunk-content path preserved verbatim for rollback.
    Invoked when PATTERN_ANALYTICS_SQL_OR_MATCHING_ENABLED is False or
    the new primary path raises.
    """
    terms = [t for t in (topic or "").replace("_", " ").split() if t]
    if not terms:
        return []

    # Build ILIKE predicate per term, OR'd together. Named params keep it
    # SQLAlchemy-text compatible. Cap terms to avoid pathological SQL length.
    terms = terms[:5]
    like_clauses: List[str] = []
    params: Dict[str, Any] = {
        "oid": organization_id,
        "limit": int(max_results),
    }
    for i, term in enumerate(terms):
        key = f"term{i}"
        like_clauses.append(f"c.content ILIKE :{key}")
        params[key] = f"%{term}%"

    where_terms = " OR ".join(like_clauses)

    sql = f"""
        SELECT DISTINCT ON (d.id)
            d.id::text                                AS document_id,
            c.metadata_json->>'primary_id'            AS primary_id,
            c.metadata_json->>'incident_number'       AS incident_number,
            c.metadata_json->>'opened_date'           AS opened_date,
            c.metadata_json->>'resolution_text'       AS resolution_text,
            c.metadata_json->>'resolution'            AS resolution_alt,
            c.metadata_json->>'sla_met'               AS sla_met,
            c.metadata_json->>'sla_target_met'        AS sla_target_met,
            c.metadata_json->>'quality_score'         AS quality_score,
            c.metadata_json->'resolution_groups'      AS resolution_groups
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE d.owner_id = :oid
          AND d.status = 'active'
          AND ({where_terms})
          AND c.metadata_json IS NOT NULL
          AND COALESCE(
              c.metadata_json->>'primary_id',
              c.metadata_json->>'incident_number'
          ) IS NOT NULL
        LIMIT :limit
    """

    try:
        with engine.connect() as conn:
            rows = conn.execute(text(sql), params).mappings().all()
    except Exception as exc:
        logger.warning(
            "[queries] load_similar_tickets_for_topic failed (topic=%r org=%s): %s",
            topic, organization_id, exc,
        )
        return []

    tickets: List[Dict[str, Any]] = []
    for row in rows:
        ticket_id = row.get("primary_id") or row.get("incident_number")
        if not ticket_id:
            continue
        tickets.append({
            "ticket_id": str(ticket_id).strip(),
            "opened_date": _parse_opened_date(row.get("opened_date")),
            "resolution_text": (
                row.get("resolution_text") or row.get("resolution_alt") or ""
            ),
            "sla_met": _coerce_bool(row.get("sla_met") or row.get("sla_target_met")),
            "quality_score": _coerce_int(row.get("quality_score")),
            "resolution_groups": row.get("resolution_groups") or [],
        })
    return tickets
