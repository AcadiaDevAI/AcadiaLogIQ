"""Sprint 10 — Cohort metadata loader.

`load_cohort_metadata(session_id) -> list[dict]` reads the existing
`tier1_sessions.top_5_match_ids` set and fetches each ticket's
`metadata_json` from `chunks`. Single round-trip (`WHERE id = ANY(:ids)`)
plus a single `tier1_sessions` lookup. Returns `[]` on any failure with
a logged warning — never raises.

Order is preserved against the session's stored rank list so callers
can rely on `result[i]` being the rank-(i+1) ticket.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy import text


logger = logging.getLogger("acadia-log-iq")


def load_cohort_metadata(session_id: str) -> List[Dict[str, Any]]:
    """Return the cohort's ticket metadata_json payloads in rank order.

    Failure modes — all return `[]`:
      - Empty / missing session_id
      - Session row not found
      - Empty `top_5_match_ids`
      - DB error (logged)
    """
    if not session_id:
        return []

    try:
        from backend.db.connection import engine
    except Exception as exc:
        logger.warning("[journey.loader] DB engine import failed: %s", exc)
        return []

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT top_5_match_ids FROM tier1_sessions WHERE id = :id"
                ),
                {"id": session_id},
            ).mappings().first()
    except Exception as exc:
        logger.warning("[journey.loader] session lookup failed: %s", exc)
        return []

    if not row:
        return []

    ids = list(row.get("top_5_match_ids") or [])
    if not ids:
        return []

    try:
        # NOTE — must NOT use bindparam(..., expanding=True) here: that
        # would expand the list into positional ($1, $2, ...) params,
        # which PostgreSQL's ANY() rejects ("op ANY/ALL (array) requires
        # array on right side"). Pass the list as a single bind value;
        # psycopg binds Python lists as PG `text[]` arrays automatically,
        # which is exactly what `ANY(:ids)` expects.
        sql = text("SELECT id, metadata_json FROM chunks WHERE id = ANY(:ids)")
        with engine.connect() as conn:
            rows = conn.execute(sql, {"ids": list(ids)}).mappings().all()
    except Exception as exc:
        logger.warning("[journey.loader] chunk fetch failed: %s", exc)
        return []

    by_id: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        meta = r.get("metadata_json")
        if isinstance(meta, dict):
            by_id[str(r["id"])] = meta

    # Preserve the session's rank order; drop any ids not found in chunks.
    return [by_id[i] for i in ids if i in by_id]
