"""
Gap Analysis — ticket lookup by ``Metadata.Incident_Number``.

This module is a deliberate fork of ``rca/ticket_lookup.py``. It is
not imported from RCA on purpose: keeping the two features fully
isolated lets the RCA team change query shape, add caching, or
swap the chunks table without risking Gap Analysis (and vice versa).

Storage shape (unchanged from RCA today)
----------------------------------------
Ingested tickets live in ``chunks.metadata_json`` (JSONB). The
canonical lookup key for a single ticket is
``metadata_json -> 'Metadata' ->> 'Incident_Number'``.

Failure-open contract
---------------------
Any DB error returns ``None`` with a logged warning. The route layer
translates ``None`` into a 404 for the frontend.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import text


logger = logging.getLogger("acadia-log-iq")


def find_ticket_by_incident_number(incident_number: str) -> Optional[Dict[str, Any]]:
    """Return the ticket's full ``metadata_json`` dict, or ``None`` when
    no row in ``chunks`` carries that ``Incident_Number``.

    Whitespace-only input short-circuits to ``None`` without a DB hit.
    """
    inc = (incident_number or "").strip()
    if not inc:
        return None

    # Lazy DB import — keeps test environments without a live engine
    # importable. Mirrors the pattern used by ``rca/ticket_lookup``.
    try:
        from backend.db.connection import engine
    except Exception as exc:
        logger.warning("[gap_analysis.lookup] DB engine import failed: %s", exc)
        return None

    # Pull the most recently ingested copy of the ticket when multiple
    # rows match (re-ingestion case). ``LIMIT 1`` because the incident
    # number is logically unique per ticket.
    sql = text(
        """
        SELECT metadata_json
          FROM chunks
         WHERE metadata_json -> 'Metadata' ->> 'Incident_Number' = :inc
           AND metadata_json IS NOT NULL
         ORDER BY created_at DESC NULLS LAST
         LIMIT 1
        """
    )

    try:
        with engine.connect() as conn:
            row = conn.execute(sql, {"inc": inc}).mappings().first()
    except Exception as exc:
        # Fallback query without ORDER BY — older deployments may not
        # carry ``created_at`` on ``chunks``. Single retry only;
        # anything else is logged and we return None.
        logger.info(
            "[gap_analysis.lookup] primary query failed (%s) — retrying "
            "without created_at ordering", exc,
        )
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT metadata_json FROM chunks "
                        "WHERE metadata_json -> 'Metadata' ->> 'Incident_Number' = :inc "
                        "AND metadata_json IS NOT NULL LIMIT 1"
                    ),
                    {"inc": inc},
                ).mappings().first()
        except Exception as exc2:
            logger.warning("[gap_analysis.lookup] fallback query failed: %s", exc2)
            return None

    if not row:
        return None
    meta = row.get("metadata_json")
    return meta if isinstance(meta, dict) else None
