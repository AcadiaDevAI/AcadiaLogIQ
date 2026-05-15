"""Sprint 13.32 — Ticket lookup by Incident_Number.

The ingested per-ticket JSON lives in ``chunks.metadata_json`` (JSONB).
The Stage 3 cohort harvester already reads from this column; we reuse
the same store here, looking up a single ticket by its
``Metadata.Incident_Number`` field instead of by ``chunks.id``.

Failure-open: any DB / shape error returns ``None`` with a logged
warning. The route translates ``None`` into a 404 for the frontend.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import text


logger = logging.getLogger("acadia-log-iq")


def find_ticket_by_incident_number(incident_number: str) -> Optional[Dict[str, Any]]:
    """Return the ticket's full ``metadata_json`` dict, or ``None`` when
    no chunk row carries that ``Incident_Number``.

    Args:
        incident_number: The Metadata.Incident_Number value entered by
            the engineer (e.g. ``"INC-LAN-88902"``). Whitespace-only
            input returns None without a DB hit.
    """
    inc = (incident_number or "").strip()
    if not inc:
        return None

    try:
        from backend.db.connection import engine
    except Exception as exc:
        logger.warning("[rca.lookup] DB engine import failed: %s", exc)
        return None

    # The JSONB path is `metadata_json -> 'Metadata' ->> 'Incident_Number'`.
    # We pull the most recent matching row (ORDER BY created_at DESC NULLS
    # LAST) so re-ingestions of the same ticket surface the latest copy.
    # LIMIT 1 because Incident_Number is logically unique per ticket.
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
        # Fall back to a query without ORDER BY in case the schema in
        # this deployment doesn't carry a created_at column on chunks
        # (older installs). One retry only — anything else is logged
        # and we return None.
        logger.info(
            "[rca.lookup] primary query failed (%s) — retrying without "
            "created_at ordering", exc,
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
            logger.warning("[rca.lookup] fallback query failed: %s", exc2)
            return None

    if not row:
        return None
    meta = row.get("metadata_json")
    return meta if isinstance(meta, dict) else None
