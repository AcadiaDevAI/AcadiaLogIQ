"""
SQL helper for the Ticket Filter route.

Pure function — no FastAPI imports, no HTTP knowledge. The route
layer (``routes.py``) calls these helpers and shapes the response.
Keeping the SQL here means we can unit-test the query against a
live DB without spinning up the whole app.

Query shape
-----------
The corpus stores one ticket as N chunks (PDF pages, structured
JSON sections, etc). The same ``Incident_Number`` appears in many
rows. To return ONE row per ticket:

  1. Filter chunks by the (SLA, Score) pair AND active document
     version (so we don't surface tickets from deleted / superseded
     documents).
  2. Use ``ROW_NUMBER() OVER (PARTITION BY Incident_Number ORDER BY
     created_at DESC)`` to pick the most recently ingested chunk for
     each ticket.
  3. Filter to ``rn = 1`` and sort the final result by the ticket's
     ``Timestamp`` metadata (ISO 8601 string — sorts lexicographically
     in the order we want).
  4. Apply LIMIT/OFFSET for pagination.

The COUNT query uses ``count(DISTINCT ...)`` so the total reflects
unique incidents, matching what the user sees in the result list.

Production-scale notes
----------------------
* The composite expression index from migration 049
  (``idx_chunks_sla_score``) covers the WHERE clause directly.
* The route caps page_size at 50 to bound per-request work.
* Both queries use the same JOIN to ``documents`` /
  ``document_versions`` to honour the active-version filter — keeps
  results consistent with the rest of the retrieval stack.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# Hard cap on page size — bounds per-request work even if a future
# UI bug submits an absurd value. Same defense-in-depth pattern the
# rest of the codebase uses for pagination.
MAX_PAGE_SIZE = 50


@dataclass
class TicketSummary:
    """One row in the filter result list — the MVP display fields.

    ``incident`` holds the short narrative from
    ``metadata_json -> Incident_Summary -> INCIDENT`` (e.g. "User
    reported ERP application was extremely slow…"). It's the
    primary thing the engineer reads on the card.

    ``customer_name`` and ``timestamp`` are kept on the wire for
    backward compatibility / future admin views, but the current UI
    does NOT render them.
    """
    incident_number: str
    customer_name: Optional[str]
    priority: Optional[str]
    ticket_status: Optional[str]
    timestamp: Optional[str]
    sla_target_met: Optional[str]
    resolution_quality_score: Optional[str]
    incident: Optional[str]


@dataclass
class FilterResult:
    """Top-level response payload returned by the route."""
    tickets: List[TicketSummary]
    total: int            # distinct-incident count across all pages
    page: int
    page_size: int
    has_more: bool


def _engine():
    """Lazy-import the SQLAlchemy engine so this module stays
    importable without a live DB (unit tests, static analysis)."""
    from backend.db.connection import engine  # type: ignore
    return engine


# ─────────────────────────────────────────────────────────────────
# SQL — defined as module-level constants so SQLAlchemy's text
# cache amortises the parse cost across requests.
# ─────────────────────────────────────────────────────────────────

_SQL_COUNT = text(
    """
    SELECT count(DISTINCT c.metadata_json->'Metadata'->>'Incident_Number') AS total
      FROM chunks c
      JOIN documents d          ON d.id = c.document_id
      JOIN document_versions dv ON dv.id = c.document_version_id
     WHERE c.metadata_json->'Metadata'->>'SLA_Target_Met' = :sla
       AND c.metadata_json->'Metadata'->>'Resolution_Quality_Score' = :score
       AND c.metadata_json->'Metadata'->>'Incident_Number' IS NOT NULL
       AND d.status = 'active'
       AND dv.is_active = TRUE
    """
)

# Window-function dedup picks the most recently ingested chunk per
# Incident_Number. The outer SELECT sorts the deduped rows by the
# ticket's ``Timestamp`` metadata (ISO-8601 strings sort correctly
# lexicographically, so no explicit cast needed).
_SQL_PAGE = text(
    """
    WITH ranked AS (
        SELECT
            c.metadata_json->'Metadata'->>'Incident_Number'              AS incident_number,
            c.metadata_json->'Metadata'->>'customer_name'                AS customer_name,
            c.metadata_json->'Metadata'->>'priority'                     AS priority,
            c.metadata_json->'Metadata'->>'ticket_status'                AS ticket_status,
            c.metadata_json->'Metadata'->>'Timestamp'                    AS ticket_timestamp,
            c.metadata_json->'Metadata'->>'SLA_Target_Met'               AS sla_target_met,
            c.metadata_json->'Metadata'->>'Resolution_Quality_Score'     AS resolution_quality_score,
            -- Short narrative from the ticket schema's Incident_Summary
            -- block — surfaced on every card. Falls back to NULL when
            -- the path is missing (older ingests / non-ticket chunks
            -- that happened to match the filter).
            c.metadata_json->'Incident_Summary'->>'INCIDENT'             AS incident,
            ROW_NUMBER() OVER (
                PARTITION BY c.metadata_json->'Metadata'->>'Incident_Number'
                ORDER BY c.created_at DESC
            ) AS rn
        FROM chunks c
        JOIN documents d          ON d.id = c.document_id
        JOIN document_versions dv ON dv.id = c.document_version_id
        WHERE c.metadata_json->'Metadata'->>'SLA_Target_Met' = :sla
          AND c.metadata_json->'Metadata'->>'Resolution_Quality_Score' = :score
          AND c.metadata_json->'Metadata'->>'Incident_Number' IS NOT NULL
          AND d.status = 'active'
          AND dv.is_active = TRUE
    )
    SELECT incident_number,
           customer_name,
           priority,
           ticket_status,
           ticket_timestamp,
           sla_target_met,
           resolution_quality_score,
           incident
      FROM ranked
     WHERE rn = 1
     ORDER BY ticket_timestamp DESC NULLS LAST
     LIMIT :page_size OFFSET :offset
    """
)


def filter_tickets(
    *,
    sla_target_met: str,
    resolution_quality_score: str,
    page: int = 1,
    page_size: int = 20,
) -> FilterResult:
    """Run the filter + dedupe + paginate, return a FilterResult.

    Caller MUST pre-validate ``sla_target_met`` and
    ``resolution_quality_score`` against their allowed value lists
    (done at the Pydantic layer in ``routes.py``). This function
    trusts the inputs and parameterises them safely against the DB.

    Page numbers are 1-indexed (1, 2, 3...) because that matches
    what the UI shows the user. Internally we compute offset =
    (page - 1) * page_size.

    On DB error we re-raise — the route's exception handler turns
    that into a 5xx for the client. We DO NOT swallow here because
    a filter that silently returns "no results" on a DB error would
    masquerade as a successful empty match — exactly the kind of
    correctness bug we want to avoid.
    """
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), MAX_PAGE_SIZE))
    offset = (page - 1) * page_size

    params = {
        "sla": sla_target_met,
        "score": resolution_quality_score,
        "page_size": page_size,
        "offset": offset,
    }

    with _engine().connect() as conn:
        # Two round-trips (count + page). At our scale negligible;
        # if it ever matters we can collapse into one query with a
        # window function but the readability win isn't worth the
        # planner complexity yet.
        total = int(conn.execute(_SQL_COUNT, params).scalar() or 0)
        rows = conn.execute(_SQL_PAGE, params).mappings().all()

    tickets = [
        TicketSummary(
            incident_number=str(r["incident_number"]),
            customer_name=r.get("customer_name"),
            priority=r.get("priority"),
            ticket_status=r.get("ticket_status"),
            timestamp=r.get("ticket_timestamp"),
            sla_target_met=r.get("sla_target_met"),
            resolution_quality_score=r.get("resolution_quality_score"),
            incident=r.get("incident"),
        )
        for r in rows
    ]

    has_more = (offset + len(tickets)) < total
    logger.info(
        "[ticket_filter] sla=%s score=%s page=%d page_size=%d total=%d returned=%d",
        sla_target_met, resolution_quality_score, page, page_size, total, len(tickets),
    )
    return FilterResult(
        tickets=tickets,
        total=total,
        page=page,
        page_size=page_size,
        has_more=has_more,
    )
