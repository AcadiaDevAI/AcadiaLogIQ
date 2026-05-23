"""
Ticket Filter — FastAPI router.

Endpoint
--------
``POST /ticket-filter`` (Clerk-auth required, per-user rate-limited)

Request
~~~~~~~
::

    {
        "sla_target_met": "True" | "False",
        "resolution_quality_score": "1" | "2" | "3" | "4" | "5",
        "page": 1,            # 1-indexed; default 1
        "page_size": 20       # default 20, max 50
    }

Both filter fields are REQUIRED — the frontend disables the Submit
button until the user selects both. Validation at the Pydantic
layer enforces the same contract server-side so a misbehaving
client can't smuggle nulls through.

Response (HTTP 200)
~~~~~~~~~~~~~~~~~~~
::

    {
        "tickets":   [ TicketSummary, ... ],
        "total":     <distinct-incident count across all pages>,
        "page":      1,
        "page_size": 20,
        "has_more":  bool
    }

Why this is its own router
--------------------------
Per the explicit ask: this feature must be fully independent of
RCA / Gap Analysis / Chat. It shares no imports with those modules,
owns its own SQL helpers, and lives at its own URL prefix. A
regression in any other flow cannot break this; a bug here cannot
break them.
"""

import logging
from typing import List, Literal, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from backend._lazy_auth import lazy_auth_dependency
from backend.observability.rate_limit import limiter

from .query import filter_tickets, MAX_PAGE_SIZE


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/ticket-filter", tags=["ticket-filter"])


# ── Request / response models ───────────────────────────────────


class TicketFilterRequest(BaseModel):
    """Required-both filter inputs.

    Both fields are typed as ``Literal`` so Pydantic rejects any
    out-of-band value (e.g. ``"yes"`` instead of ``"True"``) with a
    422 — matches the dropdown options the frontend exposes.
    """
    sla_target_met: Literal["True", "False"]
    resolution_quality_score: Literal["1", "2", "3", "4", "5"]
    page: int = Field(default=1, ge=1, le=10_000)
    page_size: int = Field(default=20, ge=1, le=MAX_PAGE_SIZE)
    model_config = ConfigDict(extra="ignore")


class TicketSummaryOut(BaseModel):
    """Wire shape for one ticket row. Mirrors the dataclass in
    ``query.py`` so the route layer is a thin pass-through.

    ``incident`` is the short narrative the UI surfaces under the
    tags — sourced from ``Incident_Summary.INCIDENT`` in the
    ticket schema. ``customer_name`` and ``timestamp`` are kept on
    the wire (backward compatibility / future admin views) but the
    current UI does NOT render them.
    """
    incident_number: str
    customer_name: Optional[str] = None
    priority: Optional[str] = None
    ticket_status: Optional[str] = None
    timestamp: Optional[str] = None
    sla_target_met: Optional[str] = None
    resolution_quality_score: Optional[str] = None
    incident: Optional[str] = None


class TicketFilterResponse(BaseModel):
    tickets: List[TicketSummaryOut]
    total: int
    page: int
    page_size: int
    has_more: bool


# ── Route ──────────────────────────────────────────────────────


@router.post("", response_model=TicketFilterResponse)
# Per-user rate limit. Conservative cap — filter queries are cheap
# in absolute terms but a runaway client could otherwise mash the
# Generate button into a thousand-request burst. 30/min matches the
# generosity of /tier1/journey while still bounding worst case.
@limiter.limit("30/minute")
async def filter_tickets_route(
    request: Request,
    payload: TicketFilterRequest = Body(...),
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> TicketFilterResponse:
    """Return the page of tickets matching the filter pair.

    Empty result set is NOT an error — returns an empty
    ``tickets`` list with ``total=0`` so the UI can render the
    "No tickets match these filters" empty state without a 404.
    """
    try:
        result = filter_tickets(
            sla_target_met=payload.sla_target_met,
            resolution_quality_score=payload.resolution_quality_score,
            page=payload.page,
            page_size=payload.page_size,
        )
    except Exception as exc:
        # We deliberately do NOT swallow DB errors here — a silent
        # empty response would masquerade as "no matches" and the
        # operator would never know the index / pool / query was
        # broken. Surface as 500 so Sentry catches it.
        logger.warning(
            "[ticket_filter] query failed sla=%s score=%s err=%s",
            payload.sla_target_met, payload.resolution_quality_score, exc,
        )
        raise HTTPException(status_code=500, detail="ticket_filter_query_failed")

    return TicketFilterResponse(
        tickets=[
            TicketSummaryOut(
                incident_number=t.incident_number,
                customer_name=t.customer_name,
                priority=t.priority,
                ticket_status=t.ticket_status,
                timestamp=t.timestamp,
                sla_target_met=t.sla_target_met,
                resolution_quality_score=t.resolution_quality_score,
                incident=t.incident,
            )
            for t in result.tickets
        ],
        total=result.total,
        page=result.page,
        page_size=result.page_size,
        has_more=result.has_more,
    )


# ─────────────────────────────────────────────────────────────────────
# Optional ServiceNow integration — added as a separate action so the
# existing filter route above is untouched. Credentials live in
# backend/env.bvk (loaded lazily inside the connector module) or in
# AWS Secrets Manager. Tighter rate limit than the standard filter
# because each call hits an external API.
# ─────────────────────────────────────────────────────────────────────


@router.get("/servicenow")
@limiter.limit("10/minute")
async def fetch_servicenow_incidents_route(
    request: Request,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
):
    """Fetch priority-1 incidents from ServiceNow and return parsed JSON.

    Response shape (envelope is stable regardless of how many
    incidents come back)::

        {
            "source":       "servicenow",
            "instance_url": "https://dev....service-now.com",
            "query":        "priority=1",
            "count":        <int>,
            "incidents":    [ {...}, ... ],
            "raw":          {...},   # full nested dict for debugging
        }

    Status codes
    ------------
    200 — success (``count`` may be 0)
    401 — Clerk JWT missing/invalid (standard auth contract)
    429 — per-user rate limit exceeded
    502 — ServiceNow unreachable / non-2xx / unparseable XML
    503 — required SERVICE_NOW_* env vars not configured
    """
    # Imported lazily so a misconfigured ServiceNow module never blocks
    # the rest of the ticket-filter router from mounting at app startup.
    try:
        from .servicenow import (
            fetch_priority_1_incidents,
            ServiceNowConfigError,
            ServiceNowAPIError,
        )
    except Exception as exc:  # pragma: no cover — import-time defensiveness
        logger.warning("[ticket_filter] servicenow module import failed: %s", exc)
        raise HTTPException(status_code=500, detail="servicenow_module_unavailable")

    # External HTTP call is blocking — push to a worker thread so the
    # asyncio event loop stays responsive while waiting on ServiceNow.
    try:
        from anyio import to_thread
        payload = await to_thread.run_sync(fetch_priority_1_incidents)
    except ServiceNowConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ServiceNowAPIError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("[ticket_filter] servicenow call failed: %s", exc)
        raise HTTPException(status_code=500, detail="servicenow_call_failed")

    return payload
