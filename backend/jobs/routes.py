"""
``GET /jobs/{job_id}`` — status polling for async report generations.

The frontend hits this every 2 s after receiving a 202 from
``POST /rca/{inc}`` or ``POST /gap-analysis/{inc}``. The response
shape is intentionally small (no markdown) — once ``status="done"``
the frontend re-POSTs to ``/rca/{inc}`` (or the gap-analysis route)
which now returns 200 directly from cache.

Auth: same Clerk-required gate as the rest of the protected routes.
This stops a stranger from polling a job_id they shouldn't see.
We do NOT cross-reference ``requested_by`` here — multiple engineers
can legitimately watch the same job (it's a shared product) and
that ACL gate would just add friction without a meaningful security
benefit. The job_id itself is a UUIDv4, unguessable.

Why a separate router (not piggybacked on /rca or /gap-analysis):
* Job kinds may grow in the future — keeping the polling endpoint
  generic at ``/jobs`` means no rewrites when we add new long-running
  endpoints (PDF generation, ingestion progress, etc.).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from backend._lazy_auth import lazy_auth_dependency
from .queue import get_job


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/jobs", tags=["jobs"])


class JobStatusResponse(BaseModel):
    """Wire shape returned by GET /jobs/{id}.

    Field naming mirrors what the frontend's polling loop needs:
      * ``status``     — pending / running / done / failed / cancelled
      * ``attempts``   — current attempt (≥1 for running/done, ≥0 for pending)
      * ``max_attempts`` — total budget; helps UI show "attempt 2 of 3"
      * ``result_kind`` / ``result_incident`` — present when done; the
        frontend uses these as the lookup key when re-fetching the
        cached markdown.
      * ``error_message`` — present when failed; surfaced to the UI.
    """
    id: str
    kind: str
    incident_number: str
    status: str
    attempts: int
    max_attempts: int
    result_kind: Optional[str] = None
    result_incident: Optional[str] = None
    error_message: Optional[str] = None


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> JobStatusResponse:
    """Return the current status of a single async job.

    404 when the id doesn't exist — covers both the "typo" case and
    the "row was deleted by the 30-day retention sweeper" case. The
    frontend distinguishes them by surfacing a fresh "regenerate"
    prompt for 404s.
    """
    jid = (job_id or "").strip()
    if not jid:
        raise HTTPException(status_code=400, detail="job_id_required")

    row: Optional[Dict[str, Any]] = get_job(jid)
    if not row:
        raise HTTPException(status_code=404, detail="job_not_found")

    return JobStatusResponse(
        id=str(row["id"]),
        kind=str(row["kind"]),
        incident_number=str(row["incident_number"]),
        status=str(row["status"]),
        attempts=int(row.get("attempts") or 0),
        max_attempts=int(row.get("max_attempts") or 0),
        result_kind=row.get("result_kind"),
        result_incident=row.get("result_incident"),
        error_message=row.get("error_message"),
    )
