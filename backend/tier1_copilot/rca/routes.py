"""Sprint 13.32 — RCA-from-incident-number FastAPI router.

POST /rca/{incident_number}
  → 200 {customer_facing_md, internal_md, incident_number}
  → 404 when no chunk row carries that Incident_Number

The two LLM calls run in parallel via asyncio.to_thread so a single
request returns in roughly the time of the slower of the two calls
rather than their sum. Failure-open per panel: if one LLM call
raises, the response still includes the other panel's markdown plus
a `*_error` flag so the frontend can show a per-panel error banner
without losing the other side.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from backend._lazy_auth import lazy_auth_dependency
from backend.observability.rate_limit import limiter
from backend.tier1_copilot._shared.report_cache import (
    FEEDBACK_DISLIKE,
    REPORT_KIND_RCA_CUSTOMER,
    REPORT_KIND_RCA_INTERNAL,
    get_cached_report,
    invalidate_cached_report,
    record_feedback,
    save_cached_report,
)
from backend.jobs import enqueue as enqueue_job
from .bedrock_claude import invoke as claude_invoke
from .prompts import CUSTOMER_FACING_PROMPT, INTERNAL_INCIDENT_PROMPT
from .ticket_lookup import find_ticket_by_incident_number


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/rca", tags=["rca"])


# Sprint 13.32.8 — Output budgets for Claude Haiku 4.5 on Bedrock.
# Customer-Facing is short (400-600 words ≈ 800 tokens), so 2048
# gives 2.5× headroom. Internal is the long pole (12 sections +
# tables + appendix ≈ 3000-5000 tokens) — 8000 leaves comfortable
# headroom against Haiku 4.5's published output ceiling. If you see
# `stop_reason=max_tokens` in the logs, the next step is either a
# longer-cap model (Sonnet 4.x family) or a prompt trim.
_MAX_TOKENS_CUSTOMER = 2048
_MAX_TOKENS_INTERNAL = 8000


class RCAResponse(BaseModel):
    incident_number: str
    customer_facing_md: str
    internal_md: str
    customer_facing_error: Optional[str] = None
    internal_error: Optional[str] = None
    # Cache-hit indicators per panel. The frontend uses these to
    # render a small "♻ Cached" tag so the engineer knows the
    # output wasn't freshly generated. Backward-compatible: legacy
    # clients that don't read these fields just ignore them.
    customer_facing_cached: bool = False
    internal_cached: bool = False


class RCAGenerateRequest(BaseModel):
    """Optional request body. Backward-compatible — clients that
    POST an empty body get default behaviour (cache lookup, fresh
    LLM only on cache miss)."""

    # When ``True``, bypass the cache for the matching panel and
    # force a fresh LLM call. Used by the "Regenerate" button in
    # each panel header. Default is two ``False`` flags — i.e.,
    # honour the cache.
    regenerate_customer: bool = False
    regenerate_internal: bool = False
    model_config = ConfigDict(extra="ignore")


class RCAFeedbackRequest(BaseModel):
    """POST body for the feedback endpoint.

    ``panel`` must be ``"customer_facing"`` or ``"internal"`` so
    we know which cache entry to invalidate when ``feedback_type``
    is ``"dislike"``.
    """
    feedback_type: str = Field(pattern="^(like|dislike)$")
    panel: str = Field(pattern="^(customer_facing|internal)$")
    model_config = ConfigDict(extra="ignore")


class RCAFeedbackResponse(BaseModel):
    ok: bool
    invalidated: bool = False


def _build_prompt(template: str, ticket_json: str) -> str:
    """Append the ticket JSON to a prompt template. The templates end
    with a paste-here marker; we tack the JSON + closing marker on so
    the model sees a clean block to read from.
    """
    return (
        f"{template}\n"
        f"{ticket_json}\n"
        f"--- END JSON ---\n\n"
        f"Generate the document now."
    )


def _call_llm(template: str, ticket_json: str, max_tokens: int) -> str:
    """Synchronous LLM call — routed through Claude on Bedrock.

    Sprint 13.32.7 — Switched off `safe_generate` (Mistral 7B) and
    onto our dedicated Claude Haiku caller. Mistral was leaking
    interface IDs / CLI / fabricated contact info despite the
    prompt's STRIP rules; Claude follows the structure reliably.
    The other LLM-touching code paths in the app still go through
    Mistral via safe_generate — this swap is RCA-only.

    Claude Haiku has a 200k context window so the full prompt + ticket
    JSON is sent verbatim; no client-side truncation, which means the
    model sees every field the prompt's extraction map references.
    """
    prompt = _build_prompt(template, ticket_json)
    return claude_invoke(prompt, max_tokens)


# ── Cache helper ────────────────────────────────────────────────
# Decides per-panel: serve cached row, or fall through to LLM.
# ``regenerate=True`` from the client forces the fall-through.
# Returns (markdown, came_from_cache).
def _cached_or_llm(
    *,
    report_kind: str,
    incident_number: str,
    template: str,
    ticket_json: str,
    max_tokens: int,
    regenerate: bool,
) -> tuple[str, bool]:
    if not regenerate:
        cached = get_cached_report(report_kind, incident_number)
        if cached:
            logger.info(
                "[rca] cache hit kind=%s inc=%s chars=%d",
                report_kind, incident_number, len(cached),
            )
            return cached, True
    md = _call_llm(template, ticket_json, max_tokens)
    if md and md.strip():
        save_cached_report(report_kind, incident_number, md.strip())
    return md, False


class RCAAsyncResponse(BaseModel):
    """202 response when one or both panels need a fresh LLM run.

    The frontend uses ``status_code == 202`` (or the presence of
    ``customer_job_id`` / ``internal_job_id`` in the body) to know
    it should switch to polling mode. When BOTH panels are already
    cached, the route returns the legacy ``RCAResponse`` with
    ``status_code == 200`` so cached requests stay fast and the
    frontend's existing render path runs unchanged.
    """
    incident_number: str
    customer_facing_cached: bool = False
    internal_cached: bool = False
    # Markdown for the panels that were already cached. Frontend can
    # render them immediately while polling the other(s).
    customer_facing_md: Optional[str] = None
    internal_md: Optional[str] = None
    # Job IDs for panels that had to be enqueued. The frontend polls
    # GET /jobs/{id} on each, then GETs /rca/{inc} for the markdown
    # once status flips to ``done``.
    customer_job_id: Optional[str] = None
    internal_job_id: Optional[str] = None


@router.post(
    "/{incident_number}",
    # Default response model is the legacy sync shape. The 202 case
    # returns RCAAsyncResponse via a manual ``Response`` so the OpenAPI
    # schema stays clean for both paths.
    response_model=None,
    responses={
        200: {"model": RCAResponse, "description": "All panels served from cache"},
        202: {"model": RCAAsyncResponse, "description": "One or both panels enqueued — poll /jobs/{id}"},
    },
)
# Phase 4 — per-user rate limit. The shared limiter (keyed on Clerk
# user_id with IP fallback) enforces 5 Generate requests per minute
# per user. A noisy engineer can't burn the Bedrock quota for the
# rest of the team. ``Request`` parameter is required by slowapi to
# resolve the key function on each call.
@limiter.limit("5/minute")
async def generate_rca(
    request: Request,
    incident_number: str,
    response: Response,
    payload: Optional[RCAGenerateRequest] = None,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
):
    """Look up the ticket by Incident_Number and return both RCA
    panels.

    Cache semantics
    ---------------
    For each panel, first try ``report_cache`` keyed on
    ``(report_kind, incident_number)``. On hit, return the cached
    Markdown without invoking the LLM. On miss (or when the
    client requests ``regenerate_*=True``), run the LLM and UPSERT
    the result into the cache.

    A 👎 from the client elsewhere (see ``record_rca_feedback``
    below) DELETES the cached row, so the NEXT call here treats
    it as a cache miss and produces fresh output.

    Per-panel failure-open: a failure on one panel never blocks
    the other.
    """
    """Generate RCA reports with the fast-path / slow-path pattern.

    Phase 1 (async jobs) rewrite — preserves backward-compatible
    response shape on the cache-hit path so existing frontends keep
    working unchanged, and switches to an HTTP-202 + job-id response
    when one or both panels need a fresh LLM call. The frontend
    branches on the response status code:

      * ``200`` → markdown is in the body; render immediately.
      * ``202`` → at least one panel is in flight; poll ``/jobs/{id}``
                  for each ``*_job_id``, then GET ``/rca/{inc}`` (with
                  a no-regenerate body) when all complete so the
                  cached markdown comes back as a 200.

    This decoupling lets long-running Bedrock calls finish inside a
    worker container without holding an HTTP request open. The
    nominal 4-min Gap Analysis (or 30-s RCA) no longer races the
    ALB / axios timeout.
    """
    inc = (incident_number or "").strip()
    if not inc:
        raise HTTPException(status_code=400, detail="incident_number_required")

    req = payload or RCAGenerateRequest()

    # Cache probe FIRST — if both panels are already cached and the
    # caller didn't ask to regenerate, we can return synchronously
    # without touching the job queue. This is the hot path.
    customer_cached_md = (
        None
        if req.regenerate_customer
        else get_cached_report(REPORT_KIND_RCA_CUSTOMER, inc)
    )
    internal_cached_md = (
        None
        if req.regenerate_internal
        else get_cached_report(REPORT_KIND_RCA_INTERNAL, inc)
    )

    if customer_cached_md and internal_cached_md:
        logger.info(
            "[rca] both panels cached inc=%s — returning 200 directly", inc,
        )
        return RCAResponse(
            incident_number=inc,
            customer_facing_md=customer_cached_md,
            internal_md=internal_cached_md,
            customer_facing_error=None,
            internal_error=None,
            customer_facing_cached=True,
            internal_cached=True,
        )

    # Slow path — at least one panel needs the LLM. Validate the
    # ticket exists FIRST so a typo'd incident gets a 404 immediately
    # regardless of which execution mode (sync vs worker) we use.
    ticket = find_ticket_by_incident_number(inc)
    if ticket is None:
        logger.info("[rca] lookup miss inc=%s", inc)
        raise HTTPException(status_code=404, detail="ticket_not_found")

    # ── Local-dev / no-worker mode ──────────────────────────────
    # When ``REPORTS_VIA_WORKER=false`` (the default), the LLM runs
    # in-process via the original ``_cached_or_llm`` + ``asyncio.to_thread``
    # pattern. Identical to the pre-Phase-1 behaviour — the response
    # is the legacy ``RCAResponse`` shape with full markdown.
    #
    # Set ``REPORTS_VIA_WORKER=true`` in production (Fargate task
    # definition) to switch to the queue-based async path below.
    from backend.config import settings as _settings
    if not getattr(_settings, "REPORTS_VIA_WORKER", False):
        try:
            ticket_json = json.dumps(ticket, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("[rca] json.dumps failed inc=%s (%s)", inc, exc)
            raise HTTPException(status_code=500, detail="ticket_serialisation_failed")

        customer_task = asyncio.to_thread(
            _cached_or_llm,
            report_kind=REPORT_KIND_RCA_CUSTOMER,
            incident_number=inc,
            template=CUSTOMER_FACING_PROMPT,
            ticket_json=ticket_json,
            max_tokens=_MAX_TOKENS_CUSTOMER,
            regenerate=req.regenerate_customer,
        )
        internal_task = asyncio.to_thread(
            _cached_or_llm,
            report_kind=REPORT_KIND_RCA_INTERNAL,
            incident_number=inc,
            template=INTERNAL_INCIDENT_PROMPT,
            ticket_json=ticket_json,
            max_tokens=_MAX_TOKENS_INTERNAL,
            regenerate=req.regenerate_internal,
        )
        customer_result, internal_result = await asyncio.gather(
            customer_task, internal_task, return_exceptions=True,
        )

        customer_md = ""
        customer_err: Optional[str] = None
        customer_cached = False
        if isinstance(customer_result, BaseException):
            logger.warning("[rca] customer call failed inc=%s (%s)", inc, customer_result)
            customer_err = "Customer-Facing RCA generation failed — please retry."
        elif isinstance(customer_result, tuple) and len(customer_result) == 2:
            md, customer_cached = customer_result
            customer_md = (md or "").strip()
            if not customer_md:
                customer_err = "Customer-Facing RCA returned empty output."

        internal_md = ""
        internal_err: Optional[str] = None
        internal_cached = False
        if isinstance(internal_result, BaseException):
            logger.warning("[rca] internal call failed inc=%s (%s)", inc, internal_result)
            internal_err = "Internal Incident RCA generation failed — please retry."
        elif isinstance(internal_result, tuple) and len(internal_result) == 2:
            md, internal_cached = internal_result
            internal_md = (md or "").strip()
            if not internal_md:
                internal_err = "Internal Incident RCA returned empty output."

        logger.info(
            "[rca] sync inc=%s customer_chars=%d cached=%s internal_chars=%d cached=%s "
            "customer_err=%s internal_err=%s",
            inc, len(customer_md), customer_cached,
            len(internal_md), internal_cached,
            bool(customer_err), bool(internal_err),
        )
        return RCAResponse(
            incident_number=inc,
            customer_facing_md=customer_md,
            internal_md=internal_md,
            customer_facing_error=customer_err,
            internal_error=internal_err,
            customer_facing_cached=customer_cached,
            internal_cached=internal_cached,
        )

    # ── Worker / async mode (REPORTS_VIA_WORKER=true) ───────────
    # Enqueue per-panel jobs and return 202 + ids. Frontend polls
    # ``/jobs/{id}`` then re-POSTs once both panels are done.
    customer_job_id: Optional[str] = None
    internal_job_id: Optional[str] = None

    if customer_cached_md is None:
        try:
            customer_job_id = enqueue_job(
                kind=REPORT_KIND_RCA_CUSTOMER,
                incident_number=inc,
                requested_by=user_id,
                payload={"regenerate": bool(req.regenerate_customer)},
            )
        except Exception as exc:
            logger.warning("[rca] enqueue customer failed inc=%s (%s)", inc, exc)
            raise HTTPException(status_code=503, detail="job_queue_unavailable")

    if internal_cached_md is None:
        try:
            internal_job_id = enqueue_job(
                kind=REPORT_KIND_RCA_INTERNAL,
                incident_number=inc,
                requested_by=user_id,
                payload={"regenerate": bool(req.regenerate_internal)},
            )
        except Exception as exc:
            logger.warning("[rca] enqueue internal failed inc=%s (%s)", inc, exc)
            raise HTTPException(status_code=503, detail="job_queue_unavailable")

    logger.info(
        "[rca] 202 inc=%s customer_job=%s internal_job=%s "
        "customer_cached=%s internal_cached=%s",
        inc, customer_job_id, internal_job_id,
        bool(customer_cached_md), bool(internal_cached_md),
    )
    response.status_code = status.HTTP_202_ACCEPTED
    return RCAAsyncResponse(
        incident_number=inc,
        customer_facing_cached=bool(customer_cached_md),
        internal_cached=bool(internal_cached_md),
        customer_facing_md=customer_cached_md,
        internal_md=internal_cached_md,
        customer_job_id=customer_job_id,
        internal_job_id=internal_job_id,
    )


# ── Feedback endpoint ──────────────────────────────────────────
# 👍 records a like in report_feedback, leaves the cache alone.
# 👎 records the dislike AND deletes the cached row so the next
#    generate runs a fresh LLM.
# Per-panel — the body carries which panel the engineer is rating.
@router.post(
    "/{incident_number}/feedback",
    response_model=RCAFeedbackResponse,
)
async def record_rca_feedback(
    incident_number: str,
    payload: RCAFeedbackRequest,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> RCAFeedbackResponse:
    inc = (incident_number or "").strip()
    if not inc:
        raise HTTPException(status_code=400, detail="incident_number_required")

    kind = (
        REPORT_KIND_RCA_CUSTOMER
        if payload.panel == "customer_facing"
        else REPORT_KIND_RCA_INTERNAL
    )

    ok = record_feedback(
        report_kind=kind,
        incident_number=inc,
        feedback_type=payload.feedback_type,
        user_id=user_id,
    )

    invalidated = False
    if payload.feedback_type == FEEDBACK_DISLIKE:
        invalidated = invalidate_cached_report(kind, inc)

    return RCAFeedbackResponse(ok=ok, invalidated=invalidated)
