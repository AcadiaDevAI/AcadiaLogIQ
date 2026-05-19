"""
Gap Analysis — FastAPI router.

Endpoint
--------
``POST /gap-analysis/{incident_number}``  (Clerk-auth required)

    200 → {
        incident_number,
        gap_analysis_md,             # LogIQ Gap Analysis report
        post_mortem_md,              # Blameless SRE post-mortem
        gap_analysis_error,          # str | null per-panel failure
        post_mortem_error,           # str | null per-panel failure
    }
    400 → empty / whitespace incident number
    404 → no ticket found with that incident number
    500 → serialisation failure (rare)

Behaviour
---------
Same parallel-LLM pattern proven in RCA:

    asyncio.gather(
        to_thread(_call_llm, GAP_ANALYSIS_MASTER_PROMPT, ticket_json, ...),
        to_thread(_call_llm, BLAMELESS_POSTMORTEM_PROMPT, ticket_json, ...),
        return_exceptions=True,
    )

Both reports return whether the other one succeeds or fails. The
frontend renders both panels and shows an inline warning on the
failed side instead of blocking the whole response. The two LLM
calls run on threadpool workers so the wall-clock latency is roughly
``max(call_a, call_b)`` rather than ``call_a + call_b``.

Why a separate router (and not adding routes to RCA's)
------------------------------------------------------
Per the design brief: Gap Analysis is a fully independent feature.
A change to RCA's routes (model, prompts, error handling, response
shape) must not silently change Gap Analysis output and vice versa.
The fork costs a small amount of duplication and buys us regression
isolation forever.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field

from backend._lazy_auth import lazy_auth_dependency
from backend.observability.rate_limit import limiter
from backend.tier1_copilot._shared.report_cache import (
    FEEDBACK_DISLIKE,
    REPORT_KIND_GAP_ANALYSIS,
    REPORT_KIND_POST_MORTEM,
    get_cached_report,
    invalidate_cached_report,
    record_feedback,
    save_cached_report,
)
from backend.jobs import enqueue as enqueue_job

from .bedrock_claude import invoke as claude_invoke
from .prompts import (
    BLAMELESS_POSTMORTEM_PROMPT,
    GAP_ANALYSIS_MASTER_PROMPT,
)
from .ticket_lookup import find_ticket_by_incident_number


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/gap-analysis", tags=["gap-analysis"])


# ─────────────────────────────────────────────────────────────────────
# Token budgets.
#
# Both reports are long-form. The Gap Analysis Master Prompt produces
# the largest output (Executive Summary + 4 detailed sections + heat
# map + tiered action plan + data-quality flags + report
# certification) and was hitting ``stop_reason=max_tokens`` at the
# previous 8192 cap, so we raise the Gap Analysis budget to 16384.
# Claude Haiku 4.5's on-demand inference profile allows well above
# this; we keep a deliberate safety margin under the model's hard
# ceiling.
#
# The Blameless Post-Mortem (13 sections + KPI scorecard) tends to
# fit comfortably inside 8192 because section-omission-on-missing-
# data is explicit in its rules, so we leave its budget alone to
# keep typical latency low. If logs ever show its calls truncating
# (``stop=max_tokens``), bump it too.
# ─────────────────────────────────────────────────────────────────────
_MAX_TOKENS_GAP_ANALYSIS = 16384
_MAX_TOKENS_POST_MORTEM = 8192


class GapAnalysisResponse(BaseModel):
    """Stable response shape for the frontend.

    Per-panel errors are nullable strings: when both calls succeed,
    both error fields are ``None``; when one call raises, the other
    panel's markdown is still returned and only its sibling's error
    field is populated. Keeps the UI simple and resilient.

    ``*_cached`` flags tell the frontend whether the panel's
    Markdown came out of ``report_cache`` (no LLM call) vs. a
    fresh generation. The UI surfaces a small "♻ Cached" tag so
    the engineer can tell at a glance.
    """

    incident_number: str
    gap_analysis_md: str
    post_mortem_md: str
    gap_analysis_error: Optional[str] = None
    post_mortem_error: Optional[str] = None
    gap_analysis_cached: bool = False
    post_mortem_cached: bool = False


class GapAnalysisGenerateRequest(BaseModel):
    """Optional request body — backward-compatible. Clients posting
    an empty body get the default behaviour (cache lookup first,
    LLM only on miss)."""

    regenerate_gap_analysis: bool = False
    regenerate_post_mortem: bool = False
    model_config = ConfigDict(extra="ignore")


class GapAnalysisFeedbackRequest(BaseModel):
    """POST body for the feedback endpoint.

    ``panel`` must be ``"gap_analysis"`` or ``"post_mortem"`` so
    the handler knows which cache row to invalidate when
    ``feedback_type`` is ``"dislike"``.
    """
    feedback_type: str = Field(pattern="^(like|dislike)$")
    panel: str = Field(pattern="^(gap_analysis|post_mortem)$")
    model_config = ConfigDict(extra="ignore")


class GapAnalysisFeedbackResponse(BaseModel):
    ok: bool
    invalidated: bool = False


# ─────────────────────────────────────────────────────────────────────
# Prompt-assembly helpers.
#
# Both prompts end with a "INPUT — INCIDENT JSON FOLLOWS BELOW" marker
# block. We append the ticket JSON immediately after, plus an
# explicit "END JSON" sentinel and a final instruction so the model
# is unambiguous about where the data ends and where to start
# writing the report.
# ─────────────────────────────────────────────────────────────────────


def _build_prompt(template: str, ticket_json: str) -> str:
    """Attach the ticket JSON onto a prompt template.

    The model receives one self-contained user-turn payload:

        <prompt template, including STRICT rules + structure>
        <ticket JSON>
        --- END JSON ---
        Generate the document now.
    """
    return (
        f"{template}\n"
        f"{ticket_json}\n"
        f"--- END JSON ---\n\n"
        f"Generate the document now."
    )


def _call_llm(template: str, ticket_json: str, max_tokens: int) -> str:
    """Synchronous LLM call routed through the dedicated Bedrock
    Claude caller for Gap Analysis. Wrapped in ``asyncio.to_thread``
    by the route handler so two prompts run on threadpool workers in
    parallel.
    """
    prompt = _build_prompt(template, ticket_json)
    return claude_invoke(prompt, max_tokens)


# ─────────────────────────────────────────────────────────────────────
# Route handler.
# ─────────────────────────────────────────────────────────────────────


# ── Cache helper ────────────────────────────────────────────────
# Cache-aware wrapper around ``_call_llm``. Returns
# (markdown, came_from_cache). When ``regenerate=True``, skips the
# cache read and forces a fresh LLM call (used by the "Regenerate"
# button on each panel).
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
                "[gap_analysis] cache hit kind=%s inc=%s chars=%d",
                report_kind, incident_number, len(cached),
            )
            return cached, True
    md = _call_llm(template, ticket_json, max_tokens)
    if md and md.strip():
        save_cached_report(report_kind, incident_number, md.strip())
    return md, False


class GapAnalysisAsyncResponse(BaseModel):
    """202 response when one or both Gap Analysis panels are enqueued."""
    incident_number: str
    gap_analysis_cached: bool = False
    post_mortem_cached: bool = False
    gap_analysis_md: Optional[str] = None
    post_mortem_md: Optional[str] = None
    gap_analysis_job_id: Optional[str] = None
    post_mortem_job_id: Optional[str] = None


@router.post(
    "/{incident_number}",
    response_model=None,
    responses={
        200: {"model": GapAnalysisResponse, "description": "Both panels served from cache"},
        202: {"model": GapAnalysisAsyncResponse, "description": "One or both panels enqueued — poll /jobs/{id}"},
    },
)
# Phase 4 — per-user rate limit, 5/min on Gap Analysis Generate.
# Same shared limiter the RCA route uses; per-user keying means a
# busy engineer doesn't lock out the rest of the team.
@limiter.limit("5/minute")
async def generate_gap_analysis(
    request: Request,
    incident_number: str,
    response: Response,
    payload: Optional[GapAnalysisGenerateRequest] = None,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
):
    """Look up the ticket by ``Incident_Number`` and return both
    Gap Analysis panels.

    Cache semantics
    ---------------
    For each panel, first check ``report_cache`` keyed on
    ``(report_kind, incident_number)``. On hit, return the cached
    Markdown without invoking the LLM. On miss (or when the client
    requests ``regenerate_*=True``), run the LLM and UPSERT the
    result into the cache.

    A 👎 from the client elsewhere DELETES the cached row, so the
    NEXT call here treats it as a miss and produces fresh output.

    Per-panel failure-open: a failure on one panel never blocks
    the other.
    """
    """Generate Gap Analysis + Blameless Post-Mortem with the fast-path
    / slow-path pattern.

    See ``backend/tier1_copilot/rca/routes.py::generate_rca`` for the
    full design rationale — same pattern: return 200 + payload when
    everything is cached, 202 + job_ids when one or both panels need
    a fresh LLM call.
    """
    inc = (incident_number or "").strip()
    if not inc:
        raise HTTPException(status_code=400, detail="incident_number_required")

    req = payload or GapAnalysisGenerateRequest()

    # Cache probe FIRST — fast path returns 200 directly with no
    # worker round-trip when nothing needs regenerating.
    gap_cached_md = (
        None
        if req.regenerate_gap_analysis
        else get_cached_report(REPORT_KIND_GAP_ANALYSIS, inc)
    )
    pm_cached_md = (
        None
        if req.regenerate_post_mortem
        else get_cached_report(REPORT_KIND_POST_MORTEM, inc)
    )

    if gap_cached_md and pm_cached_md:
        logger.info(
            "[gap_analysis] both panels cached inc=%s — returning 200 directly", inc,
        )
        return GapAnalysisResponse(
            incident_number=inc,
            gap_analysis_md=gap_cached_md,
            post_mortem_md=pm_cached_md,
            gap_analysis_error=None,
            post_mortem_error=None,
            gap_analysis_cached=True,
            post_mortem_cached=True,
        )

    # Slow path — at least one panel needs the LLM. Validate ticket
    # first so a typo returns 404 immediately regardless of execution
    # mode (sync vs worker).
    ticket = find_ticket_by_incident_number(inc)
    if ticket is None:
        logger.info("[gap_analysis] lookup miss inc=%s", inc)
        raise HTTPException(status_code=404, detail="ticket_not_found")

    # ── Local-dev / no-worker mode ──────────────────────────────
    # When ``REPORTS_VIA_WORKER=false`` (the default), the LLM calls
    # run in-process via the original ``_cached_or_llm`` +
    # ``asyncio.to_thread`` pattern. Identical to the pre-Phase-1
    # behaviour the local dev stack expects. Set
    # ``REPORTS_VIA_WORKER=true`` in production task definitions to
    # switch to the queue-based async path below.
    from backend.config import settings as _settings
    if not getattr(_settings, "REPORTS_VIA_WORKER", False):
        try:
            ticket_json = json.dumps(ticket, ensure_ascii=False, default=str)
        except Exception as exc:
            logger.warning("[gap_analysis] json.dumps failed inc=%s (%s)", inc, exc)
            raise HTTPException(status_code=500, detail="ticket_serialisation_failed")

        gap_task = asyncio.to_thread(
            _cached_or_llm,
            report_kind=REPORT_KIND_GAP_ANALYSIS,
            incident_number=inc,
            template=GAP_ANALYSIS_MASTER_PROMPT,
            ticket_json=ticket_json,
            max_tokens=_MAX_TOKENS_GAP_ANALYSIS,
            regenerate=req.regenerate_gap_analysis,
        )
        pm_task = asyncio.to_thread(
            _cached_or_llm,
            report_kind=REPORT_KIND_POST_MORTEM,
            incident_number=inc,
            template=BLAMELESS_POSTMORTEM_PROMPT,
            ticket_json=ticket_json,
            max_tokens=_MAX_TOKENS_POST_MORTEM,
            regenerate=req.regenerate_post_mortem,
        )
        gap_result, pm_result = await asyncio.gather(
            gap_task, pm_task, return_exceptions=True,
        )

        gap_md = ""
        gap_err: Optional[str] = None
        gap_cached = False
        if isinstance(gap_result, BaseException):
            logger.warning("[gap_analysis] Gap Analysis call failed inc=%s (%s)", inc, gap_result)
            gap_err = "Gap Analysis report generation failed — please retry."
        elif isinstance(gap_result, tuple) and len(gap_result) == 2:
            md, gap_cached = gap_result
            gap_md = (md or "").strip()
            if not gap_md:
                gap_err = "Gap Analysis report returned empty output."

        pm_md = ""
        pm_err: Optional[str] = None
        pm_cached = False
        if isinstance(pm_result, BaseException):
            logger.warning("[gap_analysis] Post-Mortem call failed inc=%s (%s)", inc, pm_result)
            pm_err = "Blameless Post-Mortem generation failed — please retry."
        elif isinstance(pm_result, tuple) and len(pm_result) == 2:
            md, pm_cached = pm_result
            pm_md = (md or "").strip()
            if not pm_md:
                pm_err = "Blameless Post-Mortem returned empty output."

        logger.info(
            "[gap_analysis] sync inc=%s gap_chars=%d cached=%s pm_chars=%d cached=%s "
            "gap_err=%s pm_err=%s",
            inc, len(gap_md), gap_cached, len(pm_md), pm_cached,
            bool(gap_err), bool(pm_err),
        )
        return GapAnalysisResponse(
            incident_number=inc,
            gap_analysis_md=gap_md,
            post_mortem_md=pm_md,
            gap_analysis_error=gap_err,
            post_mortem_error=pm_err,
            gap_analysis_cached=gap_cached,
            post_mortem_cached=pm_cached,
        )

    # ── Worker / async mode (REPORTS_VIA_WORKER=true) ───────────
    gap_job_id: Optional[str] = None
    pm_job_id: Optional[str] = None

    if gap_cached_md is None:
        try:
            gap_job_id = enqueue_job(
                kind=REPORT_KIND_GAP_ANALYSIS,
                incident_number=inc,
                requested_by=user_id,
                payload={"regenerate": bool(req.regenerate_gap_analysis)},
            )
        except Exception as exc:
            logger.warning("[gap_analysis] enqueue gap failed inc=%s (%s)", inc, exc)
            raise HTTPException(status_code=503, detail="job_queue_unavailable")

    if pm_cached_md is None:
        try:
            pm_job_id = enqueue_job(
                kind=REPORT_KIND_POST_MORTEM,
                incident_number=inc,
                requested_by=user_id,
                payload={"regenerate": bool(req.regenerate_post_mortem)},
            )
        except Exception as exc:
            logger.warning("[gap_analysis] enqueue pm failed inc=%s (%s)", inc, exc)
            raise HTTPException(status_code=503, detail="job_queue_unavailable")

    logger.info(
        "[gap_analysis] 202 inc=%s gap_job=%s pm_job=%s "
        "gap_cached=%s pm_cached=%s",
        inc, gap_job_id, pm_job_id,
        bool(gap_cached_md), bool(pm_cached_md),
    )
    response.status_code = status.HTTP_202_ACCEPTED
    return GapAnalysisAsyncResponse(
        incident_number=inc,
        gap_analysis_cached=bool(gap_cached_md),
        post_mortem_cached=bool(pm_cached_md),
        gap_analysis_md=gap_cached_md,
        post_mortem_md=pm_cached_md,
        gap_analysis_job_id=gap_job_id,
        post_mortem_job_id=pm_job_id,
    )


# ── Feedback endpoint ──────────────────────────────────────────
# 👍 records a like in report_feedback, leaves the cache alone.
# 👎 records the dislike AND deletes the cached row so the next
#    generate runs a fresh LLM.
@router.post(
    "/{incident_number}/feedback",
    response_model=GapAnalysisFeedbackResponse,
)
async def record_gap_analysis_feedback(
    incident_number: str,
    payload: GapAnalysisFeedbackRequest,
    user_id: Optional[str] = Depends(lazy_auth_dependency),
) -> GapAnalysisFeedbackResponse:
    inc = (incident_number or "").strip()
    if not inc:
        raise HTTPException(status_code=400, detail="incident_number_required")

    kind = (
        REPORT_KIND_GAP_ANALYSIS
        if payload.panel == "gap_analysis"
        else REPORT_KIND_POST_MORTEM
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

    return GapAnalysisFeedbackResponse(ok=ok, invalidated=invalidated)
