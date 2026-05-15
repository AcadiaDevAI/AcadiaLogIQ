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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .bedrock_claude import invoke as claude_invoke
from .prompts import CUSTOMER_FACING_PROMPT, INTERNAL_INCIDENT_PROMPT
from .ticket_lookup import find_ticket_by_incident_number


logger = logging.getLogger("acadia-log-iq")


router = APIRouter(prefix="/rca", tags=["rca"])


# Sprint 13.32.8 — Output budgets for Claude 3.5 Haiku on Bedrock.
# 3.5 Haiku caps server-side at 8192 output tokens. Customer-Facing
# is short (400-600 words ≈ 800 tokens), so 2048 gives 2.5× headroom.
# Internal is the long pole (12 sections + tables + appendix ≈
# 3000-5000 tokens) — 8000 leaves comfortable headroom and stays
# just under the 8192 server cap. If you see `stop_reason=max_tokens`
# in the logs, the next step is either a longer-cap model
# (anthropic.claude-3-5-sonnet-20241022-v2:0) or a prompt trim.
_MAX_TOKENS_CUSTOMER = 2048
_MAX_TOKENS_INTERNAL = 8000


class RCAResponse(BaseModel):
    incident_number: str
    customer_facing_md: str
    internal_md: str
    customer_facing_error: Optional[str] = None
    internal_error: Optional[str] = None


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


@router.post("/{incident_number}", response_model=RCAResponse)
async def generate_rca(incident_number: str) -> RCAResponse:
    """Look up the ticket by Incident_Number, run both LLM calls in
    parallel, return two Markdown blobs. Per-panel failure-open."""
    inc = (incident_number or "").strip()
    if not inc:
        raise HTTPException(status_code=400, detail="incident_number_required")

    ticket = find_ticket_by_incident_number(inc)
    if ticket is None:
        logger.info("[rca] lookup miss inc=%s", inc)
        raise HTTPException(status_code=404, detail="ticket_not_found")

    # Serialise once, share the string between the two LLM calls. Use
    # `default=str` so any datetime / UUID values that the schema
    # accumulated survive without raising.
    try:
        ticket_json = json.dumps(ticket, ensure_ascii=False, default=str)
    except Exception as exc:
        logger.warning("[rca] json.dumps failed inc=%s (%s)", inc, exc)
        raise HTTPException(status_code=500, detail="ticket_serialisation_failed")

    # Parallel kickoff. asyncio.to_thread offloads the blocking
    # safe_generate call so both run on threadpool workers and we
    # collect them together via gather(return_exceptions=True).
    customer_task = asyncio.to_thread(
        _call_llm, CUSTOMER_FACING_PROMPT, ticket_json, _MAX_TOKENS_CUSTOMER,
    )
    internal_task = asyncio.to_thread(
        _call_llm, INTERNAL_INCIDENT_PROMPT, ticket_json, _MAX_TOKENS_INTERNAL,
    )
    customer_result, internal_result = await asyncio.gather(
        customer_task, internal_task, return_exceptions=True,
    )

    customer_md = ""
    customer_err: Optional[str] = None
    if isinstance(customer_result, BaseException):
        logger.warning(
            "[rca] customer-facing LLM call failed inc=%s (%s)",
            inc, customer_result,
        )
        customer_err = "Customer-Facing RCA generation failed — please retry."
    elif isinstance(customer_result, str):
        customer_md = customer_result.strip()
        if not customer_md:
            customer_err = "Customer-Facing RCA returned empty output."

    internal_md = ""
    internal_err: Optional[str] = None
    if isinstance(internal_result, BaseException):
        logger.warning(
            "[rca] internal LLM call failed inc=%s (%s)",
            inc, internal_result,
        )
        internal_err = "Internal Incident RCA generation failed — please retry."
    elif isinstance(internal_result, str):
        internal_md = internal_result.strip()
        if not internal_md:
            internal_err = "Internal Incident RCA returned empty output."

    logger.info(
        "[rca] inc=%s customer_chars=%d internal_chars=%d "
        "customer_err=%s internal_err=%s",
        inc, len(customer_md), len(internal_md),
        bool(customer_err), bool(internal_err),
    )

    return RCAResponse(
        incident_number=inc,
        customer_facing_md=customer_md,
        internal_md=internal_md,
        customer_facing_error=customer_err,
        internal_error=internal_err,
    )
