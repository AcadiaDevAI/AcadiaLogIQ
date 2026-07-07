"""
Gap Analysis — dedicated Claude caller on AWS Bedrock.

Forked from ``rca/bedrock_claude.py`` so the Gap Analysis prompts can
target a different model (and a different output budget) without
disturbing RCA. By default we read the same ``BEDROCK_HAIKU_MODEL``
setting RCA uses — Claude Haiku 4.5 has the 200k context + 8192
output ceiling we need for these prompts. Operators who want to push
either report through a Sonnet variant can flip
``BEDROCK_GAP_ANALYSIS_MODEL`` in AWS Secrets Manager and restart
without touching this code.

Why a dedicated caller (and not a shared helper)
------------------------------------------------
* ``backend.api.safe_generate`` is wired to Mistral (different body
  shape) and is tuned for chat answers.
* RCA's Claude caller is owned by RCA. We don't want a refactor in
  RCA to change Gap Analysis output overnight.
* Anthropic body shape (``anthropic_version`` + ``messages``) is
  stable, but if Anthropic publishes a new contract we can adopt it
  here on Gap Analysis's schedule — not RCA's.

Failure semantics
-----------------
Raises on any boto3 / parsing error. ``routes.py`` runs both prompts
via ``asyncio.gather(return_exceptions=True)`` so a single-prompt
failure never blocks the other report.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict

from backend.services.token_usage import record_token_usage, extract_bedrock_usage

logger = logging.getLogger("acadia-log-iq")


# Dedicated Bedrock client for report generation. Built once, lazily.
# We do NOT reuse backend.api.bedrock here: that client is tuned for
# interactive chat (read_timeout=45s), but Gap Analysis / Post-Mortem are
# heavy non-streaming generations (max_tokens up to 16384) whose full
# response read routinely exceeds 45s — surfacing as "Read timeout on
# endpoint URL" and then burning ~4×45s on adaptive retries before
# failing. This client uses REPORT_LLM_READ_TIMEOUT_S (default 180s) and
# fewer attempts so the report has time to finish in one shot.
_report_bedrock = None


def _get_report_bedrock():
    global _report_bedrock
    if _report_bedrock is not None:
        return _report_bedrock

    import boto3
    from botocore.config import Config as BotoConfig
    from backend.config import settings

    cfg = BotoConfig(
        retries={
            "max_attempts": getattr(settings, "REPORT_LLM_MAX_ATTEMPTS", 2),
            "mode": "adaptive",
        },
        read_timeout=getattr(settings, "REPORT_LLM_READ_TIMEOUT_S", 180),
        connect_timeout=getattr(settings, "LLM_CONNECT_TIMEOUT_S", 10),
        tcp_keepalive=True,
    )

    kwargs: Dict[str, Any] = {
        "service_name": "bedrock-runtime",
        "region_name": settings.AWS_REGION,
        "config": cfg,
    }
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
        if settings.AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = settings.AWS_SESSION_TOKEN

    _report_bedrock = boto3.client(**kwargs)
    return _report_bedrock


# Settings precedence for model id (highest → lowest):
#   1. settings.BEDROCK_GAP_ANALYSIS_MODEL  ← Gap-Analysis-specific override
#   2. settings.BEDROCK_HAIKU_MODEL         ← shared Claude Haiku default
#   3. _FALLBACK_MODEL_ID                   ← hard fallback string
#
# Today (1) is unset on purpose — Haiku 4.5 produces excellent
# structured long-form output and runs cheaper / faster than Sonnet
# for these prompts. Flip (1) in the AWS secret if you ever want to
# split the model between RCA and Gap Analysis.
_FALLBACK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

# Anthropic Messages API contract version. Required by every request;
# pinned per AWS Bedrock docs. Do not change unless AWS publishes a
# new contract.
_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Default output budget. Both Gap Analysis prompts are long-form —
# routes.py overrides per-prompt with the right per-report budget.
# 8192 is Haiku 4.5's server-side ceiling.
_DEFAULT_MAX_TOKENS = 8192


def _resolve_model_id() -> str:
    """Resolve the Bedrock model id at call time.

    Reads from ``settings`` so a live secret rotation + uvicorn reload
    picks up the new value without any code change. Falls back to a
    hard-coded current Haiku id if the import or attribute access
    fails — never crashes the LLM call.
    """
    try:
        from backend.config import settings  # type: ignore

        # First preference — Gap-Analysis-specific override.
        mid = getattr(settings, "BEDROCK_GAP_ANALYSIS_MODEL", None)
        if isinstance(mid, str) and mid.strip():
            return mid.strip()

        # Second — shared Haiku setting (also used by RCA).
        mid = getattr(settings, "BEDROCK_HAIKU_MODEL", None)
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
    except Exception:
        pass
    return _FALLBACK_MODEL_ID


def invoke(prompt: str, max_tokens: int = _DEFAULT_MAX_TOKENS, *, temperature: float = 0.1) -> str:
    """Run a single prompt through Claude on Bedrock and return the
    response text.

    Parameters
    ----------
    prompt
        The full prompt string (system + user content + JSON payload).
        We do NOT truncate — Claude Haiku's 200k context comfortably
        holds the longest Gap Analysis prompt plus a rich ticket JSON.
    max_tokens
        Output budget. Values above the model's server-side cap (8192
        for Haiku 4.5) are silently clamped by Bedrock.
    temperature
        0.1 by default — Gap Analysis output is highly structured and
        should be near-deterministic. Mirrors RCA's tuning so behaviour
        is comparable when an operator sanity-checks a ticket through
        both reports.

    Returns
    -------
    str
        Whitespace-trimmed concatenation of every ``content[*].text``
        block in the response.

    Raises
    ------
    ValueError
        Empty / non-string prompt, or no parseable text in the
        response.
    botocore.exceptions.ClientError
        AWS-side failure (auth, throttling, model access denied).
        Caller catches via ``asyncio.gather``.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("empty prompt")

    # Dedicated long-timeout client (see _get_report_bedrock). We do NOT
    # reuse backend.api.bedrock — its 45s chat read_timeout is too tight
    # for these 8k–16k-token report generations and causes read timeouts.
    bedrock = _get_report_bedrock()

    model_id = _resolve_model_id()

    body_obj: Dict[str, Any] = {
        "anthropic_version": _ANTHROPIC_VERSION,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        # Single-turn — the entire Gap Analysis prompt is one
        # self-contained instruction block. No multi-turn needed.
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }
    body = json.dumps(body_obj).encode("utf-8")

    resp = bedrock.invoke_model(
        modelId=model_id,
        body=body,
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(resp["body"].read().decode("utf-8"))

    if not isinstance(payload, dict):
        raise ValueError("non-dict response from Bedrock Claude")

    # Claude-on-Bedrock response shape:
    #   {
    #     "id": "msg_...", "type": "message", "role": "assistant",
    #     "model": "...",
    #     "content": [{"type": "text", "text": "..."}],
    #     "stop_reason": "end_turn" | "max_tokens" | ...,
    #     "usage": {"input_tokens": N, "output_tokens": N}
    #   }
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("empty content in Bedrock Claude response")

    parts = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                parts.append(text)
    out = "\n".join(parts).strip()
    if not out:
        raise ValueError("no text blocks in Bedrock Claude response")

    stop_reason = payload.get("stop_reason")
    usage = payload.get("usage") or {}
    record_token_usage("gap_analysis", model_id, int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0))
    logger.info(
        "[gap_analysis.claude] model=%s stop=%s in_toks=%s out_toks=%s "
        "out_chars=%d",
        model_id,
        stop_reason,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        len(out),
    )
    if stop_reason == "max_tokens":
        # Caller decides whether to retry with a larger budget. The
        # partial output may still be usable so we don't raise.
        logger.warning(
            "[gap_analysis.claude] hit max_tokens cap (%d) — output may be truncated",
            max_tokens,
        )

    return out
