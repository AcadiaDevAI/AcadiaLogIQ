"""Sprint 13.32.7 — Dedicated Claude caller for the RCA module.

Why this module exists:
  * `backend.api.safe_generate` posts to `settings.BEDROCK_LLM_MODEL`
    (currently Mistral 7B). Mistral is too small to follow our RCA
    prompts' multi-level structure + STRIP rules — output leaked
    interface IDs / CLI / fabricated contact info.
  * We can't swap the global model: every other LLM-touching path
    (chat /ask, journey stages 1-5, escalation handoff) was tuned
    against Mistral's voice. Changing the global breaks them.
  * So the RCA module owns its own caller, pinned to Claude on
    Bedrock. Lives here, imports the existing bedrock client, and
    speaks the Anthropic Messages API body shape (different from
    Mistral's body shape — that's the second reason this is a
    separate function rather than a flag inside safe_generate).

Model selection:
  * Model ID is read at call time from
    ``settings.BEDROCK_HAIKU_MODEL`` (env: ``BEDROCK_HAIKU_MODEL``).
    Today the operator's .env pins it to Claude Haiku 4.5
    (``us.anthropic.claude-haiku-4-5-20251001-v1:0``) — current
    generation, large context, long output ceiling, fast.
    To swap models (e.g. point both panels at a Sonnet variant for
    higher-quality long-form output), change the env value and
    restart the backend — no code change required.
  * Earlier revisions pinned a hard-coded model ID. That created
    two operational issues:
      - When AWS flipped ``claude-3-haiku-20240307-v1:0`` to
        ``Legacy`` for inactivity, every RCA call failed with
        ``ResourceNotFoundException`` until the constant was
        edited and the backend restarted.
      - Operators with a preferred Bedrock model couldn't swap
        without a code change.
    Reading from settings fixes both — the .env value already
    points to a current, non-legacy model.

Failure semantics:
  Raises on any boto3 / parsing error. ``routes.py`` catches per-call
  via asyncio.gather(return_exceptions=True) so a single-panel failure
  never blocks the other panel.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict


logger = logging.getLogger("acadia-log-iq")


# Dedicated Bedrock client for RCA report generation. Built once, lazily.
# We do NOT reuse backend.api.bedrock: that client is tuned for
# interactive chat (read_timeout=45s), but RCA is a heavy non-streaming
# generation whose full response read routinely exceeds 45s — surfacing
# as "Read timeout on endpoint URL" and then burning ~4×45s on adaptive
# retries before failing. This client uses REPORT_LLM_READ_TIMEOUT_S
# (default 180s) and fewer attempts so the report finishes in one shot.
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


# Sprint 13.32.9 — model id is no longer hard-coded. Read it from
# settings at call time so the operator's .env (BEDROCK_HAIKU_MODEL)
# is the single source of truth. Today that env points at
# Claude Haiku 4.5 (us.anthropic.claude-haiku-4-5-20251001-v1:0).
# Fallback string mirrors the .env default in config.py; if both are
# absent the call still issues against the current Haiku ID rather
# than crashing.
_FALLBACK_MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def _resolve_model_id() -> str:
    """Resolve the Bedrock model id at call time so a live .env edit
    + uvicorn reload picks up the new value without touching code."""
    try:
        from backend.config import settings  # type: ignore
        mid = getattr(settings, "BEDROCK_HAIKU_MODEL", None)
        if isinstance(mid, str) and mid.strip():
            return mid.strip()
    except Exception:
        # Settings import / attribute error — fall through to the
        # fallback constant rather than failing the LLM call.
        pass
    return _FALLBACK_MODEL_ID


# Bedrock's pinned anthropic API version. Required field on every
# request; do not change unless AWS publishes a new one.
_ANTHROPIC_VERSION = "bedrock-2023-05-31"

# Default output budget. Callers (routes.py) override per-prompt.
# 8192 covers Claude Haiku 4.5's published output ceiling.
_DEFAULT_MAX_TOKENS = 8192


def invoke(prompt: str, max_tokens: int = _DEFAULT_MAX_TOKENS, *, temperature: float = 0.1) -> str:
    """Run a single prompt through Claude on Bedrock and return the
    response text.

    Args:
        prompt: The full prompt string (system + user content + JSON).
            No truncation is performed here — Claude Haiku's 200k
            context comfortably holds any realistic RCA payload.
        max_tokens: Output budget. 8192 is a safe default for Claude
            Haiku 4.5; values above the model's server-side ceiling are
            silently clamped.
        temperature: 0.1 by default — RCA output is structured and
            should be near-deterministic. Same value safe_generate
            uses for the Mistral path.

    Returns:
        The concatenated text from every ``content[*].text`` block in
        the response. Whitespace-trimmed.

    Raises:
        ValueError: empty / non-string prompt, or no parseable text in
            the response.
        botocore.exceptions.ClientError: any AWS-side failure (auth,
            throttling, model access denied). Caller catches.
    """
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("empty prompt")

    # Dedicated long-timeout client (see _get_report_bedrock). We do NOT
    # reuse backend.api.bedrock — its 45s chat read_timeout is too tight
    # for these large RCA generations and causes read timeouts.
    bedrock = _get_report_bedrock()

    model_id = _resolve_model_id()

    body_obj: Dict[str, Any] = {
        "anthropic_version": _ANTHROPIC_VERSION,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        # Single user turn — the entire RCA prompt is one self-
        # contained instruction block. No multi-turn needed.
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

    # Claude on Bedrock response shape:
    # {
    #   "id": "msg_...", "type": "message", "role": "assistant",
    #   "model": "...",
    #   "content": [{"type": "text", "text": "..."}],
    #   "stop_reason": "end_turn" | "max_tokens" | ...,
    #   "stop_sequence": null,
    #   "usage": {"input_tokens": N, "output_tokens": N}
    # }
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
    logger.info(
        "[rca.claude] model=%s stop=%s in_toks=%s out_toks=%s out_chars=%d",
        model_id,
        stop_reason,
        usage.get("input_tokens"),
        usage.get("output_tokens"),
        len(out),
    )
    if stop_reason == "max_tokens":
        # Not raised — the partial output may still be usable, and the
        # caller can surface a "regenerate / increase budget" hint.
        # We log loudly so the budget-tuning operator notices it.
        logger.warning(
            "[rca.claude] hit max_tokens cap (%d) — output may be truncated",
            max_tokens,
        )

    return out
