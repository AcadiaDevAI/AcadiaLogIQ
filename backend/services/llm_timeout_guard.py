"""
LLM Timeout Guard — Hard wall-clock on chat-path LLM calls, plus
fast-model fallback on Bedrock hang/throttle.

Problem
-------
Before this iteration, `safe_generate` (the chat-path LLM call) used the
shared Bedrock boto3 client with `read_timeout=120` and
`max_attempts=10`. A truly hung Bedrock endpoint would make the user
wait up to 20 minutes (10 × 120s) before any error surfaced. There was
no model fallback: if Mistral was stuck, we just kept retrying Mistral.

This module replaces that path with two complementary protections:

  1. **Tight per-attempt timeout** (caller already lowered the boto3
     read_timeout to settings.LLM_READ_TIMEOUT_S and max_attempts to
     settings.LLM_MAX_ATTEMPTS — that bounds primary-model wait at
     roughly LLM_READ_TIMEOUT_S × LLM_MAX_ATTEMPTS).

  2. **Fast-model fallback** (this module). When the primary attempt
     raises ReadTimeoutError, ThrottlingException, or ConnectionClosed,
     we make ONE more attempt against Claude Haiku 4.5 (which is fast,
     cheap, and on a different model family — so transient Mistral-side
     issues don't affect it). If Haiku also fails, we return a polite
     decline string rather than hanging the request.

Design notes
------------
* Failure detection is by exception type, NOT by elapsed time. The
  boto3 client already enforces the wall-clock; we just react to its
  errors. This avoids spawning extra threads and keeps the code simple.
* "Decline" is a string return, not an exception. `safe_generate`'s
  callers expect a string back; we preserve that contract.
* The model label used by each call is captured so callers can log
  which path actually answered. Useful when grepping the logs for
  "how often did the fallback fire?".
* Disabled-flag path: when `LLM_TIMEOUT_FALLBACK_ENABLED=False`, this
  module is a no-op pass-through — the primary function runs as it
  always did and any exception is allowed to surface.
"""

from __future__ import annotations

import logging
from typing import Callable, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Result label used in logs / context_stats so we can distinguish
# "primary answered" from "Haiku rescued the call" from "we declined".
MODEL_LABEL_PRIMARY = "primary"
MODEL_LABEL_HAIKU_FALLBACK = "haiku_fallback"
MODEL_LABEL_DECLINED = "declined"


# Exception classes we treat as "trigger the fallback path". Anything
# else (e.g. a programming error in our prompt construction) is allowed
# to propagate so we notice it.
def _is_recoverable_bedrock_error(exc: BaseException) -> bool:
    """
    Return True when this exception is the kind that fast-model
    fallback should rescue: connection timeouts, read timeouts,
    throttling, and broken-pipe-style closures. False for everything
    else so callers see real bugs instead of silent fallbacks.
    """
    name = type(exc).__name__
    # botocore raises a handful of error classes by name only — string
    # match keeps us from importing botocore here (this module is meant
    # to be lightweight).
    if name in (
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "ConnectionClosedError",
        "ResponseStreamingError",
    ):
        return True
    # ClientError covers ThrottlingException / ServiceUnavailableException.
    # We accept the broader class but only treat throttle/5xx codes as
    # recoverable — surfacing real ValidationException etc. as bugs.
    if name == "ClientError":
        code = ""
        try:
            code = (
                exc.response.get("Error", {}).get("Code", "")  # type: ignore[attr-defined]
            )
        except Exception:
            code = ""
        if code in (
            "ThrottlingException",
            "ModelTimeoutException",
            "ServiceUnavailableException",
            "ModelStreamErrorException",
        ):
            return True
    return False


def generate_with_fallback(
    *,
    prompt: str,
    max_tokens: int,
    primary_fn: Callable[[str, int], str],
) -> Tuple[str, str]:
    """
    Run `primary_fn(prompt, max_tokens)`. On a recoverable Bedrock error
    (timeout / throttle / connection drop), retry once via Claude
    Haiku 4.5. On both failures, return the configured decline message.

    Returns
    -------
    (text, model_label)
        text         — non-empty string suitable for the user
        model_label  — one of MODEL_LABEL_* constants. Use to log /
                       populate context_stats so downstream observers
                       can spot fallback usage.

    Notes
    -----
    * `primary_fn` is the existing `safe_generate`-style callable. Its
      OWN error handling is preserved — we only kick in when primary_fn
      raises (not when it swallows the error and returns a string).
    * When the flag is off, this is a no-op wrapper: primary_fn runs,
      its result returns with MODEL_LABEL_PRIMARY. Any exception
      propagates as before.
    """
    if not settings.LLM_TIMEOUT_FALLBACK_ENABLED:
        return primary_fn(prompt, max_tokens), MODEL_LABEL_PRIMARY

    # --- Primary attempt -------------------------------------------------
    try:
        text = primary_fn(prompt, max_tokens)
        return text, MODEL_LABEL_PRIMARY
    except Exception as exc:
        if not _is_recoverable_bedrock_error(exc):
            # Real bug — re-raise so the caller's existing exception
            # handler can decide what to do (typically log + return an
            # error string).
            raise
        logger.warning(
            "[llm_timeout_guard] primary model failed (%s) — "
            "attempting Haiku fallback",
            type(exc).__name__,
        )

    # --- Haiku fallback --------------------------------------------------
    try:
        # Local import keeps the module importable without boto3.
        from backend.services.bedrock_haiku import haiku_client

        fallback_text = haiku_client.invoke_text(
            prompt=prompt,
            max_tokens=settings.LLM_HAIKU_FALLBACK_MAX_TOKENS,
            context="chat_fallback",
        )
    except Exception as exc:
        # Even the Haiku call construction blew up (e.g. boto3 missing
        # in a test env). Treat as full failure → decline.
        logger.warning(
            "[llm_timeout_guard] haiku fallback raised (%s) — declining",
            type(exc).__name__,
        )
        fallback_text = ""

    if fallback_text and fallback_text.strip():
        logger.info(
            "[llm_timeout_guard] recovered via haiku fallback "
            "(chars=%d)", len(fallback_text),
        )
        return fallback_text.strip(), MODEL_LABEL_HAIKU_FALLBACK

    # --- Both failed → decline ------------------------------------------
    logger.warning(
        "[llm_timeout_guard] both primary and fallback failed — "
        "returning decline message"
    )
    return settings.LLM_FALLBACK_DECLINE_MESSAGE, MODEL_LABEL_DECLINED
