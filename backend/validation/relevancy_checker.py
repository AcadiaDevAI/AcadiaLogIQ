"""
Response Relevancy Checker — Closes the "grounded but off-topic" hole.

Problem
-------
The existing grounding checker (`grounding_checker.py`) verifies that
the answer's claims are backed by the retrieved chunks. That catches
hallucination, but not a more subtle failure: an answer that's
*faithful to the documents but doesn't address the user's question*.

  User: "What's the SLA for a P1 incident?"
  Retrieved: chunks describing P1 incident response *procedures*
  Answer: detailed (and grounded!) walkthrough of incident response
  Reality: the user asked about SLA timing, not procedures

Grounding score → 0.9 (faithful). User satisfaction → 0. Today this
sails through validation unflagged.

Solution
--------
After the answer is generated, send a tiny prompt to Claude Haiku that
asks a single yes/no question: "Does this answer address this question?"
Haiku returns a JSON object with a 0.0-1.0 score. Below the configured
threshold, the validator treats the answer as off-topic and replaces it
with a polite "please rephrase" template (Case E in `validate_answer`).

Design
------
* **Cheap.** One Haiku call with ~200 tokens of input and ~50 tokens of
  output. Costs ~$0.0001 per /ask call.
* **Fails OPEN.** If Haiku is unreachable / returns garbage, we PASS the
  answer through (passed=True, score=1.0). The opposite (fail-closed)
  would block legitimate answers during Bedrock outages — unacceptable.
* **Independent of grounding.** Runs as a separate stage so a future
  metrics dashboard can split "ungrounded" vs "off-topic" failures.
* **Question is verbatim, answer is truncated.** We cap the answer at
  the first ~600 chars so the prompt stays small and Haiku focuses on
  the lede — which is what determines on-topic-ness for the user.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# Cap on how much answer text we send to the relevancy judge. Empirically,
# the first ~600 characters carry the topic signal; piling on the rest
# just bloats input tokens and gives the LLM more room to overthink.
_ANSWER_PREVIEW_CHARS = 600


@dataclass
class RelevancyResult:
    """
    Outcome of a single relevancy check.

    Attributes
    ----------
    passed : bool
        True when score >= MIN_RELEVANCY_SCORE OR the check failed open
        (Haiku unreachable). False only when we have a confident "no".
    score : float
        0.0-1.0 from the LLM judge. Defaults to 1.0 when failing open
        so downstream weighting treats us as "no evidence of off-topic".
    reason : str
        One-line explanation from Haiku, or "" when the check was
        bypassed / failed open.
    skipped : bool
        True if the check was bypassed (disabled, empty inputs, or
        Haiku error). Useful for the eval log to distinguish "passed
        the check" from "we couldn't run the check".
    """

    passed: bool = True
    score: float = 1.0
    reason: str = ""
    skipped: bool = False


def check_relevancy(*, query: str, answer: str) -> RelevancyResult:
    """
    Ask Claude Haiku whether `answer` actually addresses `query`.

    Returns a RelevancyResult. The function NEVER raises:
      * Disabled → skipped=True, passed=True
      * Empty inputs → skipped=True, passed=True
      * Haiku error → skipped=True, passed=True (fail open)
      * Below threshold → passed=False, score from Haiku
      * At/above threshold → passed=True, score from Haiku
    """
    # Disabled / empty-input short-circuits. These count as "skipped"
    # so the eval log can distinguish them from genuine passes.
    if not settings.ENABLE_RELEVANCY_CHECK:
        return RelevancyResult(skipped=True)
    if not (query or "").strip() or not (answer or "").strip():
        return RelevancyResult(skipped=True)

    # Local import keeps this module importable in tests without boto3.
    try:
        from backend.services.bedrock_haiku import haiku_client
    except Exception as exc:
        logger.warning("[relevancy] haiku import failed: %s — fail open", exc)
        return RelevancyResult(skipped=True)

    preview = answer.strip()
    truncated = len(preview) > _ANSWER_PREVIEW_CHARS
    if truncated:
        preview = preview[:_ANSWER_PREVIEW_CHARS] + " […truncated for judge…]"

    system = (
        "You are a topical-relevance judge. "
        "Return strict JSON only. No markdown. No commentary."
    )

    # The prompt is small on purpose, but it MUST be unambiguous about
    # one thing: we are judging on-topic-ness, NOT factual accuracy.
    #
    # A previous, looser prompt produced 0.00 scores like
    #   "cannot be verified because the source document was not provided"
    # on answers that were grounded and on-topic. Haiku was interpreting
    # "relevant" as "verifiable against ground truth", and bailed out
    # whenever the question mentioned a source it hadn't seen
    # (e.g. "In the branch WAN/VPN SOP, ..."). The explicit rules
    # below stop that failure mode.
    prompt = (
        "TASK: Decide ONLY whether the proposed answer addresses the user's "
        "question on-topic. Do NOT check facts. Do NOT require a source "
        "document to verify the answer — you do not have one and that "
        "is not your job here.\n\n"
        "RULES:\n"
        " - If the answer addresses the question's subject, score HIGH "
        "(0.7–1.0) regardless of whether you can confirm correctness.\n"
        " - If the answer talks about something else entirely, score LOW.\n"
        " - If the user's question names a specific document, manual or "
        "SOP, assume the answer was drawn from it — do NOT penalize for "
        "not seeing that source.\n"
        " - 'Verifiability' is NOT a reason to score below 0.5.\n\n"
        "Question:\n"
        f"\"{query.strip()}\"\n\n"
        "Proposed answer:\n"
        f"\"{preview}\"\n\n"
        "SCORING GUIDE (about how well the answer ADDRESSES the question):\n"
        "  1.0 — answers the question directly\n"
        "  0.7 — partially answers; on topic but missing some asked-for detail\n"
        "  0.4 — discusses the topic but does not answer what was asked\n"
        "  0.0 — completely unrelated or evasive\n\n"
        "Return JSON of the form:\n"
        '  {\"score\": 0.0-1.0, \"reason\": \"one short sentence about '
        'on-topic-ness ONLY\"}\n\n'
        "JSON only:"
    )

    try:
        result = haiku_client.invoke_json(
            system=system,
            prompt=prompt,
            max_tokens=settings.RELEVANCY_MAX_TOKENS,
            context="relevancy",
        )
    except Exception as exc:
        # haiku_client.invoke_json already logs internally on retries.
        # We treat any exception as "couldn't judge" and fail open.
        logger.warning("[relevancy] haiku invoke raised: %s — fail open", exc)
        return RelevancyResult(skipped=True)

    if not isinstance(result, dict):
        logger.warning("[relevancy] haiku returned non-dict: %r — fail open", result)
        return RelevancyResult(skipped=True)

    # Parse score defensively. Haiku occasionally returns the score as
    # a string or wraps it inside an extra key — we accept both.
    raw_score = result.get("score")
    if raw_score is None:
        for key in ("relevance", "rating", "value"):
            if key in result:
                raw_score = result[key]
                break

    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        logger.warning(
            "[relevancy] could not parse score from haiku response: %r — fail open",
            result,
        )
        return RelevancyResult(skipped=True)

    # Clamp into [0, 1] so a misbehaving judge can't poison the
    # threshold comparison.
    score = max(0.0, min(1.0, score))
    reason = str(result.get("reason") or "").strip()

    threshold = settings.MIN_RELEVANCY_SCORE
    passed = score >= threshold

    logger.info(
        "[relevancy] score=%.2f threshold=%.2f passed=%s reason=%r",
        score, threshold, passed, reason[:120],
    )
    return RelevancyResult(passed=passed, score=score, reason=reason)


def get_off_topic_fallback() -> str:
    """Convenience for `validate_answer` to fetch the configured
    off-topic fallback text without importing settings directly."""
    return settings.RELEVANCY_FALLBACK_MESSAGE
