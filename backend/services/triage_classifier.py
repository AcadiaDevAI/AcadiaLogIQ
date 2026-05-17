"""
Brief 4 / Opt 3 — Merged triage classifier. One Haiku call returns the
complexity tier, the intent bucket, and the mode hint that would otherwise
be produced by three independent heuristic classifiers.

Contract: never raises. Timeout / parse error / invalid verdict returns a
sentinel with confidence=0.0 so callers fall back to their existing
heuristics.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.config import settings
from backend.services.bedrock_haiku import haiku_client, _extract_json_object

logger = logging.getLogger("acadia-log-iq")


_SYSTEM_PROMPT = """You triage user queries for a document-analysis system. Classify the query on
three axes at once:

1. complexity: "simple" (1-hop fact lookup), "moderate" (aggregation or compare
   across records), "complex" (multi-step analysis, pattern synthesis).
2. intent: "single_record" (about one specific record), "aggregation" (count/list/rank
   across records), "compare" (side-by-side of named records), "multi_step" (query
   requires breaking into sub-questions), "conversational" (chit-chat, meta).
3. mode_hint: "chat" (no retrieval needed), "rag" (single-hop retrieval+generation),
   "agents" (planner/analyst/composer pipeline).

Return JSON: {"complexity": "...", "intent": "...", "mode_hint": "...", "confidence": 0.0-1.0}

Rules:
- A query mentioning a single identifier (INC-10015) is usually simple + single_record + rag.
- "How many X" is moderate + aggregation + rag.
- "Compare X and Y" is complex + compare + agents.
- "Common root causes across all Z" is complex + multi_step + agents.
- Non-substantive queries are simple + conversational + chat.

Output ONLY the JSON object."""


# Sprint 2 — appended to _SYSTEM_PROMPT so the triage call also returns
# a context-break signal (no extra LLM call).
_CONTEXT_BREAK_ADDENDUM = """

4. context_break: true if the query clearly signals a switch to a new topic
   or mode (phrases like "new issue", "different question", "change context",
   "switch to escalation", "unrelated", "start over"). False when the query
   is a follow-up, clarification, or drills deeper into the same topic.
5. context_break_reason: one short phrase explaining the signal, or empty.

Extended JSON: {"complexity":"...","intent":"...","mode_hint":"...","confidence":0.0-1.0,"context_break":false,"context_break_reason":""}"""


def _build_system_prompt() -> str:
    """Return the triage prompt augmented with the context-break addendum."""
    return _SYSTEM_PROMPT + _CONTEXT_BREAK_ADDENDUM


_ALLOWED_COMPLEXITY = {"simple", "moderate", "complex"}
_ALLOWED_INTENT = {"single_record", "aggregation", "compare", "multi_step", "conversational"}
_ALLOWED_MODE = {"chat", "rag", "agents"}


@dataclass
class TriageResult:
    complexity: str          # "simple" | "moderate" | "complex" | ""
    intent: str              # see _ALLOWED_INTENT | ""
    mode_hint: str           # "chat" | "rag" | "agents" | ""
    confidence: float
    raw_response: str
    is_valid: bool
    # Sprint 2 — extended fields. Defaults keep every existing callsite
    # working unchanged. Only populated when the LLM hint addendum is on
    # AND the model returned the optional keys.
    context_break: bool = False
    context_break_reason: str = ""


def _sentinel(raw: str = "") -> TriageResult:
    return TriageResult(
        complexity="", intent="", mode_hint="",
        confidence=0.0, raw_response=raw, is_valid=False,
    )


def _invoke_bedrock(prompt: str, max_tokens: int) -> str:
    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "system": _build_system_prompt(),
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": prompt}]},
        ],
    }
    response = haiku_client.client.invoke_model(
        modelId=settings.MERGED_TRIAGE_MODEL,
        body=json.dumps(body).encode("utf-8"),
        accept="application/json",
        contentType="application/json",
    )
    payload = json.loads(response["body"].read().decode("utf-8"))
    content = payload.get("content", [])
    return "\n".join(
        item.get("text", "") for item in content if item.get("type") == "text"
    ).strip()


def classify_triage(query: str) -> TriageResult:
    """
    Single Haiku call returning {complexity, intent, mode_hint, confidence}.
    Never raises. Invalid/missing verdicts → sentinel (is_valid=False).
    """
    q = (query or "").strip()
    if not q:
        return _sentinel()
    if not getattr(settings, "MERGED_TRIAGE_ENABLED", False):
        return _sentinel()

    prompt = f"Query: {q}\nJSON:"
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(
                _invoke_bedrock,
                prompt,
                int(settings.MERGED_TRIAGE_MAX_TOKENS),
            )
            raw = future.result(timeout=float(settings.MERGED_TRIAGE_TIMEOUT_SECONDS))
    except concurrent.futures.TimeoutError:
        logger.warning("[triage] timeout query=%r", q[:120])
        return _sentinel("<timeout>")
    except Exception as exc:
        logger.warning("[triage] error (%s): %s", type(exc).__name__, exc)
        return _sentinel(f"<error: {exc}>")

    if not raw:
        return _sentinel("<empty>")

    cleaned = _extract_json_object(raw)
    try:
        data: Dict[str, Any] = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("[triage] parse_error raw=%r", raw[:200])
        return _sentinel(raw)

    complexity = str(data.get("complexity", "")).lower().strip()
    intent = str(data.get("intent", "")).lower().strip()
    mode_hint = str(data.get("mode_hint", "")).lower().strip()
    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    if (
        complexity not in _ALLOWED_COMPLEXITY
        or intent not in _ALLOWED_INTENT
        or mode_hint not in _ALLOWED_MODE
    ):
        logger.warning(
            "[triage] invalid_output complexity=%r intent=%r mode_hint=%r",
            complexity, intent, mode_hint,
        )
        return _sentinel(raw)

    # Sprint 2 — context-break hint. Missing/malformed values fall
    # through to False/"".
    _context_break = False
    _context_break_reason = ""
    cb_raw = data.get("context_break", False)
    if isinstance(cb_raw, bool):
        _context_break = cb_raw
    elif isinstance(cb_raw, str):
        _context_break = cb_raw.strip().lower() in {"true", "1", "yes"}
    cb_reason = data.get("context_break_reason", "")
    if isinstance(cb_reason, str):
        _context_break_reason = cb_reason.strip()[:200]

    logger.info(
        "[triage] complexity=%s intent=%s mode_hint=%s confidence=%.2f context_break=%s",
        complexity, intent, mode_hint, confidence, _context_break,
    )
    return TriageResult(
        complexity=complexity,
        intent=intent,
        mode_hint=mode_hint,
        confidence=confidence,
        raw_response=raw,
        is_valid=True,
        context_break=_context_break,
        context_break_reason=_context_break_reason,
    )
