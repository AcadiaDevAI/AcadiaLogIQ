"""
Input Guard — pre-retrieval safety check for /ask queries.
Detects prompt-injection attempts, oversized payloads, and obviously
unsafe phrasing. Fail-open: any internal error yields "safe_to_process".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Thresholds / constants
# ---------------------------------------------------------------------------
MAX_QUERY_CHARS: int = 1000      # same as the Pydantic Field upper bound
MIN_QUERY_CHARS: int = 1

GUARD_REASON_OK: str = "ok"
GUARD_REASON_EMPTY: str = "empty"
GUARD_REASON_TOO_LONG: str = "too_long"
GUARD_REASON_INJECTION: str = "prompt_injection"
GUARD_REASON_UNSAFE: str = "unsafe_phrasing"

_CANNED_FLAGGED = (
    "I can't process that request. "
    "Please rephrase your question about the uploaded documents."
)


# ---------------------------------------------------------------------------
# Pattern table — first match wins. Kept conservative to avoid false positives.
# ---------------------------------------------------------------------------
_INJECTION_PATTERNS = [
    re.compile(r"\bignore\s+(?:all|previous|prior|above)\s+(?:instructions?|prompts?|rules?)\b", re.I),
    re.compile(r"\bdisregard\s+(?:the\s+)?system\s+(?:prompt|message|instructions?)\b", re.I),
    re.compile(r"\byou\s+are\s+now\s+(?:a\s+)?(?:different|new|another)\s+(?:ai|assistant|bot)\b", re.I),
    re.compile(r"\brepeat\s+your\s+(?:system\s+)?(?:prompt|instructions?)\b", re.I),
    re.compile(r"\b(?:print|reveal|show|leak)\s+(?:the\s+)?(?:system\s+)?prompt\b", re.I),
    re.compile(r"<\|.*?\|>"),  # model control tokens
]

_UNSAFE_PATTERNS = [
    re.compile(r"\b(?:how\s+to\s+)?(?:hack|exploit|ddos|sql\s*inject)\s+(?:the\s+)?(?:backend|server|db|database|system)\b", re.I),
    re.compile(r"\bdump\s+(?:all\s+)?(?:user|pii|credentials?|passwords?)\b", re.I),
]


@dataclass
class GuardResult:
    """
    safe_to_process   — True if the request may continue through /ask
    flagged           — True if a guard rule fired
    reason            — GUARD_REASON_* constant
    matched_phrase    — substring that triggered the rule (if any)
    canned_response   — safe reply to return on flag (else "")
    """
    safe_to_process: bool = True
    flagged: bool = False
    reason: str = GUARD_REASON_OK
    matched_phrase: Optional[str] = None
    canned_response: str = ""


def check_input(query: str) -> GuardResult:
    """
    Run input safety checks. Never raises. Any internal error yields
    a fail-open GuardResult so the request continues normally.
    """
    try:
        if query is None:
            return GuardResult(safe_to_process=False, flagged=True, reason=GUARD_REASON_EMPTY,
                               canned_response=_CANNED_FLAGGED)
        q = query.strip()

        if len(q) < MIN_QUERY_CHARS:
            return GuardResult(safe_to_process=False, flagged=True, reason=GUARD_REASON_EMPTY,
                               canned_response=_CANNED_FLAGGED)

        if len(q) > MAX_QUERY_CHARS:
            return GuardResult(safe_to_process=False, flagged=True, reason=GUARD_REASON_TOO_LONG,
                               matched_phrase=q[:60], canned_response=_CANNED_FLAGGED)

        for p in _INJECTION_PATTERNS:
            m = p.search(q)
            if m:
                phrase = m.group(0)
                logger.warning("Input guard: injection pattern matched: %r", phrase)
                return GuardResult(safe_to_process=False, flagged=True,
                                   reason=GUARD_REASON_INJECTION,
                                   matched_phrase=phrase, canned_response=_CANNED_FLAGGED)

        for p in _UNSAFE_PATTERNS:
            m = p.search(q)
            if m:
                phrase = m.group(0)
                logger.warning("Input guard: unsafe phrasing matched: %r", phrase)
                return GuardResult(safe_to_process=False, flagged=True,
                                   reason=GUARD_REASON_UNSAFE,
                                   matched_phrase=phrase, canned_response=_CANNED_FLAGGED)

        return GuardResult()  # safe
    except Exception as exc:
        logger.warning("Input guard raised (%s) — failing open", exc)
        return GuardResult(safe_to_process=True, flagged=False,
                           reason=f"guard_error: {exc}")
