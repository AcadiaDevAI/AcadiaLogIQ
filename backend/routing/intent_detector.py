"""
Intent Detector — recognizes explicit user commands in /ask queries.
Returns an IntentResult with intent name, matched phrase, and an optional
suggested mode (hint only — explicit `mode` on the request always wins).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Intent constants — imported by api.py and tests to avoid magic strings
# ---------------------------------------------------------------------------
INTENT_GENERAL: str = "general"
INTENT_NOT_RESOLVED: str = "not_resolved"
INTENT_CHECK_KB: str = "check_kb"
INTENT_SHOW_SOP: str = "show_sop"
INTENT_SEARCH_DOCS: str = "search_docs"
INTENT_WHAT_NEXT: str = "what_next"


# ---------------------------------------------------------------------------
# Pattern table — ordered; first match wins
# ---------------------------------------------------------------------------
# Each entry: (intent, compiled_regex, suggested_mode_hint)
# suggested_mode_hint is only consumed when the request uses mode='auto'.
_INTENT_PATTERNS = [
    # "not resolved" / "still not resolved" / "issue not resolved"
    (
        INTENT_NOT_RESOLVED,
        re.compile(r"\b(?:still\s+)?not\s+resolv(?:ed|ing)\b|\bunresolved\b|\bdidn[’']?t\s+(?:work|resolve|help)\b", re.I),
        "multi_agent",
    ),
    # explicit "check kb" / "check the kb" / "check knowledge base"
    (
        INTENT_CHECK_KB,
        re.compile(r"\bcheck\s+(?:the\s+)?(?:kb|knowledge\s*base)\b", re.I),
        None,
    ),
    # explicit "show sop" / "show the sop" / "show sops" / "standard operating procedure"
    (
        INTENT_SHOW_SOP,
        re.compile(r"\b(?:show|display|get|pull)\s+(?:me\s+)?(?:the\s+)?sops?\b|\bstandard\s+operating\s+procedure\b", re.I),
        None,
    ),
    # explicit "search docs" / "search the docs" / "search documents"
    (
        INTENT_SEARCH_DOCS,
        re.compile(r"\bsearch\s+(?:the\s+)?(?:docs?|documents?)\b", re.I),
        None,
    ),
    # "what next" / "what's next" / "what to do next"
    (
        INTENT_WHAT_NEXT,
        re.compile(r"\bwhat(?:[’']?s| is| should I do| to do)?\s+next\b", re.I),
        "multi_agent",
    ),
]


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class IntentResult:
    """
    Outcome of intent detection.
        intent           — canonical intent name (see INTENT_* constants)
        matched          — True if any explicit intent pattern fired
        matched_phrase   — exact substring that matched (for logging/debug)
        suggested_mode   — optional hint: 'multi_agent' | 'hybrid' | None
        reason           — short human-readable explanation
    """
    intent: str = INTENT_GENERAL
    matched: bool = False
    matched_phrase: Optional[str] = None
    suggested_mode: Optional[str] = None
    reason: str = "no explicit intent detected"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_intent(query: str) -> IntentResult:
    """
    Scan the query for explicit user intents. Returns the first match found.

    Safe on empty/None input — returns INTENT_GENERAL with matched=False.
    Never raises. Never mutates input. Pure function.
    """
    if not query or not query.strip():
        return IntentResult()

    q = query.strip()

    for intent, pattern, suggested in _INTENT_PATTERNS:
        m = pattern.search(q)
        if m:
            matched_phrase = m.group(0)
            result = IntentResult(
                intent=intent,
                matched=True,
                matched_phrase=matched_phrase,
                suggested_mode=suggested,
                reason=f"matched '{matched_phrase}' → {intent}"
                + (f" (suggests {suggested})" if suggested else ""),
            )
            logger.info("Intent detected: %s", result.reason)
            return result

    return IntentResult()
