"""
Trivial Input Detector — short-circuits greetings / thanks / one-word
chit-chat so they never hit retrieval, reranking, or LLM generation.
Returns a canned reply when a match is found; otherwise signals "proceed".
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("acadia-log-iq")


TRIVIAL_KIND_GREETING: str = "greeting"
TRIVIAL_KIND_THANKS: str = "thanks"
TRIVIAL_KIND_ACK: str = "acknowledgement"
TRIVIAL_KIND_BYE: str = "farewell"


_GREETING_RE = re.compile(
    r"^\s*(?:hi|hello|hey|hiya|yo|howdy|good\s+(?:morning|afternoon|evening))[\s!.?]*$",
    re.I,
)
_THANKS_RE = re.compile(
    r"^\s*(?:thanks|thank\s*you|thx|ty|appreciate(?:d)?\s*it|cheers)[\s!.?]*$",
    re.I,
)
_ACK_RE = re.compile(
    r"^\s*(?:ok|okay|cool|great|got\s*it|understood|sure|fine|alright)[\s!.?]*$",
    re.I,
)
_BYE_RE = re.compile(
    r"^\s*(?:bye|goodbye|see\s*ya|later|cya|take\s*care)[\s!.?]*$",
    re.I,
)


_CANNED = {
    TRIVIAL_KIND_GREETING: "Hello! Ask me anything about your uploaded documents.",
    TRIVIAL_KIND_THANKS: "You're welcome. Let me know if you have another question.",
    TRIVIAL_KIND_ACK: "Got it. Anything else I can help with?",
    TRIVIAL_KIND_BYE: "Goodbye. Come back anytime.",
}


@dataclass
class TrivialResult:
    """
    matched          — True if the input was classified as trivial
    kind             — one of TRIVIAL_KIND_* (or "" if not matched)
    canned_response  — pre-written reply to return (or "" if not matched)
    """
    matched: bool = False
    kind: str = ""
    canned_response: str = ""


# Guard: only treat VERY short inputs as trivial, so real questions like
# "thanks for the info, but how does failover work?" are NOT short-circuited.
_MAX_TRIVIAL_CHARS = 40


def detect_trivial(query: str) -> TrivialResult:
    """Return TrivialResult. Pure, safe on empty/None input, never raises."""
    if not query or not query.strip():
        return TrivialResult()
    q = query.strip()
    if len(q) > _MAX_TRIVIAL_CHARS:
        return TrivialResult()

    for kind, pat in (
        (TRIVIAL_KIND_GREETING, _GREETING_RE),
        (TRIVIAL_KIND_THANKS, _THANKS_RE),
        (TRIVIAL_KIND_ACK, _ACK_RE),
        (TRIVIAL_KIND_BYE, _BYE_RE),
    ):
        if pat.match(q):
            logger.info("Trivial input detected: kind=%s query=%r", kind, q)
            return TrivialResult(matched=True, kind=kind, canned_response=_CANNED[kind])
    return TrivialResult()
