"""
Context-break phrase detector — fast regex-based classifier for
"this is a new topic" signals inside a locked-mode session.

Based on PRD Section 2 phrase list:
  - "new issue"
  - "change context"
  - "different question"
  - "switch to <mode>"
  - "this is unrelated"
  - and close paraphrases

Mirrors trivial_responder.py: normalize (lowercase + punctuation strip +
whitespace collapse), exact-match against the registry, then bounded
substring match for short queries.

Returns a ContextBreakMatch with phrase + category when detected, or None.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger("acadia-log-iq")


CONTEXT_BREAK_REGISTRY: Dict[str, List[str]] = {
    "explicit_new_topic": [
        "new issue",
        "new question",
        "new topic",
        "different question",
        "different topic",
        "different issue",
        "unrelated question",
        "unrelated topic",
        "this is unrelated",
        "change topic",
        "change context",
        "switch topic",
        "switch context",
    ],
    "switch_to_mode": [
        "switch to escalation",
        "switch to vendor",
        "switch to ticket",
        "switch to troubleshooting",
        "change to escalation",
        "change to vendor",
        "change to ticket handling",
    ],
    "meta_reset": [
        "start over",
        "start fresh",
        "reset context",
        "forget this",
        "never mind this",
    ],
}

# Mirrors trivial_responder's bound — short queries benefit from substring
# match; longer queries are almost always genuine questions about the
# current topic and should NOT be mis-classified.
_MAX_WORDS_FOR_SUBSTRING_MATCH = 8

_PUNCT_RE = re.compile(r"[.,!?;:\"'`()\[\]{}]")
_WS_RE = re.compile(r"\s+")


@dataclass
class ContextBreakMatch:
    category: str
    matched_phrase: str
    confidence: float  # 1.0 = exact/substring; lower if we add LLM hint later


def _normalize(query: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    if not query:
        return ""
    q = query.strip().lower()
    q = _PUNCT_RE.sub(" ", q)
    q = _WS_RE.sub(" ", q).strip()
    return q


def detect_context_break(query: str) -> Optional[ContextBreakMatch]:
    """
    Regex/substring check against the PRD phrase list.

    Returns a ContextBreakMatch on a hit, or None. Never raises.
    """
    if not query:
        return None
    try:
        normalized = _normalize(query)
        if not normalized:
            return None
        word_count = len(normalized.split())

        for category, phrases in CONTEXT_BREAK_REGISTRY.items():
            for phrase in phrases:
                # 1. Exact match (rare but cheap check).
                if normalized == phrase:
                    return ContextBreakMatch(
                        category=category, matched_phrase=phrase, confidence=1.0,
                    )
                # 2. Substring match — only for short queries. Multi-word
                #    phrases are safe as-is; single-word phrases require
                #    word boundaries so "switch" inside "switch port" doesn't
                #    fire.
                if word_count > _MAX_WORDS_FOR_SUBSTRING_MATCH:
                    continue
                if " " in phrase:
                    if phrase in normalized:
                        return ContextBreakMatch(
                            category=category, matched_phrase=phrase, confidence=1.0,
                        )
                else:
                    if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                        return ContextBreakMatch(
                            category=category, matched_phrase=phrase, confidence=1.0,
                        )
        return None
    except Exception as exc:
        logger.warning("[context_break] detect failed: %s", exc)
        return None
