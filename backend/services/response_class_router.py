"""
Response class classifier (Brief 5 / Part 2).

Maps each query to a response-class tier that drives the generation
`max_tokens` budget. Zero LLM calls — regex-only heuristics layered on
top of merged-triage signals when Brief 4 Opt 3 is enabled.

Tiers:
    classification   30     yes/no, is/was/how-many single-number
    short_fact      120     who/when/where/which single-hop
    explanation     350     why/what caused, root-cause, QA gaps
    walkthrough     600     tell-me-about, walk-me-through, full story
    analytical     1200     compare/patterns/common-root-causes/synthesis
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Any, Dict, Optional

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


class ResponseClass(str, Enum):
    CLASSIFICATION = "classification"
    SHORT_FACT = "short_fact"
    EXPLANATION = "explanation"
    WALKTHROUGH = "walkthrough"
    ANALYTICAL = "analytical"


# ---------------------------------------------------------------------------
# Heuristic patterns (evaluated in tier-priority order: analytical first)
# ---------------------------------------------------------------------------
_CLASSIFICATION_PATTERNS = [
    re.compile(r"^\s*(is|was|did|does|are|were)\b.{0,60}\?\s*$", re.I),
    re.compile(r"^\s*(how many|count of|total|number of)\b.{0,60}\??\s*$", re.I),
]

_SHORT_FACT_PATTERNS = [
    re.compile(r"\b(who|when|where|which)\s+(resolved|owns|works|handled|assigned|caused|opened|closed|escalated)\b", re.I),
    re.compile(r"\bSLA\s+(status|met|missed|target)\b", re.I),
    re.compile(r"\b(customer|priority|component|status|owner|assignee)\s+(of|for)\b", re.I),
    re.compile(r"\bquality\s+score\b", re.I),
]

_EXPLANATION_PATTERNS = [
    re.compile(r"\b(why|what)\b.{0,80}\b(caused|cause|happened|happen|reason|reasons)\b", re.I),
    re.compile(r"\broot\s+cause\b", re.I),
    re.compile(r"\bhow\s+(was|were|did)\b.{0,80}\b(resolved|fixed|repaired|mitigated)\b", re.I),
    re.compile(r"\bQA\s+(gap|auditor|finding)", re.I),
    re.compile(r"\b5[- ]why\b", re.I),
    re.compile(r"\bwhat\s+resolutions?\s+(were|was|applied)\b", re.I),
]

_WALKTHROUGH_PATTERNS = [
    re.compile(r"\btell me (about|everything about|more about)\b", re.I),
    re.compile(r"\b(walk (me )?through|describe|explain in detail)\b", re.I),
    re.compile(r"\bwhat\s+happened\s+(with|at|to)\b", re.I),
    re.compile(r"\bfull (story|detail|report)\b", re.I),
]

_ANALYTICAL_PATTERNS = [
    re.compile(r"\bcompare\b", re.I),
    re.compile(r"\b(patterns?|trends?|themes?|commonalities?)\s+(across|among|between|in|for)\b", re.I),
    re.compile(r"\b(common|shared)\s+root\s+causes?\b", re.I),
    re.compile(r"\banalyze\b", re.I),
    re.compile(r"\b(summarize|summary of)\b.{0,80}\b(tickets?|incidents?|cases?)\b", re.I),
    re.compile(r"\bwhat\s+are\s+the\s+common\b", re.I),
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def classify_response_class(
    query: str,
    context: Optional[Dict[str, Any]] = None,
) -> ResponseClass:
    """
    Decide the response class for a query.
    Priority:
      1. Precomputed merged-triage signals (Brief 4 Opt 3), when supplied.
      2. Regex heuristics in tier-priority order.
      3. Default: EXPLANATION (safe middle ground, still cheaper than 1500).
    """
    q = (query or "").strip().lower()

    # 1. Triage signals, if the caller threaded them in
    if context and "merged_triage" in context:
        triage = context.get("merged_triage") or {}
        intent = str(triage.get("intent") or "").lower()
        complexity = str(triage.get("complexity") or "").lower()
        if intent in {"compare", "synthesis", "analyze"} or complexity == "complex":
            return ResponseClass.ANALYTICAL
        if intent == "aggregation":
            for p in _CLASSIFICATION_PATTERNS:
                if p.search(q):
                    return ResponseClass.CLASSIFICATION
            return ResponseClass.SHORT_FACT

    # 2. Heuristic pattern match in tier-priority order
    for p in _ANALYTICAL_PATTERNS:
        if p.search(q):
            return ResponseClass.ANALYTICAL
    for p in _WALKTHROUGH_PATTERNS:
        if p.search(q):
            return ResponseClass.WALKTHROUGH
    for p in _EXPLANATION_PATTERNS:
        if p.search(q):
            return ResponseClass.EXPLANATION
    for p in _SHORT_FACT_PATTERNS:
        if p.search(q):
            return ResponseClass.SHORT_FACT
    for p in _CLASSIFICATION_PATTERNS:
        if p.search(q):
            return ResponseClass.CLASSIFICATION

    # 3. Default fallback
    return ResponseClass.EXPLANATION


def get_token_cap(response_class: ResponseClass) -> int:
    return {
        ResponseClass.CLASSIFICATION: settings.RESPONSE_TOKENS_CLASSIFICATION,
        ResponseClass.SHORT_FACT:     settings.RESPONSE_TOKENS_SHORT_FACT,
        ResponseClass.EXPLANATION:    settings.RESPONSE_TOKENS_EXPLANATION,
        ResponseClass.WALKTHROUGH:    settings.RESPONSE_TOKENS_WALKTHROUGH,
        ResponseClass.ANALYTICAL:     settings.RESPONSE_TOKENS_ANALYTICAL,
    }[response_class]


def next_tier(response_class: ResponseClass) -> Optional[ResponseClass]:
    order = [
        ResponseClass.CLASSIFICATION,
        ResponseClass.SHORT_FACT,
        ResponseClass.EXPLANATION,
        ResponseClass.WALKTHROUGH,
        ResponseClass.ANALYTICAL,
    ]
    try:
        idx = order.index(response_class)
    except ValueError:
        return None
    if idx + 1 < len(order):
        return order[idx + 1]
    return None


def is_truncated(answer: str, max_tokens: int) -> bool:
    """Cheap heuristic: answer crowds the cap AND doesn't end in sentence punct."""
    if not answer:
        return False
    approx_tokens = len(answer) // 4  # ~4 chars per token
    if approx_tokens < int(max_tokens * 0.9):
        return False
    last_char = answer.rstrip()[-1:] if answer.rstrip() else ""
    return last_char not in {".", "!", "?", '"', ")", "]", "}"}
