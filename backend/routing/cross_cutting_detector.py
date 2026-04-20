"""
Cross-cutting analytical intent detector.

Detects queries that require synthesis across multiple tickets rather than
structured aggregation (count/list/rank) or single-record retrieval. These
queries need the agent pipeline to read content, extract patterns, and
produce grouped insights — not a SQL result.

Designed to fail closed: when uncertain, returns low confidence so routing
falls through to existing classifiers. Zero risk of breaking existing paths.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Signal taxonomy — multi-category detection
# ─────────────────────────────────────────────────────────────
# Requires signals from at least 2 distinct categories to trigger.
# This prevents false positives like "list all X" (only 1 scope signal).

_PATTERN_KEYWORDS = {
    "common", "recurring", "pattern", "patterns", "theme", "themes",
    "trend", "trends", "typical", "frequent", "frequently",
}

_SYNTHESIS_VERBS = {
    "analyze", "analyzing", "analysis",
    "summarize", "synthesize",
    "identify", "identified",
    "extract",
    "recommended", "suggested", "proposed", "found", "observed",
    "emerge", "emerging", "emerged",
}

# Multi-word phrases checked via substring match.
_INSIGHT_NOUNS = {
    "root causes", "lessons", "improvements", "takeaways", "insights",
    "findings", "gaps", "weaknesses", "recommendations", "issues",
    "learnings", "observations",
    "resolution patterns",
}

_SCOPE_MARKERS = {
    "across", "across all", "across the", "overall", "in general",
    "all tickets", "every ticket", "most", "most frequently",
    "majority", "collectively", "throughout",
}

# Regex-based scope fallback: "all <noun>" / "every <noun>" broadens scope
# detection without hardcoding every domain noun (router, failures, etc).
_SCOPE_REGEXES = [
    re.compile(r"\ball\s+\w+", re.IGNORECASE),
    re.compile(r"\bevery\s+\w+", re.IGNORECASE),
]

_FREQUENCY_MARKERS = {
    "most frequently", "often", "repeatedly", "commonly",
    "most common", "most often", "usually",
}


@dataclass
class CrossCuttingResult:
    """Result from cross-cutting analytical detection."""
    is_analytical: bool
    confidence: float
    signals: List[str]
    matched_categories: List[str]
    mode: str  # "fast" or "deep"

    def __bool__(self) -> bool:
        return self.is_analytical


def _extract_signals(query_lower: str) -> dict:
    """Extract matching signals grouped by category."""
    matched = {
        "pattern": [],
        "synthesis": [],
        "insight": [],
        "scope": [],
        "frequency": [],
    }

    tokens = set(re.findall(r"\b\w+\b", query_lower))

    for kw in _PATTERN_KEYWORDS:
        if kw in tokens:
            matched["pattern"].append(kw)

    for verb in _SYNTHESIS_VERBS:
        if verb in tokens:
            matched["synthesis"].append(verb)

    for phrase in _INSIGHT_NOUNS:
        if phrase in query_lower:
            matched["insight"].append(phrase)

    for phrase in _SCOPE_MARKERS:
        if phrase in query_lower:
            matched["scope"].append(phrase)

    for rx in _SCOPE_REGEXES:
        for m in rx.findall(query_lower):
            if m not in matched["scope"]:
                matched["scope"].append(m)

    for phrase in _FREQUENCY_MARKERS:
        if phrase in query_lower:
            matched["frequency"].append(phrase)

    return matched


def _compute_confidence(matched: dict) -> float:
    """
    Compute confidence based on signal distribution.
    Rewards diversity across categories (pattern + insight = strong signal).
    """
    total_signals = sum(len(v) for v in matched.values())
    distinct_categories = sum(1 for v in matched.values() if v)

    if distinct_categories < 2:
        return 0.0

    # 2 cats=0.75, 3=0.85, 4=0.95, 5+=capped at 0.98
    base = 0.75 + (distinct_categories - 2) * 0.10

    if total_signals >= 4:
        base += 0.05
    if total_signals >= 6:
        base += 0.05

    return min(base, 0.98)


def _detect_mode(query_lower: str) -> str:
    """Detect fast vs deep analysis mode."""
    deep_triggers = {
        "deep analysis", "deeper analysis", "detailed analysis",
        "comprehensive analysis", "full analysis", "all tickets in detail",
    }
    for trigger in deep_triggers:
        if trigger in query_lower:
            return "deep"
    return "fast"


def detect_cross_cutting_analytical(query: str) -> CrossCuttingResult:
    """
    Detect whether a query requires cross-cutting analytical synthesis.

    Returns CrossCuttingResult with:
    - is_analytical: True iff 2+ distinct signal categories match above threshold
    - confidence: 0.0-0.98 score
    - signals: flat list of matched keywords/phrases
    - matched_categories: distinct categories that matched
    - mode: "fast" (default) or "deep" (user opted in)

    Fails closed — if confidence below threshold, is_analytical=False and
    routing falls through to existing classifiers.
    """
    if not query or not query.strip():
        return CrossCuttingResult(False, 0.0, [], [], "fast")

    query_lower = query.lower().strip()
    matched = _extract_signals(query_lower)
    confidence = _compute_confidence(matched)

    threshold = getattr(
        settings,
        "CROSS_CUTTING_DETECTOR_CONFIDENCE_THRESHOLD",
        0.75,
    )

    min_categories = getattr(
        settings,
        "CROSS_CUTTING_MIN_SIGNAL_CATEGORIES",
        2,
    )

    distinct_categories = sum(1 for v in matched.values() if v)
    is_analytical = (
        confidence >= threshold
        and distinct_categories >= min_categories
    )

    all_signals = [sig for signals in matched.values() for sig in signals]
    matched_categories = [cat for cat, sigs in matched.items() if sigs]
    mode = _detect_mode(query_lower)

    return CrossCuttingResult(
        is_analytical=is_analytical,
        confidence=confidence,
        signals=all_signals,
        matched_categories=matched_categories,
        mode=mode,
    )
