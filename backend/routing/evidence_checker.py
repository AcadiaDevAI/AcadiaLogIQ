"""
Evidence Checker — post-generation weak-evidence detector.
Flags answers that are too short, lack any overlap with the retrieved doc
context, or have zero sources. Adds metadata; never mutates the answer.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
MIN_ANSWER_CHARS: int = 60            # answers shorter than this look weak
MIN_OVERLAP_TOKENS: int = 3           # # of answer tokens that must appear in the doc context
MIN_CONFIDENCE: float = 0.35          # below this = weak
MIN_SOURCES: int = 1                  # at least one source expected

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{2,}")


@dataclass
class EvidenceResult:
    """
    weak              — True if any evidence check failed
    reasons           — list of reason codes that failed (for logging/stats)
    answer_chars      — len(answer.strip())
    overlap_tokens    — distinct answer-tokens that appear in doc context
    source_count      — len(source_names)
    """
    weak: bool = False
    reasons: List[str] = field(default_factory=list)
    answer_chars: int = 0
    overlap_tokens: int = 0
    source_count: int = 0


def _token_set(text: str) -> set:
    if not text:
        return set()
    return {m.group(0).lower() for m in _WORD_RE.finditer(text)}


def check_evidence(
    *,
    answer: str,
    doc_context: Optional[str],
    source_names: Optional[List[str]],
    confidence: float,
) -> EvidenceResult:
    """
    Inspect the generated answer against its retrieval context.
    Never raises; any internal error yields an EvidenceResult marked
    with reason 'check_error' and weak=False (fail-open on metadata).
    """
    try:
        a = (answer or "").strip()
        ctx = doc_context or ""
        srcs = list(source_names or [])
        result = EvidenceResult(
            answer_chars=len(a),
            source_count=len(srcs),
        )
        reasons: List[str] = []

        # --- length check ---
        if result.answer_chars < MIN_ANSWER_CHARS:
            reasons.append("short_answer")

        # --- source count ---
        if result.source_count < MIN_SOURCES:
            reasons.append("no_sources")

        # --- confidence ---
        try:
            if float(confidence) < MIN_CONFIDENCE:
                reasons.append("low_confidence")
        except (TypeError, ValueError):
            reasons.append("invalid_confidence")

        # --- token overlap between answer and doc context ---
        if a and ctx:
            a_tokens = _token_set(a)
            ctx_tokens = _token_set(ctx)
            overlap = a_tokens & ctx_tokens
            result.overlap_tokens = len(overlap)
            if len(overlap) < MIN_OVERLAP_TOKENS:
                reasons.append("low_context_overlap")
        else:
            result.overlap_tokens = 0
            reasons.append("empty_answer_or_context")

        result.weak = bool(reasons)
        result.reasons = reasons
        if result.weak:
            logger.info("Evidence check weak: reasons=%s chars=%d overlap=%d sources=%d",
                        reasons, result.answer_chars, result.overlap_tokens, result.source_count)
        return result
    except Exception as exc:
        logger.warning("Evidence checker raised (%s) — returning non-weak", exc)
        return EvidenceResult(weak=False, reasons=[f"check_error: {exc}"])
