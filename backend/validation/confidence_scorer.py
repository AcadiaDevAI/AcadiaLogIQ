"""
Confidence Scorer — computes multi-signal confidence for an answer.
Blends retrieval strength, source coverage, grounding faithfulness,
and answer consistency into a single 0.0-1.0 confidence score.
Each signal is individually auditable for debugging.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")

# Stopwords excluded from term overlap calculations
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "to", "for", "of", "in", "on", "how",
    "what", "when", "where", "why", "do", "does", "can", "i", "me", "my",
    "you", "your", "please", "tell", "about", "and", "or", "not", "this",
    "that", "with", "from", "it", "be", "was", "were", "been", "have", "has",
    "will", "would", "could", "should", "may", "might", "shall", "must",
})


@dataclass
class ConfidenceResult:
    """
    Breakdown of the confidence score with per-signal detail.

    Fields:
        score           — final blended confidence 0.0-1.0
        retrieval_score — strength of retrieval evidence
        coverage_score  — query term coverage in retrieved context
        grounding_score — answer term support in source documents
        consistency_score — answer coherence / non-hallucination
        penalties       — dict of applied penalties and reasons
        passed          — whether score meets VALIDATION_MIN_CONFIDENCE
    """
    score: float = 0.0
    retrieval_score: float = 0.0
    coverage_score: float = 0.0
    grounding_score: float = 0.0
    consistency_score: float = 0.0
    penalties: Dict[str, float] = field(default_factory=dict)
    passed: bool = True


def _extract_terms(text: str) -> set:
    """Extract meaningful terms from text, excluding stopwords."""
    return {
        t for t in re.findall(r"\w+", (text or "").lower())
        if len(t) > 2 and t not in _STOP_WORDS
    }


def score_confidence(
    *,
    query: str,
    answer: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
) -> ConfidenceResult:
    """
    Compute a multi-signal confidence score for the generated answer.

    Signals:
    1. retrieval_score  — normalized top reranked chunk score
    2. coverage_score   — what fraction of query terms appear in doc context
    3. grounding_score  — what fraction of answer's key terms appear in doc context
    4. consistency_score — answer is non-empty, not a refusal, no hallucination patterns

    Penalties:
    - Hallucination phrases detected → consistency penalty
    - Superseded source documents → version penalty
    - Very short answer → consistency penalty
    """
    result = ConfidenceResult()

    # --- Signal 1: Retrieval strength ---
    # Normalize the top reranked chunk score to 0-1
    if ranked_chunks:
        top_score = float(ranked_chunks[0][3])
        # Scores from RRF fusion are typically 0.0-0.05 range, normalize
        result.retrieval_score = min(1.0, top_score * 20)
    else:
        result.retrieval_score = 0.0

    # --- Signal 2: Source coverage ---
    # What fraction of query terms appear in the retrieved document context
    q_terms = _extract_terms(query)
    if q_terms:
        ctx_lower = (doc_context or "").lower()
        hits = sum(1 for t in q_terms if t in ctx_lower)
        result.coverage_score = hits / len(q_terms)
    else:
        result.coverage_score = 1.0  # no terms to check → assume covered

    # --- Signal 3: Grounding / faithfulness ---
    # What fraction of the answer's key terms appear in the doc context
    answer_terms = _extract_terms(answer)
    if answer_terms and doc_context:
        ctx_lower = doc_context.lower()
        grounded = sum(1 for t in answer_terms if t in ctx_lower)
        result.grounding_score = grounded / len(answer_terms)
    else:
        result.grounding_score = 0.0

    # --- Signal 4: Consistency / coherence ---
    consistency = 1.0

    # Check for empty or very short answers
    stripped = (answer or "").strip()
    if not stripped:
        consistency = 0.0
    elif len(stripped) < 20:
        consistency -= 0.3

    # Check for known refusal patterns (these are actually good — the model is honest)
    refusal_patterns = [
        "could not find supporting information",
        "not explicitly supported",
        "insufficient evidence",
    ]
    is_refusal = any(p in stripped.lower() for p in refusal_patterns)
    if is_refusal:
        # Refusals are honest — don't penalize confidence, but flag
        consistency = 0.5  # moderate: the model is being safe

    # Check for hallucination phrases
    halluc_hits = 0
    for phrase in settings.VALIDATION_HALLUCINATION_PHRASES:
        if phrase.lower() in stripped.lower():
            halluc_hits += 1
    if halluc_hits > 0:
        penalty = min(0.4, halluc_hits * 0.15)
        consistency -= penalty
        result.penalties["hallucination_phrases"] = penalty

    consistency = max(0.0, min(1.0, consistency))
    result.consistency_score = consistency

    # --- Penalties: version awareness ---
    # Check if sources include superseded documents
    if settings.VALIDATION_WARN_SUPERSEDED and ranked_chunks:
        for _, _, meta, _ in ranked_chunks[:3]:
            meta_json = meta.get("metadata_json", {})
            if isinstance(meta_json, dict):
                status = meta_json.get("status", "")
                if status == "superseded":
                    result.penalties["superseded_source"] = settings.VALIDATION_SUPERSEDED_PENALTY
                    break

    # --- Weighted blend ---
    raw_score = (
        result.retrieval_score * settings.CONF_WEIGHT_RETRIEVAL
        + result.coverage_score * settings.CONF_WEIGHT_COVERAGE
        + result.grounding_score * settings.CONF_WEIGHT_GROUNDING
        + result.consistency_score * settings.CONF_WEIGHT_CONSISTENCY
    )

    # Apply penalties
    total_penalty = sum(result.penalties.values())
    final_score = max(0.0, min(1.0, raw_score - total_penalty))

    result.score = final_score
    result.passed = final_score >= settings.VALIDATION_MIN_CONFIDENCE

    logger.info(
        "Confidence: %.3f (ret=%.2f cov=%.2f gnd=%.2f con=%.2f pen=%.2f) → %s",
        final_score, result.retrieval_score, result.coverage_score,
        result.grounding_score, result.consistency_score,
        total_penalty, "PASS" if result.passed else "FAIL",
    )

    return result
