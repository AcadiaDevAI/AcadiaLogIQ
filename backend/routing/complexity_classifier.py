"""
Complexity Classifier — scores query difficulty for model routing.
Analyzes the query, retrieval results, and context signals to produce
a 0.0-1.0 complexity score. Low = Mistral, medium = Haiku, high = Sonnet.
No LLM call needed — uses fast heuristics + retrieval metadata.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ComplexityResult:
    """
    Holds the complexity classification for a query + retrieval context.

    Fields:
        score       — 0.0 (trivial) to 1.0 (very complex)
        tier        — 'simple' | 'moderate' | 'complex'
        signals     — dict of individual signal scores for logging
        reason      — human-readable explanation
    """
    score: float = 0.5
    tier: str = "moderate"
    signals: Dict[str, float] = field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# Query complexity patterns
# ---------------------------------------------------------------------------

# Multi-step / comparison patterns → higher complexity
_MULTI_STEP_PATTERNS = [
    re.compile(r"\b(?:compare|contrast|versus|vs\.?|differ(?:ence|ent))\b", re.I),
    re.compile(r"\b(?:step.by.step|walk me through|end.to.end|workflow)\b", re.I),
    re.compile(r"\b(?:and then|after that|next step|followed by|finally)\b", re.I),
    re.compile(r"\b(?:pros? and cons?|advantages? and disadvantages?|trade.?offs?)\b", re.I),
    re.compile(r"\b(?:all .{0,15} that|list every|enumerate all|each of the)\b", re.I),
]

# Reasoning / synthesis patterns → higher complexity
_REASONING_PATTERNS = [
    re.compile(r"\b(?:why does|why would|why is|why are|root cause|because)\b", re.I),
    re.compile(r"\b(?:what.if|hypothetical|scenario|assume|given that)\b", re.I),
    re.compile(r"\b(?:implication|consequence|impact|affect|effect on)\b", re.I),
    re.compile(r"\b(?:recommend|suggest|advise|should I|best approach)\b", re.I),
    re.compile(r"\b(?:analyze|evaluate|assess|determine whether)\b", re.I),
    re.compile(r"\b(?:summarize|synthesize|consolidate|combine)\b", re.I),
]

# Simple / definition patterns → lower complexity
_SIMPLE_PATTERNS = [
    re.compile(r"^what (?:is|are|does) \w+", re.I),
    re.compile(r"^(?:define|definition of|meaning of)\b", re.I),
    re.compile(r"^(?:show|list|find|get|tell me) (?:the|a|me)?\s*\w+", re.I),
    re.compile(r"\b(?:where is|who is|when was|when did)\b", re.I),
    re.compile(r"^(?:is|does|can|will|has) \w+\s+\w+\??$", re.I),
]


# ---------------------------------------------------------------------------
# Core classification function
# ---------------------------------------------------------------------------
def classify_complexity(
    *,
    query: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    retrieval_confidence: float = 0.5,
    source_count: int = 1,
    context_chars: int = 0,
) -> ComplexityResult:
    """
    Score query complexity based on multiple signals.

    Signals (each 0.0-1.0, weighted by config):
    1. multi_step    — query asks for comparison, workflow, or multi-part answer
    2. reasoning     — query requires inference, synthesis, or root-cause analysis
    3. context_size  — large retrieved context is harder to reason over
    4. low_confidence — low retrieval confidence means model must work harder
    5. multi_doc     — answer spans multiple source documents

    Returns ComplexityResult with score, tier, and signal breakdown.
    """
    query = (query or "").strip()
    signals: Dict[str, float] = {}

    # --- Signal 1: Multi-step detection ---
    multi_step_hits = sum(1 for p in _MULTI_STEP_PATTERNS if p.search(query))
    signals["multi_step"] = min(1.0, multi_step_hits * 0.4)

    # --- Signal 2: Reasoning / synthesis detection ---
    reasoning_hits = sum(1 for p in _REASONING_PATTERNS if p.search(query))
    signals["reasoning"] = min(1.0, reasoning_hits * 0.35)

    # --- Signal 3: Context size (more context = harder to synthesize) ---
    # Normalize: 0 chars → 0.0, 20000+ chars → 1.0
    signals["context_size"] = min(1.0, context_chars / 20000)

    # --- Signal 4: Low retrieval confidence ---
    # Invert: high confidence → low signal, low confidence → high signal
    signals["low_confidence"] = max(0.0, 1.0 - retrieval_confidence)

    # --- Signal 5: Multi-document span ---
    # Multiple source docs → higher complexity
    signals["multi_doc"] = min(1.0, max(0.0, (source_count - 1) * 0.3))

    # --- Simple query suppression ---
    # If query matches simple patterns, reduce overall score
    simple_hits = sum(1 for p in _SIMPLE_PATTERNS if p.search(query))
    simplicity_discount = min(0.4, simple_hits * 0.15)

    # --- Weighted score ---
    raw_score = (
        signals["multi_step"] * settings.CX_WEIGHT_MULTI_STEP
        + signals["reasoning"] * settings.CX_WEIGHT_REASONING
        + signals["context_size"] * settings.CX_WEIGHT_CONTEXT_SIZE
        + signals["low_confidence"] * settings.CX_WEIGHT_LOW_CONFIDENCE
        + signals["multi_doc"] * settings.CX_WEIGHT_MULTI_DOC
    )

    # Apply simplicity discount
    final_score = max(0.0, min(1.0, raw_score - simplicity_discount))

    # --- Determine tier ---
    if final_score < settings.COMPLEXITY_SIMPLE_THRESHOLD:
        tier = "simple"
    elif final_score > settings.COMPLEXITY_COMPLEX_THRESHOLD:
        tier = "complex"
    else:
        tier = "moderate"

    # --- Build reason string ---
    top_signals = sorted(signals.items(), key=lambda x: x[1], reverse=True)[:3]
    signal_str = ", ".join(f"{k}={v:.2f}" for k, v in top_signals if v > 0)
    reason = f"score={final_score:.3f} tier={tier}"
    if signal_str:
        reason += f" (top: {signal_str})"
    if simplicity_discount > 0:
        reason += f" [simple discount=-{simplicity_discount:.2f}]"

    logger.info("Complexity: %s", reason)

    return ComplexityResult(
        score=final_score,
        tier=tier,
        signals=signals,
        reason=reason,
    )
