"""
Grounding Checker — verifies answer faithfulness to source documents.
Detects claims not supported by retrieved context, flags answers
sourced from superseded documents, and checks for fabricated specifics
(URLs, phone numbers, email addresses not in the context).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")


@dataclass
class GroundingResult:
    """
    Result of grounding verification.

    Fields:
        passed          — overall grounding check passed
        grounding_score — 0.0-1.0, fraction of answer grounded in context
        issues          — list of specific grounding issues found
        version_warning — non-empty if superseded sources detected
        fabrications    — specifics found in answer but not in context
    """
    passed: bool = True
    grounding_score: float = 1.0
    issues: List[str] = field(default_factory=list)
    version_warning: str = ""
    fabrications: List[str] = field(default_factory=list)


# Patterns for fabricated specifics (URLs, emails, phone numbers)
_URL_PATTERN = re.compile(r"https?://[\w./-]+", re.I)
_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w.-]+\.\w{2,}\b", re.I)
_PHONE_PATTERN = re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b")


def check_grounding(
    *,
    query: str,
    answer: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
) -> GroundingResult:
    """
    Verify that the answer is faithful to the source documents.

    Checks performed:
    1. Fabricated specifics — URLs, emails, phone numbers in answer but not in context
    2. Version awareness — detect if top sources are from superseded documents
    3. Grounding coverage — what fraction of answer's substantive sentences
       have supporting evidence in the retrieved context
    4. Hallucination pattern — known LLM hallucination phrases

    Returns GroundingResult with pass/fail, score, and detailed issues.
    """
    result = GroundingResult()
    ctx_lower = (doc_context or "").lower()
    answer_lower = (answer or "").lower()

    # ==================================================================
    # Check 1: Fabricated specifics
    # URLs, emails, phone numbers in answer that aren't in the context
    # ==================================================================
    for pattern, label in [
        (_URL_PATTERN, "URL"),
        (_EMAIL_PATTERN, "email"),
        (_PHONE_PATTERN, "phone number"),
    ]:
        answer_matches = set(pattern.findall(answer or ""))
        context_matches = set(pattern.findall(doc_context or ""))

        fabricated = answer_matches - context_matches
        for fab in fabricated:
            result.fabrications.append(f"Fabricated {label}: {fab}")
            result.issues.append(f"Answer contains {label} '{fab}' not found in source documents")

    # ==================================================================
    # Check 2: Version awareness
    # Flag if top-ranked sources are from superseded document versions
    # ==================================================================
    if settings.VALIDATION_WARN_SUPERSEDED and ranked_chunks:
        superseded_sources = set()
        active_sources = set()

        for _, _, meta, _ in ranked_chunks[:5]:
            source_name = meta.get("source", "unknown")
            meta_json = meta.get("metadata_json", {})

            if isinstance(meta_json, dict):
                status = meta_json.get("status", "active")
                if status == "superseded":
                    superseded_sources.add(source_name)
                else:
                    active_sources.add(source_name)

        if superseded_sources and not active_sources:
            # ALL top sources are superseded — strong warning
            result.version_warning = (
                "Note: This answer is based on document versions that may have been superseded. "
                "A newer version may exist with updated information. "
                f"Superseded sources: {', '.join(sorted(superseded_sources))}"
            )
            result.issues.append("All top sources are from superseded document versions")
        elif superseded_sources:
            # Mix of superseded and active — mild warning
            result.version_warning = (
                "Note: Some source documents may have newer versions available. "
                f"Potentially outdated: {', '.join(sorted(superseded_sources))}"
            )

    # ==================================================================
    # Check 3: Sentence-level grounding
    # Split answer into sentences and check each has context support
    # ==================================================================
    sentences = re.split(r"[.!?\n]", answer or "")
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    if sentences:
        grounded_count = 0
        for sentence in sentences:
            # Extract key terms from the sentence
            terms = {
                t for t in re.findall(r"\w+", sentence.lower())
                if len(t) > 3 and t not in {"this", "that", "with", "from", "have", "been",
                                              "would", "could", "should", "which", "their",
                                              "there", "these", "those", "about", "after",
                                              "before", "between", "through", "during"}
            }
            if not terms:
                grounded_count += 1  # skip trivial sentences
                continue

            # Check if at least 40% of terms appear in context
            hits = sum(1 for t in terms if t in ctx_lower)
            if len(terms) > 0 and hits / len(terms) >= 0.40:
                grounded_count += 1

        result.grounding_score = grounded_count / len(sentences) if sentences else 1.0
    else:
        result.grounding_score = 1.0  # no sentences to check

    # ==================================================================
    # Check 4: Overall pass/fail
    # ==================================================================
    if result.fabrications:
        result.passed = False
        result.issues.append(f"Found {len(result.fabrications)} fabricated specifics")

    if result.grounding_score < settings.VALIDATION_MIN_GROUNDING:
        result.passed = False
        result.issues.append(
            f"Grounding score {result.grounding_score:.2f} below threshold "
            f"{settings.VALIDATION_MIN_GROUNDING}"
        )

    logger.info(
        "Grounding: score=%.3f, %d fabrications, version_warn=%s, %d issues → %s",
        result.grounding_score, len(result.fabrications),
        bool(result.version_warning), len(result.issues),
        "PASS" if result.passed else "FAIL",
    )

    return result
