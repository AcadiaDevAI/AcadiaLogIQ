"""
Answer Validator — orchestrates confidence scoring, grounding checks,
version-awareness, and retry/fallback logic before returning the final answer.
Called by the /ask endpoint after answer generation (Phase 4/5).
Returns a validated answer with calibrated confidence and any warnings.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.validation.confidence_scorer import ConfidenceResult, score_confidence
from backend.validation.grounding_checker import GroundingResult, check_grounding

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Fallback answer templates
# ---------------------------------------------------------------------------
_FALLBACK_INSUFFICIENT = (
    "- I could not find sufficiently supported information for that question "
    "in the currently uploaded files.\n"
    "- Please try rephrasing your question or ensure the relevant documents are uploaded."
)

_FALLBACK_GROUNDING_FAIL = (
    "- The generated answer could not be fully verified against the source documents.\n"
    "- Please try a more specific question that is directly covered by the uploaded content."
)

# Soft caveat appended when grounding is low but no fabrications found
_GROUNDING_CAVEAT = (
    "\n\n⚠️ *Note: This answer may not be fully verifiable against the source documents. "
    "Please cross-check with the original document for accuracy.*"
)


# ---------------------------------------------------------------------------
# Validation result
# ---------------------------------------------------------------------------
@dataclass
class ValidationResult:
    """
    Complete result from the validation pipeline.

    Fields:
        answer          — the final answer (original, modified, or fallback)
        confidence      — calibrated confidence score 0.0-1.0
        passed          — whether validation passed
        was_modified    — whether the answer was changed by validation
        confidence_detail — full ConfidenceResult breakdown
        grounding_detail  — full GroundingResult breakdown
        version_warning   — warning about superseded sources (if any)
        issues          — list of validation issues found
        validation_ms   — time spent on validation
    """
    answer: str = ""
    confidence: float = 0.0
    passed: bool = True
    was_modified: bool = False
    confidence_detail: Optional[ConfidenceResult] = None
    grounding_detail: Optional[GroundingResult] = None
    version_warning: str = ""
    issues: List[str] = field(default_factory=list)
    validation_ms: int = 0


# ---------------------------------------------------------------------------
# Evaluation logging
# ---------------------------------------------------------------------------
def _log_eval_record(
    *,
    query: str,
    answer: str,
    validation: ValidationResult,
    source_names: List[str],
    model_used: str,
) -> None:
    """
    Write an evaluation-ready JSONL record for offline analysis.
    Contains the query, answer, confidence breakdown, grounding results,
    and metadata needed for benchmarking.
    """
    if not settings.ENABLE_EVAL_LOGGING:
        return

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "answer": answer[:1000],
        "confidence": validation.confidence,
        "passed": validation.passed,
        "was_modified": validation.was_modified,
        "retrieval_score": validation.confidence_detail.retrieval_score if validation.confidence_detail else None,
        "coverage_score": validation.confidence_detail.coverage_score if validation.confidence_detail else None,
        "grounding_score": validation.confidence_detail.grounding_score if validation.confidence_detail else None,
        "consistency_score": validation.confidence_detail.consistency_score if validation.confidence_detail else None,
        "grounding_passed": validation.grounding_detail.passed if validation.grounding_detail else None,
        "fabrications": len(validation.grounding_detail.fabrications) if validation.grounding_detail else 0,
        "version_warning": bool(validation.version_warning),
        "issues": validation.issues,
        "sources": source_names,
        "model_used": model_used,
    }

    # Log to structured logger for centralized collection
    logger.info("EVAL_RECORD: %s", json.dumps(record, default=str))

    # Optionally write to file
    if settings.EVAL_LOG_FILE:
        try:
            path = Path(settings.EVAL_LOG_FILE)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as exc:
            logger.warning("Failed to write eval log to %s: %s", settings.EVAL_LOG_FILE, exc)


# ---------------------------------------------------------------------------
# Main validation entry point
# ---------------------------------------------------------------------------
def validate_answer(
    *,
    query: str,
    answer: str,
    doc_context: str,
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    source_names: List[str],
    model_used: str = "unknown",
    retrieval_stats: Optional[Dict[str, Any]] = None,
    retry_fn: Optional[Callable[..., str]] = None,
) -> ValidationResult:
    """
    Run the full validation pipeline on a generated answer.

    Pipeline:
    1. Confidence scoring (4-signal blend)
    2. Grounding verification (fabrication + faithfulness + version checks)
    3. Decision: pass / modify / fallback
    4. Log evaluation record for offline analysis

    Decision logic:
    - If confidence passes AND grounding passes → return original answer
    - If grounding finds fabricated specifics → strip them and append caveat
    - If grounding detects only superseded sources → append version warning
    - If confidence or grounding fail → return safe fallback

    Returns ValidationResult with the final answer, confidence, and details.
    """
    # Fix 7 (c): a canonical "not found" message must bypass validation.
    # It's not a model refusal — it's a deterministic response the caller
    # produced when retrieval signalled no such record. The false-refusal
    # guard below would otherwise misread it as a failure.
    if retrieval_stats and retrieval_stats.get("search_mode") in (
        "identifier_not_found",
        "ticket_id_not_found",
    ):
        return ValidationResult(
            answer=answer,
            confidence=1.0,
            passed=True,
            was_modified=False,
        )

    if not settings.ENABLE_ANSWER_VALIDATION:
        # Validation disabled — pass through with simple confidence
        return ValidationResult(
            answer=answer,
            confidence=min(0.3 + len(ranked_chunks) * 0.1, 1.0),
            passed=True,
        )

    t_start = time.perf_counter()
    result = ValidationResult(answer=answer)

    # ==================================================================
    # Step 0: False-refusal detection
    # ==================================================================
    # If the model says "could not find" but retrieval returned strong
    # chunks, the model is incorrectly refusing. Mark it so we can
    # override later.
    # Goal 2.2: only fire on genuine "I cannot find anything" statements.
    # Partial-answer hedges ("do not contain specific X, but Y is...") are
    # acceptable and must not be misclassified as refusals.
    _NOT_FOUND_PHRASES = [
        "could not find supporting information",
        "could not find sufficiently supported",
        "not contain the answer",
        "no relevant information was found",
        "i was unable to find",
    ]
    answer_lower = answer.lower()
    is_model_refusal = any(phrase in answer_lower for phrase in _NOT_FOUND_PHRASES)
    retrieval_is_strong = (
        len(ranked_chunks) >= 3
        and float(ranked_chunks[0][3] or 0) >= 0.4
    )
    is_false_refusal = is_model_refusal and retrieval_is_strong

    if is_false_refusal:
        logger.warning(
            "FALSE REFUSAL detected: model said 'not found' but retrieval "
            "top_score=%.3f with %d chunks. Lowering confidence to trigger re-evaluation.",
            float(ranked_chunks[0][3] or 0), len(ranked_chunks),
        )

    # ==================================================================
    # Step 1: Confidence scoring
    # ==================================================================
    conf = score_confidence(
        query=query,
        answer=answer,
        doc_context=doc_context,
        ranked_chunks=ranked_chunks,
        source_names=source_names,
    )
    result.confidence_detail = conf
    result.confidence = conf.score

    # ==================================================================
    # Step 2: Grounding verification
    # ==================================================================
    grounding = check_grounding(
        query=query,
        answer=answer,
        doc_context=doc_context,
        ranked_chunks=ranked_chunks,
        source_names=source_names,
    )
    result.grounding_detail = grounding
    result.version_warning = grounding.version_warning
    result.issues = list(grounding.issues)

    # ==================================================================
    # Step 3: Decision logic
    # ==================================================================

    # Case A: Both confidence and grounding pass → return as-is
    if conf.passed and grounding.passed:
        result.passed = True

        # Append version warning if relevant (informational, not a failure)
        if grounding.version_warning:
            result.answer = answer.rstrip() + "\n\n" + grounding.version_warning
            result.was_modified = True

        logger.info("Validation PASSED: confidence=%.3f, grounding=%.3f", conf.score, grounding.grounding_score)

    # Case B: Grounding found fabricated specifics → return fallback
    elif grounding.fabrications:
        result.passed = False
        result.was_modified = True
        result.answer = _FALLBACK_GROUNDING_FAIL
        result.confidence = max(0.0, conf.score * 0.5)  # halve confidence
        result.issues.append("Answer contained fabricated specifics — replaced with safe fallback")
        logger.warning("Validation FAILED (fabrications): %s", grounding.fabrications)

    # Case C: Low confidence → fallback
    elif not conf.passed:
        result.passed = False
        result.was_modified = True
        result.answer = _FALLBACK_INSUFFICIENT
        result.confidence = conf.score
        result.issues.append(
            f"Confidence {conf.score:.3f} below threshold {settings.VALIDATION_MIN_CONFIDENCE}"
        )
        logger.warning("Validation FAILED (low confidence): %.3f", conf.score)

    # Case D: Grounding failed (low grounding score but no fabrications)
    # KEY FIX: If confidence passed, the answer is likely correct but the
    # grounding checker couldn't fully verify it (common for short queries
    # like "QoS Trust Boundaries?" where the answer spans a small section).
    # Instead of replacing the entire answer, append a soft caveat.
    # Only use full fallback if grounding is extremely low.
    elif not grounding.passed:
        if conf.passed and grounding.grounding_score >= 0.15:
            # Confidence is OK, grounding is just below threshold — keep answer + caveat
            result.passed = True
            result.was_modified = True
            result.answer = answer.rstrip() + _GROUNDING_CAVEAT
            result.confidence = max(0.0, conf.score * 0.8)
            result.issues.append(
                f"Grounding score {grounding.grounding_score:.3f} below threshold "
                f"{settings.VALIDATION_MIN_GROUNDING} — appended caveat"
            )
            logger.info(
                "Validation SOFT PASS (low grounding but confidence OK): grounding=%.3f, confidence=%.3f",
                grounding.grounding_score, conf.score,
            )
        else:
            # Grounding is very low — full fallback
            result.passed = False
            result.was_modified = True
            result.answer = _FALLBACK_GROUNDING_FAIL
            result.confidence = max(0.0, conf.score * 0.7)
            result.issues.append(
                f"Grounding score {grounding.grounding_score:.3f} below threshold "
                f"{settings.VALIDATION_MIN_GROUNDING}"
            )
            logger.warning("Validation FAILED (low grounding): %.3f", grounding.grounding_score)

    # ==================================================================
    # Step 3b: False-refusal safety net with retry path
    # ==================================================================
    # Goal 2.3: when the model hedges despite strong retrieval, give it
    # one more shot with an explicit "the answer IS in the documents"
    # directive before falling back to the canned message. Only retry when
    # retrieval was deterministic enough that we're confident the answer
    # is actually present.
    _RETRIEVAL_MODES_OK_FOR_RETRY = {
        "identifier_exact",
        "ticket_id_exact",  # legacy alias
        "hybrid_phase3 (strategy=keyword)",
        "hybrid_phase3 (strategy=semantic)",
        "hybrid_phase3 (strategy=mixed)",
    }
    _search_mode = (retrieval_stats or {}).get("search_mode") if retrieval_stats else None

    if is_false_refusal and result.passed:
        retried = False
        if retry_fn is not None and _search_mode in _RETRIEVAL_MODES_OK_FOR_RETRY:
            try:
                logger.warning(
                    "[validator] false refusal detected (mode=%s) — attempting stronger-prompt retry",
                    _search_mode,
                )
                retry_answer = retry_fn(stronger_prompt=True)
                retried = True
                if retry_answer and retry_answer.strip():
                    retry_lower = retry_answer.lower()
                    still_refused = any(p in retry_lower for p in _NOT_FOUND_PHRASES)
                    if not still_refused:
                        result.answer = retry_answer
                        result.was_modified = True
                        result.issues.append("Recovered from false refusal via stronger-prompt retry")
                        logger.info("[validator] retry succeeded — returning retried answer")
                        result.validation_ms = int((time.perf_counter() - t_start) * 1000)
                        _log_eval_record(
                            query=query,
                            answer=result.answer,
                            validation=result,
                            source_names=source_names,
                            model_used=model_used,
                        )
                        return result
                    logger.warning("[validator] false refusal survived retry")
            except Exception as retry_exc:
                logger.warning("[validator] retry_fn raised: %s", retry_exc)

        result.passed = False
        result.was_modified = True
        result.confidence = 0.1
        result.answer = (
            "I found relevant documents but was unable to extract the answer. "
            "This is a known issue being addressed. Please try asking again."
        )
        result.issues.append(
            f"False refusal: model said 'not found' but retrieval top_score="
            f"{float(ranked_chunks[0][3] or 0):.3f} with {len(ranked_chunks)} chunks"
            + (" (retry also refused)" if retried else "")
        )
        logger.error(
            "FALSE REFUSAL: model refused despite strong retrieval "
            "(top_score=%.3f, %d chunks, mode=%s, retried=%s).",
            float(ranked_chunks[0][3] or 0), len(ranked_chunks), _search_mode, retried,
        )

    result.validation_ms = int((time.perf_counter() - t_start) * 1000)

    # ==================================================================
    # Step 4: Evaluation logging
    # ==================================================================
    _log_eval_record(
        query=query,
        answer=result.answer,
        validation=result,
        source_names=source_names,
        model_used=model_used,
    )

    logger.info(
        "Validation complete: passed=%s, confidence=%.3f, modified=%s, %d issues, %dms",
        result.passed, result.confidence, result.was_modified,
        len(result.issues), result.validation_ms,
    )

    return result