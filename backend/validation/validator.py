"""
Answer Validator — orchestrates confidence scoring, grounding checks,
version-awareness, and retry/fallback logic before returning the final answer.
Called by the /ask endpoint after answer generation (Phase 4/5).
Returns a validated answer with calibrated confidence and any warnings.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from backend.config import settings
from backend.validation.confidence_scorer import ConfidenceResult, score_confidence
from backend.validation.grounding_checker import GroundingResult, check_grounding
from backend.validation.relevancy_checker import (
    RelevancyResult,
    check_relevancy,
    get_off_topic_fallback,
)

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Fallback answer templates
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Dynamic (phrase-list-free) refusal detection
# ---------------------------------------------------------------------------
# Detect refusals by STRUCTURE, not exact phrases. The LLM produces dozens
# of refusal wording variants:
#   "I cannot find specific information about ..."
#   "Based on the provided documents, the X is not detailed."
#   "The retrieved context does not contain ..."
#   "I'm unable to determine the exact X from ..."
#   "There is no information about ..."
#   "Documents do not specify ..."
# Maintaining a phrase list for every variant is brittle. The two regexes
# below catch the underlying structure instead:
#   (1) an opening pattern that begins with a refusal (covers >80% of cases)
#   (2) a negated modal/auxiliary within ~50 chars of a content-search verb
# Both fire only when the answer is short — long answers that incidentally
# include "cannot find" inside a paragraph are real answers, not refusals.

# Pattern (1): refusal openings — match at the start of the answer.
_REFUSAL_OPENING_RE = re.compile(
    r"^\s*(?:"
    r"i\s+(?:cannot|can'?t|could\s+not|couldn'?t|do\s+not|don'?t|"
    r"am\s+unable|don'?t\s+have|lack|fail\s+to)"
    r"|"
    r"(?:the|these|those)\s+(?:document|file|source|context|chunk|"
    r"passage|provided|available|retrieved|given)s?\s+"
    r"(?:do(?:es)?\s+not|don'?t|doesn'?t)"
    r"|"
    r"there\s+(?:is|are)\s+no\b"
    r"|"
    r"based\s+on\s+(?:the\s+)?(?:provided|available|retrieved|given|"
    r"current)\b[^.]{0,80}(?:cannot|can'?t|do(?:es)?\s+not|don'?t|"
    r"doesn'?t|unable|no\s+(?:information|specific|mention))"
    r")",
    re.IGNORECASE,
)

# Pattern (2): negated modal/auxiliary near a content-search verb.
# Catches "cannot find", "do not contain", "unable to provide",
# "lack details about", "have no information on", etc.
_HEDGE_NEGATION_RE = re.compile(
    r"\b(?:"
    r"cannot|can'?t|"
    r"could\s+not|couldn'?t|"
    r"do(?:es)?\s+not|do(?:es)?n'?t|"
    r"did\s+not|didn'?t|"
    r"am\s+unable|is\s+unable|are\s+unable|unable\s+to|"
    r"is\s+not|are\s+not|isn'?t|aren'?t|was\s+not|were\s+not|wasn'?t|weren'?t|"
    r"have\s+no|has\s+no|had\s+no|"
    r"don'?t\s+have|doesn'?t\s+have|"
    r"lack(?:s|ed)?|"
    r"no\s+(?:information|specific|mention|details?|content|reference|data)"
    r")\b"
    r"[\s\S]{0,50}?"
    r"\b(?:"
    r"find|finding|found|"
    r"contain|contains?|containing|"
    r"see|seen|seeing|"
    r"show|shown|showing|"
    r"provide|provided|providing|"
    r"mention|mentions?|mentioned|"
    r"specif(?:y|ies|ied|ic|ically)|"
    r"detail(?:ed|s)?|"
    r"include|included|including|"
    r"cover|covered|covering|"
    r"state|stated|stating|"
    r"address(?:ed|es)?|"
    r"answer|answering|answered|"
    r"discuss(?:ed|es)?|"
    r"describe|described|"
    r"reference|references?|"
    r"information|content|specifics?|"
    r"verif(?:y|ied)"
    r")\b",
    re.IGNORECASE,
)

# Refusals are typically terse — a real grounded answer with detail is
# longer than this. A long answer that includes "cannot find" buried in
# a sub-clause is almost always a real answer that hedges on one point,
# not a wholesale refusal.
_REFUSAL_MAX_CHARS = 600


def _looks_like_structural_refusal(answer: str) -> bool:
    """
    Return True when the answer exhibits refusal STRUCTURE.

    Designed to replace ever-growing phrase lists. Two signals fire:
      * `_REFUSAL_OPENING_RE` — the answer opens with a known refusal
        construction (covers ~80% of cases).
      * `_HEDGE_NEGATION_RE` — a negated modal verb is within ~50
        chars of a content-search verb (catches in-body refusals).

    Both signals are gated on `len(answer) <= _REFUSAL_MAX_CHARS` so a
    long, substantive answer that incidentally uses "cannot find" is
    NOT flagged.

    Never raises. Returns False on empty input.
    """
    if not answer:
        return False
    stripped = answer.strip()
    if not stripped or len(stripped) > _REFUSAL_MAX_CHARS:
        return False
    if _REFUSAL_OPENING_RE.search(stripped):
        return True
    if _HEDGE_NEGATION_RE.search(stripped):
        return True
    return False


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
    # New: response-relevancy judge result. None when the check was
    # disabled / skipped. Allows the eval log + context_stats to surface
    # whether the answer was off-topic separately from other failures.
    relevancy_detail: Optional[RelevancyResult] = None
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
    # Fast-path: exact-phrase match for the handful of refusal strings
    # we've historically seen verbatim. Kept for back-compat / cheap hit.
    _NOT_FOUND_PHRASES = [
        "could not find supporting information",
        "could not find sufficiently supported",
        "not contain the answer",
        "no relevant information was found",
        "i was unable to find",
    ]
    answer_lower = answer.lower()
    # Structural detector covers the long tail of refusal phrasings that
    # the static list above can't enumerate. See
    # `_looks_like_structural_refusal` for the patterns. ORed together so
    # either path is sufficient — false-refusal RETRY logic downstream
    # then decides whether to attempt a stronger-prompt re-generation.
    is_model_refusal = (
        any(phrase in answer_lower for phrase in _NOT_FOUND_PHRASES)
        or _looks_like_structural_refusal(answer)
    )
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
    # Step 2b: Response-relevancy check (Haiku judge)
    # ==================================================================
    # Catches the "grounded but off-topic" failure mode: the answer is
    # faithful to the documents but does not address the user's question.
    # Skipped when:
    #   * disabled via ENABLE_RELEVANCY_CHECK, OR
    #   * grounding already detected fabrications (Case B takes priority
    #     and we don't waste a Haiku call on an answer we're throwing
    #     away anyway).
    # Fails open: if Haiku is unreachable, RelevancyResult.skipped=True
    # and the answer passes this stage. Better than blocking legitimate
    # answers during a Bedrock outage.
    if not grounding.fabrications:
        relevancy = check_relevancy(query=query, answer=answer)
    else:
        relevancy = RelevancyResult(skipped=True)
    result.relevancy_detail = relevancy

    # ==================================================================
    # Step 3: Decision logic
    # ==================================================================

    # Case A: Both confidence and grounding pass → check relevancy, then return.
    if conf.passed and grounding.passed:
        # Case E (off-topic) takes precedence over Case A when the
        # relevancy judge produced a confident "no". `skipped` results
        # do NOT block — see check_relevancy fail-open contract.
        if (
            relevancy is not None
            and not relevancy.skipped
            and not relevancy.passed
        ):
            # ──────────────────────────────────────────────────────────
            # Soft-refusal retry (relevancy-driven).
            # ──────────────────────────────────────────────────────────
            # A confidence-and-grounding-passing answer that the relevancy
            # judge still rejected is the classic "Haiku discussed the
            # topic but hedged on the specific ask" pattern. Length is
            # typically too long for the structural-refusal regex to
            # catch (>600 chars), so the existing Step 3b retry path
            # never fires. Give it one stronger-prompt retry HERE,
            # before returning the off-topic fallback. Only fires when:
            #   * retry_fn is wired (api.py always passes _retry_generate)
            #   * retrieval was strong (matches the hard-refusal retry
            #     criterion: >=3 chunks, top_score >= 0.4)
            # Strictly one retry, then fall through if it also fails —
            # bounded cost, no recursive retry loop.
            soft_refusal_recovered = False
            soft_refusal_retried = False
            if retry_fn is not None and retrieval_is_strong:
                try:
                    logger.warning(
                        "[validator] soft refusal via relevancy (%.2f) "
                        "on strong retrieval — attempting stronger-prompt retry",
                        relevancy.score,
                    )
                    retry_answer = retry_fn(stronger_prompt=True)
                    soft_refusal_retried = True
                    if retry_answer and retry_answer.strip():
                        # Re-judge ONLY the dimension that failed
                        # (relevancy). Grounding and confidence were
                        # already passing on the original answer drawn
                        # from the same chunks, so the retry is over-
                        # whelmingly likely to keep passing them. We
                        # save a Haiku call by not re-running grounding.
                        retry_relevancy = check_relevancy(
                            query=query, answer=retry_answer,
                        )
                        if (
                            retry_relevancy is not None
                            and not retry_relevancy.skipped
                            and retry_relevancy.passed
                        ):
                            # Retry produced a relevant answer — promote it.
                            result.answer = retry_answer
                            result.relevancy_detail = retry_relevancy
                            result.was_modified = True
                            result.passed = True
                            result.confidence = conf.score
                            result.issues.append(
                                "Recovered from soft refusal via stronger-prompt "
                                f"retry (relevancy {relevancy.score:.2f} -> "
                                f"{retry_relevancy.score:.2f})"
                            )
                            soft_refusal_recovered = True
                            logger.info(
                                "[validator] soft-refusal retry succeeded "
                                "(relevancy %.2f -> %.2f)",
                                relevancy.score, retry_relevancy.score,
                            )
                        else:
                            new_score = (
                                retry_relevancy.score if retry_relevancy else 0.0
                            )
                            logger.warning(
                                "[validator] soft-refusal retry still off-topic "
                                "(relevancy %.2f)",
                                new_score,
                            )
                except Exception as soft_retry_exc:
                    logger.warning(
                        "[validator] soft-refusal retry raised: %s",
                        soft_retry_exc,
                    )

            if not soft_refusal_recovered:
                result.passed = False
                result.was_modified = True
                result.answer = get_off_topic_fallback()
                # Halve confidence so downstream UI / cache / eval can spot
                # off-topic outcomes even if the score field is glanced at.
                result.confidence = max(0.0, conf.score * 0.5)
                result.issues.append(
                    f"Relevancy score {relevancy.score:.2f} below threshold "
                    f"{settings.MIN_RELEVANCY_SCORE} — answer did not address the question"
                )
                if relevancy.reason:
                    result.issues.append(f"Relevancy reason: {relevancy.reason}")
                logger.warning(
                    "Validation FAILED (off-topic): relevancy=%.2f reason=%r%s",
                    relevancy.score, relevancy.reason[:120],
                    " (retry also off-topic)" if soft_refusal_retried else "",
                )
        else:
            result.passed = True

            # Append version warning if relevant (informational, not a failure)
            if grounding.version_warning:
                result.answer = answer.rstrip() + "\n\n" + grounding.version_warning
                result.was_modified = True

            logger.info(
                "Validation PASSED: confidence=%.3f, grounding=%.3f, relevancy=%s",
                conf.score, grounding.grounding_score,
                f"{relevancy.score:.2f}" if relevancy and not relevancy.skipped else "skipped",
            )

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