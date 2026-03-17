"""
Evaluation Harness — offline testing utilities for validation quality.
Provides run_eval_suite() for benchmarking against predefined test cases,
and parse_eval_log() for analyzing JSONL evaluation records.
Used for development/QA, not at runtime.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.validation.confidence_scorer import score_confidence
from backend.validation.grounding_checker import check_grounding
from backend.validation.validator import validate_answer

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Eval test case
# ---------------------------------------------------------------------------
@dataclass
class EvalCase:
    """
    A single evaluation test case for benchmarking validation quality.

    Fields:
        query            — the user question
        answer           — the generated answer to validate
        doc_context      — the retrieved document context
        source_names     — list of source document names
        expected_pass    — whether validation should pass
        expected_issues  — expected issue substrings to look for
        description      — human-readable description of the test
    """
    query: str
    answer: str
    doc_context: str
    source_names: List[str] = field(default_factory=lambda: ["test_doc.pdf"])
    expected_pass: bool = True
    expected_issues: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class EvalResult:
    """Result from running a single eval case."""
    case: EvalCase
    passed: bool = False
    correct: bool = False  # did actual result match expected_pass?
    confidence: float = 0.0
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Predefined eval cases
# ---------------------------------------------------------------------------
EVAL_SUITE: List[EvalCase] = [
    EvalCase(
        description="Well-grounded answer should pass",
        query="What is the maximum file upload size?",
        answer="- The maximum file upload size is 100MB as specified in the configuration.",
        doc_context="Configuration: MAX_FILE_SIZE_MB is set to 100. Files larger than this are rejected.",
        expected_pass=True,
    ),
    EvalCase(
        description="Answer with fabricated URL should fail",
        query="Where can I find the documentation?",
        answer="- Documentation is available at https://fake-docs.example.com/guide",
        doc_context="The system documentation is maintained internally. Contact the admin team.",
        expected_pass=False,
        expected_issues=["fabricat"],
    ),
    EvalCase(
        description="Answer with hallucination phrase should be penalized",
        query="How does the backup system work?",
        answer="- Based on my training data, the backup runs nightly at 2 AM.",
        doc_context="Backups are scheduled via cron. The default schedule is daily at 02:00 UTC.",
        expected_pass=True,  # may pass with penalty, depends on grounding
    ),
    EvalCase(
        description="Empty answer should fail",
        query="What is the deployment process?",
        answer="",
        doc_context="Deployment uses Docker Compose. Run docker-compose up to start services.",
        expected_pass=False,
    ),
    EvalCase(
        description="Completely ungrounded answer should fail",
        query="What is the server architecture?",
        answer="- The system uses a quantum computing cluster with 1000 qubits for processing.",
        doc_context="Architecture: FastAPI backend, PostgreSQL database, React frontend.",
        expected_pass=False,
        expected_issues=["grounding"],
    ),
    EvalCase(
        description="Honest refusal should pass with moderate confidence",
        query="What is the CEO's phone number?",
        answer="- I could not find supporting information for that question in the currently uploaded files.",
        doc_context="Company handbook: general policies and procedures.",
        expected_pass=True,
    ),
    EvalCase(
        description="Answer with fabricated email should fail",
        query="Who should I contact for support?",
        answer="- Contact support at help@totally-fake-company.com for assistance.",
        doc_context="For support issues, escalate to the Tier-2 engineering team.",
        expected_pass=False,
        expected_issues=["fabricat"],
    ),
]


# ---------------------------------------------------------------------------
# Run eval suite
# ---------------------------------------------------------------------------
def run_eval_suite(
    cases: Optional[List[EvalCase]] = None,
) -> List[EvalResult]:
    """
    Run a suite of evaluation cases through the validation pipeline.
    Returns per-case results with pass/fail and correctness assessment.

    Usage:
        from backend.validation.eval_harness import run_eval_suite, EVAL_SUITE
        results = run_eval_suite(EVAL_SUITE)
        for r in results:
            print(f"{'✓' if r.correct else '✗'} {r.case.description}")
    """
    cases = cases or EVAL_SUITE
    results: List[EvalResult] = []

    for case in cases:
        validation = validate_answer(
            query=case.query,
            answer=case.answer,
            doc_context=case.doc_context,
            ranked_chunks=[],  # no ranked chunks in offline eval
            source_names=case.source_names,
            model_used="eval_harness",
        )

        # Check if actual result matches expected
        actual_pass = validation.passed
        correct = actual_pass == case.expected_pass

        # Check expected issues
        if case.expected_issues:
            for expected_issue in case.expected_issues:
                found = any(expected_issue.lower() in issue.lower() for issue in validation.issues)
                if not found:
                    correct = False

        results.append(EvalResult(
            case=case,
            passed=actual_pass,
            correct=correct,
            confidence=validation.confidence,
            issues=validation.issues,
            details={
                "was_modified": validation.was_modified,
                "version_warning": validation.version_warning,
                "validation_ms": validation.validation_ms,
            },
        ))

        status = "✓" if correct else "✗"
        logger.info(
            "Eval %s: %s | expected_pass=%s actual_pass=%s confidence=%.3f",
            status, case.description, case.expected_pass, actual_pass, validation.confidence,
        )

    # Summary
    total = len(results)
    correct_count = sum(1 for r in results if r.correct)
    logger.info("Eval suite: %d/%d correct (%.0f%%)", correct_count, total, 100 * correct_count / max(1, total))

    return results


# ---------------------------------------------------------------------------
# Parse eval log file
# ---------------------------------------------------------------------------
def parse_eval_log(log_path: str) -> List[Dict[str, Any]]:
    """
    Parse a JSONL evaluation log file into a list of records.
    Useful for offline analysis and dashboarding.

    Usage:
        records = parse_eval_log("/path/to/eval.jsonl")
        failed = [r for r in records if not r["passed"]]
    """
    records = []
    path = Path(log_path)
    if not path.exists():
        logger.warning("Eval log not found: %s", log_path)
        return records

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    logger.info("Parsed %d eval records from %s", len(records), log_path)
    return records
