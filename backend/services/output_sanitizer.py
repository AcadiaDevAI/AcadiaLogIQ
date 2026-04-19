"""
Output sanitizer (Brief 5 / Part 3, Layer 3).

Post-generation scrub for PII tokens and system-prompt leakage. Never
blocks — the answer is always returned, possibly with redactions. The
list of issues is returned for logging/observability.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from backend.config import settings
from backend.services.input_guardrails import PII_PATTERNS

logger = logging.getLogger("acadia-log-iq")


_PROMPT_LEAK_HINTS = [
    "<|system|>",
    "<|im_start|>",
    "system prompt:",
    "you are an aggregation",
    "return a json",
    "anthropic_version",
]


def sanitize_output(answer: str, original_query: str = "") -> Tuple[str, List[str]]:
    """Return (sanitized_answer, [issue_tags]). Never raises."""
    if not settings.OUTPUT_SANITIZER_ENABLED:
        return answer or "", []
    if not answer:
        return answer or "", []

    result = answer
    issues: List[str] = []

    try:
        if settings.OUTPUT_SANITIZER_SCRUB_PII:
            for name, pattern in PII_PATTERNS.items():
                if pattern.search(result):
                    issues.append(f"pii:{name}")
                    result = pattern.sub(f"[REDACTED_{name.upper()}]", result)

        lowered = result.lower()
        for hint in _PROMPT_LEAK_HINTS:
            if hint.lower() in lowered:
                issues.append(f"prompt_leak:{hint}")

    except Exception as exc:
        logger.warning("[sanitizer] failure (%s) — returning unchanged", exc)
        return answer, []

    if issues:
        logger.warning("[sanitizer] output issues: %s", issues)

    return result, issues
