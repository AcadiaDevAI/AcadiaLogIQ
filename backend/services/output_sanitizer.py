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
from backend.services.input_guardrails import PII_PATTERNS, _should_redact_pii_match

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

    # Per-org opt-out for EMAIL redaction only (US Pharma: vendor / escalation
    # contact emails are the requested content, not PII). All other PII types
    # stay scrubbed. Resolved from the request org context; any failure keeps
    # the default (redact) so we never accidentally leak. See OrgProfile.
    _skip_email = False
    try:
        from backend.orgs.context import resolve_current_profile
        _skip_email = not bool(
            getattr(resolve_current_profile(), "redact_contact_emails", True)
        )
    except Exception:
        _skip_email = False

    try:
        if settings.OUTPUT_SANITIZER_SCRUB_PII:
            for name, pattern in PII_PATTERNS.items():
                if name == "email" and _skip_email:
                    continue
                def _repl(m, _name=name):
                    candidate = m.group(0)
                    if not _should_redact_pii_match(_name, candidate):
                        logger.debug(
                            "[sanitizer] skipping Luhn-invalid digit sequence: %s (likely ticket ID)",
                            candidate,
                        )
                        return candidate
                    if f"pii:{_name}" not in issues:
                        issues.append(f"pii:{_name}")
                    return f"[REDACTED_{_name.upper()}]"
                result = pattern.sub(_repl, result)

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
