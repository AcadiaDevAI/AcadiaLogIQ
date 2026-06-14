"""
Mode Selector — resolves the /ask `mode` parameter into a pipeline decision.
'auto' is hybrid-first: it routes to standard RAG unless a specific agent
signal fires (agent-eligible pattern OR complex-tier + high score). 'hybrid'
forces standard RAG; 'multi_agent' forces the agent pipeline.
"""

from __future__ import annotations

import logging
from typing import Tuple

from backend.config import settings
# Import the pattern list from the agent orchestrator so the two sides
# agree on what counts as "agent-eligible" — avoids drift from duplication.
from backend.agents.orchestrator import (
    should_escalate_to_agents,
    _AGENT_ELIGIBLE_PATTERNS,
)

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Mode constants — imported by api.py so callers avoid magic strings
# ---------------------------------------------------------------------------
MODE_AUTO: str = "auto"
MODE_HYBRID: str = "hybrid"
MODE_MULTI_AGENT: str = "multi_agent"

_VALID_MODES = {MODE_AUTO, MODE_HYBRID, MODE_MULTI_AGENT}


def _normalize_mode(mode: str) -> str:
    """Normalize aliases to canonical mode names. Unknown input → MODE_AUTO."""
    if not mode:
        return MODE_AUTO
    m = str(mode).strip().lower().replace("-", "_")
    if m in ("agent", "agents", "multi_agents", "multiagent"):
        return MODE_MULTI_AGENT
    if m in ("standard", "rag", "single"):
        return MODE_HYBRID
    if m in _VALID_MODES:
        return m
    logger.warning("Unknown mode '%s' — falling back to '%s'", mode, MODE_AUTO)
    return MODE_AUTO


def _matched_agent_pattern(query: str) -> str:
    """Return the first agent-eligible pattern the query matches, else ''."""
    if not query:
        return ""
    for pat in _AGENT_ELIGIBLE_PATTERNS:
        m = pat.search(query)
        if m:
            return m.group(0)
    return ""


def resolve_mode(
    *,
    mode: str,
    query: str,
    complexity_score: float,
    complexity_tier: str,
    source_count: int,
) -> Tuple[bool, str]:
    """
    Decide whether the agent pipeline should run.

    Modes:
        'auto'        → hybrid-first. Escalate to agents only when the
                        query matches an agent-eligible pattern, OR when
                        the classifier marks it complex with a score at
                        or above AGENT_COMPLEXITY_THRESHOLD. Otherwise
                        answer with standard hybrid RAG.
        'hybrid'      → never use agents (explicit override).
        'multi_agent' → always use agents (requires ≥1 source).

    Returns:
        (use_agent: bool, reason: str)
    """
    canonical = _normalize_mode(mode)

    # Explicit hybrid override — never run agents
    if canonical == MODE_HYBRID:
        return False, "mode=hybrid (forced standard RAG)"

    # Forced multi-agent
    if canonical == MODE_MULTI_AGENT:
        if source_count < 1:
            return False, "mode=multi_agent but no sources available (fallback to standard)"
        return True, "mode=multi_agent (forced agent pipeline)"

    # MODE_AUTO — hybrid-first default
    # Gate 1: no sources → nothing to ground against, stay on hybrid.
    if source_count < 1:
        return False, "mode=auto -> hybrid (no sources available)"

    # Gate 2: agent-eligible pattern is the strongest positive signal —
    # queries like "compare X and Y" or "walk me through" intrinsically
    # need multi-step reasoning even at low complexity scores.
    matched = _matched_agent_pattern(query)
    if matched:
        return True, f"mode=auto -> agents (pattern matched: '{matched}')"

    # Gate 3: complexity classifier fallback — complex-tier queries with
    # a high score still escalate even without a pattern hit.
    if (
        complexity_tier == "complex"
        and complexity_score >= settings.AGENT_COMPLEXITY_THRESHOLD
    ):
        return True, (
            f"mode=auto -> agents (complex tier, score="
            f"{complexity_score:.3f} >= {settings.AGENT_COMPLEXITY_THRESHOLD})"
        )

    # Default — standard hybrid RAG handles the query.
    return False, (
        f"mode=auto -> hybrid (no agent-eligible pattern, tier={complexity_tier})"
    )
