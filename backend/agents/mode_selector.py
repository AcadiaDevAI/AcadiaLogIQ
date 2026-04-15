"""
Mode Selector — resolves the /ask `mode` parameter into a pipeline decision.
'auto' now routes to the agent pipeline by default (agent-first); 'hybrid'
still forces the standard path as an override; 'multi_agent' unchanged.
The old should_escalate_to_agents gate is kept for telemetry / reason strings.
"""

from __future__ import annotations

import logging
from typing import Tuple

from backend.agents.orchestrator import should_escalate_to_agents

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
        'auto'        → route to agents by default (agent-first).
                        If no sources are available, fall back to hybrid.
                        For telemetry the reason string also includes the
                        classical 5-gate outcome.
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

    # MODE_AUTO — agent-first default
    if source_count < 1:
        # Nothing to ground against — fall back to the standard path
        return False, "mode=auto: no sources → fallback to standard RAG"

    # Call the classical gate only to surface its reason in telemetry
    classical_use, classical_reason = should_escalate_to_agents(
        query=query,
        complexity_score=complexity_score,
        complexity_tier=complexity_tier,
        source_count=source_count,
    )
    return True, (
        f"mode=auto (agent-first default) [classical_gate="
        f"{'escalate' if classical_use else 'skip'}: {classical_reason}]"
    )
