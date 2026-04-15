"""
Stage Enforcer — applies per-stage policy to the /ask pipeline.
Stages: 'tickets' filters retrieval to ticket-sourced chunks; 'docs' tracks
repeated unresolved follow-ups per session and upgrades them to multi_agent.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("acadia-log-iq")


# ---------------------------------------------------------------------------
# Stage constants
# ---------------------------------------------------------------------------
STAGE_GENERAL: str = "general"
STAGE_TICKETS: str = "tickets"
STAGE_DOCS: str = "docs"

_VALID_STAGES = {STAGE_GENERAL, STAGE_TICKETS, STAGE_DOCS}

# Upgrade to multi_agent once a session crosses this threshold of
# repeated "not resolved" follow-ups in STAGE_DOCS.
UNRESOLVED_ESCALATE_THRESHOLD: int = 2


# ---------------------------------------------------------------------------
# Per-session unresolved counter (in-memory, thread-safe)
# ---------------------------------------------------------------------------
_unresolved_counts: Dict[str, int] = {}
_unresolved_lock = threading.Lock()


def _bump_unresolved(session_id: Optional[str]) -> int:
    if not session_id:
        return 0
    try:
        with _unresolved_lock:
            _unresolved_counts[session_id] = _unresolved_counts.get(session_id, 0) + 1
            return _unresolved_counts[session_id]
    except Exception as exc:
        logger.warning("Unresolved counter bump failed (%s)", exc)
        return 0


def _reset_unresolved(session_id: Optional[str]) -> None:
    if not session_id:
        return
    try:
        with _unresolved_lock:
            _unresolved_counts.pop(session_id, None)
    except Exception:
        pass


def get_unresolved_count(session_id: Optional[str]) -> int:
    if not session_id:
        return 0
    try:
        with _unresolved_lock:
            return int(_unresolved_counts.get(session_id, 0))
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------
def normalize_stage(stage: Optional[str]) -> str:
    """Return a canonical stage value, defaulting to STAGE_GENERAL."""
    if not stage:
        return STAGE_GENERAL
    s = str(stage).strip().lower()
    if s in _VALID_STAGES:
        return s
    logger.warning("Unknown stage '%s' — using '%s'", stage, STAGE_GENERAL)
    return STAGE_GENERAL


# ---------------------------------------------------------------------------
# Source filtering for tickets stage
# ---------------------------------------------------------------------------
def _looks_like_ticket(metadata: Dict[str, Any], source_name: str) -> bool:
    """
    Heuristic: a chunk is a ticket if its metadata or source name suggests it.
    We check common fields used by the ingestion layer; callers that use
    custom metadata keys can extend this heuristic safely.
    """
    if not isinstance(metadata, dict):
        metadata = {}
    file_type = str(metadata.get("file_type") or metadata.get("type") or "").lower()
    if file_type in {"ticket", "tickets", "incident"}:
        return True
    source_type = str(metadata.get("source_type") or "").lower()
    if source_type in {"ticket", "tickets", "incident"}:
        return True
    name_l = (source_name or "").lower()
    return ("ticket" in name_l) or ("incident" in name_l)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    """
    stage                     — normalized stage ('general'|'tickets'|'docs')
    filtered_chunks           — ranked chunks after stage filter (may equal input)
    filter_applied            — True if any chunks were removed by the filter
    original_chunk_count      — count before filter
    filtered_chunk_count      — count after filter
    enforced_mode             — optional mode override ('multi_agent' | None)
    escalate_reason           — human-readable reason when enforced_mode is set
    unresolved_count          — per-session counter, after this call
    notes                     — free-form list of applied actions (for stats)
    """
    stage: str = STAGE_GENERAL
    filtered_chunks: List[Tuple[str, str, Dict[str, Any], float]] = field(default_factory=list)
    filter_applied: bool = False
    original_chunk_count: int = 0
    filtered_chunk_count: int = 0
    enforced_mode: Optional[str] = None
    escalate_reason: str = ""
    unresolved_count: int = 0
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def enforce_stage(
    *,
    stage: Optional[str],
    session_id: Optional[str],
    ranked_chunks: List[Tuple[str, str, Dict[str, Any], float]],
    intent_name: str = "general",
) -> StageResult:
    """
    Apply stage policy. Never raises — any internal error returns a
    pass-through StageResult with original chunks.

    Policy:
        tickets → filter ranked_chunks to ticket-sourced chunks only
        docs    → if intent_name == 'not_resolved', bump the unresolved
                  counter for this session; once >= threshold, force
                  enforced_mode='multi_agent'
        general → pass-through
    """
    try:
        normalized = normalize_stage(stage)
        original = list(ranked_chunks or [])
        result = StageResult(
            stage=normalized,
            filtered_chunks=original,
            original_chunk_count=len(original),
            filtered_chunk_count=len(original),
        )

        # --- Tickets stage: enforce ticket-only sources ---
        if normalized == STAGE_TICKETS:
            filtered = [
                row for row in original
                if _looks_like_ticket(row[2] if len(row) > 2 else {}, row[1] if len(row) > 1 else "")
            ]
            result.filtered_chunks = filtered
            result.filtered_chunk_count = len(filtered)
            result.filter_applied = len(filtered) != len(original)
            if result.filter_applied:
                result.notes.append(
                    f"tickets stage: filtered {len(original)} → {len(filtered)} ticket-only chunks"
                )
            else:
                result.notes.append("tickets stage: no non-ticket chunks to filter")

        # --- Docs stage: track unresolved follow-ups ---
        elif normalized == STAGE_DOCS:
            if intent_name == "not_resolved":
                count = _bump_unresolved(session_id)
                result.unresolved_count = count
                result.notes.append(f"docs stage: unresolved count for session={count}")
                if count >= UNRESOLVED_ESCALATE_THRESHOLD:
                    result.enforced_mode = "multi_agent"
                    result.escalate_reason = (
                        f"docs stage: repeated unresolved ({count}>={UNRESOLVED_ESCALATE_THRESHOLD}) "
                        f"→ forcing multi_agent"
                    )
            else:
                # Any non-unresolved question resets the streak
                _reset_unresolved(session_id)
                result.unresolved_count = 0

        # --- General stage: pass through ---
        else:
            result.notes.append("general stage: no enforcement")

        return result

    except Exception as exc:
        logger.warning("Stage enforcer raised (%s) — passing through", exc)
        passthrough = list(ranked_chunks or [])
        return StageResult(
            stage=normalize_stage(stage),
            filtered_chunks=passthrough,
            original_chunk_count=len(passthrough),
            filtered_chunk_count=len(passthrough),
            notes=[f"stage_enforcer_error: {exc}"],
        )
