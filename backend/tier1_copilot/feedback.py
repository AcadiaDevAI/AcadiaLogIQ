"""Tier-1 feedback persistence + follow-up dispatcher.

Two responsibilities:

1. `record_feedback` — append to tier1_feedback telemetry.
2. `handle_follow_up_action` — map the 5 thumb-down actions to their
   side effects (currently: cache invalidation on next_best_solution;
   no-op for the other four in MVP — they return their action label
   so the frontend can render guidance copy).
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger("acadia-log-iq")


_KNOWN_ACTIONS = {
    "next_best_solution",
    "deeper_diagnostics",
    "escalation_note",
    "search_kb_sop",
    "explain_recommendation",
}

# Sprint 7 — additional event_type values that may land in the extended
# tier1_feedback table. The frontend never sets these directly; they
# come from server-side detectors (stuck modal, escalation open).
_KNOWN_EVENT_TYPES = {
    "feedback",        # legacy Sprint 6 row
    "stuck_nudge",     # stuck-detection modal shown
    "escalation_open", # escalation package retrieved
    "match_cycle",     # arrow pagination
    "action_log",      # what_tried append
}


def record_feedback(
    *,
    response_id: str,
    session_id: str,
    helpful: bool,
    follow_up_action: Optional[str] = None,
) -> bool:
    """Insert a feedback row. Returns True on success, False on error.

    The schema (migration 038) enforces NOT NULL on response_id,
    session_id, helpful. follow_up_action is nullable and constrained
    to the known set in Python (DB stores the raw string)."""
    if not response_id or not session_id:
        return False
    if follow_up_action is not None and follow_up_action not in _KNOWN_ACTIONS:
        logger.warning(
            "[tier1_copilot] unknown follow_up_action rejected: %s",
            follow_up_action,
        )
        return False
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO tier1_feedback
                        (response_id, session_id, helpful, follow_up_action)
                    VALUES (:rid, :sid, :h, :act)
                    """
                ),
                {
                    "rid": response_id,
                    "sid": session_id,
                    "h": helpful,
                    "act": follow_up_action,
                },
            )
        return True
    except Exception as exc:
        logger.warning("[tier1_copilot] feedback insert failed: %s", exc)
        return False


def handle_follow_up_action(
    *,
    response_id: str,
    action: Optional[str],
) -> Optional[str]:
    """Dispatch the 👎 action. Returns the action label on success,
    None if the action isn't recognized or has no side effect.

    Current MVP behaviors:
      - next_best_solution: invalidate this response's cache row so
        the next identical intake returns rank-2 from retrieval.
      - deeper_diagnostics / escalation_note / search_kb_sop /
        explain_recommendation: tracked as telemetry only; the
        frontend already has deterministic copy for each.
    """
    if not action or action not in _KNOWN_ACTIONS:
        return None

    if action == "next_best_solution":
        try:
            from sqlalchemy import text
            from backend.db.connection import engine
            with engine.begin() as conn:
                conn.execute(
                    text(
                        """
                        DELETE FROM tier1_answer_cache
                        WHERE signature_hash IN (
                            SELECT DISTINCT response_id
                            FROM tier1_feedback
                            WHERE response_id = :rid
                        )
                        """
                    ),
                    {"rid": response_id},
                )
        except Exception as exc:
            logger.warning(
                "[tier1_copilot] next_best cache bust failed rid=%s: %s",
                response_id, exc,
            )

    return action


# ─────────────────────────────────────────────────────────────
# Sprint 7 — event logging
# Uses the extended tier1_feedback table (migration 039 added event_type
# + session_elapsed_seconds). Fails silently so telemetry never blocks
# the Tier-1 flow on a DB outage.
# ─────────────────────────────────────────────────────────────
def record_event(
    *,
    response_id: str,
    session_id: str,
    event_type: str,
    follow_up_action: Optional[str] = None,
    session_elapsed_seconds: Optional[int] = None,
) -> bool:
    if not response_id or not session_id:
        return False
    if event_type not in _KNOWN_EVENT_TYPES:
        logger.warning(
            "[tier1_copilot:sprint7] unknown event_type rejected: %s", event_type,
        )
        return False
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO tier1_feedback
                        (response_id, session_id, helpful, follow_up_action,
                         event_type, session_elapsed_seconds)
                    VALUES (:rid, :sid, FALSE, :act, :etype, :elapsed)
                    """
                ),
                {
                    "rid": response_id,
                    "sid": session_id,
                    "act": follow_up_action,
                    "etype": event_type,
                    "elapsed": session_elapsed_seconds,
                },
            )
        return True
    except Exception as exc:
        logger.warning(
            "[tier1_copilot:sprint7] event insert failed etype=%s: %s",
            event_type, exc,
        )
        return False
