"""Sprint 10 — Tier-1 journey telemetry.

Append-only inserts into `tier1_journey_events` (migration 040).
Fire-and-forget: on any DB error we log a warning and return False;
the engineer's UX is never blocked by a telemetry hiccup.

Same pattern as Sprint 7's `feedback.record_event`.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional


logger = logging.getLogger("acadia-log-iq")


_VALID_STAGES = {
    "stage_0",
    "stage_1a", "stage_1b",   # Sprint 10 legacy — historical rows only
    "pivot_insights",          # Sprint 10.2 — merged 1A+1B panel
    "stage_2", "stage_3", "stage_4", "stage_5",
}

_VALID_EVENT_TYPES = {
    "stage_rendered",
    "helpful_clicked",
    "next_stage_clicked",
    "abandoned",
    # Sprint 10.5 §3.2 — fired when an engineer clicks "Escalate
    # Ticket" inline on a chat message footer (the chat originated
    # from a Stage 4 Search-KB handoff). Recorded BEFORE the navigate
    # to /tier1/journey/{id}?stage=5 so the event survives the route
    # unmount; see JourneyMessageActions.handleEscalate.
    "escalation_initiated_from_chat",
    # Sprint 10.7 §3 — fired when the engineer reaches a new stage.
    # /resume-state filters on this event_type exclusively to derive
    # the engineer's current stage on remount.
    "stage_advanced",
    # Sprint 11 — fired AFTER the first /ask round-trip in a Stage 4
    # spawned chat session (Stage4SearchKBHandoff.handleOpenChat
    # posts after ADD_ASSISTANT_MESSAGE lands). Distinguishes "user
    # opened the KB chat and got an answer" from "user clicked Open
    # Chat and immediately escalated without engaging". The
    # JourneyEventRequest schema and this allowlist must stay in
    # sync — both gate the event end-to-end.
    "kb_chat_engaged",
}


def record_event(
    *,
    session_id: str,
    stage: str,
    event_type: str,
    payload: Optional[Dict[str, Any]] = None,
    user_id: Optional[str] = None,
) -> bool:
    """Insert one row into tier1_journey_events. Returns True on
    success, False on any error.

    All inputs are validated against the schema's allowed values
    before hitting the DB — unknown stage / event_type returns False
    without an INSERT (mirrors `feedback.record_feedback`'s posture)."""
    if not session_id or not stage or not event_type:
        return False
    if stage not in _VALID_STAGES:
        logger.warning("[journey.telemetry] unknown stage rejected: %s", stage)
        return False
    if event_type not in _VALID_EVENT_TYPES:
        logger.warning("[journey.telemetry] unknown event_type rejected: %s", event_type)
        return False

    try:
        from sqlalchemy import text
        from backend.db.connection import engine
    except Exception as exc:
        logger.warning("[journey.telemetry] DB import failed: %s", exc)
        return False

    payload_json = json.dumps(payload, ensure_ascii=False) if payload else None

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO tier1_journey_events
                        (session_id, user_id, stage, event_type, payload_json)
                    VALUES (:sid, :uid, :stage, :etype, CAST(:pj AS JSONB))
                    """
                ),
                {
                    "sid": session_id,
                    "uid": user_id,
                    "stage": stage,
                    "etype": event_type,
                    "pj": payload_json,
                },
            )
        return True
    except Exception as exc:
        logger.warning("[journey.telemetry] insert failed: %s", exc)
        return False
