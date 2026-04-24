"""Sprint 7 — per-alert session state (tier1_sessions table).

Thin CRUD helpers over migration 039's tier1_sessions schema. Every
helper is exception-safe — a DB failure logs a warning and returns
None/False so the caller (routes) can fall back gracefully.

The Tier1SessionRecord dataclass is returned to callers so in-memory
tests can construct fake records without touching SQLAlchemy.
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("acadia-log-iq")


@dataclass
class Tier1SessionRecord:
    id: str
    alert_signature: str
    alert_payload: Dict[str, Any] = field(default_factory=dict)
    top_5_match_ids: List[str] = field(default_factory=list)
    current_match_index: int = 0
    thumbs_down_count: int = 0
    what_tried: List[Dict[str, Any]] = field(default_factory=list)
    stuck_nudge_shown: bool = False
    resolved: bool = False
    escalated: bool = False
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    last_activity_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _coerce_ts(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _row_to_record(row: Any) -> Tier1SessionRecord:
    alert_payload = row["alert_payload"]
    if isinstance(alert_payload, str):
        try:
            alert_payload = json.loads(alert_payload)
        except Exception:
            alert_payload = {}
    what_tried = row["what_tried"]
    if isinstance(what_tried, str):
        try:
            what_tried = json.loads(what_tried)
        except Exception:
            what_tried = []
    return Tier1SessionRecord(
        id=row["id"],
        alert_signature=row["alert_signature"],
        alert_payload=alert_payload or {},
        top_5_match_ids=list(row["top_5_match_ids"] or []),
        current_match_index=int(row["current_match_index"] or 0),
        thumbs_down_count=int(row["thumbs_down_count"] or 0),
        what_tried=what_tried or [],
        stuck_nudge_shown=bool(row["stuck_nudge_shown"]),
        resolved=bool(row["resolved"]),
        escalated=bool(row["escalated"]),
        created_at=_coerce_ts(row["created_at"]),
        last_activity_at=_coerce_ts(row["last_activity_at"]),
    )


# ─────────────────────────────────────────────────────────────
# CRUD
# ─────────────────────────────────────────────────────────────
def create_session(
    *,
    alert_signature: str,
    alert_payload: Dict[str, Any],
    top_5_match_ids: Optional[List[str]] = None,
    session_id: Optional[str] = None,
) -> Optional[Tier1SessionRecord]:
    sid = session_id or f"sess_{uuid.uuid4().hex[:12]}"
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO tier1_sessions
                        (id, alert_signature, alert_payload, top_5_match_ids)
                    VALUES
                        (:id, :sig, CAST(:payload AS JSONB), :ids)
                    ON CONFLICT (id) DO NOTHING
                    """
                ),
                {
                    "id": sid,
                    "sig": alert_signature,
                    "payload": json.dumps(alert_payload, ensure_ascii=False),
                    "ids": list(top_5_match_ids or []),
                },
            )
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] session create failed: %s", exc)
        return None
    return get_session(sid)


def get_session(session_id: str) -> Optional[Tier1SessionRecord]:
    if not session_id:
        return None
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT * FROM tier1_sessions WHERE id = :id"),
                {"id": session_id},
            ).mappings().first()
    except Exception as exc:
        logger.warning(
            "[tier1_copilot:sprint7] session fetch failed id=%s: %s",
            session_id, exc,
        )
        return None
    if not row:
        return None
    try:
        return _row_to_record(row)
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] row decode failed: %s", exc)
        return None


def touch_activity(session_id: str) -> bool:
    if not session_id:
        return False
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE tier1_sessions
                    SET last_activity_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": session_id},
            )
        return True
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] touch failed: %s", exc)
        return False


def update_match_index(session_id: str, index: int) -> bool:
    if not session_id:
        return False
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE tier1_sessions
                    SET current_match_index = :idx,
                        last_activity_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": session_id, "idx": int(index)},
            )
        return True
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] match_index update failed: %s", exc)
        return False


def increment_thumbs_down(session_id: str) -> Optional[int]:
    if not session_id:
        return None
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    UPDATE tier1_sessions
                    SET thumbs_down_count = thumbs_down_count + 1,
                        last_activity_at = NOW()
                    WHERE id = :id
                    RETURNING thumbs_down_count
                    """
                ),
                {"id": session_id},
            ).first()
        return int(row[0]) if row else None
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] thumbs_down update failed: %s", exc)
        return None


def mark_stuck_shown(session_id: str) -> bool:
    if not session_id:
        return False
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE tier1_sessions
                    SET stuck_nudge_shown = TRUE,
                        last_activity_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": session_id},
            )
        return True
    except Exception as exc:
        logger.warning(
            "[tier1_copilot:sprint7] stuck_nudge_shown update failed: %s", exc,
        )
        return False


def append_what_tried(
    session_id: str,
    entry: Dict[str, Any],
) -> Optional[List[Dict[str, Any]]]:
    if not session_id or not entry:
        return None
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            row = conn.execute(
                text(
                    """
                    UPDATE tier1_sessions
                    SET what_tried = COALESCE(what_tried, '[]'::jsonb)
                                     || CAST(:e AS JSONB),
                        last_activity_at = NOW()
                    WHERE id = :id
                    RETURNING what_tried
                    """
                ),
                {
                    "id": session_id,
                    "e": json.dumps([entry], ensure_ascii=False),
                },
            ).first()
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] what_tried append failed: %s", exc)
        return None
    if not row:
        return None
    val = row[0]
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            val = []
    return list(val or [])


def mark_escalated(session_id: str) -> bool:
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE tier1_sessions
                    SET escalated = TRUE, last_activity_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": session_id},
            )
        return True
    except Exception as exc:
        logger.warning("[tier1_copilot:sprint7] mark_escalated failed: %s", exc)
        return False
