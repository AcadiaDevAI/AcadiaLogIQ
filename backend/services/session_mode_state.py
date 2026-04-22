"""
Session Mode State — persistent read/write for the guided workflow.

Backs the PRD session fields (selected_mode, sub_mode,
conversation_context_active, customer_name, technology_domain, ticket_id,
issue_summary, last_recommendation, form_data) on the chat_sessions table.

All operations are fail-safe: any exception is logged and translated into
an is_valid=False sentinel so /ask can fall back to classic behavior
without breaking the user's turn. Mirrors stage_enforcer's
dataclass+try/except pattern for consistency across the codebase.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text

from backend.config import settings  # noqa: F401  (kept for forward-compat flag checks)
from backend.db.connection import engine

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────
# Mode constants
# ─────────────────────────────────────────────────────────────
MODE_TROUBLESHOOTING: str = "troubleshooting"
MODE_TICKET_HANDLING: str = "ticket_handling"
MODE_ESCALATION: str = "escalation"
MODE_VENDOR_OEM: str = "vendor_oem"

_VALID_MODES = {
    MODE_TROUBLESHOOTING,
    MODE_TICKET_HANDLING,
    MODE_ESCALATION,
    MODE_VENDOR_OEM,
}

# Troubleshooting sub-modes (PRD Section 3)
SUB_CUSTOMER_SPECIFIC: str = "customer_specific"
SUB_TECHNOLOGY_SPECIFIC: str = "technology_specific"

# Ticket handling sub-modes (PRD Section 8A) — declared now for Sprint 4.
SUB_TICKET_CREATE: str = "ticket_create"
SUB_TICKET_UPDATE: str = "ticket_update"
SUB_TICKET_CLOSE: str = "ticket_close"
SUB_TICKET_VALIDATE: str = "ticket_validate"

_VALID_SUB_MODES = {
    SUB_CUSTOMER_SPECIFIC,
    SUB_TECHNOLOGY_SPECIFIC,
    SUB_TICKET_CREATE,
    SUB_TICKET_UPDATE,
    SUB_TICKET_CLOSE,
    SUB_TICKET_VALIDATE,
}


# ─────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────
@dataclass
class SessionMode:
    """
    Snapshot of the mode-state fields for a single session.

    is_valid=False indicates a read/write error. Callers should treat an
    invalid result as "no mode set" and let the pipeline run classic behavior.
    """
    session_id: str = ""
    selected_mode: Optional[str] = None
    sub_mode: Optional[str] = None
    conversation_context_active: bool = False
    customer_name: Optional[str] = None
    technology_domain: Optional[str] = None
    ticket_id: Optional[str] = None
    issue_summary: Optional[str] = None
    last_recommendation: Optional[Dict[str, Any]] = None
    form_data: Optional[Dict[str, Any]] = None
    mode_set_at: Optional[datetime] = None
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe dict for API responses."""
        d = asdict(self)
        if isinstance(self.mode_set_at, datetime):
            d["mode_set_at"] = self.mode_set_at.isoformat()
        return d


# ─────────────────────────────────────────────────────────────
# Validators
# ─────────────────────────────────────────────────────────────
def _normalize_mode(mode: Optional[str]) -> Optional[str]:
    if not mode:
        return None
    m = str(mode).strip().lower().replace("-", "_")
    return m if m in _VALID_MODES else None


def _normalize_sub_mode(sub_mode: Optional[str]) -> Optional[str]:
    if not sub_mode:
        return None
    s = str(sub_mode).strip().lower().replace("-", "_")
    return s if s in _VALID_SUB_MODES else None


def _as_dict(v: Any) -> Optional[Dict[str, Any]]:
    """JSONB fields may arrive as dict (psycopg3) or str — normalize both."""
    if v is None:
        return None
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────
def get_session_mode(session_id: str) -> SessionMode:
    """
    Read the mode-state row for a session. Returns a default SessionMode
    with is_valid=True when the session has no mode set yet; returns
    is_valid=False only on DB failure.
    """
    if not session_id:
        return SessionMode(session_id="", is_valid=False)

    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT
                        id,
                        selected_mode,
                        sub_mode,
                        conversation_context_active,
                        customer_name,
                        technology_domain,
                        ticket_id,
                        issue_summary,
                        last_recommendation,
                        form_data,
                        mode_set_at
                    FROM chat_sessions
                    WHERE id = :sid
                    LIMIT 1
                    """
                ),
                {"sid": session_id},
            ).mappings().first()
    except Exception as exc:
        logger.warning("[session_mode] get failed for sid=%s: %s", session_id, exc)
        return SessionMode(session_id=session_id, is_valid=False)

    if not row:
        return SessionMode(session_id=session_id)

    return SessionMode(
        session_id=str(row["id"]),
        selected_mode=row["selected_mode"],
        sub_mode=row["sub_mode"],
        conversation_context_active=bool(row["conversation_context_active"] or False),
        customer_name=row["customer_name"],
        technology_domain=row["technology_domain"],
        ticket_id=row["ticket_id"],
        issue_summary=row["issue_summary"],
        last_recommendation=_as_dict(row["last_recommendation"]),
        form_data=_as_dict(row["form_data"]),
        mode_set_at=row["mode_set_at"],
        is_valid=True,
    )


# ─────────────────────────────────────────────────────────────
# Write — set mode / sub-mode on landing page submit
# ─────────────────────────────────────────────────────────────
def set_session_mode(
    *,
    session_id: str,
    selected_mode: str,
    sub_mode: Optional[str] = None,
    form_data: Optional[Dict[str, Any]] = None,
) -> SessionMode:
    """
    Lock a session to a mode (and optional sub-mode / form_data).

    Called when the user clicks Continue on the landing page. Sets
    conversation_context_active=True and stamps mode_set_at.
    """
    mode = _normalize_mode(selected_mode)
    sub = _normalize_sub_mode(sub_mode)
    if not session_id or not mode:
        logger.warning(
            "[session_mode] set rejected sid=%r mode=%r sub=%r",
            session_id, selected_mode, sub_mode,
        )
        return SessionMode(session_id=session_id or "", is_valid=False)

    payload = json.dumps(form_data) if form_data else None
    now = datetime.now(timezone.utc)

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE chat_sessions
                    SET
                        selected_mode = :mode,
                        sub_mode = :sub,
                        conversation_context_active = TRUE,
                        form_data = CAST(:form AS JSONB),
                        mode_set_at = :ts
                    WHERE id = :sid
                    """
                ),
                {
                    "mode": mode,
                    "sub": sub,
                    "form": payload,
                    "ts": now,
                    "sid": session_id,
                },
            )
    except Exception as exc:
        logger.warning("[session_mode] set failed for sid=%s: %s", session_id, exc)
        return SessionMode(session_id=session_id, is_valid=False)

    logger.info(
        "[session_mode] SET sid=%s mode=%s sub=%s",
        session_id, mode, sub or "-",
    )
    return get_session_mode(session_id)


# ─────────────────────────────────────────────────────────────
# Reset — "Start New" / "Change Context"
# ─────────────────────────────────────────────────────────────
def reset_session_mode(session_id: str) -> SessionMode:
    """
    Clear all mode-state fields for a session. The session row itself is
    preserved (so chat history stays visible in the sidebar); only the
    mode-related columns are nulled out.
    """
    if not session_id:
        return SessionMode(session_id="", is_valid=False)

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE chat_sessions
                    SET
                        selected_mode = NULL,
                        sub_mode = NULL,
                        conversation_context_active = FALSE,
                        customer_name = NULL,
                        technology_domain = NULL,
                        ticket_id = NULL,
                        issue_summary = NULL,
                        last_recommendation = NULL,
                        form_data = NULL,
                        mode_set_at = NULL
                    WHERE id = :sid
                    """
                ),
                {"sid": session_id},
            )
    except Exception as exc:
        logger.warning("[session_mode] reset failed for sid=%s: %s", session_id, exc)
        return SessionMode(session_id=session_id, is_valid=False)

    logger.info("[session_mode] RESET sid=%s", session_id)
    return get_session_mode(session_id)


# ─────────────────────────────────────────────────────────────
# Patch — partial update used by Sprint 2 flows (form submit, etc.)
# Declared now so endpoints in Sprint 1 forward-compatibly accept
# partial payloads without needing another release.
# ─────────────────────────────────────────────────────────────
def patch_session_mode(
    session_id: str,
    updates: Dict[str, Any],
) -> SessionMode:
    """
    Apply a partial update to mode-state fields. Only whitelisted keys
    are honored; unknown keys are silently ignored.
    """
    if not session_id or not updates:
        return get_session_mode(session_id)

    allowed_scalar = {
        "selected_mode", "sub_mode", "conversation_context_active",
        "customer_name", "technology_domain", "ticket_id", "issue_summary",
    }
    allowed_jsonb = {"last_recommendation", "form_data"}

    set_clauses = []
    params: Dict[str, Any] = {"sid": session_id}

    for key, value in (updates or {}).items():
        if key in allowed_scalar:
            if key == "selected_mode":
                value = _normalize_mode(value)
            elif key == "sub_mode":
                value = _normalize_sub_mode(value)
            set_clauses.append(f"{key} = :{key}")
            params[key] = value
        elif key in allowed_jsonb:
            set_clauses.append(f"{key} = CAST(:{key} AS JSONB)")
            params[key] = json.dumps(value) if value is not None else None

    if not set_clauses:
        return get_session_mode(session_id)

    try:
        with engine.begin() as conn:
            conn.execute(
                text(
                    f"""
                    UPDATE chat_sessions
                    SET {", ".join(set_clauses)}
                    WHERE id = :sid
                    """
                ),
                params,
            )
    except Exception as exc:
        logger.warning("[session_mode] patch failed for sid=%s: %s", session_id, exc)
        return SessionMode(session_id=session_id, is_valid=False)

    logger.info(
        "[session_mode] PATCH sid=%s keys=%s",
        session_id, sorted(updates.keys()),
    )
    return get_session_mode(session_id)
