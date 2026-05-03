"""Sprint 11 — Journey-aware retrieval context for /ask.

When an engineer clicks "Ask in chat" beside a step on Stage 0 / Stage 3,
the chat session created via /tier1/journey/<sid>/search-kb-handoff
carries the journey's `journey_session_id` in its first-message
`_session_metadata` blob (the existing Sprint 10.4 round-trip).

This module gives /ask a clean way to look up that journey id and
fetch the originating ticket's intake context (severity, asset name,
alert type, customer, location, notes). The caller uses the returned
dict to bias retrieval — typically by prepending a compact context
string to the BM25 / keyword query — without touching the LLM prompt
or the visible chat message.

Why this lives here, not inline in api.py:
  - Easier to unit-test (no /ask integration harness needed).
  - Single source of truth for the metadata-extraction shape, so a
    future stage that wants the same enrichment doesn't rebuild it.
  - The tier1_sessions read is wrapped in a tolerant try/except — a
    failure here must NEVER block /ask. Worst case is missing context
    enrichment; the bare query still works.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# Fields lifted from tier1_sessions.alert_payload that genuinely help
# retrieval. Order matters for the prefix string (most discriminating
# first). Keep this list short — every additional token is BM25 noise
# if it doesn't appear in the matching corpus chunks.
_CONTEXT_FIELDS = (
    "asset_name",
    "alert_type",
    "component_category",
    "customer",
    "severity",
    "location",
    "ip_or_device_id",
    "error_code",
    "notes",
)


def _read_journey_session_id(db, chat_session_id: str) -> Optional[str]:
    """Pull journey_session_id from the chat session's first message.

    The Sprint 10.4 round-trip embeds the metadata under
    `sources_json._session_metadata.journey_session_id` on the FIRST
    chat_messages row. Returns None when no journey is attached or
    the row doesn't exist.
    """
    if not chat_session_id:
        return None
    row = db.execute(
        text(
            """
            SELECT sources_json
            FROM chat_messages
            WHERE session_id = :sid
            ORDER BY id ASC
            LIMIT 1
            """
        ),
        {"sid": chat_session_id},
    ).mappings().first()
    if not row:
        return None
    raw = row.get("sources_json")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (TypeError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    meta = raw.get("_session_metadata")
    if not isinstance(meta, dict):
        return None
    sid = meta.get("journey_session_id")
    return str(sid) if sid else None


def _read_alert_payload(db, journey_session_id: str) -> Optional[Dict[str, Any]]:
    """Load the engineer's original alert from tier1_sessions.alert_payload.
    Returns the dict or None when the row / payload is missing."""
    if not journey_session_id:
        return None
    row = db.execute(
        text(
            "SELECT alert_payload FROM tier1_sessions WHERE id = :id"
        ),
        {"id": journey_session_id},
    ).mappings().first()
    if not row:
        return None
    payload = row.get("alert_payload")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except (TypeError, ValueError):
            return None
    if not isinstance(payload, dict):
        return None
    return payload


def load_journey_context(
    *, chat_session_id: str, db_session_factory
) -> Optional[Dict[str, Any]]:
    """Returns a context dict (or None) for use in retrieval enrichment.

    Args:
      chat_session_id: the /ask `session_id` (the chat session, not the
        journey).
      db_session_factory: a callable that returns a SQLAlchemy session
        (typically `SessionLocal`). Passed in for test injection.

    Behaviour:
      - Failure is silent — any exception logs a warning and returns
        None. /ask must never break because of context enrichment.
      - Returned dict only contains fields with non-empty values. A
        ticket missing severity won't carry "severity": None.
    """
    if not chat_session_id:
        return None
    try:
        with db_session_factory() as db:
            journey_sid = _read_journey_session_id(db, chat_session_id)
            if not journey_sid:
                return None
            payload = _read_alert_payload(db, journey_sid)
            if not payload:
                return None
            ctx: Dict[str, Any] = {}
            for key in _CONTEXT_FIELDS:
                value = payload.get(key)
                if value not in (None, "", [], {}):
                    ctx[key] = value
            if not ctx:
                return None
            ctx["_journey_session_id"] = journey_sid
            return ctx
    except Exception as exc:
        # Fail-open — context enrichment is best-effort; a transient
        # DB blip must not break /ask.
        logger.warning(
            "[journey_context] load failed for chat=%s (%s) — falling back",
            chat_session_id, exc,
        )
        return None


def build_query_prefix(ctx: Optional[Dict[str, Any]]) -> str:
    """Render the context dict as a compact BM25-friendly prefix.

    Output shape: "[Context: asset=V-Desktop alert=Slowness customer=Acme
    severity=P3] " — short, space-separated key=value pairs that BM25
    tokenisation handles well. The square-bracket framing is there so a
    log scrape can spot enriched-vs-bare queries at a glance.

    Returns "" when ctx is None or empty so the caller can do
    `prefix + query` unconditionally.
    """
    if not ctx:
        return ""
    parts = []
    for key in _CONTEXT_FIELDS:
        value = ctx.get(key)
        if value is None or value == "" or value == [] or value == {}:
            continue
        # Collapse whitespace so the prefix stays one line; this matters
        # for long `notes` values that may contain newlines.
        text_val = " ".join(str(value).split())
        parts.append(f"{key}={text_val}")
    if not parts:
        return ""
    return f"[Context: {' '.join(parts)}] "
