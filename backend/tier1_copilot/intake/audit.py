"""Sprint 9 — Audit logger for intake_extractions.

Insert on extract; update on engineer pick / edit / reject. All helpers
are exception-safe (returns None / False on error) so a DB hiccup never
cascades into a user-facing failure.
"""
from __future__ import annotations

import hashlib
import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from backend.tier1_copilot.intake.schemas import ValidatedCandidate

logger = logging.getLogger("acadia-log-iq")


def _hash_text(raw: str) -> str:
    return hashlib.sha1((raw or "").encode("utf-8")).hexdigest()


def log_extraction(
    *,
    source: str,
    raw_text: str,
    candidates: List[ValidatedCandidate],
    session_id: Optional[str] = None,
) -> Optional[str]:
    """Insert a fresh row, return its id. None on failure."""
    extraction_id = f"intk_{uuid.uuid4().hex[:16]}"
    payload_json = json.dumps(
        [c.model_dump() for c in candidates], ensure_ascii=False,
    )
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    INSERT INTO intake_extractions
                        (id, session_id, source, raw_text, raw_text_hash,
                         candidates_json, was_rejected)
                    VALUES (:id, :sid, :src, :raw, :hash,
                            CAST(:cands AS JSONB), FALSE)
                    """
                ),
                {
                    "id": extraction_id,
                    "sid": session_id,
                    "src": source,
                    "raw": raw_text,
                    "hash": _hash_text(raw_text),
                    "cands": payload_json,
                },
            )
        return extraction_id
    except Exception as exc:
        logger.warning("[intake] audit log_extraction failed: %s", exc)
        return None


def log_extraction_feedback(
    *,
    extraction_id: str,
    picked_index: Optional[int] = None,
    edits: Optional[Dict[str, Any]] = None,
    was_rejected: bool = False,
) -> bool:
    """Update the row with engineer feedback. Idempotent — multiple
    edits overwrite the prior `edits_json` rather than appending so the
    final row is the latest known truth."""
    if not extraction_id:
        return False
    edits_json = json.dumps(edits, ensure_ascii=False) if edits else None
    try:
        from sqlalchemy import text
        from backend.db.connection import engine
        with engine.begin() as conn:
            conn.execute(
                text(
                    """
                    UPDATE intake_extractions
                    SET picked_index = :idx,
                        edits_json   = CAST(:edits AS JSONB),
                        was_rejected = :rej,
                        feedback_at  = NOW()
                    WHERE id = :id
                    """
                ),
                {
                    "id": extraction_id,
                    "idx": picked_index,
                    "edits": edits_json,
                    "rej": bool(was_rejected),
                },
            )
        return True
    except Exception as exc:
        logger.warning("[intake] audit log_feedback failed: %s", exc)
        return False
