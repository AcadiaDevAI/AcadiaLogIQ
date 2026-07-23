"""Shared US Pharma store-scoped KB chat opener helpers (Idea A).

Both entry points into a store-scoped KB chat open with the SAME 2-line
store-orientation summary + the SAME clickable sample questions:

  1. Journey Stage 4 handoff (search-kb-handoff route).
  2. The standalone "Discuss Store Specific with LogIQ" store-scoped chat
     (POST /chat/sessions/store-scoped), launched from the landing tile,
     the stages quick-actions pill, and the intake form.

Keeping the summary builder + the sample questions here means one source
of truth, so the two flows never drift.
"""
from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger("acadia-log-iq")


def store_sample_questions(store_id: str) -> List[str]:
    """The clickable sample questions rendered under the store opener."""
    sid = (store_id or "").strip()
    return [
        f"Who is the primary network provider for store {sid}?",
        f"What is the escalation procedure for store {sid}?",
        f"Who is the backup network provider for store {sid}?",
    ]


def build_store_summary(store_id: str) -> Optional[str]:
    """A short 2-line orientation for the store-scoped KB chat opener.

    Reads the store's single config chunk and asks Haiku for a tight,
    2-line summary (location + primary network provider). Uses a plain,
    constrained prompt — NOT the KB-Search structured prompt — so the
    output stays two lines with no headings or `## Source documents`
    footer. Returns a deterministic fallback if the chunk or the LLM is
    unavailable; never raises.
    """
    sid = (store_id or "").strip()
    if not sid:
        return None

    fallback = (
        f"You're now in a chat scoped to Store {sid} — every answer below is "
        f"limited to this store's data. Ask about its network, vendors, or incidents."
    )

    # 1. Fetch the single store-config chunk (one per store; the exact
    #    nested path the store-scoped retrieval filters on, migration 065).
    chunk_text = ""
    try:
        from sqlalchemy import text as _text
        from backend.db.connection import engine as _engine
        with _engine.connect() as conn:
            row = conn.execute(
                _text(
                    "SELECT COALESCE(contextualized_content, content) AS body "
                    "FROM chunks "
                    "WHERE (metadata_json->'Metadata'->>'store_id') = :sid "
                    "  AND chunk_type = 'store_config' "
                    "LIMIT 1"
                ),
                {"sid": sid},
            ).mappings().first()
        if row and row.get("body"):
            chunk_text = str(row["body"])
    except Exception as exc:
        logger.warning(
            "[store_kb] store-config chunk read failed sid=%s err=%s", sid, exc
        )

    if not chunk_text:
        return fallback

    # 2. Tight, constrained LLM call. Two lines, plain text, no sources.
    prompt = (
        "You are orienting a support engineer who just opened a chat scoped to "
        f"US Pharma Store {sid}. Using ONLY the store configuration below, write "
        "EXACTLY two short lines:\n"
        f"Line 1: Store {sid} and its city/state location.\n"
        "Line 2: the primary network provider (and backup provider if present).\n"
        "Plain text only — no markdown, no headings, no bullet points, no sources, "
        "no preamble. If a fact is missing, omit it gracefully.\n\n"
        f"STORE CONFIGURATION:\n{chunk_text[:6000]}"
    )
    try:
        from backend.tier1_copilot.routes import _invoke_haiku
        out = (_invoke_haiku(prompt) or "").strip()
    except Exception as exc:
        logger.warning(
            "[store_kb] store summary LLM failed sid=%s err=%s", sid, exc
        )
        out = ""
    return out or fallback
