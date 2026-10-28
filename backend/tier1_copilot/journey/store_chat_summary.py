"""US Pharma — "Discuss Store Specifics with LogIQ" questions summary.

When a Tier-1 engineer escalates a store incident to Tier-2, the
Operational Handoff panel already shows *how long* the engineer spent
on the KB / store chats (compute_journey_time_metrics). This module
adds the missing *what*: a short LLM-generated digest of the topics
the engineer asked about across ALL their store-scoped chats, so the
Tier-2 colleague picking up the escalation can see which areas were
already explored without reading every transcript.

Design notes / contract:

  * Store-specific + cumulative. The journey session carries the
    Store ID in `tier1_sessions.alert_payload.store_id`. Every chat
    the engineer opened for that store — the standalone "Discuss Store
    Specifics with LogIQ" chats (POST /chat/sessions/store-scoped) AND
    the Stage-4 store KB handoff chats — persists `scope_store_id` on
    its `chat_sessions` row. We gather the engineer's questions across
    *all* of them (owner-scoped), so a summary regenerated after the
    engineer went back and asked more reflects both the earlier and
    the latest questions.

  * Store-switch safe. Because we key strictly on the journey's
    current `store_id`, changing the store number naturally scopes the
    summary to that store's chats only.

  * No cache. The route builds this fresh on every call, so each
    click of Escalation / Handoff picks up the latest interaction.

  * Empty-case honest. No store_id, no store chats, or no user
    questions → returns summary=None. The frontend then hides the
    section and the escalation report is exactly what it was before
    (time-spent-on-KB only). We never fabricate a summary.

  * Failure-open. Any DB / LLM error returns an empty result and logs
    a warning; this feature never blocks or blanks the escalation
    panel.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")

# Cap the number of questions fed to the LLM so a very chatty session
# can't blow the prompt budget. Oldest→newest order is preserved; if we
# truncate we keep the MOST RECENT questions (the tail) since "latest
# interaction" is what the escalation reader cares about most.
_MAX_QUESTIONS = 80

# Hard cap on bullets in the rendered digest. Haiku follows the "2–4
# bullets" instruction well, but we clamp server-side too so a stray
# extra line or trailing narrative can never bloat the escalation card.
_MAX_BULLETS = 4

# System prompt — kept terse and imperative so Haiku holds the format.
_SUMMARY_SYSTEM = (
    "You summarize a support engineer's chat questions into a short, "
    "skimmable digest for a Tier-2 colleague picking up an escalation. "
    "You never invent answers, findings, or details that are not present "
    "in the questions. You always answer with 2 to 4 bullet points and "
    "nothing else."
)


def _fetch_store_id_for_session(session_id: str) -> Optional[str]:
    """Read the Store ID this journey is scoped to from
    `tier1_sessions.alert_payload.store_id`. Returns None when the
    session has no store (non-US-Pharma journeys) or on any error."""
    if not session_id:
        return None
    try:
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT alert_payload FROM tier1_sessions WHERE id = :id"),
                {"id": session_id},
            ).mappings().first()
        if row and isinstance(row.get("alert_payload"), dict):
            sid = row["alert_payload"].get("store_id")
            sid = (str(sid).strip() if sid is not None else "")
            return sid or None
    except Exception as exc:
        logger.warning(
            "[store_chat_summary] store_id lookup failed sid=%s err=%s",
            session_id, exc,
        )
    return None


def _collect_store_questions(store_id: str, owner_id: str) -> List[str]:
    """Return every `role='user'` message the engineer typed across all
    chat sessions scoped to this store (owner-scoped), oldest→newest.

    Both entry points that open a store chat persist
    `chat_sessions.scope_store_id`, so a single filter on that column
    captures the standalone Discuss chats and the Stage-4 store KB
    handoff chats alike. Assistant openers/answers are excluded — only
    the engineer's own questions feed the summary."""
    if not store_id or not owner_id:
        return []
    try:
        from backend.db.connection import engine
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT cm.content AS content
                    FROM chat_messages cm
                    JOIN chat_sessions cs ON cs.id = cm.session_id
                    WHERE cs.scope_store_id = :store
                      AND cs.owner_id = :owner
                      AND cm.role = 'user'
                    ORDER BY cm.created_at ASC, cm.id ASC
                    """
                ),
                {"store": str(store_id).strip(), "owner": owner_id},
            ).mappings().all()
    except Exception as exc:
        logger.warning(
            "[store_chat_summary] question collection failed store=%s err=%s",
            store_id, exc,
        )
        return []

    questions: List[str] = []
    for r in rows:
        q = (r.get("content") or "").strip()
        if q:
            questions.append(q)
    return questions


def _build_prompt(store_id: str, questions: List[str]) -> str:
    numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
    return (
        "A pharmacy support engineer opened one or more "
        f"\"Discuss Store Specifics with LogIQ\" chat sessions for Store "
        f"{store_id} while triaging an incident, and is now escalating to "
        "Tier 2. Below are the questions the engineer asked, across all of "
        "those chats, in the order they were asked:\n\n"
        f"{numbered}\n\n"
        "Summarize the DISTINCT topics / areas the engineer explored, so a "
        "Tier-2 colleague can see what was already covered. Rules:\n"
        "- Output 2 to 4 bullet points, each starting with \"- \".\n"
        "- Group related questions into a single topic; do NOT restate each "
        "question individually.\n"
        "- Keep the whole thing under about 50 words.\n"
        "- Summarize ONLY what the questions reveal; do not invent answers "
        "or details.\n"
        "- If the questions are only greetings or trivial chatter, output a "
        "single bullet saying no substantive topics were discussed.\n"
        "Output only the bullet points, nothing else."
    )


def _postprocess_summary(raw: str) -> Optional[str]:
    """Normalize Haiku's output to at most _MAX_BULLETS clean bullet
    lines. Strips any leading bullet marker, drops blank/non-content
    lines (e.g. a stray preamble or trailing narrative), and clamps the
    count. Returns None when nothing usable remains."""
    if not raw:
        return None
    bullets: List[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        # Keep only lines that read as bullets; a model that ignores the
        # format and returns a paragraph leaves us with no bullets, which
        # we treat as "no usable summary" rather than dumping prose into
        # the card.
        if text[0] in "-*•":
            text = text.lstrip("-*• \t").strip()
            if text:
                bullets.append(text)
        if len(bullets) >= _MAX_BULLETS:
            break
    if not bullets:
        return None
    return "\n".join(f"- {b}" for b in bullets)


def build_store_chat_summary(session_id: str, owner_id: str) -> Dict[str, Any]:
    """Assemble the store-chat topics digest for the escalation screen.

    Returns a dict:
        {
          "summary": Optional[str],   # None when nothing to summarize
          "store_id": Optional[str],
          "question_count": int,
          "chat_count": int,          # sessions that contributed ≥1 question
        }

    Never raises."""
    result: Dict[str, Any] = {
        "summary": None,
        "store_id": None,
        "question_count": 0,
        "chat_count": 0,
    }

    store_id = _fetch_store_id_for_session(session_id)
    if not store_id:
        return result
    result["store_id"] = store_id

    questions = _collect_store_questions(store_id, owner_id)
    result["question_count"] = len(questions)
    # chat_count — distinct sessions that contributed a question. Cheap
    # secondary query; failure-open (leaves count at 0).
    try:
        from backend.db.connection import engine
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    """
                    SELECT COUNT(DISTINCT cm.session_id) AS n
                    FROM chat_messages cm
                    JOIN chat_sessions cs ON cs.id = cm.session_id
                    WHERE cs.scope_store_id = :store
                      AND cs.owner_id = :owner
                      AND cm.role = 'user'
                    """
                ),
                {"store": store_id, "owner": owner_id},
            ).mappings().first()
        if row and row.get("n") is not None:
            result["chat_count"] = int(row["n"])
    except Exception:
        pass

    if not questions:
        # Empty case — no user questions for this store. Frontend falls
        # back to the existing time-spent-on-KB report unchanged.
        return result

    # Keep the most recent questions when over the cap (see _MAX_QUESTIONS).
    if len(questions) > _MAX_QUESTIONS:
        logger.info(
            "[store_chat_summary] store=%s truncating %d→%d questions",
            store_id, len(questions), _MAX_QUESTIONS,
        )
        questions = questions[-_MAX_QUESTIONS:]

    try:
        from backend.services.bedrock_haiku import haiku_client
        prompt = _build_prompt(store_id, questions)
        # Haiku (Claude) holds the "2–4 bullets" format far better than
        # the Mistral primary; invoke_text returns "" on any failure, so
        # a Bedrock hiccup just yields no summary and the UI falls back.
        # 200 tokens is ample for ≤4 short bullets and keeps cost tiny.
        raw = haiku_client.invoke_text(
            prompt=prompt,
            system=_SUMMARY_SYSTEM,
            max_tokens=200,
            context="store_chat_summary",
        )
        result["summary"] = _postprocess_summary(raw)
    except Exception as exc:
        logger.warning(
            "[store_chat_summary] LLM summary failed store=%s err=%s",
            store_id, exc,
        )

    return result
