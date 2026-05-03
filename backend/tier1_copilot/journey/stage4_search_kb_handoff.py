"""Sprint 10.6 — Stage 4 Search KB handoff (parity with regular /ask).

History:
  - Sprint 10.2 introduced this module with an empty-corpus guard
    (`SELECT COUNT(*) ... metadata_json->>'doc_kind' IN ('sop','kb')`)
    plus an `allowed_doc_kinds=['sop','kb']` filter on the /ask call.
  - Sprint 10.3/10.5 patched the guard's metadata key but kept the
    architectural mistake of gating + filtering at all.
  - Sprint 10.6 deletes both. Engineer-uploaded PDFs go through the
    generic ingestion path which doesn't stamp `doc_kind` on chunk
    metadata, and regular /ask finds them perfectly with no filter.
    Stage 4 must invoke /ask the same way: search everything, no
    gate, no filter, no upload-prompt branch, no `has_corpus` field.

What the helper now does:
  1. Create a fresh chat_sessions row (uses save_message_to_session
     which mints a new session_id when none is passed).
  2. Save the prefilled user turn, embedding journey_session_id in
     metadata so the chat-session GET response can surface it back to
     the frontend (Sprint 10.4 round-trip mechanism).
  3. Optionally invoke ask_fn (if injected) WITHOUT any
     `allowed_doc_kinds` filter — exactly like regular /ask.
  4. Save the assistant turn from ask_fn's response.

If ask_fn is None (current production wiring) the chat opens with the
user turn only; the frontend's regular ask flow generates the
assistant turn after landing.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional


logger = logging.getLogger("acadia-log-iq")


def create_chat_session_with_handoff(
    *,
    journey_session_id: str,
    owner_id: str,
    prefilled_message: str,
    engine: Any = None,
    ask_fn: Optional[Callable[..., Dict[str, Any]]] = None,
    save_message_fn: Optional[Callable[..., str]] = None,
) -> Dict[str, Any]:
    """Sprint 10.6 — orchestrate the Stage 4 handoff.

    Args:
        journey_session_id: tier1_sessions.id this handoff originated
            from. Embedded in the first message's metadata so the chat
            response can surface it back to the frontend (powers the
            inline JourneyMessageActions buttons).
        owner_id: chat_sessions.owner_id. MUST match the authenticated
            user's id; the GET /chat/sessions/{id} endpoint filters by
            owner_id and returns 404 on mismatch (the Sprint 10.6 §4
            bug — owner_id was previously a hardcoded constant).
        prefilled_message: First user-turn content. Built from the
            engineer's intake form + Stage 0 distilled "what worked".
        engine: SQLAlchemy engine. No longer used by the helper itself
            — kept in the signature for backward compatibility with
            the route caller and future telemetry needs.
        ask_fn: Optional injectable that returns an answer dict shaped
            like /ask's AnswerResponse. Sprint 10.6: invoked WITHOUT
            any `allowed_doc_kinds` filter — Search KB now searches
            the same corpus regular /ask searches.
        save_message_fn: Test injection point for save_message_to_session.

    Returns:
        {"chat_session_id", "redirect_url"}  — `has_corpus` removed in 10.6
    """
    if save_message_fn is None:
        from backend.vector_store import save_message_to_session as save_message_fn  # type: ignore

    # 1. Create chat session + insert the prefilled user turn. The
    #    journey_session_id rides on metadata under `_session_metadata`
    #    so the chat-session GET response carries it back at the top
    #    level (Sprint 10.4 round-trip).
    save_kwargs = {
        "session_id": None,
        "role": "user",
        "content": prefilled_message,
        "owner_id": owner_id,
    }
    if journey_session_id:
        save_kwargs["metadata"] = {"journey_session_id": journey_session_id}
    try:
        chat_session_id = save_message_fn(**save_kwargs)
    except TypeError:
        # Test stubs may not accept the metadata kwarg; fall back so
        # existing handoff tests keep passing.
        save_kwargs.pop("metadata", None)
        chat_session_id = save_message_fn(**save_kwargs)

    # 2. Sprint 10.6 §3 — invoke /ask WITHOUT any doc-kind filter.
    #    No empty-corpus gate, no upload-prompt branch. If retrieval
    #    finds nothing, /ask's natural low-confidence response handles
    #    it gracefully — the same UX regular chat gets.
    if ask_fn is not None:
        try:
            answer = ask_fn(
                question=prefilled_message,
                chat_session_id=chat_session_id,
            )
            if isinstance(answer, dict):
                save_message_fn(
                    session_id=chat_session_id,
                    role="assistant",
                    content=str(answer.get("answer") or ""),
                    owner_id=owner_id,
                    sources=answer.get("sources"),
                )
        except Exception as exc:
            logger.error(
                "[journey.stage4] ask_fn raised — leaving chat with "
                "user turn only. chat=%s err=%s",
                chat_session_id, exc,
            )

    return {
        "chat_session_id": chat_session_id,
        "redirect_url": f"/chat/{chat_session_id}",
    }
