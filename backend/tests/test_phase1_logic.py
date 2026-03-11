"""
Reference tests for Phase 1 behavior.

These show the critical flows you should validate:
- same-name upload supersedes old active doc
- chat messages persist in PostgreSQL
"""

from backend.vector_store import (
    delete_all_chat_sessions,
    list_chat_sessions,
    save_message_to_session,
)


def test_chat_session_persistence_helpers_work():
    owner_id = "pytest-user"
    delete_all_chat_sessions(owner_id)

    session_id = save_message_to_session(
        session_id=None,
        role="user",
        content="hello phase one",
        owner_id=owner_id,
    )
    save_message_to_session(
        session_id=session_id,
        role="assistant",
        content="hello back",
        owner_id=owner_id,
        sources={"docs": []},
    )

    sessions = list_chat_sessions(owner_id)
    assert sessions, "Expected at least one saved chat session"