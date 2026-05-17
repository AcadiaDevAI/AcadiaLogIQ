"""Sprint 10.2 — Stage 4 Search KB handoff helper tests.

The helper is the testable surface (the route handler is a thin
wrapper around it). Tests inject a mock save_message_fn so we never
touch the real chat_sessions / chat_messages tables; ask_fn is
similarly injectable so we never invoke real Bedrock.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.tier1_copilot.journey.stage4_search_kb_handoff import (
    create_chat_session_with_handoff,
)


def _stub_engine_with_corpus_count(count: int):
    """Mock engine whose corpus-count query returns the given count."""
    fake_result = MagicMock()
    fake_result.first.return_value = {"c": count}
    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.return_value = fake_result
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm
    return engine


def _stub_save_message_fn():
    """Mock save_message_to_session that returns a fixed session_id and
    captures every call for later assertions.

    Sprint 10.4 — accepts the new optional `metadata` kwarg so the
    handoff helper can forward `journey_session_id` without falling
    through to its TypeError fallback path."""
    saved = []

    def fake_save(*, session_id, role, content, owner_id,
                  sources=None, metadata=None):
        new_id = session_id or "chat-NEW-12345"
        saved.append({
            "session_id": new_id,
            "role": role,
            "content": content,
            "owner_id": owner_id,
            "sources": sources,
            "metadata": metadata,
        })
        return new_id

    return fake_save, saved


def test_handoff_invokes_ask_without_doc_kind_filter():
    """Sprint 10.6 §3.5 — replaces the deleted Sprint 10.2 tests for
    has_corpus / allowed_doc_kinds. Search KB now invokes /ask exactly
    the way regular chat does: no doc-kind gate, no filter.

    Asserts:
      - ask_fn IS invoked (no empty-corpus gate suppressing it).
      - ask_fn is called WITHOUT the `allowed_doc_kinds` kwarg.
      - User + assistant turns are persisted.
      - Response shape contains chat_session_id + redirect_url ONLY —
        the has_corpus boolean is gone."""
    save_fn, saved = _stub_save_message_fn()
    ask_calls = []

    def fake_ask_fn(**kwargs):
        ask_calls.append(kwargs)
        return {
            "answer": "Found it in incident_records.pdf, page 3.",
            "sources": [{"file": "incident_records.pdf"}],
        }

    result = create_chat_session_with_handoff(
        journey_session_id="journey-sess-parity",
        owner_id="test-owner",
        prefilled_message="What's the fix for VDI desktop slowness?",
        engine=None,  # 10.6 — helper no longer queries the corpus
        ask_fn=fake_ask_fn,
        save_message_fn=save_fn,
    )

    # Sprint 10.6 §3.4 — has_corpus removed from response.
    assert "has_corpus" not in result
    assert result["chat_session_id"] == "chat-NEW-12345"
    assert result["redirect_url"] == "/chat/chat-NEW-12345"

    # ask_fn invoked exactly once, WITHOUT allowed_doc_kinds.
    assert len(ask_calls) == 1
    call = ask_calls[0]
    assert "allowed_doc_kinds" not in call, (
        "Sprint 10.6: Stage 4 must invoke /ask the same way regular "
        "chat does — no doc-kind filter. The PDF the engineer uploads "
        "has no doc_kind metadata; filtering excluded it."
    )
    assert call["question"] == "What's the fix for VDI desktop slowness?"
    assert call["chat_session_id"] == "chat-NEW-12345"

    # User + assistant turns saved.
    assert len(saved) == 2
    assert saved[0]["role"] == "user"
    assert saved[1]["role"] == "assistant"
    assert saved[1]["content"] == "Found it in incident_records.pdf, page 3."
    assert saved[1]["sources"] == [{"file": "incident_records.pdf"}]


def test_handoff_finds_uploaded_pdf_with_no_metadata_tag():
    """Sprint 10.6 §3.5 — regression for the architectural mistake. A
    chunk indexed via the generic file path has NO `doc_kind` field
    in its metadata_json (the engineer's `incident_records.pdf` is
    exactly this case). The previous gate query returned 0 and the
    upload-prompt branch fired, even though regular /ask finds the
    same PDF perfectly with no filter.

    Post-10.6: no gate query is run, no upload-prompt branch fires,
    and ask_fn is invoked unconditionally — proving Search KB now
    behaves like regular /ask."""
    save_fn, saved = _stub_save_message_fn()
    ask_invoked = []

    def fake_ask_fn(**kwargs):
        ask_invoked.append(True)
        # ask_fn never sees a filter for this corpus shape.
        assert "allowed_doc_kinds" not in kwargs
        # The PDF chunk has plain text content, no doc_kind metadata —
        # exactly the corpus shape that broke pre-10.6. The unfiltered
        # retrieval pipeline finds it.
        return {
            "answer": "Per incident_records.pdf, restart the DDC service.",
            "sources": [{"file": "incident_records.pdf", "chunk": 42}],
        }

    # An engine value the helper would have queried pre-10.6. Post-
    # 10.6 it's not used at all — the helper takes no DB-call path
    # for the corpus check.
    engine = MagicMock(name="should-not-be-queried")

    result = create_chat_session_with_handoff(
        journey_session_id="journey-sess-pdf",
        owner_id="test-owner",
        prefilled_message="What does incident_records.pdf say about VDI?",
        engine=engine,
        ask_fn=fake_ask_fn,
        save_message_fn=save_fn,
    )

    # ask_fn fired regardless of doc_kind metadata shape.
    assert ask_invoked == [True]
    # No has_corpus / no upload-prompt branching.
    assert "has_corpus" not in result
    # Engine was not queried — corpus-count SQL is fully gone.
    assert engine.connect.called is False
    assert engine.begin.called is False
    # The grounded answer from the unfiltered pipeline is persisted.
    assert saved[-1]["role"] == "assistant"
    assert "incident_records.pdf" in saved[-1]["content"]


def test_creates_new_chat_session_row():
    """Sprint 10.2 §7 — a new chat_sessions row is inserted (delegated
    via save_message_fn whose first call carries session_id=None to
    trigger session creation in vector_store.save_message_to_session).
    redirect_url matches /chat/{id} pattern."""
    engine = _stub_engine_with_corpus_count(0)  # empty corpus to keep things simple
    save_fn, saved = _stub_save_message_fn()

    result = create_chat_session_with_handoff(
        journey_session_id="journey-sess-2",
        owner_id="test-owner",
        prefilled_message="Test message",
        engine=engine,
        ask_fn=None,
        save_message_fn=save_fn,
    )

    # First save_message_to_session call carried session_id=None →
    # forces session creation. The returned id is the new chat_session_id.
    assert saved[0]["session_id"] == "chat-NEW-12345"  # what our stub returns

    # Subsequent saves on the same session reuse the id we got back.
    if len(saved) > 1:
        for s in saved[1:]:
            assert s["session_id"] == "chat-NEW-12345"

    # Redirect URL matches /chat/{id} pattern
    assert result["chat_session_id"] == "chat-NEW-12345"
    assert result["redirect_url"] == "/chat/chat-NEW-12345"


def test_handoff_persists_session_to_chat_sessions_table():
    """Sprint 10.3 §6 — the handoff must create a row in chat_sessions
    (delegated via save_message_fn whose first call carries
    session_id=None to trigger the create path inside
    vector_store.save_message_to_session). Asserts the persistence call
    is invoked with the right shape."""
    engine = _stub_engine_with_corpus_count(0)
    save_fn, saved = _stub_save_message_fn()

    create_chat_session_with_handoff(
        journey_session_id="journey-sess-create",
        owner_id="owner-42",
        prefilled_message="Persist me",
        engine=engine,
        ask_fn=None,
        save_message_fn=save_fn,
    )

    # The very first save_message_fn call drives chat_sessions row
    # creation (session_id=None → vector_store.save_message_to_session
    # mints a new id). The mock returns "chat-NEW-12345" and the helper
    # threads that id back through subsequent saves.
    assert len(saved) >= 1
    first = saved[0]
    assert first["session_id"] == "chat-NEW-12345"
    assert first["role"] == "user"
    assert first["owner_id"] == "owner-42"


def test_handoff_persists_first_user_message():
    """Sprint 10.3 §6 — the prefilled message must be persisted verbatim
    as the first user turn on the new chat session. This guarantees
    `SELECT * FROM chat_messages WHERE session_id = :id ORDER BY
    created_at ASC LIMIT 1` returns the engineer's intake summary."""
    engine = _stub_engine_with_corpus_count(0)
    save_fn, saved = _stub_save_message_fn()

    prefill = (
        "Severity P2 — Desktop Slowness on V-Desktop Environment. "
        "Stage 0 distilled: Restart DDC and purge stale lock."
    )
    create_chat_session_with_handoff(
        journey_session_id="journey-sess-msg",
        owner_id="owner-42",
        prefilled_message=prefill,
        engine=engine,
        ask_fn=None,
        save_message_fn=save_fn,
    )

    # The first persisted message is the prefilled user turn, content-
    # identical (no truncation, no escaping mutation).
    assert saved[0]["role"] == "user"
    assert saved[0]["content"] == prefill


def test_chat_session_response_includes_journey_session_id():
    """Sprint 10.4 §5.5 — verify that:
      (a) the handoff helper forwards journey_session_id as metadata
          on the FIRST message save, and
      (b) get_chat_session round-trips it as a top-level `metadata`
          dict on the response payload.

    Together these two halves let the frontend gate the "Back to
    Resolution Journey" banner on `activeSession.metadata.journey_session_id`."""
    # Half (a) — handoff helper forwards metadata.
    engine = _stub_engine_with_corpus_count(0)
    save_fn, saved = _stub_save_message_fn()

    create_chat_session_with_handoff(
        journey_session_id="sess_journey_xyz",
        owner_id="owner-42",
        prefilled_message="Severity P2 — Desktop Slowness",
        engine=engine,
        ask_fn=None,
        save_message_fn=save_fn,
    )
    assert saved[0]["metadata"] == {"journey_session_id": "sess_journey_xyz"}

    # Half (b) — get_chat_session surfaces the metadata back at read
    # time. Mock the vector_store DB layer so we can verify the
    # extraction logic without a real Postgres.
    from datetime import datetime, timezone
    now = datetime(2026, 4, 30, tzinfo=timezone.utc)
    fake_session_row = {
        "id": "chat-NEW-12345",
        "title": "Severity P2 — Desktop Slowness",
        "created_at": now,
        "updated_at": now,
    }
    fake_first_msg = {
        "role": "user",
        "content": "Severity P2 — Desktop Slowness",
        "sources_json": {
            "_session_metadata": {"journey_session_id": "sess_journey_xyz"},
        },
        "feedback": None,
        "created_at": now,
    }

    class FakeMappings:
        def __init__(self, payload):
            self._payload = payload

        def first(self):
            return self._payload[0] if self._payload else None

        def all(self):
            return list(self._payload)

    class FakeResult:
        def __init__(self, rows):
            self._rows = rows

        def mappings(self):
            return FakeMappings(self._rows)

    class FakeDB:
        def __init__(self):
            self._step = 0

        def execute(self, *_a, **_kw):
            self._step += 1
            if self._step == 1:
                return FakeResult([fake_session_row])
            return FakeResult([fake_first_msg])

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    from unittest.mock import patch
    from backend import vector_store as vs

    with patch.object(vs, "SessionLocal", FakeDB):
        out = vs.get_chat_session("chat-NEW-12345", "owner-42")

    assert out is not None
    assert "metadata" in out, "response must surface a top-level metadata field"
    assert out["metadata"] == {"journey_session_id": "sess_journey_xyz"}
    # The private namespaced key must be stripped from the user-facing
    # sources blob — engineers shouldn't see _session_metadata in chat.
    first_msg_out = out["messages"][0]
    if first_msg_out["sources"] is not None:
        assert "_session_metadata" not in first_msg_out["sources"]


def test_ask_fn_failure_does_not_propagate():
    """If ask_fn itself raises, the helper logs and returns; only the
    user turn is saved. User experience: chat opens with the prefilled
    message, retry naturally on the next user input."""
    save_fn, saved = _stub_save_message_fn()

    def boom(**kwargs):
        raise RuntimeError("Bedrock unavailable")

    result = create_chat_session_with_handoff(
        journey_session_id="journey-sess-4",
        owner_id="test-owner",
        prefilled_message="Q",
        engine=None,  # Sprint 10.6 — helper takes no DB-call path
        ask_fn=boom,
        save_message_fn=save_fn,
    )

    # Sprint 10.6 — has_corpus removed from response.
    assert "has_corpus" not in result
    assert "chat_session_id" in result
    # Only the user turn was saved; ask_fn raised so no assistant turn
    assert len(saved) == 1
    assert saved[0]["role"] == "user"


def test_handoff_route_returns_200_with_valid_token_not_401():
    """Sprint 10.6.1 hotfix — when the handoff endpoint is hit with a
    valid Clerk token (mocked here via FastAPI's dependency_overrides),
    it must return 200, NOT 401.

    Background: Sprint 10.6 §4 added Depends(_lazy_auth_dependency) on
    /search-kb-handoff to fix the GET-404 caused by an unowned
    chat_sessions row. That introduced a 401 surface — and the
    frontend's journey API client (`journeyApi.js`) had a separate
    bare axios instance with no Clerk request interceptor, so it
    never attached Authorization: Bearer ..., producing 401 in 4.86ms.

    The frontend fix is to route journey calls through the
    authenticated `api` instance from services/api.js. This backend
    test guards the contract: when the auth dependency yields a
    user_id, the route reaches the handler (200), not the 401 path.
    The dependency_overrides mechanism is FastAPI's idiomatic way to
    inject a stand-in user for an authenticated endpoint without
    booting Clerk."""
    import pytest
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.config import settings
    from backend.tier1_copilot.journey.routes import (
        router,
        _lazy_auth_dependency,
        _reset_cache,
    )

    _reset_cache()
    app = FastAPI()
    app.include_router(router)

    # Stand in for an authenticated Clerk session — returns a Clerk-
    # shaped user_id. The override matches FastAPI's dependency-injection
    # contract; the real _lazy_auth_dependency would have called
    # backend.api.auth_dependency which talks to Clerk.
    async def _fake_authenticated_user():
        return "user_2qFAKE_CLERK_ID"

    app.dependency_overrides[_lazy_auth_dependency] = _fake_authenticated_user

    # Patch the helper so we don't touch the real DB. Returns the
    # shape the route expects.
    fake_helper_result = {
        "chat_session_id": "chat-AUTH-OK",
        "redirect_url": "/chat/chat-AUTH-OK",
    }

    with patch(
             "backend.tier1_copilot.journey.routes.load_cohort_metadata",
             return_value=[],
         ), \
         patch(
             "backend.tier1_copilot.journey.routes.create_chat_session_with_handoff",
             return_value=fake_helper_result,
         ):
        # Stub the corpus-aggregate engine call Stage 0 makes inside
        # the route (the route runs compute_stage0 to build the
        # dominant root cause string before the handoff helper).
        fake_result = MagicMock()
        fake_result.first.return_value = {"corpus_size": 0, "platform_median_minutes": None}
        fake_conn = MagicMock()
        fake_conn.execute.return_value.mappings.return_value = fake_result
        cm = MagicMock()
        cm.__enter__.return_value = fake_conn
        cm.__exit__.return_value = False
        engine = MagicMock()
        engine.connect.return_value = cm

        with patch("backend.db.connection.engine", engine):
            client = TestClient(app)
            r = client.post("/tier1/journey/sess_test/search-kb-handoff")

    # The critical assertion: with an authenticated user injected, the
    # route reaches the handler. 200 (success), NOT 401 (the bug we
    # are guarding against).
    assert r.status_code == 200, (
        f"Expected 200 with valid auth, got {r.status_code}. "
        f"If this is 401, the auth dependency override isn't being "
        f"honoured — the route may have been re-wired to bypass "
        f"FastAPI's dependency system."
    )

    body = r.json()
    assert body["chat_session_id"] == "chat-AUTH-OK"
    assert body["redirect_url"] == "/chat/chat-AUTH-OK"
    # has_corpus removed in Sprint 10.6 §3.4
    assert "has_corpus" not in body


def test_handoff_route_returns_401_without_auth():
    """Sprint 10.6.1 hotfix — companion to the 200 test above.
    Without an authenticated user the route must return 401 (Clerk's
    rejection), proving the auth dependency is wired and active.
    Without this test, a future refactor that accidentally drops the
    Depends(_lazy_auth_dependency) would silently re-introduce the
    Sprint 10.6 §4 ownership bug (404 from owner_id mismatch on the
    follow-up GET)."""
    import pytest
    from fastapi import FastAPI, HTTPException, status
    from fastapi.testclient import TestClient
    from backend.config import settings
    from backend.tier1_copilot.journey.routes import (
        router,
        _lazy_auth_dependency,
        _reset_cache,
    )

    _reset_cache()
    app = FastAPI()
    app.include_router(router)

    # Override the auth dependency to simulate Clerk rejecting an
    # unauthenticated request — the same 401 the real Clerk middleware
    # raises when no/invalid Bearer token is present.
    async def _reject_unauthenticated():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please sign in.",
        )

    app.dependency_overrides[_lazy_auth_dependency] = _reject_unauthenticated

    client = TestClient(app)
    r = client.post("/tier1/journey/sess_no_auth/search-kb-handoff")

    assert r.status_code == 401


def test_handoff_chat_session_is_immediately_fetchable():
    """Sprint 10.6 §4.5 — the new chat_sessions row must carry the
    authenticated caller's owner_id so a follow-up
    GET /chat/sessions/{id} (which filters by owner_id) returns the
    row. Pre-10.6 the route hardcoded owner_id="tier1-journey", which
    made the row invisible to the engineer's auth context — the 404
    the user reported.

    This test verifies the helper threads owner_id through to the
    save_message_fn call. The route-level test for the auth-injected
    owner_id is covered by the handoff route itself; this is the
    helper-layer regression."""
    save_fn, saved = _stub_save_message_fn()
    AUTHENTICATED_USER = "user_2qABCDEF"   # Clerk-shaped id

    create_chat_session_with_handoff(
        journey_session_id="journey-sess-auth",
        owner_id=AUTHENTICATED_USER,
        prefilled_message="Search for fix",
        engine=None,
        ask_fn=None,
        save_message_fn=save_fn,
    )

    # The first save call (which mints the chat_sessions row) must
    # carry the authenticated user's id. If the helper ever defaults
    # back to a constant or drops the kwarg, the new row's owner_id
    # would mismatch the GET filter and 404 would return — exactly the
    # Sprint 10.6 §4 regression.
    assert len(saved) >= 1
    assert saved[0]["owner_id"] == AUTHENTICATED_USER, (
        "owner_id must match the authenticated caller so the immediate "
        "follow-up GET /chat/sessions/{id} doesn't 404"
    )
