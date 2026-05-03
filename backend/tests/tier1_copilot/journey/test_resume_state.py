"""Sprint 10.7 §3.4 — /resume-state endpoint tests.

Five backend tests verbatim from the spec:
  - fresh-session → stage_0
  - stage_advanced events present → returns latest stage
  - helpful_clicked alone is NOT a stage advance
  - DB error → graceful stage_0 fallback (no 500)
  - missing auth → 401
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException, status
from fastapi.testclient import TestClient

from backend.config import settings


@pytest.fixture
def app_with_journey_authed():
    """Mounts the journey router with the auth dependency overridden
    to return a fixed user_id. Mirrors Sprint 10.6's auth pattern."""
    from backend.tier1_copilot.journey.routes import (
        router,
        _lazy_auth_dependency,
        _reset_cache,
    )
    _reset_cache()
    app = FastAPI()
    app.include_router(router)

    async def _fake_user():
        return "user_2qFAKE_CLERK_ID"
    app.dependency_overrides[_lazy_auth_dependency] = _fake_user
    return app


def _stub_engine_returning_row(row):
    """Engine whose execute().mappings().first() returns `row`."""
    fake_result = MagicMock()
    fake_result.first.return_value = row
    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.return_value = fake_result
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm
    return engine


def test_resume_state_returns_stage_0_for_fresh_session(app_with_journey_authed):
    """A new session has no stage_advanced events → fresh-cohort
    default. The endpoint must return current_stage='stage_0' so the
    journey paints exactly the pre-10.7 first-load behaviour."""
    engine = _stub_engine_returning_row(None)  # no event row

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch("backend.db.connection.engine", engine):
        client = TestClient(app_with_journey_authed)
        r = client.get("/tier1/journey/sess_fresh/resume-state")

    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess_fresh"
    assert body["current_stage"] == "stage_0"
    assert body["last_event_at"] is None


def test_resume_state_returns_latest_stage_advanced(app_with_journey_authed):
    """Session has events pivot_insights → stage_2 → stage_3. The DB
    query orders by created_at DESC LIMIT 1, so the response must
    surface stage_3 (the latest advance)."""
    last_at = datetime(2026, 4, 30, 12, 0, 0, tzinfo=timezone.utc)
    engine = _stub_engine_returning_row({
        "stage": "stage_3",
        "created_at": last_at,
    })

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch("backend.db.connection.engine", engine):
        client = TestClient(app_with_journey_authed)
        r = client.get("/tier1/journey/sess_advanced/resume-state")

    assert r.status_code == 200
    body = r.json()
    assert body["current_stage"] == "stage_3"
    assert body["last_event_at"] == last_at.isoformat()


def test_resume_state_ignores_helpful_events(app_with_journey_authed):
    """Engineer who only clicked Helpful (no next-stage advance) must
    resume at Stage 0. Implementation detail: the SQL filter is
    `WHERE event_type = 'stage_advanced'`, so helpful_clicked rows are
    invisible to the query — the engine stub returns no row, mirroring
    that filter at the test boundary."""
    captured_sql = []

    def _capture_execute(stmt, params=None):
        captured_sql.append(str(stmt))
        # Empty result → simulates the WHERE clause filtering out the
        # helpful_clicked row that physically exists in the table.
        result = MagicMock()
        mappings = MagicMock()
        mappings.first.return_value = None
        result.mappings.return_value = mappings
        return result

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = _capture_execute
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch("backend.db.connection.engine", engine):
        client = TestClient(app_with_journey_authed)
        r = client.get("/tier1/journey/sess_helpful_only/resume-state")

    assert r.status_code == 200
    assert r.json()["current_stage"] == "stage_0"

    # Critical assertion — the SQL must filter on stage_advanced
    # exclusively. A future refactor that broadens the filter to
    # `IN (...)` or drops it entirely would fail this test, catching
    # the regression where Helpful clicks would falsely advance the
    # resume pointer.
    sql = " ".join(captured_sql).lower()
    assert "event_type = 'stage_advanced'" in sql, (
        "resume-state must filter on event_type='stage_advanced' so "
        "helpful_clicked / stage_rendered rows don't move the pointer"
    )


def test_resume_state_handles_db_error_gracefully(app_with_journey_authed):
    """Mock the DB to raise — endpoint must return 200 with stage_0,
    not a 500. The journey UI relies on this graceful degrade: if
    the events table is unreachable, paint Stage 0 (the pre-10.7
    behaviour) rather than break the entire journey load."""
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("DB unreachable")

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch("backend.db.connection.engine", engine):
        client = TestClient(app_with_journey_authed)
        r = client.get("/tier1/journey/sess_db_error/resume-state")

    assert r.status_code == 200
    assert r.json()["current_stage"] == "stage_0"
    assert r.json()["last_event_at"] is None


def test_resume_state_requires_auth():
    """Sprint 10.6 pattern — auth-gated endpoint must reject without
    a Bearer token (returns 401, not 500). Keeps per-session
    navigation history private to the authenticated owner."""
    from backend.tier1_copilot.journey.routes import (
        router,
        _lazy_auth_dependency,
        _reset_cache,
    )
    _reset_cache()
    app = FastAPI()
    app.include_router(router)

    async def _reject_unauthenticated():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required.",
        )
    app.dependency_overrides[_lazy_auth_dependency] = _reject_unauthenticated

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True):
        client = TestClient(app)
        r = client.get("/tier1/journey/sess_x/resume-state")

    assert r.status_code == 401
