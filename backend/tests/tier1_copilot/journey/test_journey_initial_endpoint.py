"""Sprint 10 — /initial endpoint smoke test.

Boots a FastAPI TestClient with the journey flag on, mocks
`load_cohort_metadata` and the Stage 0 SQL aggregate, then asserts:
  - /initial returns 200 with all three initial-payload stages populated
  - flag-off → 404
  - POST /event writes a telemetry row (DB mocked)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings


@pytest.fixture
def app_with_journey():
    """Build a fresh FastAPI app with only the journey router mounted.
    Avoids booting the full backend (which would need DB / Bedrock)."""
    from backend.tier1_copilot.journey.routes import router, _reset_cache
    _reset_cache()
    app = FastAPI()
    app.include_router(router)
    return app


def _phoenix_cohort():
    """Five DISTINCT cohort tickets that all share the same pivot signal.
    Distinct incident numbers matter — Stage 1A counts unique incidents
    per group when checking the 40% threshold, not raw occurrences."""
    return [
        {
            "Metadata": {
                "Incident_Number": f"INC-PHOENIX-{402 + i}",
                "component_category": "router",
                "Target_Service": f"BGP-EDGE-RTR-{i:02d}",
            },
            "Symptom_Solution_Mapping": {"Detected_Symptom": "BGP session flap"},
            "Knowledge_Base": {
                "the_mental_pivot": {
                    "pivot_data_point": "BFD timer below carrier minimum",
                    "shift_in_logic": "Skip neighbour check; restore BFD min interval",
                },
                "diagnostic_logic": {"the_pivot_signal": "set bfd-interval 300"},
            },
        }
        for i in range(5)
    ]


def _stub_engine(corpus_row):
    """Sprint 10.2 — Stage 0 makes a single narrow corpus aggregate
    call; the cohort-distillation work happens in Python from the
    mocked load_cohort_metadata payload."""
    fake_result = MagicMock()
    fake_result.first.return_value = corpus_row
    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.return_value = fake_result
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm
    return engine


def test_initial_flag_off_returns_404(app_with_journey):
    """LOGIQ_TIER1_JOURNEY_BACKEND=False → 404."""
    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", False):
        client = TestClient(app_with_journey)
        r = client.get("/tier1/journey/sess-X/initial")
    assert r.status_code == 404
    assert r.json()["detail"] == "tier1_journey_flag_off"


def test_initial_flag_on_returns_merged_pivot_insights(app_with_journey):
    """Sprint 10.2 — flag on, mocked cohort → JourneyInitial with the
    new shape: stage_0 (Best-Ticket Distillation) + pivot_insights
    wrapper containing both smoking_gun and do_not_chase. Old top-
    level stage_1a / stage_1b keys must NOT appear in the response."""
    cohort = _phoenix_cohort()
    # Add Resolution_Quality_Score + Primary_Fix to the first ticket so
    # Stage 0's distillation has something to feed the headline.
    cohort[0]["Metadata"]["Resolution_Quality_Score"] = "5"
    cohort[0]["Metadata"]["time_to_resolve_minutes"] = "42"
    cohort[0]["Symptom_Solution_Mapping"]["Primary_Fix"] = "Restore BFD min interval to 300ms"
    cohort[0]["Executive_Sharable_RCA"] = {
        "Resolution_Steps": ["Verify BFD config", "Apply 300ms baseline"],
    }

    corpus_row = {"corpus_size": 918, "platform_median_minutes": 134.0}

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch(
             "backend.tier1_copilot.journey.routes.load_cohort_metadata",
             return_value=cohort,
         ), \
         patch(
             "backend.db.connection.engine",
             _stub_engine(corpus_row),
         ):
        client = TestClient(app_with_journey)
        r = client.get("/tier1/journey/sess-PHOENIX/initial")

    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-PHOENIX"

    # Sprint 10.2 — old top-level keys must be ABSENT
    assert "stage_1a" not in body
    assert "stage_1b" not in body

    # Stage 0 — best-ticket distillation
    s0 = body["stage_0"]
    assert s0["sparse"] is False
    assert s0["cohort_size"] == 5
    assert s0["best_incident"] == "INC-PHOENIX-402"
    assert s0["best_quality_score"] == 5
    assert s0["best_time_minutes"] == 42
    assert "BFD" in s0["what_worked"]
    assert s0["corpus_size"] == 918
    assert s0["platform_median_minutes"] == 134

    # Sprint 10.2 — pivot_insights wraps both panels
    pi = body["pivot_insights"]
    assert "smoking_gun" in pi
    assert "do_not_chase" in pi
    # Smoking Gun — 5 of 5 cohort tickets share the pivot
    assert pi["smoking_gun"]["empty"] is False
    assert "BFD" in pi["smoking_gun"]["pivot_signal"]
    # Do Not Chase — no false-path data in this fixture
    assert pi["do_not_chase"]["empty"] is True
    assert pi["do_not_chase"]["entries"] == []


def test_journey_initial_after_cache_hit_returns_cohort(app_with_journey):
    """Sprint 10.3 §3.4 #4 — full path regression: when the analyze
    handler's cache-hit branch has written cohort_ids onto the new
    session row (via the §3.3.3 fix), the journey's /initial endpoint
    must see those IDs and render a populated cohort.

    This test simulates the post-cache-hit state: the session row has
    top_5_match_ids populated, so load_cohort_metadata returns the 5-
    ticket Phoenix cohort. The Stage 1A panel must render with
    empty=False and cohort=5 — exactly what the bug was suppressing
    pre-10.3."""
    cohort = _phoenix_cohort()
    # Add Resolution_Quality_Score + Primary_Fix to the rank-0 ticket so
    # Stage 0's distillation has something to feed the headline.
    cohort[0]["Metadata"]["Resolution_Quality_Score"] = "5"
    cohort[0]["Symptom_Solution_Mapping"]["Primary_Fix"] = "Restore BFD"
    corpus_row = {"corpus_size": 100, "platform_median_minutes": 50.0}

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch(
             "backend.tier1_copilot.journey.routes.load_cohort_metadata",
             return_value=cohort,
         ), \
         patch(
             "backend.db.connection.engine",
             _stub_engine(corpus_row),
         ):
        client = TestClient(app_with_journey)
        r = client.get("/tier1/journey/sess_cache_hit/initial")

    assert r.status_code == 200
    body = r.json()
    # The post-cache-hit cohort is fully visible: 5 tickets, Stage 1A
    # populated, Stage 0 cohort_size matches.
    assert body["stage_0"]["cohort_size"] == 5
    pi = body["pivot_insights"]
    assert pi["smoking_gun"]["empty"] is False, (
        "Cache-hit cohort regression: Stage 1A should render with "
        "the 5-ticket Phoenix cohort. If empty=True here, the "
        "Sprint 10.3 §3.3.3 fix didn't land and the journey is "
        "seeing top_5=0 on cache hits."
    )


def test_event_post_writes_telemetry_row(app_with_journey):
    """POST /event with valid stage + event_type → ok=true."""
    fake_conn = MagicMock()
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.begin.return_value = cm

    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch("backend.db.connection.engine", engine):
        client = TestClient(app_with_journey)
        r = client.post(
            "/tier1/journey/sess-1/event",
            json={"stage": "stage_1a", "event_type": "helpful_clicked"},
        )

    assert r.status_code == 200
    assert r.json() == {"ok": True}
    # Confirm INSERT was actually attempted on the mocked engine
    fake_conn.execute.assert_called_once()


def test_event_post_rejects_unknown_stage(app_with_journey):
    """Unknown stage → 422 from Pydantic Literal validation."""
    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True):
        client = TestClient(app_with_journey)
        r = client.post(
            "/tier1/journey/sess-1/event",
            json={"stage": "stage_99", "event_type": "helpful_clicked"},
        )
    assert r.status_code == 422


def test_event_flag_off_returns_404(app_with_journey):
    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", False):
        client = TestClient(app_with_journey)
        r = client.post(
            "/tier1/journey/sess-1/event",
            json={"stage": "stage_1a", "event_type": "helpful_clicked"},
        )
    assert r.status_code == 404
