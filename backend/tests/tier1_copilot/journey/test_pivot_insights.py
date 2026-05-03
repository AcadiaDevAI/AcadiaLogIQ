"""Sprint 10.2 — Pivot Insights endpoint test.

Verifies the JourneyInitial response shape change: the old top-level
stage_1a + stage_1b fields are gone; both panels live nested under
pivot_insights. Also exercises the standalone /pivot-insights endpoint.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.config import settings


@pytest.fixture
def app_with_journey():
    from backend.tier1_copilot.journey.routes import _reset_cache, router
    _reset_cache()
    app = FastAPI()
    app.include_router(router)
    return app


def _cohort_with_kb():
    """Five distinct cohort tickets sharing the same pivot signal."""
    return [
        {
            "Metadata": {
                "Incident_Number": f"INC-{402 + i}",
                "Resolution_Quality_Score": "5",
                "component_category": "router",
            },
            "Knowledge_Base": {
                "the_mental_pivot": {
                    "pivot_data_point": "BFD timer below carrier minimum",
                    "shift_in_logic": "Restore BFD min interval to 300ms",
                },
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "Restore BFD baseline"},
        }
        for i in range(5)
    ]


def _stub_corpus_engine():
    fake_result = MagicMock()
    fake_result.first.return_value = {"corpus_size": 918, "platform_median_minutes": 134.0}
    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.return_value = fake_result
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm
    return engine


def test_initial_endpoint_returns_merged_pivot_insights(app_with_journey):
    """Sprint 10.2 §7 — JourneyInitial.pivot_insights.smoking_gun and
    .do_not_chase both populate; old top-level stage_1a / stage_1b
    keys are ABSENT (the schema swap from Sprint 10 → 10.2 is complete)."""
    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch(
             "backend.tier1_copilot.journey.routes.load_cohort_metadata",
             return_value=_cohort_with_kb(),
         ), \
         patch("backend.db.connection.engine", _stub_corpus_engine()):
        client = TestClient(app_with_journey)
        r = client.get("/tier1/journey/sess-X/initial")

    assert r.status_code == 200
    body = r.json()

    # ── Old top-level stage_1a / stage_1b must NOT appear ──
    assert "stage_1a" not in body
    assert "stage_1b" not in body

    # ── pivot_insights wrapper present with both children ──
    assert "pivot_insights" in body
    pi = body["pivot_insights"]
    assert "smoking_gun" in pi
    assert "do_not_chase" in pi

    # ── Smoking Gun populated (5 of 5 share the pivot) ──
    assert pi["smoking_gun"]["empty"] is False
    assert "BFD" in pi["smoking_gun"]["pivot_signal"]

    # ── Do Not Chase well-formed (empty in this fixture) ──
    assert pi["do_not_chase"]["empty"] is True
    assert pi["do_not_chase"]["entries"] == []


def test_pivot_insights_standalone_endpoint(app_with_journey):
    """Sprint 10.2 — the new /pivot-insights endpoint exposes the same
    merged content for callers that want just the Stage 1 panel."""
    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", True), \
         patch(
             "backend.tier1_copilot.journey.routes.load_cohort_metadata",
             return_value=_cohort_with_kb(),
         ), \
         patch("backend.db.connection.engine", _stub_corpus_engine()):
        client = TestClient(app_with_journey)
        r = client.get("/tier1/journey/sess-X/pivot-insights")

    assert r.status_code == 200
    body = r.json()
    assert "smoking_gun" in body
    assert "do_not_chase" in body
    assert body["smoking_gun"]["empty"] is False


def test_pivot_insights_endpoint_flag_off_returns_404(app_with_journey):
    with patch.object(settings, "LOGIQ_TIER1_JOURNEY_BACKEND", False):
        client = TestClient(app_with_journey)
        r = client.get("/tier1/journey/sess-X/pivot-insights")
    assert r.status_code == 404
