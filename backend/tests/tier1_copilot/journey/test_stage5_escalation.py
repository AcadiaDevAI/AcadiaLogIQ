"""Sprint 10 Stage 5 — escalation wrapper unit tests."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from backend.tier1_copilot.journey import stage5_escalation


def _ts(h: int, m: int, s: int = 0):
    return datetime(2026, 4, 29, h, m, s, tzinfo=timezone.utc)


def _stub_engine(rows):
    fake_result = MagicMock()
    fake_result.all.return_value = rows
    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.return_value = fake_result
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm
    return engine


def test_fetch_traversal_log_empty_for_blank_session():
    assert stage5_escalation.fetch_traversal_log("") == []


def test_fetch_traversal_log_db_error_returns_empty():
    broken = MagicMock()
    broken.connect.side_effect = RuntimeError("no table")
    with patch("backend.db.connection.engine", broken):
        out = stage5_escalation.fetch_traversal_log("sess-1")
    assert out == []


def test_fetch_traversal_log_rolls_up_per_stage():
    """Multiple events for the same stage roll up into one entry; stages
    sort by first-seen timestamp."""
    rows = [
        {"stage": "stage_0", "event_type": "stage_rendered", "created_at": _ts(12, 0)},
        {"stage": "stage_1a", "event_type": "stage_rendered", "created_at": _ts(12, 1)},
        {"stage": "stage_1a", "event_type": "helpful_clicked", "created_at": _ts(12, 2)},
        {"stage": "stage_1a", "event_type": "next_stage_clicked", "created_at": _ts(12, 3)},
        {"stage": "stage_2", "event_type": "stage_rendered", "created_at": _ts(12, 4)},
    ]
    with patch("backend.db.connection.engine", _stub_engine(rows)):
        out = stage5_escalation.fetch_traversal_log("sess-1")
    # Three stages traversed
    assert len(out) == 3
    # Order by first-seen
    assert out[0]["step"].endswith("Confidence Lead")
    assert out[1]["step"].endswith("Smoking Gun")
    assert out[2]["step"].endswith("Historical Matches")
    # Stage 1A had viewed + Helpful + advanced
    s1a = out[1]
    assert "viewed" in s1a["result"]
    assert "marked Helpful" in s1a["result"]
    # Sprint 11 — time format now includes date + UTC zone label
    # plus second-level precision so rapid stage transitions don't
    # collapse into identical minute-level timestamps.
    assert "advanced at 2026-04-29 12:03:00 UTC" in s1a["result"]
    # Stage 2 only viewed
    assert out[2]["result"] == "viewed"


def test_build_package_called_with_merged_what_tried():
    """The wrapper must call Sprint 7 build_package with extra_what_tried
    + traversal-log merged into client_what_tried, plus pass through
    ticket_metadata + alert_payload."""
    rows = [
        {"stage": "stage_0", "event_type": "stage_rendered", "created_at": _ts(12, 0)},
    ]

    sentinel_pkg = MagicMock(name="returned_package")
    captured = {}

    def fake_build_package(**kwargs):
        captured.update(kwargs)
        return sentinel_pkg

    fake_module = MagicMock()
    fake_module.build_package = fake_build_package

    extra = [{"step": "manual probe", "result": "abnormal"}]
    ticket_md = {"Metadata": {"Incident_Number": "INC-X"}}
    alert = {"severity": "P2", "asset_name": "srv-01", "alert_type": "cpu spike"}

    with patch("backend.db.connection.engine", _stub_engine(rows)), \
         patch.dict(
             "sys.modules",
             {"backend.tier1_copilot.diagnostics.escalation_package": fake_module},
         ):
        result = stage5_escalation.build_journey_escalation_package(
            session_id="sess-1",
            ticket_metadata=ticket_md,
            alert_payload=alert,
            session_what_tried=None,
            extra_what_tried=extra,
        )

    assert result is sentinel_pkg
    assert captured["ticket_metadata"] == ticket_md
    assert captured["alert_payload"] == alert
    # client_what_tried = extra (1) + traversal log (1) = 2 entries
    cwt = captured["client_what_tried"]
    assert len(cwt) == 2
    assert cwt[0] == {"step": "manual probe", "result": "abnormal"}
    assert cwt[1]["step"].endswith("Confidence Lead")


def test_blank_traversal_log_still_calls_build_package():
    """No journey events yet (engineer escalated immediately) → still
    delegates to Sprint 7 with the empty traversal merged in."""
    sentinel_pkg = MagicMock()
    captured = {}

    def fake_build_package(**kwargs):
        captured.update(kwargs)
        return sentinel_pkg

    fake_module = MagicMock()
    fake_module.build_package = fake_build_package

    with patch("backend.db.connection.engine", _stub_engine([])), \
         patch.dict(
             "sys.modules",
             {"backend.tier1_copilot.diagnostics.escalation_package": fake_module},
         ):
        result = stage5_escalation.build_journey_escalation_package(
            session_id="sess-1",
            ticket_metadata={},
            alert_payload={"severity": "P3", "asset_name": "x", "alert_type": "y"},
        )
    assert result is sentinel_pkg
    assert captured["client_what_tried"] == []


# ─────────────────────────────────────────────────────────────
# Sprint 11 — KB chat engagement counter + UTC zone in time labels.
# ─────────────────────────────────────────────────────────────
def test_kb_chat_engagement_renders_when_present():
    """Stage 4 with kb_chat_engaged event(s) → traversal log shows
    'opened KB chat (N exchanges)' so Tier-2 sees real engagement."""
    rows = [
        {"stage": "stage_4", "event_type": "stage_rendered", "created_at": _ts(12, 0)},
        {"stage": "stage_4", "event_type": "kb_chat_engaged", "created_at": _ts(12, 1)},
        {"stage": "stage_4", "event_type": "kb_chat_engaged", "created_at": _ts(12, 5)},
        {"stage": "stage_4", "event_type": "next_stage_clicked", "created_at": _ts(12, 9)},
    ]
    with patch("backend.db.connection.engine", _stub_engine(rows)):
        out = stage5_escalation.fetch_traversal_log("sess-1")
    assert len(out) == 1
    s4 = out[0]
    assert "viewed" in s4["result"]
    assert "opened KB chat (2 exchanges)" in s4["result"]
    assert "advanced at 2026-04-29 12:09:00 UTC" in s4["result"]


def test_kb_chat_engagement_singular_for_one_exchange():
    """1 engagement uses singular 'exchange', not 'exchanges'."""
    rows = [
        {"stage": "stage_4", "event_type": "stage_rendered", "created_at": _ts(12, 0)},
        {"stage": "stage_4", "event_type": "kb_chat_engaged", "created_at": _ts(12, 1)},
    ]
    with patch("backend.db.connection.engine", _stub_engine(rows)):
        out = stage5_escalation.fetch_traversal_log("sess-1")
    assert "opened KB chat (1 exchange)" in out[0]["result"]
    assert "exchanges" not in out[0]["result"]


def test_kb_chat_engagement_absent_when_only_viewed():
    """The user's reported case: clicked Open Chat (kb_chat_engaged
    NEVER posted because /ask never returned), then immediately
    escalated. Traversal log should NOT claim the chat was engaged."""
    rows = [
        {"stage": "stage_4", "event_type": "stage_rendered", "created_at": _ts(12, 0)},
        {"stage": "stage_4", "event_type": "next_stage_clicked", "created_at": _ts(12, 1)},
    ]
    with patch("backend.db.connection.engine", _stub_engine(rows)):
        out = stage5_escalation.fetch_traversal_log("sess-1")
    assert "opened KB chat" not in out[0]["result"]
    assert "viewed" in out[0]["result"]
    assert "advanced at 2026-04-29 12:01:00 UTC" in out[0]["result"]


def test_rapid_stage_transitions_render_distinct_seconds():
    """Sprint 11 — engineer who clicks through 4 stages in 14 seconds
    must see 4 distinct second-level timestamps in the escalation
    log. The earlier %H:%M format collapsed all 4 into the same
    minute string, hiding the actual cadence from Tier-2."""
    rows = [
        # pivot_insights → next_stage at 12:24:06
        {"stage": "pivot_insights", "event_type": "stage_rendered",
         "created_at": _ts(12, 24, 0)},
        {"stage": "pivot_insights", "event_type": "next_stage_clicked",
         "created_at": _ts(12, 24, 6)},
        # stage_2 → next_stage at 12:24:10 (4s later)
        {"stage": "stage_2", "event_type": "stage_rendered",
         "created_at": _ts(12, 24, 8)},
        {"stage": "stage_2", "event_type": "next_stage_clicked",
         "created_at": _ts(12, 24, 10)},
        # stage_3 → next_stage at 12:24:15 (5s later)
        {"stage": "stage_3", "event_type": "stage_rendered",
         "created_at": _ts(12, 24, 11)},
        {"stage": "stage_3", "event_type": "next_stage_clicked",
         "created_at": _ts(12, 24, 15)},
        # stage_4 → next_stage at 12:24:20 (5s later)
        {"stage": "stage_4", "event_type": "stage_rendered",
         "created_at": _ts(12, 24, 16)},
        {"stage": "stage_4", "event_type": "next_stage_clicked",
         "created_at": _ts(12, 24, 20)},
    ]
    with patch("backend.db.connection.engine", _stub_engine(rows)):
        out = stage5_escalation.fetch_traversal_log("sess-1")
    assert len(out) == 4
    # Each stage shows the second when its next_stage_clicked fired.
    by_stage = {entry["step"]: entry["result"] for entry in out}
    pivot = next(v for k, v in by_stage.items() if "pivot_insights" in k)
    s2 = next(v for k, v in by_stage.items() if "Historical Matches" in k)
    s3 = next(v for k, v in by_stage.items() if "Troubleshooting Approach" in k)
    s4 = next(v for k, v in by_stage.items() if "Search KB / SOP" in k)
    assert "12:24:06 UTC" in pivot
    assert "12:24:10 UTC" in s2
    assert "12:24:15 UTC" in s3
    assert "12:24:20 UTC" in s4
