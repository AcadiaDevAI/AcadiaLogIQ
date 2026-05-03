"""Sprint 10 Stage 4 — KB handoff unit tests."""
from __future__ import annotations

from backend.tier1_copilot.journey.stage4_kb_handoff import build_stage4


def test_full_handoff_with_root_cause_and_notes():
    out = build_stage4(
        severity="P2",
        asset_name="V-Desktop Environment",
        alert_type="Desktop Slowness",
        notes="Affects only the Aetheris VDI pool.",
        dominant_root_cause="Stale session profile lock on the delivery controller",
    )
    assert "Severity P2" in out.prefilled_message
    assert "Desktop Slowness on V-Desktop Environment" in out.prefilled_message
    assert "Past tickets suggest" in out.prefilled_message
    assert "Stale session profile lock" in out.prefilled_message
    assert "Aetheris VDI pool" in out.prefilled_message
    assert out.allowed_doc_kinds == ["sop", "kb"]


def test_handoff_without_optional_fields():
    """No notes, no dominant root cause → just the alert summary line."""
    out = build_stage4(
        severity="P3",
        asset_name="BGP-EDGE-RTR-01",
        alert_type="link flapping",
    )
    assert out.prefilled_message == "Severity P3 — link flapping on BGP-EDGE-RTR-01."
    assert out.allowed_doc_kinds == ["sop", "kb"]


def test_handoff_truncates_long_root_cause():
    long_rc = "x" * 600
    out = build_stage4(
        severity="P1",
        asset_name="auth-srv",
        alert_type="auth failures",
        dominant_root_cause=long_rc,
    )
    # Root cause line truncated to ≤ 240 chars + ellipsis
    rc_line = next(
        line for line in out.prefilled_message.split("\n")
        if line.startswith("Past tickets suggest")
    )
    assert len(rc_line) <= 280  # "Past tickets suggest " (21) + 240 + "."
    assert rc_line.endswith("….")


def test_handoff_handles_missing_severity():
    """Default severity P3 when not provided."""
    out = build_stage4(severity=None, asset_name="x", alert_type="y")
    assert "Severity P3" in out.prefilled_message


def test_handoff_strips_blank_notes():
    """Empty / whitespace-only notes don't add a trailing line."""
    out = build_stage4(
        severity="P2",
        asset_name="srv-01",
        alert_type="cpu spike",
        notes="   ",
    )
    assert out.prefilled_message == "Severity P2 — cpu spike on srv-01."
