"""Sprint 10 Stage 1A — smoking_gun aggregator unit tests."""
from __future__ import annotations

from backend.tier1_copilot.journey.stage1_smoking_gun import build_stage1a


def _ticket(inc: str, pivot: str = None, shift: str = None, action: str = None):
    kb = {}
    if pivot or shift:
        kb["the_mental_pivot"] = {}
        if pivot:
            kb["the_mental_pivot"]["pivot_data_point"] = pivot
        if shift:
            kb["the_mental_pivot"]["shift_in_logic"] = shift
    if action:
        kb["diagnostic_logic"] = {"the_pivot_signal": action}
    return {
        "Metadata": {"Incident_Number": inc},
        "Knowledge_Base": kb,
    }


def test_empty_cohort():
    out = build_stage1a([])
    assert out.empty is True
    assert out.pivot_signal is None


def test_threshold_boundary_below_40_percent():
    """N=5, ratio=0.4 → ceil(2.0)=2. Every signal appears in only 1
    of 5 tickets, so none qualifies — empty=True."""
    cohort = [_ticket(f"INC-{i}", pivot=f"signal-{i}") for i in range(5)]
    out = build_stage1a(cohort)
    assert out.empty is True


def test_threshold_boundary_at_40_percent():
    """N=5, ratio=0.4 → only "MTU mismatch" (2 of 5) qualifies; the
    other 3 tickets each have a unique signal so none of THEM clears
    the threshold."""
    cohort = [
        _ticket("INC-1", pivot="MTU mismatch", shift="Skip neighbour checks"),
        _ticket("INC-2", pivot="MTU mismatch", shift="Skip neighbour checks; align MTU on both ends"),
        _ticket("INC-3", pivot="distinct-signal-A"),
        _ticket("INC-4", pivot="distinct-signal-B"),
        _ticket("INC-5", pivot="distinct-signal-C"),
    ]
    out = build_stage1a(cohort)
    assert out.empty is False
    assert out.pivot_signal == "MTU mismatch"
    # Longest verbatim shift_in_logic wins
    assert out.bypass_instruction == "Skip neighbour checks; align MTU on both ends"
    assert out.frequency_in_cohort_percent == 40
    assert out.seen_in_incidents == ["INC-1", "INC-2"]


def test_casing_variance_groups_together():
    """Same pivot in different cases must collapse into one group."""
    cohort = [
        _ticket("INC-1", pivot="MTU Mismatch"),
        _ticket("INC-2", pivot="mtu mismatch"),
        _ticket("INC-3", pivot="MTU MISMATCH"),
    ]
    out = build_stage1a(cohort, min_frequency_ratio=0.4)
    assert out.empty is False
    assert out.frequency_in_cohort_percent == 100
    assert len(out.seen_in_incidents) == 3
    # Canonical = most-frequent verbatim casing — three tied; any single
    # one of the three input strings is acceptable.
    assert out.pivot_signal in {"MTU Mismatch", "mtu mismatch", "MTU MISMATCH"}


def test_empty_knowledge_base():
    """Tickets with no Knowledge_Base section must not raise."""
    cohort = [{"Metadata": {"Incident_Number": "INC-X"}} for _ in range(5)]
    out = build_stage1a(cohort)
    assert out.empty is True


def test_quality_weighted_canonical_pick():
    """Sprint 10.1 — when picking the canonical text from a group,
    sort by Resolution_Quality_Score DESC, then text length DESC.
    A score-5 ticket's pivot description trumps a score-3 ticket's
    even if the score-5 text is shorter."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-LO", "Resolution_Quality_Score": "3"},
            "Knowledge_Base": {
                "the_mental_pivot": {
                    # Score-3 ticket has the LONGER pivot description.
                    "pivot_data_point": "much longer descriptive pivot signal text from low quality ticket",
                    "shift_in_logic": "verbose shift_in_logic from a low quality ticket that should NOT win",
                },
                "diagnostic_logic": {"the_pivot_signal": "low-quality action"},
            },
        },
        {
            "Metadata": {"Incident_Number": "INC-HI", "Resolution_Quality_Score": "5"},
            "Knowledge_Base": {
                "the_mental_pivot": {
                    # Score-5 ticket has a SHORTER pivot, but quality wins.
                    "pivot_data_point": "much longer descriptive pivot signal text",  # same normalized form (lowercased) — needs to match
                    "shift_in_logic": "concise high-quality shift",
                },
                "diagnostic_logic": {"the_pivot_signal": "hi-quality action"},
            },
        },
    ]
    # Same normalized pivot key (whitespace + case identical for both)
    # — they group together; we then quality-weight inside the group.
    cohort[0]["Knowledge_Base"]["the_mental_pivot"]["pivot_data_point"] = "MTU mismatch"
    cohort[1]["Knowledge_Base"]["the_mental_pivot"]["pivot_data_point"] = "MTU mismatch"
    # Score-3 ticket has the longer shift; score-5 has the shorter shift
    # (verbatim, no further mutation). Re-set after group-key fix:
    cohort[0]["Knowledge_Base"]["the_mental_pivot"]["shift_in_logic"] = (
        "verbose shift_in_logic from a low quality ticket that should NOT win"
    )
    cohort[1]["Knowledge_Base"]["the_mental_pivot"]["shift_in_logic"] = (
        "concise high-quality shift"
    )
    cohort[0]["Knowledge_Base"]["diagnostic_logic"]["the_pivot_signal"] = "low-quality action"
    cohort[1]["Knowledge_Base"]["diagnostic_logic"]["the_pivot_signal"] = "hi-quality action"

    out = build_stage1a(cohort, min_frequency_ratio=0.4)
    assert out.empty is False
    assert out.derived_from == "mental_pivot_aggregate"
    # The high-quality ticket's CONCISE shift wins — quality > length.
    assert out.bypass_instruction == "concise high-quality shift"
    # The high-quality ticket's action text wins.
    assert out.recommended_action == "hi-quality action"


def test_primary_fix_fallback_when_kb_empty():
    """Sprint 10.1 — when the cohort has zero
    Knowledge_Base.the_mental_pivot.pivot_data_point entries, distil
    from the highest-quality ticket's Symptom_Solution_Mapping.Primary_Fix
    + Executive_Sharable_RCA.Root_Cause_Technical_High_Level. Mark
    derived_from='primary_fix_fallback'."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-LO", "Resolution_Quality_Score": "3"},
            "Symptom_Solution_Mapping": {"Primary_Fix": "low-quality fix text"},
            "Executive_Sharable_RCA": {
                "Root_Cause_Technical_High_Level": "low-quality root cause",
            },
        },
        {
            "Metadata": {"Incident_Number": "INC-HI", "Resolution_Quality_Score": "5"},
            "Symptom_Solution_Mapping": {"Primary_Fix": "high-quality fix text"},
            "Executive_Sharable_RCA": {
                "Root_Cause_Technical_High_Level": "high-quality root cause",
            },
        },
    ]
    # No Knowledge_Base sections at all — fallback path must trigger.
    out = build_stage1a(cohort, min_frequency_ratio=0.4)
    assert out.empty is False
    assert out.derived_from == "primary_fix_fallback"
    # Highest-score ticket's Primary_Fix → bypass_instruction
    assert out.bypass_instruction == "high-quality fix text"
    # Highest-score ticket's Root_Cause → pivot_signal
    assert out.pivot_signal == "high-quality root cause"
    # Single-ticket distillation — frequency_in_cohort_percent stays 0
    assert out.frequency_in_cohort_percent == 0
    # The high-quality incident is recorded for traceability.
    assert out.seen_in_incidents == ["INC-HI"]


def test_fallback_returns_empty_when_neither_pivot_nor_primary_fix():
    """Edge case: KB is empty AND Primary_Fix is unavailable across the
    cohort → Stage 1A is genuinely empty with derived_from='empty'."""
    cohort = [
        {"Metadata": {"Incident_Number": "INC-1", "Resolution_Quality_Score": "3"}},
        {"Metadata": {"Incident_Number": "INC-2", "Resolution_Quality_Score": "4"}},
    ]
    out = build_stage1a(cohort, min_frequency_ratio=0.4)
    assert out.empty is True
    assert out.derived_from == "empty"


def test_recommended_action_picked_from_pivot_signal_text():
    """the_pivot_signal feeds recommended_action."""
    cohort = [
        _ticket("INC-1", pivot="MTU mismatch", action="set mtu 9000"),
        _ticket("INC-2", pivot="MTU mismatch", action="set mtu 9000"),
        _ticket("INC-3", pivot="MTU mismatch", action="set mtu 1500"),
    ]
    out = build_stage1a(cohort, min_frequency_ratio=0.4)
    assert out.recommended_action == "set mtu 9000"
