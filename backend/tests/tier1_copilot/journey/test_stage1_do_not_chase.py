"""Sprint 10 Stage 1B — do_not_chase aggregator unit tests."""
from __future__ import annotations

from backend.tier1_copilot.journey.stage1_do_not_chase import build_stage1b


def _ticket_with_kb(inc: str, kb: dict):
    return {"Metadata": {"Incident_Number": inc}, "Knowledge_Base": kb}


def test_empty_cohort():
    out = build_stage1b([])
    assert out.empty is True
    assert out.entries == []


def test_count_of_one_excluded():
    """Sprint 10.4 — when a caller explicitly passes min_count=2,
    single occurrences must still be excluded. Default min_count is 1
    in 10.4, so we pass the threshold explicitly here."""
    cohort = [
        _ticket_with_kb("INC-1", {"false_path_red_herrings": [
            {"misleading_signal": "DNS lookup failure", "rule_out_logic": "DNS resolved fine"},
        ]}),
    ]
    out = build_stage1b(cohort, min_count=2)
    assert out.empty is True


def test_threshold_min_count_is_one():
    """Sprint 10.4 §4.5 — single-ticket red herring surfaces in entries
    at the default min_count=1 (lowered from 2 in 10.4). reason must be
    'populated' since the entry cleared the threshold."""
    cohort = [
        _ticket_with_kb("INC-LONELY", {"false_path_red_herrings": [
            {
                "misleading_signal": "DNS lookup failure",
                "rule_out_logic": "DNS resolved fine in this incident",
            },
        ]}),
    ]
    out = build_stage1b(cohort)
    assert out.empty is False
    assert out.reason == "populated"
    assert len(out.entries) == 1
    assert out.entries[0].misleading_signal == "DNS lookup failure"
    assert out.entries[0].occurrence_count == 1
    assert out.entries[0].seen_in_incidents == ["INC-LONELY"]


def test_dedup_across_all_four_field_families():
    """Same signal coming from elimination_checklist + differential_diagnosis
    + false_path_red_herrings + Diagnostic_Tests_Executed should collapse
    into one entry."""
    cohort = [
        _ticket_with_kb("INC-1", {
            "diagnostic_pathway": {"elimination_checklist": [{"signal": "CPU spike"}]},
        }),
        _ticket_with_kb("INC-2", {
            "diagnostic_logic": {"differential_diagnosis": [
                {"signal": "cpu spike", "rule_out_logic": "CPU stayed below 60%"},
            ]},
        }),
        _ticket_with_kb("INC-3", {
            "false_path_red_herrings": [{"misleading_signal": "CPU SPIKE"}],
        }),
        {
            "Metadata": {"Incident_Number": "INC-4"},
            "Troubleshooting_Ledger": {
                "Diagnostic_Tests_Executed": [
                    {"test": "cpu spike", "outcome": "ruled_out"},
                ],
            },
        },
    ]
    out = build_stage1b(cohort, min_count=2)
    assert out.empty is False
    assert len(out.entries) == 1
    e = out.entries[0]
    assert e.occurrence_count == 4
    assert sorted(e.seen_in_incidents) == ["INC-1", "INC-2", "INC-3", "INC-4"]
    # First non-empty rule_out wins
    assert e.rule_out_logic == "CPU stayed below 60%"


def test_diagnostic_tests_executed_filter_by_outcome():
    """Tests with non-ruled_out outcome must NOT contribute."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-1"},
            "Troubleshooting_Ledger": {"Diagnostic_Tests_Executed": [
                {"test": "memory leak check", "outcome": "abnormal"},  # not ruled out
                {"test": "swap pressure check", "outcome": "normal"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-2"},
            "Troubleshooting_Ledger": {"Diagnostic_Tests_Executed": [
                {"test": "swap pressure check", "outcome": "healthy"},
            ]},
        },
    ]
    out = build_stage1b(cohort, min_count=2)
    assert out.empty is False
    sigs = [e.misleading_signal.lower() for e in out.entries]
    assert "swap pressure check" in sigs
    assert "memory leak check" not in sigs


def test_cap_at_8_entries():
    """When more than 8 distinct signals qualify, cap at 8 by frequency."""
    # Build a cohort where 10 different signals each occur in 2 tickets
    cohort = []
    for i in range(10):
        sig = f"signal-{i:02d}"
        for inc in ("a", "b"):
            cohort.append(_ticket_with_kb(f"INC-{i}-{inc}", {
                "false_path_red_herrings": [{"misleading_signal": sig, "rule_out_logic": "ok"}],
            }))
    out = build_stage1b(cohort, min_count=2, max_entries=8)
    assert len(out.entries) == 8


def test_sort_descending_by_occurrence_count():
    """Higher-frequency signals come first."""
    cohort = [
        _ticket_with_kb(f"INC-A-{i}", {
            "false_path_red_herrings": [{"misleading_signal": "signal-frequent", "rule_out_logic": "x"}],
        }) for i in range(5)
    ] + [
        _ticket_with_kb(f"INC-B-{i}", {
            "false_path_red_herrings": [{"misleading_signal": "signal-rare", "rule_out_logic": "x"}],
        }) for i in range(2)
    ]
    out = build_stage1b(cohort, min_count=2)
    assert len(out.entries) == 2
    assert out.entries[0].misleading_signal == "signal-frequent"
    assert out.entries[0].occurrence_count == 5
    assert out.entries[1].misleading_signal == "signal-rare"


def test_reason_field_no_data():
    """Sprint 10.1 — when the cohort has zero false_path_red_herrings,
    elimination_checklist, differential_diagnosis, AND zero
    Diagnostic_Tests_Executed entries with ruled-out outcomes, Stage 1B
    must report empty=True with reason='no_data' (not 'no_recurring' —
    no entries existed at all to deduplicate)."""
    cohort = [
        {"Metadata": {"Incident_Number": f"INC-{i}"}}
        for i in range(5)
    ]
    out = build_stage1b(cohort)
    assert out.empty is True
    assert out.reason == "no_data"
    assert out.entries == []


def test_reason_field_no_recurring_when_each_ticket_unique():
    """Sprint 10.1 — when every ticket explored a distinct false path
    (no signal repeats across tickets), reason='no_recurring' so the
    frontend can show the right empty-state copy."""
    cohort = [
        _ticket_with_kb(f"INC-{i}", {
            "false_path_red_herrings": [
                {"misleading_signal": f"unique-signal-{i}", "rule_out_logic": "ok"},
            ],
        })
        for i in range(5)
    ]
    out = build_stage1b(cohort, min_count=2)
    assert out.empty is True
    assert out.reason == "no_recurring"


def test_reason_field_populated_when_entries_qualify():
    """Sprint 10.1 — when entries clear the threshold,
    reason='populated' alongside empty=False."""
    cohort = [
        _ticket_with_kb(f"INC-{i}", {
            "false_path_red_herrings": [
                {"misleading_signal": "shared-signal", "rule_out_logic": "ok"},
            ],
        })
        for i in range(3)
    ]
    out = build_stage1b(cohort, min_count=2)
    assert out.empty is False
    assert out.reason == "populated"
    assert len(out.entries) == 1


def test_default_rule_out_when_all_blank():
    """Fallback copy fires when no rule_out_logic was ever populated."""
    cohort = [
        _ticket_with_kb("INC-1", {"false_path_red_herrings": [{"misleading_signal": "fan-speed alarm"}]}),
        _ticket_with_kb("INC-2", {"false_path_red_herrings": [{"misleading_signal": "fan-speed alarm"}]}),
    ]
    out = build_stage1b(cohort, min_count=2)
    assert len(out.entries) == 1
    assert "Checked and found healthy" in out.entries[0].rule_out_logic
