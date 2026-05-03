"""Sprint 10 Stage 3 — troubleshooting approach unit tests."""
from __future__ import annotations

from backend.tier1_copilot.journey.stage3_troubleshooting import build_stage3


def test_empty_cohort():
    out = build_stage3([])
    assert out.steps == []


def test_dedup_case_insensitive_across_tickets():
    """Same action in different cases collapses into one step."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-1"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Check BFD timer", "rationale": "verify aggressive timer"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-2"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "check bfd timer", "rationale": "verify timing"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-3"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "CHECK BFD TIMER"},
            ]},
        },
    ]
    out = build_stage3(cohort)
    assert len(out.steps) == 1
    s = out.steps[0]
    assert s.step_number == 1
    # Canonical = longest verbatim — all three are different lengths;
    # longest is "CHECK BFD TIMER" (15) vs "Check BFD timer" (15) vs
    # "check bfd timer" (15). Tie → first one wins.
    assert s.action.lower() == "check bfd timer"
    assert sorted(s.seen_in_incidents) == ["INC-1", "INC-2", "INC-3"]
    # Intent — first non-empty wins (or join of up to 2 distinct)
    assert s.intent is not None
    assert "verify" in s.intent.lower()


def test_ordinal_based_sequencing():
    """Steps sequence by average original ordinal."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-1"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Step alpha"},   # ordinal 0
                {"action": "Step beta"},    # ordinal 1
                {"action": "Step gamma"},   # ordinal 2
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-2"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Step alpha"},   # ordinal 0
                {"action": "Step gamma"},   # ordinal 1
                {"action": "Step beta"},    # ordinal 2  ← shifted later in this ticket
            ]},
        },
    ]
    out = build_stage3(cohort)
    actions = [s.action for s in out.steps]
    # alpha: avg 0 ; beta: avg (1+2)/2=1.5 ; gamma: avg (2+1)/2=1.5
    assert actions[0] == "Step alpha"
    # beta and gamma both at 1.5 — break by larger incident-count first;
    # they tied so stable sort wins (insertion order).
    assert set(actions[1:3]) == {"Step beta", "Step gamma"}


def test_branching_detection_two_distinct_critical_interventions():
    """Two cohort tickets with different Critical_Intervention strings
    that DON'T collapse → tagged as Primary + Alt A. The more-frequent
    CI wins 'Primary'."""
    cohort = [
        # Three tickets favour CI "restart DDC"
        {
            "Metadata": {"Incident_Number": f"INC-{i}"},
            "Forensic_Performance_Audit": {"Critical_Intervention": "restart DDC service"},
        }
        for i in range(3)
    ] + [
        # One ticket favours CI "purge stale lock"
        {
            "Metadata": {"Incident_Number": "INC-X"},
            "Forensic_Performance_Audit": {"Critical_Intervention": "purge stale profile lock"},
        },
    ]
    out = build_stage3(cohort)
    primary = [s for s in out.steps if s.branch_label == "Primary"]
    alts = [s for s in out.steps if s.branch_label == "Alt A"]
    assert len(primary) == 1
    assert primary[0].action == "restart DDC service"
    assert len(alts) == 1
    assert alts[0].action == "purge stale profile lock"


def test_no_branch_label_when_only_one_ci_or_none():
    """Single Critical_Intervention across the cohort → no branch labels."""
    cohort = [
        {
            "Metadata": {"Incident_Number": f"INC-{i}"},
            "Forensic_Performance_Audit": {"Critical_Intervention": "restart DDC"},
        }
        for i in range(3)
    ]
    out = build_stage3(cohort)
    assert all(s.branch_label is None for s in out.steps)


def test_max_8_cap():
    """More than 8 deduped steps → only the first 8 by ordinal survive."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-1"},
        "Operational_SOP": {"diagnostic_logic_chunks": [
            {"action": f"step number {i:02d}"} for i in range(15)
        ]},
    }]
    out = build_stage3(cohort, max_steps=8)
    assert len(out.steps) == 8
    # Step numbers are sequential 1..8
    assert [s.step_number for s in out.steps] == list(range(1, 9))


def test_resolution_steps_string_falls_through():
    """Resolution_Steps as a single string also produces a step."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-1"},
        "Executive_Sharable_RCA": {"Resolution_Steps": "Restart the service"},
    }]
    out = build_stage3(cohort)
    assert len(out.steps) == 1
    assert out.steps[0].action == "Restart the service"


# Sprint 10.8.1 §6 — `test_harvests_all_seven_source_fields` deleted:
# Technical_Snapshot and Diagnostic_Tests_Executed are 0/27 populated
# in the verified corpus. Five-source coverage replaces it via the
# §7 tests below (test_harvests_diagnostic_logic_chunks_with_*,
# test_harvests_timeline_actions_via_array_step, etc.).


def test_dedup_collapses_normalized_duplicates():
    """Sprint 10.8 §3.4 — case-insensitive, whitespace-collapsed,
    trailing-punctuation-stripped equality merges into one step
    with both incidents in seen_in_incidents."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-A"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Check CPU usage"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-B"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "check  cpu  usage."},
            ]},
        },
    ]
    out = build_stage3(cohort)
    matching = [
        s for s in out.steps if "cpu" in s.action.lower()
    ]
    assert len(matching) == 1
    assert sorted(matching[0].seen_in_incidents) == ["INC-A", "INC-B"]


def test_dedup_picks_longest_display_text():
    """Sprint 10.8 §3.4 — display text is the longest non-normalized
    action when N steps dedup together (preserves the most detail)."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-1"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Restart DDC"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-2"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Restart Citrix Delivery Controller"},
            ]},
        },
    ]
    out = build_stage3(cohort)
    matching = [
        s for s in out.steps
        if "restart" in s.action.lower() and "controller" in s.action.lower()
    ]
    assert len(matching) == 1
    assert matching[0].action == "Restart Citrix Delivery Controller"


def test_intent_pivot_only_from_diagnostic_logic_chunks():
    """Sprint 10.8 §3.7 — the architectural commitment. Steps sourced
    from diagnostic_logic_chunks have Intent + Pivot; every other
    source path leaves both null. No LLM imagination fills gaps."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-1"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {
                    "action": "Inspect BFD config",
                    "rationale": "Validate timer baseline",
                    "branching_logic": "If aggressive → revert",
                },
            ]},
            "Forensic_Performance_Audit": {
                "Critical_Intervention": "Reset session",
            },
        },
    ]
    out = build_stage3(cohort)
    by_action = {s.action.lower(): s for s in out.steps}
    bfd = by_action.get("inspect bfd config")
    reset = by_action.get("reset session")
    assert bfd is not None
    assert bfd.intent == "Validate timer baseline"
    assert bfd.pivot == "If aggressive → revert"
    assert reset is not None
    # Sprint 10.8 §3.7 — Critical_Intervention path must NOT carry
    # Intent/Pivot. No LLM imagination fills the gaps.
    assert reset.intent is None
    assert reset.pivot is None


def test_sequencing_diagnostic_then_timeline_then_intervention():
    """Sprint 10.8 §3.5 — output ledger orders categories
    diagnostic → timeline → intervention regardless of harvest
    insertion order."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-MIX"},
        "Forensic_Performance_Audit": {
            "Key_Movements_Timeline": [{"Action": "tl-step"}],
            "Critical_Intervention": "intv-step",
        },
        "Operational_SOP": {"diagnostic_logic_chunks": [
            {"action": "diag-step"},
        ]},
    }]
    out = build_stage3(cohort)
    cats = [s.category for s in out.steps]
    # diagnostic first, then timeline, then intervention.
    assert cats == ["diagnostic", "timeline", "intervention"], (
        f"sequencing wrong: {cats}"
    )


def test_fallback_marker_when_two_distinct_interventions():
    """Sprint 10.8 §3.6 — two non-overlapping intervention groups →
    second one is is_fallback=True."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-A"},
            "Forensic_Performance_Audit": {
                "Critical_Intervention": "Restart DDC",
            },
        },
        {
            "Metadata": {"Incident_Number": "INC-B"},
            "Forensic_Performance_Audit": {
                "Critical_Intervention": "Clear profile lock",
            },
        },
    ]
    out = build_stage3(cohort)
    interventions = [s for s in out.steps if s.category == "intervention"]
    assert len(interventions) == 2
    assert interventions[0].is_fallback is False
    assert interventions[1].is_fallback is True


def test_no_fallback_when_intervention_dedups():
    """Sprint 10.8 §3.6 — the same intervention across tickets dedups
    into one group → no fallback marker (only one intervention emitted,
    nothing to "fall back" to)."""
    cohort = [
        {
            "Metadata": {"Incident_Number": f"INC-{i}"},
            "Forensic_Performance_Audit": {
                "Critical_Intervention": "Restart DDC",
            },
        }
        for i in range(3)
    ]
    out = build_stage3(cohort)
    interventions = [s for s in out.steps if s.category == "intervention"]
    assert len(interventions) == 1
    assert interventions[0].is_fallback is False


def test_caps_at_eight_steps():
    """Sprint 10.8 §3.10 — 15 unique diagnostic steps → output capped
    at 8, total_unique_steps_before_cap exposes the pre-cap count."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-MANY"},
        "Operational_SOP": {"diagnostic_logic_chunks": [
            {"action": f"step number {i:02d}"} for i in range(15)
        ]},
    }]
    out = build_stage3(cohort, max_steps=8)
    assert len(out.steps) == 8
    assert out.total_unique_steps_before_cap == 15


def test_seen_in_incidents_preserves_rank_order():
    """Sprint 10.8 §3.4 — incidents in seen_in_incidents are listed in
    cohort rank order (the order tickets appeared in the cohort), not
    insertion-into-set order."""
    cohort = [
        {  # rank 0
            "Metadata": {"Incident_Number": "INC-RANK0"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "shared step"},
            ]},
        },
        {  # rank 1 — does NOT have this step
            "Metadata": {"Incident_Number": "INC-RANK1"},
        },
        {  # rank 2 — has the step
            "Metadata": {"Incident_Number": "INC-RANK2"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "shared step"},
            ]},
        },
        {  # rank 3 — does NOT
            "Metadata": {"Incident_Number": "INC-RANK3"},
        },
        {  # rank 4 — has the step
            "Metadata": {"Incident_Number": "INC-RANK4"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "shared step"},
            ]},
        },
    ]
    out = build_stage3(cohort)
    matching = [s for s in out.steps if "shared" in s.action.lower()]
    assert len(matching) == 1
    # Cohort rank order: 0 → 2 → 4 (no 1 or 3 because they didn't
    # contribute the step).
    assert matching[0].seen_in_incidents == [
        "INC-RANK0", "INC-RANK2", "INC-RANK4",
    ]


# Sprint 10.8.1 §6 — `test_handles_dict_shape_for_diagnostic_test_executed`
# deleted: Troubleshooting_Ledger.Diagnostic_Tests_Executed is 0/27
# populated in the verified corpus and was dropped from the harvest
# sources entirely.


def test_completely_empty_cohort_returns_empty_steps_no_crash():
    """Sprint 10.8 §3.10 — every ticket missing all 7 fields → no
    exception, steps=[]."""
    cohort = [
        {"Metadata": {"Incident_Number": f"INC-{i}"}}
        for i in range(5)
    ]
    out = build_stage3(cohort)
    assert out.steps == []
    assert out.cohort_size == 5
    assert out.total_unique_steps_before_cap == 0


def test_harvests_diagnostic_logic_chunks_with_intent_pivot_command():
    """Sprint 10.8.1 §7 — diagnostic_logic_chunks is the only source
    that produces a step with all four populated: action, intent
    (from `context`), pivot (from `branching_logic`), command."""
    from backend.tier1_copilot.journey.stage3_troubleshooting import (
        _harvest_steps,
    )
    ticket = {
        "Metadata": {"Incident_Number": "INC-DLC"},
        "Operational_SOP": {"diagnostic_logic_chunks": [{
            "action": "Check controller mode",
            "context": "Confirm controller is in expected forwarding state",
            "branching_logic": "If standalone, fail over to HA peer",
            "command": "show controllers ha-state",
        }]},
    }
    raws = _harvest_steps(ticket)
    dlc = [r for r in raws if r.source_field == "diagnostic_logic_chunks"]
    assert len(dlc) == 1
    s = dlc[0]
    assert s.action == "Check controller mode"
    assert s.intent == "Confirm controller is in expected forwarding state"
    assert s.pivot == "If standalone, fail over to HA peer"
    assert s.command == "show controllers ha-state"


def test_harvests_timeline_actions_via_array_step():
    """Sprint 10.8.1 §7 — Forensic_Performance_Audit is a list in
    the real corpus. _safe_get auto-steps into [0] before reading
    Key_Movements_Timeline. A Sprint 10.8 ticket with FPA-as-dict
    won't fail (the auto-step is a no-op); a 10.8.1 ticket with
    FPA-as-list must produce timeline steps."""
    from backend.tier1_copilot.journey.stage3_troubleshooting import (
        _harvest_steps,
    )
    ticket = {
        "Metadata": {"Incident_Number": "INC-TL"},
        "Forensic_Performance_Audit": [{
            "Key_Movements_Timeline": [
                {"Time": "10:01", "Action": "First responder paged"},
                {"Time": "10:14", "Action": "Workaround applied"},
            ],
        }],
    }
    raws = _harvest_steps(ticket)
    timeline = [r for r in raws if r.source_field == "Key_Movements_Timeline"]
    assert len(timeline) == 2
    assert timeline[0].action == "First responder paged"
    assert timeline[1].action == "Workaround applied"
    # All timeline steps tagged with the timeline category.
    assert all(r.category == "timeline" for r in timeline)


def test_harvests_resolution_steps_as_intervention_category():
    """Sprint 10.8.1 §7 — Resolution_Steps as a 4-element list yields
    4 raw steps, all tagged category='intervention'."""
    from backend.tier1_copilot.journey.stage3_troubleshooting import (
        _harvest_steps,
    )
    ticket = {
        "Metadata": {"Incident_Number": "INC-RS"},
        "Executive_Sharable_RCA": {"Resolution_Steps": [
            "step 1", "step 2", "step 3", "step 4",
        ]},
    }
    raws = _harvest_steps(ticket)
    rs = [r for r in raws if r.source_field == "Resolution_Steps"]
    assert len(rs) == 4
    assert all(r.category == "intervention" for r in rs)


def test_intent_pivot_only_present_for_diagnostic_logic_chunks_source():
    """Sprint 10.8.1 §7 / §3.7 — Intent + Pivot populated ONLY on
    steps from diagnostic_logic_chunks. Every other source path has
    them as None."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-MIX"},
        "Operational_SOP": {"diagnostic_logic_chunks": [{
            "action": "diag-only-step",
            "context": "the intent",
            "branching_logic": "the pivot",
        }]},
        "Forensic_Performance_Audit": [{
            "Critical_Intervention": "intervention-only-step",
            "Key_Movements_Timeline": [{"Action": "timeline-only-step"}],
        }],
        "Executive_Sharable_RCA": {"Resolution_Steps": ["resolution-only-step"]},
        "Key_Contributors": {"Key_Impact_Players": [{
            "Hero_Action": "hero-only-step",
        }]},
    }]
    out = build_stage3(cohort)
    by_action = {s.action: s for s in out.steps}
    diag = by_action["diag-only-step"]
    assert diag.intent == "the intent"
    assert diag.pivot == "the pivot"
    # Every other surviving step has intent + pivot as None.
    for action_text in (
        "intervention-only-step",
        "timeline-only-step",
        "resolution-only-step",
        "hero-only-step",
    ):
        s = by_action.get(action_text)
        assert s is not None, f"missing step {action_text}"
        assert s.intent is None, f"{action_text} unexpectedly has intent={s.intent!r}"
        assert s.pivot is None, f"{action_text} unexpectedly has pivot={s.pivot!r}"


def test_command_field_only_present_for_diagnostic_logic_chunks_source():
    """Sprint 10.8.1 §7 — `command` field populated ONLY on
    diagnostic_logic_chunks-sourced steps. Mirrors the intent/pivot
    rule for the new CLI display field."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-CMD"},
        "Operational_SOP": {"diagnostic_logic_chunks": [{
            "action": "Check uptime",
            "command": "show version | inc uptime",
        }]},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["a resolution step"]},
        "Forensic_Performance_Audit": [{"Critical_Intervention": "an intervention"}],
    }]
    out = build_stage3(cohort)
    by_action = {s.action: s for s in out.steps}
    diag = by_action["Check uptime"]
    assert diag.command == "show version | inc uptime"
    assert by_action["a resolution step"].command is None
    assert by_action["an intervention"].command is None


def test_dedupe_collapses_normalized_duplicates_across_tickets():
    """Sprint 10.8.1 §7 / Sprint 10.8 §3.4 — case-insensitive,
    whitespace-collapsed, trailing-punctuation-stripped equality
    merges into one step with both tickets in seen_in_incidents."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-A"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "Check line status"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-B"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "check  line status."},
            ]},
        },
    ]
    out = build_stage3(cohort)
    matching = [s for s in out.steps if "line status" in s.action.lower()]
    assert len(matching) == 1
    assert sorted(matching[0].seen_in_incidents) == ["INC-A", "INC-B"]


def test_caps_at_eight_steps_realigned_corpus():
    """Sprint 10.8.1 §7 — 12 unique diagnostic steps in cohort →
    output capped at 8. Renamed from Sprint 10.8's
    `test_caps_at_eight_steps` so both run side-by-side without
    name collision."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-MANY"},
        "Operational_SOP": {"diagnostic_logic_chunks": [
            {"action": f"unique step {i:02d}"} for i in range(12)
        ]},
    }]
    out = build_stage3(cohort, max_steps=8)
    assert len(out.steps) == 8
    assert out.total_unique_steps_before_cap == 12


def test_does_not_crash_when_forensic_performance_audit_missing():
    """Sprint 10.8.1 §7 — ticket without Forensic_Performance_Audit
    must not raise. Just no timeline/critical-intervention steps in
    the output (only DLC + Resolution_Steps + Hero_Action survive)."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-NOFPA"},
        "Operational_SOP": {"diagnostic_logic_chunks": [
            {"action": "DLC-step"},
        ]},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["RS-step"]},
        # NB: no Forensic_Performance_Audit key at all.
    }]
    out = build_stage3(cohort)
    actions = {s.action for s in out.steps}
    assert "DLC-step" in actions
    assert "RS-step" in actions
    # No timeline / Critical_Intervention rows.
    assert all(s.category != "timeline" for s in out.steps)


def test_multiple_field_families_merged():
    """Action that appears in multiple of the surviving field families
    collapses into one group with all incidents tracked.
    Sprint 10.8.1 — fixture updated: Diagnostic_Tests_Executed dropped
    (0/27 populated in real corpus); the cross-source merge is now
    diagnostic_logic_chunks vs. Resolution_Steps with the same
    action text."""
    cohort = [
        {
            "Metadata": {"Incident_Number": "INC-1"},
            "Operational_SOP": {"diagnostic_logic_chunks": [
                {"action": "check link state"},
            ]},
        },
        {
            "Metadata": {"Incident_Number": "INC-2"},
            "Executive_Sharable_RCA": {
                "Resolution_Steps": ["check link state"],
            },
        },
    ]
    out = build_stage3(cohort)
    matching = [s for s in out.steps if "link state" in s.action.lower()]
    assert len(matching) == 1
    assert sorted(matching[0].seen_in_incidents) == ["INC-1", "INC-2"]
