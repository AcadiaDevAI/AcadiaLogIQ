"""Sprint 11 — Per-ticket detail unit tests for Stage 3.

Companion to test_stage3_troubleshooting.py. The consolidated `steps`
list is covered there; this file focuses on `per_ticket_details` —
the raw verbatim per-cohort-ticket breakdown that the frontend renders
as a collapsible accordion below the consolidated playbook.

Coverage:
  - Every JSON path on the spec list extracts correctly when present
  - Missing fields stay at their schema default (None / [])
  - List ordering matches cohort rank order
  - INC-ALPHA-001-shaped fixture (real corpus shape, file1)
  - Forward-compat for Troubleshooting_Ledger.Diagnostic_Tests_Executed
  - Empty cohort → empty per_ticket_details
"""
from __future__ import annotations

from backend.tier1_copilot.journey.stage3_troubleshooting import (
    _build_ticket_detail,
    build_stage3,
)


def _alpha_001_fixture():
    """Mirrors the real INC-ALPHA-001 source shape (file1):
      - Resolution_Steps: list of strings
      - diagnostic_logic_chunks: 1 entry with step_id+action+context
      - Key_Movements_Timeline: 1 entry with Time+Action
      - Critical_Intervention: string
      - Hero_Action: string at Key_Impact_Players[0]
      - Technical_Snapshot: missing (file1 has 0/27 populated)
    """
    return {
        "Metadata": {"Incident_Number": "INC-ALPHA-001"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": [
                "1. Incident detection via customer self-service reporting.",
                "2. Technical assessment of fax line.",
                "3. Confirm carrier-side outage.",
                "4. Close ticket after recurrence check.",
            ],
        },
        "Operational_SOP": {
            "diagnostic_logic_chunks": [
                {
                    "step_id": "Step 1",
                    "action": "Check line status",
                    "context": "Remote testing of circuit",
                    "branching_logic": "If down, proceed to Step 2",
                    "command": "test-line --id <FAX_NUMBER>",
                },
            ],
        },
        "Forensic_Performance_Audit": [
            {
                "Critical_Intervention": "Accurate initial logging of site symptoms.",
                "Key_Movements_Timeline": [
                    {"Time": "09:34", "Action": "Created ticket and logged symptoms"},
                ],
            },
        ],
        "Key_Contributors": {
            "Key_Impact_Players": [
                {"name": "Tech A", "Hero_Action": "Managed the resolution tasks and verified fix."},
            ],
        },
    }


def test_empty_cohort_yields_empty_per_ticket_details():
    out = build_stage3([])
    assert out.per_ticket_details == []


def test_per_ticket_detail_pulls_every_alpha_001_field():
    """End-to-end: real INC-ALPHA-001 shape produces a populated
    detail with every field on the user's spec list."""
    out = build_stage3([_alpha_001_fixture()])
    assert len(out.per_ticket_details) == 1
    d = out.per_ticket_details[0]
    assert d.rank == 1
    assert d.incident_number == "INC-ALPHA-001"
    # Technical_Snapshot is absent in file1 — must omit, not raise.
    assert d.technical_snapshot is None
    assert len(d.resolution_steps) == 4
    assert d.resolution_steps[0].startswith("1.")
    assert len(d.diagnostic_logic) == 1
    dl = d.diagnostic_logic[0]
    assert dl.action == "Check line status"
    assert dl.intent == "Remote testing of circuit"
    assert dl.pivot == "If down, proceed to Step 2"
    assert dl.command == "test-line --id <FAX_NUMBER>"
    assert len(d.timeline) == 1
    assert d.timeline[0].time == "09:34"
    assert d.timeline[0].action == "Created ticket and logged symptoms"
    assert d.critical_intervention == "Accurate initial logging of site symptoms."
    assert d.hero_action == "Managed the resolution tasks and verified fix."
    # Forward-compat: Troubleshooting_Ledger absent → empty list, not None.
    assert d.diagnostic_tests_executed == []


def test_per_ticket_detail_picks_up_technical_snapshot_when_present():
    """file3 / Hypothetical_goldschema shape — Technical_Snapshot
    populated as a numbered narrative string."""
    ticket = _alpha_001_fixture()
    ticket["Executive_Sharable_RCA"]["Technical_Snapshot"] = (
        "1. Latency alarms triggered. 2. Path analysis confirmed asymmetric routing."
    )
    d = _build_ticket_detail(rank=2, ticket=ticket)
    assert d.technical_snapshot.startswith("1. Latency alarms")


def test_per_ticket_detail_walks_diagnostic_logic_key_cascade():
    """Sprint 10.8.1 §2 — `context` / `branching_logic` are canonical;
    `rationale` / `intent` / `pivot` legacy names still resolve."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-LEGACY"},
        "Operational_SOP": {
            "diagnostic_logic_chunks": [
                # Legacy fixture-style keys.
                {
                    "action": "Run diag",
                    "intent": "Verify state",
                    "pivot": "If unhealthy, escalate",
                },
            ],
        },
    }
    d = _build_ticket_detail(rank=1, ticket=ticket)
    assert len(d.diagnostic_logic) == 1
    dl = d.diagnostic_logic[0]
    assert dl.action == "Run diag"
    assert dl.intent == "Verify state"
    assert dl.pivot == "If unhealthy, escalate"


def test_per_ticket_detail_resolution_steps_handle_dict_shape():
    """Resolution_Steps may be list-of-dict in some corpora.
    description / action / step keys all flatten to the same string."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-DICT"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": [
                {"step": 1, "description": "Restart service"},
                {"step": 2, "action": "Verify uptime"},
            ],
        },
    }
    d = _build_ticket_detail(rank=1, ticket=ticket)
    assert d.resolution_steps == ["Restart service", "Verify uptime"]


def test_per_ticket_detail_omits_missing_sections():
    """Sparse ticket (only Metadata) → detail with rank + incident_number,
    every other field at schema default."""
    ticket = {"Metadata": {"Incident_Number": "INC-SPARSE"}}
    d = _build_ticket_detail(rank=1, ticket=ticket)
    assert d.incident_number == "INC-SPARSE"
    assert d.technical_snapshot is None
    assert d.resolution_steps == []
    assert d.diagnostic_logic == []
    assert d.timeline == []
    assert d.critical_intervention is None
    assert d.hero_action is None
    assert d.diagnostic_tests_executed == []


def test_per_ticket_details_ordered_by_cohort_rank():
    """rank field on each detail matches cohort index + 1.

    Sprint 11 — pass filter_empty_details=False because the fixtures
    here only carry Incident_Number; production filter would drop
    them. The test asserts rank ordering, not the filter behaviour."""
    cohort = [
        {"Metadata": {"Incident_Number": f"INC-{i}"}}
        for i in range(1, 4)
    ]
    out = build_stage3(cohort, filter_empty_details=False)
    assert [d.rank for d in out.per_ticket_details] == [1, 2, 3]
    assert [d.incident_number for d in out.per_ticket_details] == [
        "INC-1", "INC-2", "INC-3",
    ]


def test_per_ticket_detail_diagnostic_tests_executed_when_present():
    """Forward-compat — when a future upload populates
    Troubleshooting_Ledger.Diagnostic_Tests_Executed, the field
    auto-lights up. List-of-string and list-of-dict both supported."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-FUTURE"},
        "Troubleshooting_Ledger": {
            "Diagnostic_Tests_Executed": [
                "Ping gateway",
                {"name": "Traceroute"},
                {"description": "Check ARP table"},
            ],
        },
    }
    d = _build_ticket_detail(rank=1, ticket=ticket)
    assert d.diagnostic_tests_executed == [
        "Ping gateway", "Traceroute", "Check ARP table",
    ]


def test_consolidated_steps_unaffected_by_per_ticket_addition():
    """Sprint 11 must NOT change the consolidated `steps` output —
    existing Sprint 10.x callers (analytics, fallback detection)
    keep their contract."""
    out = build_stage3([_alpha_001_fixture()])
    # The consolidated playbook still surfaces from the harvest.
    assert len(out.steps) >= 1
    # Cohort size still set.
    assert out.cohort_size == 1
    # And the per-ticket detail rides alongside, not in place of.
    assert len(out.per_ticket_details) == 1


# ─────────────────────────────────────────────────────────────
# Sprint 11 — Useful-detail filter + max_details_shown contract.
# Mirrors the Stage 2 filter tests added in the same sprint so both
# stages stay in sync on what counts as "useful" enough to render.
# ─────────────────────────────────────────────────────────────
def test_filter_drops_detail_with_only_incident_number():
    """The user's reported INC-10000 case (slim chunk from the
    unbackfilled 5th file): only Incident_Number set, all body
    sections empty → detail must NOT appear in production."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-10000"},
        "Executive_Sharable_RCA": {},
        "Operational_SOP": {},
    }]
    out = build_stage3(cohort)
    assert out.per_ticket_details == []
    assert out.total_available_details == 0


def test_filter_drops_detail_with_unknown_incident_number():
    """A cohort ticket with no Metadata.Incident_Number falls back to
    UNKNOWN-N. Even with body sections populated, the filter drops
    it because we have no way to identify the ticket."""
    cohort = [{
        # No Metadata.Incident_Number → builder falls back to UNKNOWN-1.
        "Executive_Sharable_RCA": {
            "Resolution_Steps": ["real step 1", "real step 2"],
        },
    }]
    out = build_stage3(cohort)
    assert out.per_ticket_details == []


def test_filter_drops_detail_with_only_trivial_placeholders():
    """Body sections filled with placeholders ('Data Not Present in
    Log', 'N/A', etc.) → detail dropped. The placeholder is not a
    real signal."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-PLACEHOLDER"},
        "Executive_Sharable_RCA": {
            "Technical_Snapshot": "Data Not Present in Log",
            "Resolution_Steps": ["N/A", "Unknown"],
        },
        "Forensic_Performance_Audit": [{
            "Critical_Intervention": "TBD",
        }],
    }]
    out = build_stage3(cohort)
    assert out.per_ticket_details == []


def test_filter_keeps_detail_with_real_resolution_steps():
    """Real Resolution_Steps content survives the filter."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-REAL"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": ["Restore BFD min interval to 300ms"],
        },
    }]
    out = build_stage3(cohort)
    assert len(out.per_ticket_details) == 1
    assert out.per_ticket_details[0].incident_number == "INC-REAL"


def test_filter_keeps_detail_with_only_diagnostic_logic():
    """A ticket whose only signal is Operational_SOP.diagnostic_logic_chunks
    still surfaces — the diagnostic section is real signal."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-DIAG"},
        "Operational_SOP": {"diagnostic_logic_chunks": [
            {"action": "Check line", "context": "remote test"},
        ]},
    }]
    out = build_stage3(cohort)
    assert len(out.per_ticket_details) == 1


def test_filter_re_ranks_after_sibling_drops():
    """Empty siblings dropped → survivors renumber 1..N so the
    visible labels stay sequential."""
    rich = {
        "Metadata": {"Incident_Number": "INC-RICH-A"},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["fix it"]},
    }
    rich2 = {
        "Metadata": {"Incident_Number": "INC-RICH-B"},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["fix it harder"]},
    }
    empty = {"Metadata": {"Incident_Number": "INC-EMPTY"}}
    cohort = [rich, empty, empty, rich2]
    out = build_stage3(cohort)
    assert [d.rank for d in out.per_ticket_details] == [1, 2]
    assert [d.incident_number for d in out.per_ticket_details] == [
        "INC-RICH-A", "INC-RICH-B",
    ]


def test_max_details_shown_default_is_5():
    cohort = [
        {
            "Metadata": {"Incident_Number": f"INC-{i}"},
            "Executive_Sharable_RCA": {"Resolution_Steps": ["a step"]},
        }
        for i in range(3)
    ]
    out = build_stage3(cohort)
    assert out.max_details_shown == 5
    # All 3 useful details returned (no backend cap); FE caps display.
    assert len(out.per_ticket_details) == 3
    assert out.total_available_details == 3


def test_returns_more_than_5_useful_details_when_cohort_supports_it():
    """Forward-compat: when retrieval expands beyond top-5, build_stage3
    returns ALL useful details. Frontend caps display via max_details_shown
    and surfaces a 'View more ticket details' button."""
    cohort = [
        {
            "Metadata": {"Incident_Number": f"INC-{i}"},
            "Executive_Sharable_RCA": {"Resolution_Steps": ["a step"]},
        }
        for i in range(7)
    ]
    out = build_stage3(cohort)
    assert len(out.per_ticket_details) == 7
    assert out.total_available_details == 7
    assert out.max_details_shown == 5
