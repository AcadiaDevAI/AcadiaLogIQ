"""Sprint 10 Stage 2 — historical_matches builder unit tests."""
from __future__ import annotations

from backend.tier1_copilot.journey.stage2_historical import build_stage2


def _phoenix_full_ticket():
    return {
        "Metadata": {
            "Incident_Number": "INC-PHOENIX-402",
            "outage_cause": "Stale session profile lock",
            "customer_name": "Aetheris Corp",
            "time_to_resolve_minutes": "38",
            "ticket_status": "closed",
        },
        "Incident_Summary": {"INCIDENT": "BGP session flap, edge router"},
        "Executive_Sharable_RCA": {
            "Executive_Summary": "Recurring BGP adjacency flaps after BFD timer change",
            "Technical_Snapshot": "Sub-second BFD timers caused premature peer down on uplink",
            "Root_Cause_Technical_High_Level": "BFD timer below carrier minimum",
            "Resolution_Steps": [
                "Restore BFD min interval to 300ms",
                "Restart BGP process on affected peer",
            ],
        },
        "Forensic_Performance_Audit": {"Critical_Intervention": "Reapply BFD baseline policy"},
        "Key_Contributors": {"Hero_Action": "Engineer caught it from BFD log signature"},
        "Operational_SOP": {"signal_identification": {"human_symptom": "session bouncing every 90s"}},
        "QA_Auditor_Feedback": {"Rework_Detected": False},
    }


def test_full_card_phoenix_402():
    out = build_stage2([_phoenix_full_ticket()])
    assert len(out.matches) == 1
    c = out.matches[0]
    assert c.rank == 1
    assert c.incident_number == "INC-PHOENIX-402"
    assert c.headline == "BGP session flap, edge router"
    assert "Recurring" in c.summary
    # Sprint 11 — technical_snapshot reinstated. Direct file inspection
    # of the four reachable corpus files shows 72/180 populated; the
    # phoenix fixture above carries it.
    assert c.technical_snapshot == "Sub-second BFD timers caused premature peer down on uplink"
    assert c.symptoms == "session bouncing every 90s"
    assert c.root_cause == "Stale session profile lock — BFD timer below carrier minimum"
    assert c.resolution == [
        "Restore BFD min interval to 300ms",
        "Restart BGP process on affected peer",
        "Reapply BFD baseline policy",
        "Engineer caught it from BFD log signature",
    ]
    assert c.customer == "Aetheris Corp"
    assert c.time_to_resolve_minutes == 38
    assert c.closed_without_recurrence is True


def test_missing_fields_omitted_not_n_a():
    """Missing optional fields → None on the model, never the string 'N/A'.

    Sprint 11 — call with filter_empty=False so the resilience check
    survives the new useful-card filter (a card with only
    Incident_Number would be filtered as empty in production)."""
    sparse = {"Metadata": {"Incident_Number": "INC-X"}}
    out = build_stage2([sparse], filter_empty=False)
    c = out.matches[0]
    assert c.incident_number == "INC-X"
    assert c.headline is None
    assert c.summary is None
    assert c.symptoms is None
    assert c.root_cause is None
    assert c.resolution == []
    # And nothing should equal the literal "N/A"
    for attr in ("headline", "summary", "symptoms", "root_cause"):
        assert getattr(c, attr) != "N/A"


def test_technical_snapshot_absent_when_source_lacks_it():
    """Sprint 11 — Technical_Snapshot is populated in 72/180 reachable
    tickets; for the other 108 the field must be None on the card
    (frontend omits the row entirely)."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-NO-SNAPSHOT"},
        "Executive_Sharable_RCA": {
            "Executive_Summary": "summary",
            # No Technical_Snapshot key.
            "Resolution_Steps": ["fix"],
        },
    }
    out = build_stage2([ticket])
    c = out.matches[0]
    assert c.technical_snapshot is None


def test_resolution_string_input_renders_as_single_bullet():
    """Resolution_Steps as a single string (not a list) still produces
    one bullet."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-1"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": "Single fix step",
        },
    }
    out = build_stage2([ticket])
    c = out.matches[0]
    assert c.resolution == ["Single fix step"]


def test_resolution_list_of_dicts_extracts_description():
    """Resolution_Steps may be list-of-dicts with a 'description' key."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-1"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": [
                {"description": "Step one"},
                {"action": "Step two"},
                {"step": "Step three"},
            ],
        },
    }
    out = build_stage2([ticket])
    assert out.matches[0].resolution == ["Step one", "Step two", "Step three"]


def test_root_cause_em_dash_join_only_when_both_present():
    """outage_cause + Root_Cause_Technical_High_Level joined with ' — '."""
    only_outage = {
        "Metadata": {"Incident_Number": "INC-1", "outage_cause": "Lock contention"},
    }
    only_root = {
        "Metadata": {"Incident_Number": "INC-2"},
        "Executive_Sharable_RCA": {"Root_Cause_Technical_High_Level": "BFD too aggressive"},
    }
    both = {
        "Metadata": {"Incident_Number": "INC-3", "outage_cause": "A"},
        "Executive_Sharable_RCA": {"Root_Cause_Technical_High_Level": "B"},
    }
    out = build_stage2([only_outage, only_root, both])
    assert out.matches[0].root_cause == "Lock contention"
    assert out.matches[1].root_cause == "BFD too aggressive"
    assert out.matches[2].root_cause == "A — B"


def test_rank_order_preserved_and_5_card_ordering():
    """Cards numbered 1..N in input order.

    Sprint 11 — filter_empty=False because the fixtures here only
    carry Incident_Number; production filter would drop them. The
    test is asserting rank ordering, not the filter behaviour."""
    cohort = [{"Metadata": {"Incident_Number": f"INC-{i}"}} for i in range(5)]
    out = build_stage2(cohort, filter_empty=False)
    assert len(out.matches) == 5
    for i, c in enumerate(out.matches):
        assert c.rank == i + 1
        assert c.incident_number == f"INC-{i}"


def test_card_pulls_all_five_sections_when_fully_populated():
    """Sprint 11 — every listed JSON path populated → card has
    incident_number + 5 content sections (incident_summary,
    technical_snapshot, symptoms, root_cause, resolution_approach)
    all non-None.

    Sprint 10.8.1 §6 had dropped technical_snapshot citing a 0/27
    populated rate in file1's 27-ticket audit; the broader 180-ticket
    inspection across the four reachable source files found 72/180
    populated, so the field is reinstated."""
    ticket = _phoenix_full_ticket()
    out = build_stage2([ticket])
    c = out.matches[0]
    assert c.incident_number == "INC-PHOENIX-402"
    # All FIVE content sections populate.
    assert c.incident_summary is not None
    assert c.technical_snapshot is not None
    assert c.symptoms is not None
    assert c.root_cause is not None
    assert c.resolution_approach is not None


def test_card_concatenates_root_cause_with_em_dash():
    """Sprint 10.8 §2.7 — outage_cause + Root_Cause_Technical_High_Level
    joined with ' — '."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-RC", "outage_cause": "<outage>"},
        "Executive_Sharable_RCA": {"Root_Cause_Technical_High_Level": "<technical>"},
    }
    out = build_stage2([ticket])
    assert out.matches[0].root_cause == "<outage> — <technical>"


def test_card_concatenates_resolution_three_fields():
    """Sprint 10.8 §2.7 — Resolution_Steps + Critical_Intervention +
    Hero_Action joined em-dash → all three components in
    resolution_approach in order.
    Sprint 10.8.1 — fixture realigned to the verified corpus shape:
    Forensic_Performance_Audit is `[{...}]` and Hero_Action lives at
    Key_Contributors.Key_Impact_Players[0].Hero_Action."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-RES"},
        "Executive_Sharable_RCA": {"Resolution_Steps": "<steps>"},
        "Forensic_Performance_Audit": [{"Critical_Intervention": "<critical>"}],
        "Key_Contributors": {
            "Key_Impact_Players": [{"Hero_Action": "<hero>"}],
        },
    }
    out = build_stage2([ticket])
    assert out.matches[0].resolution_approach == "<steps> — <critical> — <hero>"


def test_card_skips_missing_field_in_concat():
    """Sprint 10.8 §2.7 — only outage_cause set, the technical-side
    field missing → root_cause is just <outage> with no trailing
    em-dash."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-X", "outage_cause": "Lock contention"},
    }
    out = build_stage2([ticket])
    rc = out.matches[0].root_cause
    assert rc == "Lock contention"
    assert " — " not in rc


def test_card_omits_section_when_all_sources_missing():
    """Sprint 10.8 §2.7 — neither outage_cause NOR
    Root_Cause_Technical_High_Level present → root_cause is None
    (frontend hides the row).

    Sprint 11 — filter_empty=False; the fixture is intentionally
    sparse so the production filter would drop it."""
    ticket = {"Metadata": {"Incident_Number": "INC-NONE"}}
    out = build_stage2([ticket], filter_empty=False)
    assert out.matches[0].root_cause is None


def test_card_handles_list_value_for_resolution_steps():
    """Sprint 10.8 §2.3 — Resolution_Steps as a list of strings is
    flattened into one ; -joined value before the em-dash join."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-LIST"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": ["Step alpha", "Step beta", "Step gamma"],
        },
    }
    out = build_stage2([ticket])
    ra = out.matches[0].resolution_approach
    assert ra is not None
    assert "Step alpha" in ra
    assert "Step beta" in ra
    assert "Step gamma" in ra
    # The list joiner is "; " (per _flatten_field).
    assert "Step alpha; Step beta; Step gamma" in ra


def test_card_handles_dict_value_for_human_symptom():
    """Sprint 10.8 §2.3 — human_symptom as a dict with
    {description: '...'} → extracts the description string."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-DICT"},
        "Operational_SOP": {"signal_identification": {
            "human_symptom": {"description": "intermittent UI freeze"},
        }},
    }
    out = build_stage2([ticket])
    assert out.matches[0].symptoms == "intermittent UI freeze"


def test_card_resilient_to_completely_missing_top_level_key():
    """Sprint 10.8 §2.4 — ticket has no Executive_Sharable_RCA at all
    → no exception, all RCA-derived fields are None.

    Sprint 11 — filter_empty=False; the fixture is intentionally
    sparse so the production filter would drop it."""
    ticket = {"Metadata": {"Incident_Number": "INC-BARE"}}
    out = build_stage2([ticket], filter_empty=False)
    c = out.matches[0]
    assert c.incident_summary is None
    assert c.resolution_approach is None
    assert c.root_cause is None


def test_card_extracts_all_four_sections_from_real_ticket():
    """Sprint 10.8.1 §7 — fixture mirrors INC-ALPHA-001's real shape:
    Forensic_Performance_Audit is a list, Key_Contributors holds a
    Key_Impact_Players inner list. All 4 content fields must populate
    with em-dash visible in root_cause and resolution_approach."""
    ticket = {
        "Metadata": {
            "Incident_Number": "INC-ALPHA-001",
            "outage_cause": "Wireless handheld AP roam latency",
        },
        "Incident_Summary": {"INCIDENT": "Handheld disconnect on warehouse floor"},
        "Executive_Sharable_RCA": {
            "Executive_Summary": "Recurring drop during shift handover at AP boundary",
            "Root_Cause_Technical_High_Level": "Roam timer above carrier minimum",
            "Resolution_Steps": ["Lower roam timer", "Re-survey channel plan"],
        },
        "Operational_SOP": {"signal_identification": {"human_symptom": "device disassociates mid-aisle"}},
        "Forensic_Performance_Audit": [{
            "Critical_Intervention": "Push timer profile to controller",
        }],
        "Key_Contributors": {"Key_Impact_Players": [{
            "Hero_Action": "Engineer flagged roam-timer mismatch from controller log",
        }]},
    }
    out = build_stage2([ticket])
    c = out.matches[0]
    assert c.incident_summary is not None
    assert c.symptoms is not None
    assert c.root_cause is not None
    assert c.resolution_approach is not None
    # Em-dash visible in 2-source root_cause and 3-source resolution.
    assert " — " in c.root_cause
    assert c.resolution_approach.count(" — ") == 2


def test_card_traverses_forensic_performance_audit_array():
    """Sprint 10.8.1 §7 — the corpus reality: FPA is `[{...}]`, not
    `{...}`. _safe_get must auto-step into [0] before reading the
    Critical_Intervention key. Sprint 10.8 missed this and silently
    dropped resolution_approach for every cohort."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-FPA"},
        "Forensic_Performance_Audit": [
            {"Critical_Intervention": "X"},
        ],
    }
    out = build_stage2([ticket])
    c = out.matches[0]
    assert c.resolution_approach is not None
    assert "X" in c.resolution_approach


def test_card_traverses_key_impact_players_array():
    """Sprint 10.8.1 §7 — Hero_Action lives at
    Key_Contributors.Key_Impact_Players[0].Hero_Action. The two-level
    array nesting is what Sprint 10.8 missed."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-KIP"},
        "Key_Contributors": {
            "Key_Impact_Players": [{"Hero_Action": "Y"}],
        },
    }
    out = build_stage2([ticket])
    c = out.matches[0]
    assert c.resolution_approach is not None
    assert "Y" in c.resolution_approach


def test_card_resolution_joins_three_sources_with_em_dash():
    """Sprint 10.8.1 §7 — when all three resolution sources populate,
    the output has exactly two em-dash separators."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-3SRC"},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["a"]},
        "Forensic_Performance_Audit": [{"Critical_Intervention": "b"}],
        "Key_Contributors": {"Key_Impact_Players": [{"Hero_Action": "c"}]},
    }
    out = build_stage2([ticket])
    ra = out.matches[0].resolution_approach
    assert ra is not None
    assert ra.count(" — ") == 2


def test_card_skips_missing_resolution_source_no_trailing_separator():
    """Sprint 10.8.1 §7 — only Resolution_Steps populated → no leading
    or trailing ' — '. The other two sources (FPA, hero action) absent
    must not leave separator artifacts."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-1SRC"},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["only step"]},
    }
    out = build_stage2([ticket])
    ra = out.matches[0].resolution_approach
    assert ra == "only step"
    assert not ra.startswith(" — ")
    assert not ra.endswith(" — ")


def test_card_resolution_steps_list_joined_with_semicolons():
    """Sprint 10.8.1 §3.1 / §7 — Resolution_Steps as list-of-str is
    flattened to one display string with '; ' joiners (the
    _flatten contract for list-of-string)."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-LIST"},
        "Executive_Sharable_RCA": {"Resolution_Steps": ["a", "b", "c"]},
    }
    out = build_stage2([ticket])
    ra = out.matches[0].resolution_approach
    assert ra == "a; b; c"


def test_card_handles_completely_empty_ticket_no_crash():
    """Sprint 10.8.1 §7 — ticket is `{}` → no exception, card has
    incident_number == 'UNKNOWN-{rank}', all other fields None.

    Sprint 11 — filter_empty=False; this fixture is the canonical
    case the production filter is designed to drop."""
    out = build_stage2([{}], filter_empty=False)
    c = out.matches[0]
    assert c.incident_number == "UNKNOWN-1"
    assert c.incident_summary is None
    assert c.symptoms is None
    assert c.root_cause is None
    assert c.resolution_approach is None


def test_closed_with_rework_not_clean():
    """Rework_Detected=True → closed_without_recurrence stays False.

    Sprint 11 — filter_empty=False; the test only sets Metadata +
    QA_Auditor_Feedback so the production filter would drop it."""
    ticket = {
        "Metadata": {"Incident_Number": "INC-X", "ticket_status": "closed"},
        "QA_Auditor_Feedback": {"Rework_Detected": True},
    }
    out = build_stage2([ticket], filter_empty=False)
    assert out.matches[0].closed_without_recurrence is False


# ─────────────────────────────────────────────────────────────
# Sprint 11 — Useful-card filter + max_matches_shown contract.
# ─────────────────────────────────────────────────────────────
def _useful_ticket(incident="INC-USEFUL"):
    """Ticket with one body field set so it survives the filter."""
    return {
        "Metadata": {"Incident_Number": incident},
        "Executive_Sharable_RCA": {"Root_Cause_Technical_High_Level": "BFD timer"},
    }


def test_filter_drops_card_with_only_incident_number():
    """The user's reported INC-10000 case (slim chunk from the
    unbackfilled 5th file): only Incident_Number + customer set,
    all body fields empty → card must NOT appear in production."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-10000", "customer_name": "Nebula-Corp"},
        "Executive_Sharable_RCA": {},  # empty, no Root_Cause / Resolution_Steps
    }]
    out = build_stage2(cohort)
    assert out.matches == []
    assert out.total_available_matches == 0


def test_filter_drops_card_with_unknown_incident_number():
    """The MATCH 4 OF 5 — UNKNOWN-4 case (chunk had no
    Metadata.Incident_Number): even if body fields are populated,
    the card is dropped because we have no way to identify the
    ticket — UNKNOWN-N labels were a placeholder, not a real value."""
    cohort = [{
        # No Metadata.Incident_Number → falls back to UNKNOWN-1.
        "Executive_Sharable_RCA": {"Root_Cause_Technical_High_Level": "real RC"},
    }]
    out = build_stage2(cohort)
    assert out.matches == []


def test_filter_drops_card_with_only_trivial_placeholders():
    """Body fields filled with 'Data Not Present in Log' / 'N/A'
    style placeholders count as empty — the placeholder is not a
    real signal."""
    cohort = [{
        "Metadata": {"Incident_Number": "INC-PLACEHOLDER"},
        "Executive_Sharable_RCA": {
            "Root_Cause_Technical_High_Level": "Data Not Present in Log",
            "Executive_Summary": "N/A",
        },
        "Operational_SOP": {"signal_identification": {"human_symptom": "Unknown"}},
    }]
    out = build_stage2(cohort)
    assert out.matches == []


def test_filter_keeps_card_with_real_root_cause():
    cohort = [{
        "Metadata": {"Incident_Number": "INC-REAL"},
        "Executive_Sharable_RCA": {
            "Root_Cause_Technical_High_Level": "Stale BFD lock under aggressive timers",
        },
    }]
    out = build_stage2(cohort)
    assert len(out.matches) == 1
    assert out.matches[0].incident_number == "INC-REAL"


def test_filter_keeps_card_with_real_resolution_steps():
    cohort = [{
        "Metadata": {"Incident_Number": "INC-STEPS"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": ["Restore BFD min interval", "Restart BGP peer"],
        },
    }]
    out = build_stage2(cohort)
    assert len(out.matches) == 1
    # resolution list survives, resolution_approach concatenation works.
    assert "Restore BFD min interval" in (out.matches[0].resolution_approach or "")


def test_filter_re_ranks_after_sibling_drops():
    """After filtering out empty cards, surviving cards renumber 1..N.
    Engineer sees clean 'MATCH 1 OF 2' / 'MATCH 2 OF 2' instead of
    confusing original ranks like 1 and 4."""
    cohort = [
        _useful_ticket("INC-A"),
        {"Metadata": {"Incident_Number": "INC-EMPTY-1"}},   # dropped
        {"Metadata": {"Incident_Number": "INC-EMPTY-2"}},   # dropped
        _useful_ticket("INC-B"),
    ]
    out = build_stage2(cohort)
    assert [c.rank for c in out.matches] == [1, 2]
    assert [c.incident_number for c in out.matches] == ["INC-A", "INC-B"]
    assert out.total_available_matches == 2


def test_max_matches_shown_default_is_5():
    out = build_stage2([_useful_ticket(f"INC-{i}") for i in range(3)])
    assert out.max_matches_shown == 5
    # All 3 useful cards returned (no backend cap); FE caps display.
    assert len(out.matches) == 3
    assert out.total_available_matches == 3


def test_returns_more_than_5_useful_cards_when_cohort_supports_it():
    """Forward-compat: when retrieval expands beyond top-5 candidates,
    build_stage2 returns ALL useful cards. The frontend caps display
    via max_matches_shown and surfaces a 'View more matches' button.
    Today's retrieval still returns 5 max so this case is forward-only."""
    cohort = [_useful_ticket(f"INC-{i}") for i in range(8)]
    out = build_stage2(cohort)
    assert len(out.matches) == 8
    assert out.total_available_matches == 8
    assert out.max_matches_shown == 5  # FE renders 5, hides 3 behind reveal.
