"""Sprint 10 Stage 0 — Best-Ticket Distillation unit tests.

Sprint 10.2 redesign: Stage 0 distils the highest-quality cohort
ticket's actual resolution. Corpus stats come from a separate narrow
SQL aggregate (no nullable filter params → no parameter-binding bug
surface). Tests stub `engine.connect()` so we never hit a real DB.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.tier1_copilot.journey.stage0_confidence import (
    _derive_profile,
    compute_stage0,
)


# ─────────────────────────────────────────────────────────────
# Stub engine helpers
# ─────────────────────────────────────────────────────────────
def _stub_corpus_engine(corpus_size: int, platform_median: float | None):
    """Engine whose single execute()→mappings()→first() returns the
    narrow corpus aggregate row. No nullable filter params in the
    SQL, so this is the only DB call Stage 0 makes."""
    fake_result = MagicMock()
    fake_result.first.return_value = {
        "corpus_size": corpus_size,
        "platform_median_minutes": platform_median,
    }
    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.return_value = fake_result
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False
    engine = MagicMock()
    engine.connect.return_value = cm
    return engine, fake_conn


def _broken_engine():
    engine = MagicMock()
    engine.connect.side_effect = RuntimeError("connection refused")
    return engine


# ─────────────────────────────────────────────────────────────
# Profile-signature tests (unchanged from prior sprint)
# ─────────────────────────────────────────────────────────────
def test_derive_profile_modal_extraction():
    """Profile builder picks the modal value of each field."""
    cohort = [
        {"Metadata": {"component_category": "router", "Target_Service": "BGP-EDGE-RTR-01"},
         "Symptom_Solution_Mapping": {"Detected_Symptom": "BGP flap"}},
        {"Metadata": {"component_category": "router", "Target_Service": "BGP-EDGE-RTR-02"},
         "Symptom_Solution_Mapping": {"Detected_Symptom": "BGP flap"}},
        {"Metadata": {"component_category": "switch", "Target_Service": "AUTH-SRV-01"},
         "Symptom_Solution_Mapping": {"Detected_Symptom": "auth timeout"}},
    ]
    component, family, symptom, sig = _derive_profile(cohort)
    assert component == "router"
    assert symptom == "BGP flap"
    assert family is not None
    assert "BGP flap" in sig
    assert "router" in sig


# ─────────────────────────────────────────────────────────────
# Sparse path: empty cohort → sparse=True (Sprint 10.2 §3.2(5))
# ─────────────────────────────────────────────────────────────
def test_empty_cohort_returns_sparse():
    """Empty cohort → sparse=True with profile_match populated; no
    SQL fired."""
    out = compute_stage0([])
    assert out.sparse is True
    assert out.cohort_size == 0
    assert out.best_incident is None
    assert out.what_worked is None
    assert out.how_they_did_it == []


def test_db_error_does_not_break_distillation():
    """Sprint 10.2 — corpus aggregate SQL exception leaves corpus_size=0
    and platform_median_minutes=None, but the cohort-distilled fields
    still populate."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-PHOENIX-402",
            "Resolution_Quality_Score": "5",
            "time_to_resolve_minutes": "42",
        },
        "Symptom_Solution_Mapping": {"Primary_Fix": "Restart DDC; purge stale lock"},
        "Executive_Sharable_RCA": {
            "Resolution_Steps": ["Verified DDC health", "Restarted DDC", "Purged lock"],
        },
    }]
    with patch("backend.db.connection.engine", _broken_engine()):
        out = compute_stage0(cohort)

    # Cohort-side fields still populate
    assert out.sparse is False
    assert out.best_incident == "INC-PHOENIX-402"
    assert out.best_quality_score == 5
    assert out.what_worked == "Restart DDC; purge stale lock"
    assert len(out.how_they_did_it) == 3
    # Corpus-side fields default to 0/None
    assert out.corpus_size == 0
    assert out.platform_median_minutes is None


# ─────────────────────────────────────────────────────────────
# Sprint 10.2 §7 — required new tests
# ─────────────────────────────────────────────────────────────
def test_aggregate_binds_params_as_single_dict():
    """Sprint 10.2 — the corpus aggregate must be invoked with a
    SINGLE dict as the second arg to conn.execute(), NOT a list-of-
    dicts. Passing [dict] would make SQLAlchemy interpret it as
    executemany() and trip on missing-param errors. Capture the
    execute() call args and assert the second arg is a dict."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-001",
            "Resolution_Quality_Score": "4",
        },
    }]
    engine, fake_conn = _stub_corpus_engine(corpus_size=918, platform_median=134.0)

    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.cohort_size == 1
    assert out.corpus_size == 918
    # Confirm exec was called and the params arg is a dict (not a list)
    call_args = fake_conn.execute.call_args
    assert call_args is not None
    params_arg = call_args.args[1]
    assert isinstance(params_arg, dict), (
        "Stage 0 must pass params as a single dict, not a list-of-dicts; "
        "list-of-dicts triggers SQLAlchemy executemany() and breaks the "
        "single-row fetch."
    )


def test_distillation_picks_highest_quality_ticket():
    """Sprint 10.2 — sort cohort by Resolution_Quality_Score DESC,
    then by recency DESC as tiebreaker. The winner becomes the
    best_ticket whose fields populate the headline."""
    cohort = [
        {  # rank 0 in input order, but score 3 — should NOT win
            "Metadata": {
                "Incident_Number": "INC-LOW",
                "Resolution_Quality_Score": "3",
                "Created_At": "2026-01-01T00:00:00Z",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "low-quality fix"},
        },
        {  # score 5 + later date — should WIN
            "Metadata": {
                "Incident_Number": "INC-HIGH",
                "Resolution_Quality_Score": "5",
                "Created_At": "2026-04-01T00:00:00Z",
                "time_to_resolve_minutes": "42",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "high-quality fix wins"},
            "Executive_Sharable_RCA": {
                "Resolution_Steps": ["step a", "step b"],
            },
        },
        {  # score 5 + earlier date — should LOSE on recency tiebreak
            "Metadata": {
                "Incident_Number": "INC-OLDER-5",
                "Resolution_Quality_Score": "5",
                "Created_At": "2025-12-01T00:00:00Z",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "older fix should not win"},
        },
    ]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.best_incident == "INC-HIGH"
    assert out.best_quality_score == 5
    assert out.best_time_minutes == 42
    assert out.what_worked == "high-quality fix wins"


def test_distillation_extracts_resolution_steps():
    """Sprint 10.2 — Resolution_Steps populates how_they_did_it,
    capped at 5 entries. Handles list-of-strings + list-of-dicts +
    single-string shapes (the existing Stage 2 normalizer pattern)."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-LISTY",
            "Resolution_Quality_Score": "5",
        },
        "Executive_Sharable_RCA": {
            "Resolution_Steps": [
                "step 1",                              # plain string
                {"description": "step 2 with dict"},   # dict with description
                {"action": "step 3 with action key"},  # dict with action
                {"step": "step 4 with step key"},      # dict with step
                "step 5",
                "step 6 SHOULD NOT APPEAR",            # over the cap
            ],
        },
    }]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert len(out.how_they_did_it) == 5
    assert out.how_they_did_it == [
        "step 1",
        "step 2 with dict",
        "step 3 with action key",
        "step 4 with step key",
        "step 5",
    ]
    assert "SHOULD NOT APPEAR" not in " ".join(out.how_they_did_it)


def test_corpus_stats_resilient_to_query_failure():
    """Sprint 10.2 — when the corpus aggregate SQL fails, corpus_size
    and platform_median_minutes stay at their zero/None defaults and
    the rest of the payload still populates from cohort."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-X",
            "Resolution_Quality_Score": "5",
        },
        "Symptom_Solution_Mapping": {"Primary_Fix": "the fix"},
    }]
    with patch("backend.db.connection.engine", _broken_engine()):
        out = compute_stage0(cohort)

    # Cohort-distilled fields populate
    assert out.best_incident == "INC-X"
    assert out.what_worked == "the fix"
    # Corpus-side fields default
    assert out.corpus_size == 0
    assert out.platform_median_minutes is None
    # Output is NOT marked sparse — sparse means cohort empty, not corpus failure
    assert out.sparse is False


# ─────────────────────────────────────────────────────────────
# Cohort stats tail (collapsed-by-default frontend section)
# ─────────────────────────────────────────────────────────────
def test_cohort_stats_clean_resolution_count():
    """clean_resolution_count tallies cohort tickets with score >= 4;
    clean_resolution_percent is rounded."""
    cohort = [
        {"Metadata": {"Incident_Number": f"INC-{i}", "Resolution_Quality_Score": str(s)}}
        for i, s in enumerate([5, 4, 3, 4, 2])
    ]
    # Add a Primary_Fix on the highest-score ticket so the headline
    # populates (otherwise this test conflates two concerns)
    cohort[0]["Symptom_Solution_Mapping"] = {"Primary_Fix": "fixed"}

    engine, _ = _stub_corpus_engine(corpus_size=918, platform_median=134.0)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.cohort_size == 5
    assert out.clean_resolution_count == 3       # scores 5, 4, 4
    assert out.clean_resolution_percent == 60    # 3 of 5 → 60%


def test_cohort_avg_minutes_excludes_unparseable():
    """avg_minutes_to_resolve_cohort averages only cohort tickets with
    parseable numeric time_to_resolve_minutes. Returns None when none
    have a numeric value."""
    cohort = [
        {"Metadata": {
            "Incident_Number": "INC-A",
            "Resolution_Quality_Score": "5",
            "time_to_resolve_minutes": "30",
        }},
        {"Metadata": {
            "Incident_Number": "INC-B",
            "Resolution_Quality_Score": "5",
            "time_to_resolve_minutes": "60",
        }},
        {"Metadata": {
            "Incident_Number": "INC-C",
            "Resolution_Quality_Score": "4",
            "time_to_resolve_minutes": "N/A",      # unparseable
        }},
    ]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.avg_minutes_to_resolve_cohort == 45  # mean of 30 + 60


def test_picker_breaks_score_ties_by_rank():
    """Sprint 10.3 §4.2 — when every cohort ticket shares the same
    Resolution_Quality_Score, the picker must prefer the rank-0 ticket
    (the cohort's first element). The cohort is already in retrieval-
    rank order from ticket_loader, and Python's stable sort on
    (-score, idx) preserves that ordering on score ties.

    Regression: Sprint 10.2 picked INC-ALPHA-028 over INC-ALPHA-027
    when both scored 3, because the recency-key fallback to
    incident-number lex order broke the tie wrong."""
    cohort = [
        {  # rank 0 — should WIN on tiebreak
            "Metadata": {
                "Incident_Number": "INC-ALPHA-027",
                "Resolution_Quality_Score": "3",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "rank-0 fix"},
        },
        {  # rank 1 — same score, lower rank
            "Metadata": {
                "Incident_Number": "INC-ALPHA-028",
                "Resolution_Quality_Score": "3",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "rank-1 fix"},
        },
        {  # rank 2 — same score, lex-smallest incident number
            "Metadata": {
                "Incident_Number": "INC-ALPHA-001",
                "Resolution_Quality_Score": "3",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "rank-2 fix"},
        },
    ]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.best_incident == "INC-ALPHA-027"
    assert out.best_quality_score == 3
    assert out.what_worked == "rank-0 fix"


def test_picker_picks_highest_score_when_distinct():
    """Sprint 10.3 §4.2 — when scores are distinct, the highest-score
    ticket wins regardless of rank. Scores [3, 5, 4] → the rank-1
    score-5 ticket wins."""
    cohort = [
        {  # rank 0, score 3
            "Metadata": {
                "Incident_Number": "INC-RANK-0",
                "Resolution_Quality_Score": "3",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "score-3 fix"},
        },
        {  # rank 1, score 5 — should WIN
            "Metadata": {
                "Incident_Number": "INC-RANK-1",
                "Resolution_Quality_Score": "5",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "score-5 fix wins"},
        },
        {  # rank 2, score 4
            "Metadata": {
                "Incident_Number": "INC-RANK-2",
                "Resolution_Quality_Score": "4",
            },
            "Symptom_Solution_Mapping": {"Primary_Fix": "score-4 fix"},
        },
    ]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.best_incident == "INC-RANK-1"
    assert out.best_quality_score == 5
    assert out.what_worked == "score-5 fix wins"


def test_evidence_strength_adequate_when_score_3():
    """Sprint 10.3 §4.3 — score == 3 → evidence_strength="adequate"."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-3",
            "Resolution_Quality_Score": "3",
        },
        "Symptom_Solution_Mapping": {"Primary_Fix": "ok fix"},
    }]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.evidence_strength == "adequate"
    assert out.best_quality_score == 3


def test_evidence_strength_weak_when_score_2():
    """Sprint 10.3 §4.3 — score in (1, 2) → evidence_strength="weak"."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-2",
            "Resolution_Quality_Score": "2",
        },
        "Symptom_Solution_Mapping": {"Primary_Fix": "weak fix"},
    }]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.evidence_strength == "weak"
    assert out.best_quality_score == 2


def test_evidence_strength_strong_when_score_5():
    """Sprint 10.3 §4.3 — score >= 4 → evidence_strength="strong"."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-5",
            "Resolution_Quality_Score": "5",
        },
        "Symptom_Solution_Mapping": {"Primary_Fix": "strong fix"},
    }]
    engine, _ = _stub_corpus_engine(0, None)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.evidence_strength == "strong"
    assert out.best_quality_score == 5


def test_full_happy_path_phoenix_402():
    """Doc's worked example — best-ticket distillation populates all
    headline fields; corpus stats embed the platform median."""
    cohort = [{
        "Metadata": {
            "Incident_Number": "INC-PHOENIX-402",
            "Resolution_Quality_Score": "5",
            "time_to_resolve_minutes": "42",
            "Created_At": "2026-04-15T00:00:00Z",
        },
        "Symptom_Solution_Mapping": {
            "Primary_Fix": "Restart Citrix delivery controller and purge stale session profile lock on the affected pool",
        },
        "Executive_Sharable_RCA": {
            "Resolution_Steps": [
                "Verified DDC health via Get-BrokerController",
                "Identified stuck profile lock on VDI-Pool-A",
                "Restarted DDC service; purged lock; verified launch < 20s",
            ],
        },
        "Forensic_Performance_Audit": {
            "Critical_Intervention": "Manual DDC restart while profile broker was idle",
        },
    }]
    engine, _ = _stub_corpus_engine(corpus_size=918, platform_median=134.0)
    with patch("backend.db.connection.engine", engine):
        out = compute_stage0(cohort)

    assert out.cohort_size == 1
    assert out.best_incident == "INC-PHOENIX-402"
    assert out.best_quality_score == 5
    assert out.best_time_minutes == 42
    assert "Restart Citrix" in out.what_worked
    assert len(out.how_they_did_it) == 3
    assert "DDC restart" in out.critical_intervention
    # Cohort tail
    assert out.clean_resolution_count == 1
    assert out.clean_resolution_percent == 100
    assert out.avg_minutes_to_resolve_cohort == 42
    # Corpus tail
    assert out.corpus_size == 918
    assert out.platform_median_minutes == 134
    assert out.sparse is False
