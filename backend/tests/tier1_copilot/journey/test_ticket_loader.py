"""Sprint 10 — ticket_loader unit tests.

Critical invariants:
  1. The chunks fetch uses `WHERE id = ANY(:ids)` with the list bound
     as a single param (NOT `bindparam(expanding=True)` which would
     produce `id = ANY(($1, $2, ...))` and PostgreSQL would reject it
     with "op ANY/ALL (array) requires array on right side").
  2. Rank order from `tier1_sessions.top_5_match_ids` is preserved in
     the returned list, even though Postgres returns ANY-array results
     in arbitrary order.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from backend.tier1_copilot.journey.ticket_loader import load_cohort_metadata


def _stub_engine(session_row, chunk_rows):
    """Stub engine.connect() so the first execute() (session lookup)
    returns session_row and the second (chunk fetch) returns chunk_rows.
    Captures all execute() calls so tests can assert on the SQL + params."""
    fake_session_result = MagicMock()
    fake_session_result.first.return_value = session_row
    fake_chunks_result = MagicMock()
    fake_chunks_result.all.return_value = chunk_rows

    fake_conn = MagicMock()
    fake_conn.execute.return_value.mappings.side_effect = [
        fake_session_result, fake_chunks_result,
    ]
    cm = MagicMock()
    cm.__enter__.return_value = fake_conn
    cm.__exit__.return_value = False

    engine = MagicMock()
    engine.connect.return_value = cm
    return engine, fake_conn


def test_empty_session_id_returns_empty():
    out = load_cohort_metadata("")
    assert out == []


def test_session_not_found_returns_empty():
    engine, _conn = _stub_engine(session_row=None, chunk_rows=[])
    with patch("backend.db.connection.engine", engine):
        out = load_cohort_metadata("missing-sess")
    assert out == []


def test_session_with_empty_top5_returns_empty():
    engine, _conn = _stub_engine(
        session_row={"top_5_match_ids": []},
        chunk_rows=[],
    )
    with patch("backend.db.connection.engine", engine):
        out = load_cohort_metadata("sess-empty")
    assert out == []


def test_db_error_returns_empty():
    broken = MagicMock()
    broken.connect.side_effect = RuntimeError("connection refused")
    with patch("backend.db.connection.engine", broken):
        out = load_cohort_metadata("sess-X")
    assert out == []


def test_5_id_round_trip_preserves_rank_order():
    """Top-5 chunk IDs are passed in rank order; Postgres returns the
    rows in arbitrary order; the loader must re-sort them to match the
    input id order."""
    ids = ["chunk-a", "chunk-b", "chunk-c", "chunk-d", "chunk-e"]
    # Deliberately return rows in REVERSE order to simulate Postgres
    # ordering by chunk_id (or some other internal heuristic) instead of
    # the input rank order.
    shuffled_rows = [
        {"id": "chunk-e", "metadata_json": {"Metadata": {"Incident_Number": "INC-005"}}},
        {"id": "chunk-c", "metadata_json": {"Metadata": {"Incident_Number": "INC-003"}}},
        {"id": "chunk-a", "metadata_json": {"Metadata": {"Incident_Number": "INC-001"}}},
        {"id": "chunk-d", "metadata_json": {"Metadata": {"Incident_Number": "INC-004"}}},
        {"id": "chunk-b", "metadata_json": {"Metadata": {"Incident_Number": "INC-002"}}},
    ]
    engine, conn = _stub_engine(
        session_row={"top_5_match_ids": ids},
        chunk_rows=shuffled_rows,
    )

    with patch("backend.db.connection.engine", engine):
        out = load_cohort_metadata("sess-rank-test")

    # Returned cohort matches the input rank order regardless of DB shuffle
    assert len(out) == 5
    assert [t["Metadata"]["Incident_Number"] for t in out] == [
        "INC-001", "INC-002", "INC-003", "INC-004", "INC-005",
    ]


def test_chunk_fetch_uses_any_array_not_expanding():
    """Regression guard: the SQL must be the single-bind ANY(:ids) form,
    NOT `bindparam(..., expanding=True)`. Expanding would produce
    `ANY(($1, $2, ...))` and PostgreSQL rejects that."""
    ids = ["a", "b", "c"]
    engine, conn = _stub_engine(
        session_row={"top_5_match_ids": ids},
        chunk_rows=[
            {"id": i, "metadata_json": {"x": idx}} for idx, i in enumerate(ids)
        ],
    )

    with patch("backend.db.connection.engine", engine):
        load_cohort_metadata("sess-sql-shape")

    # Two execute calls — the second is the chunk fetch
    chunk_call = conn.execute.call_args_list[1]
    stmt_arg = chunk_call.args[0]
    params_arg = chunk_call.args[1]

    sql_text = str(stmt_arg)
    assert "ANY(:ids)" in sql_text
    # Params: a single bind named "ids" with the WHOLE list
    assert "ids" in params_arg
    assert params_arg["ids"] == ids
    assert isinstance(params_arg["ids"], list)


def test_missing_chunk_dropped_silently():
    """If a chunk_id from top_5_match_ids has been deleted from the
    chunks table, the loader skips it without raising."""
    ids = ["chunk-a", "chunk-MISSING", "chunk-c"]
    rows = [
        {"id": "chunk-a", "metadata_json": {"Metadata": {"Incident_Number": "INC-A"}}},
        {"id": "chunk-c", "metadata_json": {"Metadata": {"Incident_Number": "INC-C"}}},
    ]
    engine, _ = _stub_engine(
        session_row={"top_5_match_ids": ids},
        chunk_rows=rows,
    )
    with patch("backend.db.connection.engine", engine):
        out = load_cohort_metadata("sess-partial")
    assert len(out) == 2
    assert [t["Metadata"]["Incident_Number"] for t in out] == ["INC-A", "INC-C"]


def test_non_dict_metadata_json_dropped():
    """Defensive: if metadata_json comes back as a non-dict type
    (NULL, malformed, etc.), drop the row rather than propagating it."""
    ids = ["chunk-a", "chunk-bad", "chunk-c"]
    rows = [
        {"id": "chunk-a", "metadata_json": {"Metadata": {"Incident_Number": "INC-A"}}},
        {"id": "chunk-bad", "metadata_json": None},
        {"id": "chunk-c", "metadata_json": "not-a-dict"},
    ]
    engine, _ = _stub_engine(
        session_row={"top_5_match_ids": ids},
        chunk_rows=rows,
    )
    with patch("backend.db.connection.engine", engine):
        out = load_cohort_metadata("sess-bad-md")
    assert len(out) == 1
    assert out[0]["Metadata"]["Incident_Number"] == "INC-A"
