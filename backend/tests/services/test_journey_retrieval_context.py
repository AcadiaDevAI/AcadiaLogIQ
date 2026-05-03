"""Sprint 11 — journey_retrieval_context tests.

Validates the helper that biases /ask retrieval with the originating
ticket's intake when the chat session was opened via Stage 0 / Stage 3
"Ask in chat" links.

Tests use a fake DB session that mimics the SQLAlchemy `execute().mappings().first()`
shape — the helper is small enough that an in-memory fake is the
right tool here (no Postgres in the unit-test loop, no heavy mocks).

Run with: py -m pytest backend/tests/services/test_journey_retrieval_context.py
"""
from __future__ import annotations

import json
import unittest

from backend.services import journey_retrieval_context as jrc


# ─────────────────────────────────────────────────────────────
# Fake SQLAlchemy session — minimal surface the helper uses.
# ─────────────────────────────────────────────────────────────
class _FakeRow:
    def __init__(self, mapping):
        self._mapping = mapping
    def get(self, key, default=None):
        return self._mapping.get(key, default)


class _FakeMappings:
    def __init__(self, rows):
        self._rows = rows
    def first(self):
        return _FakeRow(self._rows[0]) if self._rows else None


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows
    def mappings(self):
        return _FakeMappings(self._rows)


class _FakeSession:
    """Routes execute() based on which SQL fragment is being run.

    The helper issues two queries:
      - SELECT sources_json FROM chat_messages WHERE session_id = :sid
      - SELECT alert_payload FROM tier1_sessions WHERE id = :id
    We dispatch on the table name in the SQL string."""
    def __init__(self, *, chat_messages_rows=None, tier1_sessions_rows=None):
        self._chat = chat_messages_rows or []
        self._tier1 = tier1_sessions_rows or []
    def execute(self, statement, params=None):
        sql = str(statement).lower()
        if "from chat_messages" in sql:
            return _FakeResult(self._chat)
        if "from tier1_sessions" in sql:
            return _FakeResult(self._tier1)
        return _FakeResult([])
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def _factory_for(*, chat=None, tier1=None):
    return lambda: _FakeSession(
        chat_messages_rows=chat or [],
        tier1_sessions_rows=tier1 or [],
    )


# ─────────────────────────────────────────────────────────────
# load_journey_context tests
# ─────────────────────────────────────────────────────────────
class LoadJourneyContextTests(unittest.TestCase):

    def test_returns_none_when_chat_session_id_empty(self):
        result = jrc.load_journey_context(
            chat_session_id="", db_session_factory=_factory_for(),
        )
        self.assertIsNone(result)

    def test_returns_none_when_no_first_message(self):
        result = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(chat=[]),
        )
        self.assertIsNone(result)

    def test_returns_none_when_no_session_metadata(self):
        # Chat message exists but sources_json has no _session_metadata.
        result = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(
                chat=[{"sources_json": {"docs": [{"x": 1}]}}],
            ),
        )
        self.assertIsNone(result)

    def test_returns_none_when_journey_id_missing_inside_metadata(self):
        result = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(
                chat=[{"sources_json": {"_session_metadata": {"foo": "bar"}}}],
            ),
        )
        self.assertIsNone(result)

    def test_returns_none_when_tier1_session_missing(self):
        # Journey id present but tier1_sessions has no row.
        result = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(
                chat=[{"sources_json": {
                    "_session_metadata": {"journey_session_id": "sess_x"},
                }}],
                tier1=[],
            ),
        )
        self.assertIsNone(result)

    def test_returns_none_when_alert_payload_is_empty_dict(self):
        result = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(
                chat=[{"sources_json": {
                    "_session_metadata": {"journey_session_id": "sess_x"},
                }}],
                tier1=[{"alert_payload": {}}],
            ),
        )
        self.assertIsNone(result)

    def test_returns_context_dict_when_journey_and_payload_exist(self):
        ctx = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(
                chat=[{"sources_json": {
                    "_session_metadata": {"journey_session_id": "sess_alpha"},
                }}],
                tier1=[{"alert_payload": {
                    "severity": "P3",
                    "asset_name": "V-Fax-Line-12",
                    "alert_type": "Faxlinesunreachable",
                    "customer": "Aetheris",
                    "location": "Singapore Hub",
                    "ip_or_device_id": "10.0.12.4",
                    "error_code": None,    # filtered out
                    "notes": "",           # filtered out
                    "extra_field": "ign",  # not in _CONTEXT_FIELDS
                }}],
            ),
        )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["severity"], "P3")
        self.assertEqual(ctx["asset_name"], "V-Fax-Line-12")
        self.assertEqual(ctx["alert_type"], "Faxlinesunreachable")
        self.assertEqual(ctx["customer"], "Aetheris")
        self.assertEqual(ctx["location"], "Singapore Hub")
        self.assertEqual(ctx["ip_or_device_id"], "10.0.12.4")
        self.assertNotIn("error_code", ctx)
        self.assertNotIn("notes", ctx)
        self.assertNotIn("extra_field", ctx)
        self.assertEqual(ctx["_journey_session_id"], "sess_alpha")

    def test_handles_string_encoded_sources_json(self):
        # Some DB drivers return JSONB columns as strings.
        ctx = jrc.load_journey_context(
            chat_session_id="chat-1",
            db_session_factory=_factory_for(
                chat=[{"sources_json": json.dumps({
                    "_session_metadata": {"journey_session_id": "sess_x"},
                })}],
                tier1=[{"alert_payload": json.dumps({
                    "asset_name": "Router-1",
                    "alert_type": "BGP flap",
                })}],
            ),
        )
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx["asset_name"], "Router-1")
        self.assertEqual(ctx["alert_type"], "BGP flap")

    def test_returns_none_on_db_exception(self):
        # Any exception in the factory must fail-open (return None).
        def _broken_factory():
            raise RuntimeError("simulated DB outage")
        ctx = jrc.load_journey_context(
            chat_session_id="chat-1", db_session_factory=_broken_factory,
        )
        self.assertIsNone(ctx)


# ─────────────────────────────────────────────────────────────
# build_query_prefix tests
# ─────────────────────────────────────────────────────────────
class BuildQueryPrefixTests(unittest.TestCase):

    def test_empty_for_none(self):
        self.assertEqual(jrc.build_query_prefix(None), "")

    def test_empty_for_empty_dict(self):
        self.assertEqual(jrc.build_query_prefix({}), "")

    def test_empty_when_only_internal_keys(self):
        # _journey_session_id is an internal marker, not a context field;
        # the prefix must skip it.
        self.assertEqual(
            jrc.build_query_prefix({"_journey_session_id": "sess_x"}),
            "",
        )

    def test_renders_compact_prefix_in_field_order(self):
        prefix = jrc.build_query_prefix({
            "asset_name": "V-Fax-Line-12",
            "alert_type": "Faxlinesunreachable",
            "customer": "Aetheris",
            "severity": "P3",
            "_journey_session_id": "sess_x",
        })
        # Must start with the framing marker.
        self.assertTrue(prefix.startswith("[Context: "))
        # Must end with closing bracket and a trailing space (so the
        # caller can do `prefix + query` cleanly).
        self.assertTrue(prefix.endswith("] "))
        # Field order matches _CONTEXT_FIELDS — asset_name first.
        body = prefix[len("[Context: "):-len("] ")]
        parts = body.split(" ")
        self.assertEqual(parts[0], "asset_name=V-Fax-Line-12")
        # alert_type comes next (per _CONTEXT_FIELDS ordering).
        self.assertEqual(parts[1], "alert_type=Faxlinesunreachable")

    def test_collapses_whitespace_in_values(self):
        prefix = jrc.build_query_prefix({
            "notes": "Line 1\nLine 2\n  trailing  spaces  ",
        })
        # Newlines + multi-space collapse to single spaces inside
        # the value, so the prefix stays one line.
        self.assertNotIn("\n", prefix)
        self.assertIn("notes=Line 1 Line 2 trailing spaces", prefix)


if __name__ == "__main__":
    unittest.main()
