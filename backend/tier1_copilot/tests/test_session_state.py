"""Sprint 7 — session_state CRUD (offline, mocked engine)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import datetime, timezone
from unittest import mock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _fake_begin():
    """Return a (patch target, conn) pair where the MagicMock `conn`
    supports .execute().first() and the `with engine.begin() as conn`
    pattern."""
    fake_conn = mock.MagicMock()
    fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
    fake_conn.__exit__ = mock.MagicMock(return_value=False)
    fake_engine = mock.MagicMock()
    fake_engine.begin.return_value = fake_conn
    fake_engine.connect.return_value = fake_conn
    return fake_engine, fake_conn


class CreateSessionTests(unittest.TestCase):
    def test_create_inserts_and_returns_record(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        fake_engine, fake_conn = _fake_begin()

        # SELECT row used by get_session after create returns:
        select_row = {
            "id": "sess_abc",
            "alert_signature": "p2 | a | b",
            "alert_payload": {"severity": "P2"},
            "top_5_match_ids": ["c1", "c2"],
            "current_match_index": 0,
            "thumbs_down_count": 0,
            "what_tried": [],
            "stuck_nudge_shown": False,
            "resolved": False,
            "escalated": False,
            "created_at": datetime.now(timezone.utc),
            "last_activity_at": datetime.now(timezone.utc),
        }
        mapping_result = mock.MagicMock()
        mapping_result.first.return_value = select_row
        # execute().mappings() returns mapping_result; execute() alone
        # is used for the INSERT which just discards.
        fake_conn.execute.return_value.mappings.return_value = mapping_result

        with mock.patch("backend.db.connection.engine", fake_engine):
            rec = ts.create_session(
                alert_signature="p2 | a | b",
                alert_payload={"severity": "P2"},
                top_5_match_ids=["c1", "c2"],
                session_id="sess_abc",
            )
        self.assertIsNotNone(rec)
        self.assertEqual(rec.id, "sess_abc")
        self.assertEqual(rec.top_5_match_ids, ["c1", "c2"])

    def test_db_error_returns_none(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        broken = mock.MagicMock()
        broken.begin.side_effect = RuntimeError("no table")
        with mock.patch("backend.db.connection.engine", broken):
            rec = ts.create_session(
                alert_signature="x", alert_payload={}, top_5_match_ids=[],
            )
        self.assertIsNone(rec)


class MutatorTests(unittest.TestCase):
    def test_update_match_index(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        fake_engine, fake_conn = _fake_begin()
        with mock.patch("backend.db.connection.engine", fake_engine):
            self.assertTrue(ts.update_match_index("sess_abc", 3))
        sql = fake_conn.execute.call_args.args[0]
        self.assertIn("UPDATE tier1_sessions", str(sql))
        self.assertIn("current_match_index", str(sql))

    def test_increment_thumbs_down_returns_count(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        fake_engine, fake_conn = _fake_begin()
        fake_conn.execute.return_value.first.return_value = (4,)
        with mock.patch("backend.db.connection.engine", fake_engine):
            n = ts.increment_thumbs_down("sess_abc")
        self.assertEqual(n, 4)

    def test_append_what_tried_returns_list(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        fake_engine, fake_conn = _fake_begin()
        fake_conn.execute.return_value.first.return_value = (
            [{"step": "ping", "result": "ok"}],
        )
        with mock.patch("backend.db.connection.engine", fake_engine):
            out = ts.append_what_tried("sess_abc", {"step": "ping", "result": "ok"})
        self.assertEqual(out, [{"step": "ping", "result": "ok"}])

    def test_empty_inputs_short_circuit(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        self.assertIsNone(ts.get_session(""))
        self.assertFalse(ts.update_match_index("", 1))
        self.assertIsNone(ts.append_what_tried("", {"step": "x", "result": "y"}))

    def test_mark_stuck_shown_swallows_errors(self):
        from backend.tier1_copilot.session_state import tier1_session as ts
        broken = mock.MagicMock()
        broken.begin.side_effect = RuntimeError("no table")
        with mock.patch("backend.db.connection.engine", broken):
            self.assertFalse(ts.mark_stuck_shown("sess"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
