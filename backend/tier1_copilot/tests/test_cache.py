"""Sprint 6 — cache helpers (offline, mocked engine)."""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class CacheGetSetTests(unittest.TestCase):
    def test_set_writes_upsert_sql(self):
        from backend.tier1_copilot import cache
        fake_conn = mock.MagicMock()
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_engine = mock.MagicMock()
        fake_engine.begin.return_value = fake_conn
        with mock.patch("backend.db.connection.engine", fake_engine):
            ok = cache.set_cached_answer(
                signature_hash="hash123",
                alert_signature="p2 | a | b",
                answer={"hello": "world"},
                confidence="High",
                matched_chunk_id="chunk-1",
            )
        self.assertTrue(ok)
        sql_call = fake_conn.execute.call_args
        self.assertIn("INSERT INTO tier1_answer_cache", str(sql_call.args[0]))
        params = sql_call.args[1]
        self.assertEqual(params["h"], "hash123")
        self.assertEqual(params["sig"], "p2 | a | b")
        # answer is JSON-serialized string, not dict.
        self.assertEqual(json.loads(params["a"]), {"hello": "world"})

    def test_get_returns_none_on_expired(self):
        from datetime import datetime, timedelta, timezone
        from backend.tier1_copilot import cache

        row = {
            "answer_json": {"x": 1},
            "confidence": "High",
            "matched_chunk_id": "c1",
            "alert_signature": "s",
            "expires_at": datetime.now(timezone.utc) - timedelta(days=1),
        }
        fake_result = mock.MagicMock()
        fake_result.first.return_value = row
        fake_conn = mock.MagicMock()
        fake_conn.execute.return_value.mappings.return_value = fake_result
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_engine = mock.MagicMock()
        fake_engine.connect.return_value = fake_conn
        with mock.patch("backend.db.connection.engine", fake_engine):
            out = cache.get_cached_answer("hash")
        self.assertIsNone(out)

    def test_get_returns_payload_on_fresh(self):
        from datetime import datetime, timedelta, timezone
        from backend.tier1_copilot import cache

        row = {
            "answer_json": {"x": 1},
            "confidence": "High",
            "matched_chunk_id": "c1",
            "alert_signature": "s",
            "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
        }
        fake_result = mock.MagicMock()
        fake_result.first.return_value = row
        fake_conn = mock.MagicMock()
        fake_conn.execute.return_value.mappings.return_value = fake_result
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_engine = mock.MagicMock()
        fake_engine.connect.return_value = fake_conn
        with mock.patch("backend.db.connection.engine", fake_engine):
            out = cache.get_cached_answer("hash")
        self.assertIsNotNone(out)
        self.assertEqual(out["confidence"], "High")
        self.assertEqual(out["answer_json"], {"x": 1})

    def test_get_db_error_returns_none(self):
        from backend.tier1_copilot import cache
        broken = mock.MagicMock()
        broken.connect.side_effect = RuntimeError("no table")
        with mock.patch("backend.db.connection.engine", broken):
            self.assertIsNone(cache.get_cached_answer("x"))

    def test_set_db_error_returns_false(self):
        from backend.tier1_copilot import cache
        broken = mock.MagicMock()
        broken.begin.side_effect = RuntimeError("no table")
        with mock.patch("backend.db.connection.engine", broken):
            ok = cache.set_cached_answer(
                signature_hash="h",
                alert_signature="s",
                answer={"a": 1},
                confidence="High",
                matched_chunk_id=None,
            )
        self.assertFalse(ok)

    def test_empty_inputs_short_circuit(self):
        from backend.tier1_copilot import cache
        self.assertIsNone(cache.get_cached_answer(""))
        self.assertFalse(cache.set_cached_answer(
            signature_hash="",
            alert_signature="",
            answer={},
            confidence="None",
            matched_chunk_id=None,
        ))

    def test_set_cached_answer_persists_top5_ids(self):
        """Sprint 10.3 — set_cached_answer must accept top_5_match_ids
        and bind it as the :top5 SQL param so cache hits can repopulate
        the new session's cohort. Without this, the journey sees an
        empty cohort on every cache hit."""
        from backend.tier1_copilot import cache
        fake_conn = mock.MagicMock()
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_engine = mock.MagicMock()
        fake_engine.begin.return_value = fake_conn
        with mock.patch("backend.db.connection.engine", fake_engine):
            ok = cache.set_cached_answer(
                signature_hash="hash-top5",
                alert_signature="p2 | a | b",
                answer={"hello": "world"},
                confidence="High",
                matched_chunk_id="chunk-1",
                top_5_match_ids=["chunk-1", "chunk-2", "chunk-3",
                                 "chunk-4", "chunk-5"],
            )
        self.assertTrue(ok)
        sql_call = fake_conn.execute.call_args
        sql_str = str(sql_call.args[0])
        self.assertIn("top_5_match_ids", sql_str)
        params = sql_call.args[1]
        # List cast at SQL boundary per spec §8 — psycopg + TEXT[] is
        # finicky; tuples and numpy arrays fail silently.
        self.assertIsInstance(params["top5"], list)
        self.assertEqual(
            params["top5"],
            ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
        )

    def test_get_cached_answer_returns_top5_ids(self):
        """Sprint 10.3 — round-trip read of top_5_match_ids on cache
        hit. Cached payload returns a list of the cohort chunk_ids the
        original answer was derived from."""
        from datetime import datetime, timedelta, timezone
        from backend.tier1_copilot import cache

        row = {
            "answer_json": {"x": 1},
            "confidence": "High",
            "matched_chunk_id": "c1",
            "alert_signature": "s",
            "top_5_match_ids": ["c1", "c2", "c3", "c4", "c5"],
            "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
        }
        fake_result = mock.MagicMock()
        fake_result.first.return_value = row
        fake_conn = mock.MagicMock()
        fake_conn.execute.return_value.mappings.return_value = fake_result
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_engine = mock.MagicMock()
        fake_engine.connect.return_value = fake_conn
        with mock.patch("backend.db.connection.engine", fake_engine):
            out = cache.get_cached_answer("hash")
        self.assertIsNotNone(out)
        self.assertIn("top_5_match_ids", out)
        self.assertIsInstance(out["top_5_match_ids"], list)
        self.assertEqual(out["top_5_match_ids"],
                         ["c1", "c2", "c3", "c4", "c5"])

    def test_signature_collision_avoidance(self):
        """Two alerts with same required triple but different optional
        fields must produce different signature_hash values (regression
        guard for §5 of the spec)."""
        from backend.tier1_copilot.normalizer import normalize_alert
        from backend.tier1_copilot.schemas import Tier1AnalyzeRequest
        base = dict(
            severity="P2",
            asset_name="V-Desktop Environment",
            alert_type="Desktop Slowness",
            session_id="s",
        )
        a = normalize_alert(Tier1AnalyzeRequest(**base), alias_dict=None)
        b = normalize_alert(
            Tier1AnalyzeRequest(**base, customer="Aetheris"), alias_dict=None,
        )
        self.assertNotEqual(a["signature_hash"], b["signature_hash"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
