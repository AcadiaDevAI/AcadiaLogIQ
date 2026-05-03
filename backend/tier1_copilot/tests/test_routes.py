"""Sprint 6 — routes: flag gating + request validation (offline).

We exercise the router via fastapi.testclient without mounting against
the real app assembly, so DB / Bedrock are never called.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _make_client(flag_on: bool):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.tier1_copilot.routes import router
    from backend.config import settings

    app = FastAPI()
    app.include_router(router)

    patched = mock.patch.object(settings, "LOGIQ_TIER1_COPILOT_BACKEND", flag_on)
    patched.start()
    client = TestClient(app)
    return client, patched


class FlagOffTests(unittest.TestCase):
    def test_analyze_returns_404_when_flag_off(self):
        client, patch_ctx = _make_client(flag_on=False)
        try:
            r = client.post("/tier1/analyze", json={
                "severity": "P2",
                "asset_name": "a",
                "alert_type": "b",
                "session_id": "s",
            })
            self.assertEqual(r.status_code, 404)
        finally:
            patch_ctx.stop()

    def test_feedback_returns_404_when_flag_off(self):
        client, patch_ctx = _make_client(flag_on=False)
        try:
            r = client.post("/tier1/feedback", json={
                "response_id": "x",
                "helpful": True,
                "session_id": "s",
            })
            self.assertEqual(r.status_code, 404)
        finally:
            patch_ctx.stop()


class ValidationTests(unittest.TestCase):
    def test_malformed_payload_returns_422(self):
        client, patch_ctx = _make_client(flag_on=True)
        try:
            r = client.post("/tier1/analyze", json={
                "asset_name": "a", "alert_type": "b", "session_id": "s",
            })  # missing severity
            self.assertEqual(r.status_code, 422)
        finally:
            patch_ctx.stop()

    def test_severity_enum_rejected_when_bad(self):
        client, patch_ctx = _make_client(flag_on=True)
        try:
            r = client.post("/tier1/analyze", json={
                "severity": "P9", "asset_name": "a",
                "alert_type": "b", "session_id": "s",
            })
            self.assertEqual(r.status_code, 422)
        finally:
            patch_ctx.stop()


class HealthEndpointTests(unittest.TestCase):
    def test_health_reports_flag_state_on(self):
        client, patch_ctx = _make_client(flag_on=True)
        try:
            r = client.get("/tier1/health")
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertTrue(body["flag_on"])
            self.assertIn("alias_term_count", body)
            self.assertIn("cache_size", body)
        finally:
            patch_ctx.stop()


class AnalyzeNoEvidencePathTests(unittest.TestCase):
    """When retrieval returns [] the route should still 200 with a
    template-fallback answer (confidence=None, no LLM invoked)."""

    def test_no_evidence_returns_200_and_none_confidence(self):
        from backend.tier1_copilot import routes
        client, patch_ctx = _make_client(flag_on=True)
        try:
            with mock.patch.object(routes, "retrieve_top_matches", return_value=[]), \
                 mock.patch.object(routes, "get_cached_answer", return_value=None), \
                 mock.patch.object(routes, "set_cached_answer", return_value=True), \
                 mock.patch.object(routes, "_invoke_haiku", return_value=""):
                r = client.post("/tier1/analyze", json={
                    "severity": "P3",
                    "asset_name": "Unknown Asset",
                    "alert_type": "Mystery",
                    "session_id": "s1",
                })
            self.assertEqual(r.status_code, 200)
            body = r.json()
            self.assertEqual(body["confidence"], "None")
            self.assertEqual(body["similar_count"], 0)
            self.assertFalse(body["cache_hit"])
            self.assertIsNone(body["matched_incident"])
        finally:
            patch_ctx.stop()


class CacheHitCohortPopulationTests(unittest.TestCase):
    """Sprint 10.3 §3.4 — on a cache hit the analyze handler must
    populate the new session row's top_5_match_ids from the cached
    payload. Without this, the Resolution Journey sees an empty cohort
    on every cache hit and renders blank panels."""

    def test_cache_hit_populates_session_cohort(self):
        from backend.tier1_copilot import routes
        from backend.config import settings

        cached_payload = {
            "answer_json": {
                "matched_incident": "INC-CACHED-001",
                "similar_count": 5,
                "answer": {},
            },
            "confidence": "High",
            "matched_chunk_id": "chunk-1",
            "alert_signature": "p2 | a | b",
            "top_5_match_ids": [
                "chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5",
            ],
        }

        # Capture the call args sent to create_session so we can assert
        # the cohort got forwarded onto the session row.
        captured: dict = {}

        class FakeSession:
            id = "sess_cached_xyz"
            from datetime import datetime, timezone
            created_at = datetime(2026, 4, 1, tzinfo=timezone.utc)

        def fake_create_session(*, alert_signature, alert_payload,
                                top_5_match_ids):
            captured["alert_signature"] = alert_signature
            captured["alert_payload"] = alert_payload
            captured["top_5_match_ids"] = list(top_5_match_ids or [])
            return FakeSession()

        client, patch_ctx = _make_client(flag_on=True)
        try:
            with mock.patch.object(
                routes, "get_cached_answer", return_value=cached_payload,
            ), mock.patch.object(
                routes, "create_session", side_effect=fake_create_session,
            ), mock.patch.object(
                routes, "retrieve_top_matches",
                side_effect=AssertionError(
                    "retrieval must NOT run on cache hit",
                ),
            ), mock.patch.object(
                settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", True,
            ):
                r = client.post("/tier1/analyze", json={
                    "severity": "P2",
                    "asset_name": "V-Desktop Environment",
                    "alert_type": "Desktop Slowness",
                    "session_id": "s",
                })
            self.assertEqual(r.status_code, 200)
            body = r.json()

            # Cache hit was honoured; retrieval did not run (else the
            # AssertionError side_effect would have raised).
            self.assertTrue(body["cache_hit"])

            # The cached cohort flowed onto the new session row.
            self.assertEqual(
                captured["top_5_match_ids"],
                ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
            )

            # The response also exposes the cohort so callers (frontend)
            # know which tickets the journey will paint.
            self.assertEqual(
                body["top_5_match_ids"],
                ["chunk-1", "chunk-2", "chunk-3", "chunk-4", "chunk-5"],
            )
            self.assertEqual(body["session_id"], "sess_cached_xyz")
        finally:
            patch_ctx.stop()


if __name__ == "__main__":
    unittest.main(verbosity=2)
