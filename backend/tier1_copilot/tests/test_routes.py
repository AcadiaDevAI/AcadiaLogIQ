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


if __name__ == "__main__":
    unittest.main(verbosity=2)
