"""Sprint 9 — /intake/* routes (offline; LLM + DB mocked)."""
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


def _make_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from backend.tier1_copilot.intake.routes import router
    from backend.tier1_copilot.intake.catalogs import reset_singleton_for_tests

    reset_singleton_for_tests()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class ValidationTests(unittest.TestCase):
    def test_invalid_source_returns_422(self):
        client = _make_client()
        r = client.post(
            "/intake/extract",
            json={"source": "fax", "raw_text": "x"},
        )
        self.assertEqual(r.status_code, 422)

    def test_empty_text_returns_422(self):
        client = _make_client()
        r = client.post(
            "/intake/extract",
            json={"source": "email", "raw_text": "   "},
        )
        self.assertEqual(r.status_code, 422)

    def test_oversized_text_returns_422(self):
        client = _make_client()
        big = "a" * 11000
        r = client.post(
            "/intake/extract",
            json={"source": "email", "raw_text": big},
        )
        self.assertEqual(r.status_code, 422)


class HealthTests(unittest.TestCase):
    def test_health_reports_ok(self):
        from backend.tier1_copilot.intake import routes as routes_mod
        client = _make_client()
        with mock.patch.object(routes_mod, "get_intake_catalogs") as mocked:
            cat = mock.MagicMock()
            cat.health_snapshot.return_value = (4, 2, 3, 1)
            cat.is_built.return_value = True
            mocked.return_value = cat
            r = client.get("/intake/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["flag_on"])
        self.assertTrue(body["catalogs_built"])
        self.assertEqual(body["asset_families"], 2)


class ExtractHappyPathTests(unittest.TestCase):
    def test_extract_returns_validated_diversified(self):
        from backend.tier1_copilot.intake import routes as routes_mod
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs

        cat = IntakeCatalogs()
        cat.rebuild_from_rows([
            {"metadata_json": {
                "Metadata": {
                    "Affected_Assets": ["VPN-EDGE-01"],
                    "customer_name": "Aetheris",
                },
                "Symptom_Solution_Mapping": {
                    "Detected_Symptom": "Login failure",
                },
            }},
        ])

        # LLM returns 3 candidates: two duplicates + one different.
        payload = json.dumps([
            {
                "severity": "P2", "asset_name": "VPN-EDGE-01",
                "alert_type": "Login failure", "customer": "Aetheris",
                "evidence": {"severity": "20 users", "asset_name": "VPN",
                             "alert_type": "login", "customer": "Aetheris"},
            },
            {
                "severity": "P2", "asset_name": "VPN-EDGE-01",
                "alert_type": "Login failure", "customer": "Aetheris",
                "evidence": {"severity": "20 users", "asset_name": "VPN",
                             "alert_type": "login", "customer": "Aetheris"},
            },
            {
                "severity": "P3", "asset_name": "VPN-EDGE-01",
                "alert_type": "Login failure", "customer": "Aetheris",
                "evidence": {"severity": "minor", "asset_name": "VPN",
                             "alert_type": "login", "customer": "Aetheris"},
            },
        ])

        client = _make_client()
        # Patch the extractor's default LLM invoker so the route
        # handler's call chain uses our canned JSON payload.
        with mock.patch.object(
            routes_mod, "get_intake_catalogs", return_value=cat,
        ), mock.patch(
            "backend.tier1_copilot.intake.extractor._default_invoker",
            return_value=payload,
        ), mock.patch.object(
            routes_mod, "log_extraction", return_value="intk_test123",
        ):
            r = client.post(
                "/intake/extract",
                json={
                    "source": "email",
                    "raw_text": "20 users in Chicago can't connect",
                    "session_id": "sess_abc",
                },
            )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["extraction_id"], "intk_test123")
        # Diversifier dedupes the two P2 duplicates first, then
        # pads back the dropped one (max_cards=4 leaves room) — so
        # the engineer sees the unique P2, the P3, and the
        # duplicate P2 last. The first two slots are the unique
        # interpretations.
        self.assertGreaterEqual(len(body["candidates"]), 2)
        self.assertEqual(body["candidates"][0]["severity"], "P2")
        self.assertEqual(body["candidates"][1]["severity"], "P3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
