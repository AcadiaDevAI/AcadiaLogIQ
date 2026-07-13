"""Tests for US Pharma store-scoped tier-1 matching.

Covers the pure surface — the store-filter SQL fragment (added only when a
store_id is present, so Acadia's shared query stays byte-identical) and the
normalizer's store-aware cache hashing + optional asset_name — without a DB.

Run with: py -m unittest backend.tests.test_tier1_store_filter
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("AWS_SECRETS_DISABLED", "true")

from backend.tier1_copilot.retrieval import _store_filter_sql  # noqa: E402
from backend.tier1_copilot.normalizer import normalize_alert  # noqa: E402
from backend.tier1_copilot.schemas import Tier1AnalyzeRequest  # noqa: E402


class TestStoreFilterSql(unittest.TestCase):
    def test_empty_when_no_store_id(self):
        # Acadia path — no filter, so the shared SQL is byte-identical.
        self.assertEqual(_store_filter_sql(""), "")
        self.assertEqual(_store_filter_sql(None), "")

    def test_fragment_when_store_id(self):
        frag = _store_filter_sql("3001")
        self.assertIn(":store_id", frag)           # bound param, not interpolated
        self.assertNotIn("3001", frag)             # value must NOT be inlined
        self.assertIn("metadata_json->'Metadata'->>'store_id'", frag)
        self.assertTrue(frag.strip().startswith("AND"))

    def test_alias_prefix_for_stage2(self):
        self.assertIn("c.metadata_json", _store_filter_sql("3001", "c.metadata_json"))


class TestNormalizerStoreScope(unittest.TestCase):
    def _req(self, **kw):
        base = dict(alert_type="bgp flap", session_id="s1")
        base.update(kw)
        return Tier1AnalyzeRequest(**base)

    def test_asset_name_optional_does_not_crash(self):
        # US Pharma sends Store ID + symptom, no asset_name.
        n = normalize_alert(self._req(store_id="3001"), None)
        self.assertIn("signature_hash", n)
        self.assertIn("search_text", n)

    def test_store_id_changes_cache_hash(self):
        h1 = normalize_alert(self._req(store_id="3001"), None)["signature_hash"]
        h2 = normalize_alert(self._req(store_id="3012"), None)["signature_hash"]
        self.assertNotEqual(h1, h2)

    def test_no_store_id_hash_is_deterministic(self):
        # Acadia-style requests (no store_id) → stable, and store_id absence
        # must not perturb the hash across calls.
        r = dict(asset_name="edge-router", alert_type="circuit down")
        h1 = normalize_alert(self._req(**r), None)["signature_hash"]
        h2 = normalize_alert(self._req(**r), None)["signature_hash"]
        self.assertEqual(h1, h2)

    def test_store_id_absent_vs_present_differs(self):
        r = dict(asset_name="edge-router", alert_type="circuit down")
        no_store = normalize_alert(self._req(**r), None)["signature_hash"]
        with_store = normalize_alert(self._req(store_id="3001", **r), None)["signature_hash"]
        self.assertNotEqual(no_store, with_store)


class TestFingerprintRerank(unittest.TestCase):
    """The reranker must score the user's symptom against the ticket's real
    Metadata.Fingerprints (fingerprints_text is empty for US Pharma tickets),
    so the fingerprint-matching ticket ranks #1."""

    @staticmethod
    def _cand(chunk_id, fingerprints, vsim=0.1):
        return {
            "chunk_id": chunk_id,
            "metadata_json": {
                "Metadata": {"Fingerprints": fingerprints, "Incident_Number": chunk_id}
            },
            "fingerprints_text": "",   # empty, as it is for US Pharma gold tickets
            "component_category": "Network",
            "_vector_sim": vsim,
            "_ts_rank": 0.0,
        }

    def test_matching_fingerprint_ranks_first(self):
        from backend.tier1_copilot.retrieval import _weighted_rank

        # US Pharma alert — symptom in alert_type, no asset_name.
        alert = {
            "alert_type": "LOS on WAN1",
            "asset_name": None,
            "store_id": "3001",
            "error_code": None,
            "technology": None,
            "customer": None,
        }
        match = self._cand(
            "TKT-3001-01",
            ["%BGP-5-ADJCHANGE: neighbor Down", "LOS Alarm on WAN1"],
        )
        other = self._cand(
            "TKT-3001-05",
            ["DHCP Discover drops", "No IP addresses available in pool"],
        )
        # Pass the non-matching one first to prove it's the score, not order.
        ranked = _weighted_rank([other, match], alert, {})
        self.assertEqual(ranked[0]["chunk_id"], "TKT-3001-01")
        self.assertGreater(ranked[0]["final_score"], ranked[1]["final_score"])

    def test_acadia_no_store_id_is_unaffected(self):
        # No store_id (Acadia): the Metadata.Fingerprints enrichment must NOT
        # apply, so a fingerprint-matching ticket gets NO boost and ties a
        # non-matching one (same vector sim) — proving Acadia's ranking is
        # byte-identical to before.
        from backend.tier1_copilot.retrieval import _weighted_rank

        alert = {
            "alert_type": "LOS on WAN1",
            "asset_name": "edge-router",   # Acadia always sends asset_name
            "error_code": None,
            "technology": None,
            "customer": None,
            # no store_id
        }
        match = self._cand("A", ["LOS Alarm on WAN1"], vsim=0.2)
        other = self._cand("B", ["DHCP drops"], vsim=0.2)
        ranked = _weighted_rank([match, other], alert, {})
        self.assertEqual(ranked[0]["final_score"], ranked[1]["final_score"])
        self.assertEqual(
            ranked[0]["_score_components"]["fingerprint_match"],
            ranked[1]["_score_components"]["fingerprint_match"],
        )


class TestStoreFingerprintFilter(unittest.TestCase):
    """US Pharma (store-scoped) narrows the store cohort to tickets whose
    Fingerprints match the symptom — so store_id + fingerprint pinpoints the
    ticket. Falls back to the full cohort when nothing matches; Acadia (no
    store_id) is never filtered."""

    @staticmethod
    def _cand(chunk_id, fingerprints):
        return {
            "chunk_id": chunk_id,
            "metadata_json": {
                "Metadata": {"Fingerprints": fingerprints, "Incident_Number": chunk_id}
            },
            "fingerprints_text": "",
            "final_score": 0.5,
        }

    def test_filters_to_fingerprint_match_when_store_scoped(self):
        from backend.tier1_copilot.retrieval import _apply_store_fingerprint_filter

        alert = {"alert_type": "LOS Alarm on WAN1", "store_id": "3001"}
        match = self._cand("TKT-3001-01", ["%BGP-5-ADJCHANGE: neighbor Down", "LOS Alarm on WAN1"])
        other = self._cand("TKT-3001-05", ["DHCP Discover drops", "No IP addresses available in pool"])
        out = _apply_store_fingerprint_filter([match, other], alert)
        self.assertEqual([c["chunk_id"] for c in out], ["TKT-3001-01"])

    def test_fallback_to_full_cohort_when_no_fingerprint_match(self):
        from backend.tier1_copilot.retrieval import _apply_store_fingerprint_filter

        alert = {"alert_type": "completely unrelated zzz qqq", "store_id": "3001"}
        a = self._cand("A", ["LOS Alarm on WAN1"])
        b = self._cand("B", ["DHCP drops"])
        out = _apply_store_fingerprint_filter([a, b], alert)
        self.assertEqual(len(out), 2)  # never empty — fall back to ranked cohort

    def test_acadia_no_store_id_is_never_filtered(self):
        from backend.tier1_copilot.retrieval import _apply_store_fingerprint_filter

        alert = {"alert_type": "LOS Alarm on WAN1"}  # no store_id → Acadia
        a = self._cand("A", ["LOS Alarm on WAN1"])
        b = self._cand("B", ["DHCP drops"])
        out = _apply_store_fingerprint_filter([a, b], alert)
        self.assertEqual(len(out), 2)  # both kept — Acadia cohort untouched


if __name__ == "__main__":
    unittest.main()
