"""Sprint 7 — ranking boost + asset-family tests (offline)."""
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


class DeriveAssetFamilyTests(unittest.TestCase):
    def test_hyphenated_numeric_suffix(self):
        from backend.tier1_copilot.retrieval import derive_asset_family
        self.assertEqual(derive_asset_family("NY4-CORE-RTR-01"), "ny4-core-rtr")
        self.assertEqual(derive_asset_family("SFO-EDGE-FW-02"), "sfo-edge-fw")

    def test_compound_suffix(self):
        from backend.tier1_copilot.retrieval import derive_asset_family
        # The regex strips the trailing `[-_]?\d+$`, leaving the session
        # portion intact until its OWN numeric suffix — the function is
        # one-pass, so "V-Desktop-Session-17" → "v-desktop-session".
        self.assertEqual(
            derive_asset_family("V-Desktop-Session-17"),
            "v-desktop-session",
        )

    def test_underscore_suffix(self):
        from backend.tier1_copilot.retrieval import derive_asset_family
        self.assertEqual(derive_asset_family("switch_stack_07"), "switch_stack")

    def test_no_suffix(self):
        from backend.tier1_copilot.retrieval import derive_asset_family
        # No trailing digits → lowercased as-is.
        self.assertEqual(derive_asset_family("CoreSwitch"), "coreswitch")

    def test_blank_returns_none(self):
        from backend.tier1_copilot.retrieval import derive_asset_family
        self.assertIsNone(derive_asset_family(""))
        self.assertIsNone(derive_asset_family("   "))
        self.assertIsNone(derive_asset_family(None))


class WeightsSumToOneTests(unittest.TestCase):
    """Spec §3 invariant — TIER1_RANKING_WEIGHTS must sum to exactly 1.00.
    If a future engineer tweaks a weight, this test catches it."""

    def test_weights_sum_to_one(self):
        from backend.config import settings
        total = sum(settings.TIER1_RANKING_WEIGHTS.values())
        # Round to 3dp to tolerate float noise from the declaration.
        self.assertAlmostEqual(total, 1.00, places=3)


class SameCustomerBoostTests(unittest.TestCase):
    """With the Sprint 7 flag on, a ticket whose customer matches the
    alert's customer should beat an otherwise-identical ticket with a
    different customer."""

    def _candidate(self, *, customer: str, chunk_id: str) -> dict:
        return {
            "chunk_id": chunk_id,
            "metadata_json": {
                "Metadata": {
                    "Target_Service": "V-Desktop Environment",
                    "customer_name": customer,
                    "Resolution_Quality_Score": 3,
                    "component_category": "Citrix",
                },
                "Symptom_Solution_Mapping": {
                    "Detected_Symptom": "Desktop Slowness",
                },
            },
            "fingerprints_text": "ctx-slow-01",
            "component_category": "citrix",
            "asset_family": "v-desktop",
            "_vector_sim": 0.5,
        }

    def test_same_customer_beats_different_customer(self):
        from backend.config import settings
        from backend.tier1_copilot.retrieval import _weighted_rank

        alert_input = {
            "asset_name": "V-Desktop Environment",
            "alert_type": "Desktop Slowness",
            "technology": "Citrix",
            "error_code": "CTX-SLOW-01",
            "customer": "Aetheris Corp",
        }

        cand_same = self._candidate(customer="Aetheris Corp", chunk_id="c_same")
        cand_diff = self._candidate(customer="Blueshift Ltd", chunk_id="c_diff")

        with mock.patch.object(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", True):
            ranked = _weighted_rank(
                [cand_diff, cand_same], alert_input, {},
            )

        self.assertEqual(ranked[0]["chunk_id"], "c_same")
        self.assertGreater(
            ranked[0]["_score_components"]["same_customer_boost"],
            ranked[1]["_score_components"]["same_customer_boost"],
        )

    def test_flag_off_keeps_sprint6_weights(self):
        """When Sprint 7 flag is OFF, the customer-equal vs customer-diff
        candidates must score IDENTICALLY — Sprint 6 weights have no
        customer-boost component."""
        from backend.config import settings
        from backend.tier1_copilot.retrieval import _weighted_rank

        alert_input = {
            "asset_name": "V-Desktop Environment",
            "alert_type": "Desktop Slowness",
            "technology": "Citrix",
            "error_code": "CTX-SLOW-01",
            "customer": "Aetheris Corp",
        }
        cand_same = self._candidate(customer="Aetheris Corp", chunk_id="c_same")
        cand_diff = self._candidate(customer="Blueshift Ltd", chunk_id="c_diff")

        with mock.patch.object(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", False):
            ranked = _weighted_rank(
                [cand_diff, cand_same], alert_input, {},
            )
        # Same feature tokens → same score.
        self.assertEqual(
            ranked[0]["final_score"], ranked[1]["final_score"],
        )


class SameAssetFamilyBoostTests(unittest.TestCase):
    def test_asset_family_boost_only_on_match(self):
        from backend.config import settings
        from backend.tier1_copilot.retrieval import _weighted_rank

        alert_input = {
            "asset_name": "NY4-CORE-RTR-07",
            "alert_type": "BGP flap",
            "customer": "X",
        }

        cand_match = {
            "chunk_id": "cm",
            "metadata_json": {
                "Metadata": {
                    "Target_Service": "edge-router",
                    "Affected_Assets": ["NY4-CORE-RTR-12"],
                    "Resolution_Quality_Score": 3,
                },
                "Symptom_Solution_Mapping": {"Detected_Symptom": "bgp flap"},
            },
            "fingerprints_text": "bgp-5-adjchange",
            "component_category": "bgp",
            "asset_family": "ny4-core-rtr",
            "_vector_sim": 0.4,
        }
        cand_nomatch = {
            "chunk_id": "cn",
            "metadata_json": {
                "Metadata": {
                    "Target_Service": "edge-router",
                    "Affected_Assets": ["SFO-EDGE-FW-02"],
                    "Resolution_Quality_Score": 3,
                },
                "Symptom_Solution_Mapping": {"Detected_Symptom": "bgp flap"},
            },
            "fingerprints_text": "bgp-5-adjchange",
            "component_category": "bgp",
            "asset_family": "sfo-edge-fw",
            "_vector_sim": 0.4,
        }

        with mock.patch.object(settings, "LOGIQ_TIER1_PROGRESSIVE_BACKEND", True):
            ranked = _weighted_rank(
                [cand_nomatch, cand_match], alert_input, {},
            )
        self.assertEqual(ranked[0]["chunk_id"], "cm")
        self.assertEqual(
            ranked[0]["_score_components"]["same_asset_family"], 1.0,
        )
        self.assertEqual(
            ranked[1]["_score_components"]["same_asset_family"], 0.0,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
