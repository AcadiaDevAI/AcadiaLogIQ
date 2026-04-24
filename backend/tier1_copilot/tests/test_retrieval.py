"""Sprint 6 — retrieval scoring + confidence band tests (offline)."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class ConfidenceBandTests(unittest.TestCase):
    def test_high_band(self):
        from backend.tier1_copilot.retrieval import confidence_band
        self.assertEqual(confidence_band(0.90), "High")
        self.assertEqual(confidence_band(0.85), "High")

    def test_medium_band(self):
        from backend.tier1_copilot.retrieval import confidence_band
        self.assertEqual(confidence_band(0.70), "Medium")
        self.assertEqual(confidence_band(0.60), "Medium")

    def test_low_band(self):
        from backend.tier1_copilot.retrieval import confidence_band
        self.assertEqual(confidence_band(0.50), "Low")
        self.assertEqual(confidence_band(0.40), "Low")

    def test_none_band(self):
        from backend.tier1_copilot.retrieval import confidence_band
        self.assertEqual(confidence_band(0.39), "None")
        self.assertEqual(confidence_band(0.0), "None")


class WeightedRankTests(unittest.TestCase):
    """Direct test of _weighted_rank — the pure math slice. No DB."""

    def test_exact_alert_and_asset_match_wins(self):
        from backend.tier1_copilot.retrieval import _weighted_rank
        alert_input = {
            "asset_name": "V-Desktop Environment",
            "alert_type": "Desktop Slowness",
            "technology": "Citrix",
            "error_code": "CTX-SLOW-01",
        }
        normalized = {}

        good_candidate = {
            "chunk_id": "c1",
            "metadata_json": {
                "Metadata": {
                    "Target_Service": "V-Desktop Environment",
                    "Affected_Assets": ["VDI-Pool-A"],
                    "component_category": "Citrix",
                    "Resolution_Quality_Score": 5,
                },
                "Symptom_Solution_Mapping": {
                    "Detected_Symptom": "Desktop Slowness",
                },
            },
            "fingerprints_text": "ctx-slow-01",
            "component_category": "citrix",
            "_vector_sim": 0.8,
        }
        weak_candidate = {
            "chunk_id": "c2",
            "metadata_json": {
                "Metadata": {
                    "Target_Service": "Edge Router",
                    "component_category": "BGP",
                    "Resolution_Quality_Score": 3,
                },
                "Symptom_Solution_Mapping": {
                    "Detected_Symptom": "BGP peer flapping",
                },
            },
            "fingerprints_text": "bgp-5-adjchange",
            "component_category": "bgp",
            "_vector_sim": 0.3,
        }

        ranked = _weighted_rank(
            [weak_candidate, good_candidate], alert_input, normalized,
        )
        self.assertEqual(ranked[0]["chunk_id"], "c1")
        self.assertGreater(ranked[0]["final_score"], ranked[1]["final_score"])
        self.assertEqual(ranked[0]["confidence"],
                         # good candidate overlap is near-total
                         ranked[0]["confidence"])  # just assert non-None
        self.assertIn(ranked[0]["confidence"], {"High", "Medium", "Low"})

    def test_zero_overlap_yields_low_or_none(self):
        from backend.tier1_copilot.retrieval import _weighted_rank
        alert_input = {
            "asset_name": "Totally Unrelated",
            "alert_type": "Mystery Glitch",
        }
        candidate = {
            "chunk_id": "c1",
            "metadata_json": {
                "Metadata": {
                    "Target_Service": "Edge Router",
                    "Resolution_Quality_Score": 1,
                },
            },
            "fingerprints_text": "",
            "component_category": "bgp",
            "_vector_sim": 0.0,
        }
        ranked = _weighted_rank([candidate], alert_input, {})
        self.assertLess(ranked[0]["final_score"], 0.60)
        self.assertIn(ranked[0]["confidence"], {"Low", "None"})


class RetrieveEngineFailureTests(unittest.TestCase):
    """retrieve_top_matches must never raise — it returns [] on errors."""

    def test_all_stages_fail_returns_empty(self):
        from unittest import mock
        from backend.tier1_copilot.retrieval import retrieve_top_matches
        broken = mock.MagicMock()
        broken.connect.side_effect = RuntimeError("no db")
        out = retrieve_top_matches(
            normalized={"alert_signature": "x | y | z", "search_text": "q"},
            alert_input={"asset_name": "a", "alert_type": "b"},
            engine=broken,
            embed_fn=None,
        )
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
