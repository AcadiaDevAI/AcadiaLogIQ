"""Sprint 7 — explain recommendation math + field match (offline)."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _top_match(**overrides):
    base = {
        "chunk_id": "c1",
        "metadata_json": {
            "Metadata": {
                "Incident_Number": "INC-1",
                "customer_name": "Aetheris Corp",
                "component_category": "Citrix",
                "priority": "P2",
                "Resolution_Quality_Score": 4,
                "Affected_Assets": ["V-Desktop Environment"],
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": "Desktop Slowness",
                "Primary_Fix": "Restart DDC",
            },
            "QA_Auditor_Feedback": {"Rework_Detected": False},
        },
        "final_score": 0.78,
        "_score_components": {
            "alert_type_match": 0.87,
            "asset_match": 0.90,
            "fingerprint_match": 0.75,
            "technology_match": 0.60,
            "vector_similarity": 0.62,
            "resolution_quality": 0.80,
            "recency": 0.45,
            "success_frequency": 0.80,
            "same_customer_boost": 1.00,
            "same_asset_family": 0.0,
        },
    }
    base.update(overrides)
    return base


class BuildExplainTests(unittest.TestCase):
    def test_score_breakdown_mirrors_components(self):
        from backend.tier1_copilot.diagnostics.explain_recommendation import (
            build_explain,
        )
        alert = {
            "severity": "P2",
            "asset_name": "V-Desktop Environment",
            "alert_type": "Desktop Slowness",
            "customer": "Aetheris Corp",
            "technology": "Citrix",
        }
        out = build_explain(alert_payload=alert, top_matches=[_top_match()])
        self.assertEqual(out.matched_incident, "INC-1")
        self.assertAlmostEqual(out.score_breakdown.final_score, 0.78)
        self.assertAlmostEqual(out.score_breakdown.same_customer_boost, 1.00)
        self.assertAlmostEqual(out.score_breakdown.asset_match, 0.90)

    def test_field_match_true_false_partial(self):
        from backend.tier1_copilot.diagnostics.explain_recommendation import (
            build_explain,
        )
        alert = {
            "severity": "P2",
            "asset_name": "V-Desktop Environment",
            "alert_type": "Slowness",                # partial overlap
            "customer": "Blueshift Ltd",             # different customer
            "technology": "Citrix",                  # exact
        }
        out = build_explain(alert_payload=alert, top_matches=[_top_match()])
        by_field = {f.field: f for f in out.matched_fields}
        self.assertIs(by_field["Asset"].match, True)
        self.assertIs(by_field["Severity"].match, True)
        self.assertIs(by_field["Technology"].match, True)
        self.assertIs(by_field["Customer"].match, False)
        self.assertEqual(by_field["Alert type"].match, "partial")

    def test_historical_success_counts(self):
        from backend.tier1_copilot.diagnostics.explain_recommendation import (
            build_explain,
        )
        good = _top_match()
        bad = _top_match(chunk_id="c2")
        bad["metadata_json"]["Metadata"]["Resolution_Quality_Score"] = 1
        bad["metadata_json"]["Metadata"]["Incident_Number"] = "INC-2"
        bad["metadata_json"]["QA_Auditor_Feedback"] = {"Rework_Detected": True}
        partial = _top_match(chunk_id="c3")
        partial["metadata_json"]["Metadata"]["Incident_Number"] = "INC-3"

        alert = {"severity": "P2", "customer": "Aetheris Corp"}
        out = build_explain(
            alert_payload=alert, top_matches=[good, bad, partial],
        )
        self.assertEqual(out.historical_success.total_similar, 3)
        self.assertEqual(out.historical_success.succeeded_count, 2)
        self.assertEqual(out.historical_success.failed_count, 1)
        self.assertEqual(out.historical_success.success_rate_percent, 67)
        self.assertEqual(out.historical_success.primary_fix, "Restart DDC")

    def test_empty_matches_returns_empty_breakdown(self):
        from backend.tier1_copilot.diagnostics.explain_recommendation import (
            build_explain,
        )
        out = build_explain(alert_payload={}, top_matches=[])
        self.assertIsNone(out.matched_incident)
        self.assertEqual(out.score_breakdown.final_score, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
