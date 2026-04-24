"""Sprint 6 — alias dictionary build + lookup (offline)."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


_SAMPLE_ROWS = [
    {
        "metadata_json": {
            "Metadata": {
                "component_category": "Citrix",
                "Fingerprints": ["CTX-SLOW-01", "CTX-HUNG-02"],
                "Target_Service": "V-Desktop Environment",
                "Affected_Assets": ["VDI-Pool-A"],
            },
            "Symptom_Solution_Mapping": {"Detected_Symptom": "Desktop Slowness"},
            "Knowledge_Base": [{
                "semantic_unit_educational": {
                    "knowledge_id": "KB-CTX-01",
                    "related_signals": ["session_launch_lag"],
                }
            }],
        },
    },
    {
        "metadata_json": {
            "Metadata": {
                "component_category": "BGP",
                "Fingerprints": ["BGP-5-ADJCHANGE"],
                "Target_Service": "Edge Router",
            }
        }
    },
]


class AliasBuildTests(unittest.TestCase):
    def test_build_from_sample_rows(self):
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        ad = AliasDictionary()
        ad.rebuild_from_rows(_SAMPLE_ROWS)
        self.assertTrue(ad.is_built())
        self.assertGreater(ad.term_count(), 0)

    def test_component_to_fingerprint_linkage(self):
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        ad = AliasDictionary()
        ad.rebuild_from_rows(_SAMPLE_ROWS)
        self.assertIn("ctx-slow-01", ad.get_aliases("citrix"))
        self.assertIn("citrix", ad.get_aliases("ctx-slow-01"))

    def test_target_to_asset_linkage(self):
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        ad = AliasDictionary()
        ad.rebuild_from_rows(_SAMPLE_ROWS)
        self.assertIn("vdi-pool-a", ad.get_aliases("v-desktop environment"))

    def test_related_signal_linkage(self):
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        ad = AliasDictionary()
        ad.rebuild_from_rows(_SAMPLE_ROWS)
        self.assertIn(
            "session_launch_lag",
            ad.get_aliases("ctx-slow-01"),
        )

    def test_empty_term_returns_empty_set(self):
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        ad = AliasDictionary()
        ad.rebuild_from_rows(_SAMPLE_ROWS)
        self.assertEqual(ad.get_aliases(""), set())
        self.assertEqual(ad.get_aliases("nonexistent-term"), set())

    def test_rebuild_is_atomic(self):
        """A second rebuild fully replaces the prior map."""
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        ad = AliasDictionary()
        ad.rebuild_from_rows(_SAMPLE_ROWS)
        ad.rebuild_from_rows([])
        self.assertEqual(ad.term_count(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
