"""Sprint 6 — normalizer tests (offline)."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _make_req(**overrides):
    from backend.tier1_copilot.schemas import Tier1AnalyzeRequest
    base = dict(
        severity="P2",
        asset_name="V-Desktop Environment",
        alert_type="Desktop Slowness",
        session_id="abc123",
    )
    base.update(overrides)
    return Tier1AnalyzeRequest(**base)


class SignatureConsistencyTests(unittest.TestCase):
    def test_signature_is_lowercase_pipe_joined(self):
        from backend.tier1_copilot.normalizer import normalize_alert
        req = _make_req()
        out = normalize_alert(req, alias_dict=None)
        self.assertEqual(
            out["alert_signature"],
            "p2 | v-desktop environment | desktop slowness",
        )

    def test_signature_whitespace_collapsed(self):
        from backend.tier1_copilot.normalizer import normalize_alert
        req = _make_req(
            asset_name="  V-Desktop   Environment  ",
            alert_type="Desktop    Slowness",
        )
        out = normalize_alert(req, alias_dict=None)
        self.assertEqual(
            out["alert_signature"],
            "p2 | v-desktop environment | desktop slowness",
        )

    def test_hash_differs_with_optional_fields(self):
        from backend.tier1_copilot.normalizer import normalize_alert
        a = normalize_alert(_make_req(), alias_dict=None)
        b = normalize_alert(_make_req(customer="Aetheris Corp"), alias_dict=None)
        # Same required triple, different optionals → different hashes.
        self.assertEqual(a["alert_signature"], b["alert_signature"])
        self.assertNotEqual(a["signature_hash"], b["signature_hash"])

    def test_hash_stable_across_calls(self):
        from backend.tier1_copilot.normalizer import normalize_alert
        a = normalize_alert(_make_req(), alias_dict=None)
        b = normalize_alert(_make_req(), alias_dict=None)
        self.assertEqual(a["signature_hash"], b["signature_hash"])


class AliasExpansionTests(unittest.TestCase):
    def test_expansion_from_dict(self):
        from backend.tier1_copilot.alias_dictionary import AliasDictionary
        from backend.tier1_copilot.normalizer import normalize_alert

        ad = AliasDictionary()
        ad.rebuild_from_rows([{
            "metadata_json": {
                "Metadata": {
                    "component_category": "Citrix",
                    "Fingerprints": ["CTX-SLOW-01"],
                    "Target_Service": "V-Desktop Environment",
                    "Affected_Assets": ["VDI-Pool-A"],
                },
                "Symptom_Solution_Mapping": {"Detected_Symptom": "Desktop Slowness"},
            }
        }])

        req = _make_req(technology="Citrix")
        out = normalize_alert(req, alias_dict=ad)
        # Citrix was the tech; its alias 'ctx-slow-01' should be pulled in.
        self.assertIn("ctx-slow-01", out["search_terms"])

    def test_no_alias_dict_means_base_terms_only(self):
        from backend.tier1_copilot.normalizer import normalize_alert
        out = normalize_alert(_make_req(), alias_dict=None)
        self.assertIn("v-desktop environment", out["search_terms"])
        self.assertIn("desktop slowness", out["search_terms"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
