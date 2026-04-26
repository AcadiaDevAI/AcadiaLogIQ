"""Sprint 9 — IntakeCatalogs build + normalisation (offline)."""
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
                "Affected_Assets": ["NY4-CORE-RTR-01"],
                "Target_Service": "Edge Routing",
                "customer_name": "Aetheris Corp",
                "Fingerprints": ["BGP-5-ADJCHANGE"],
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": "BGP peer flapping",
                "Origin_Event": "BFD session flap",
            },
            "doc_kind": "ticket",
        }
    },
    {
        "metadata_json": {
            "Metadata": {
                "Affected_Assets": ["V-Desktop Environment"],
                "customer_name": "Phoenix Industries",
                "Customer_Name": None,
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": "Desktop slowness in VDI pool",
            },
            "doc_kind": "ticket",
        }
    },
    {
        "metadata_json": {
            "Metadata": {
                "Affected_Assets": ["NY4-CORE-RTR-15"],
                "customer_name": "Aetheris Corp",
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": "BGP peer flapping",
            },
            "doc_kind": "ticket",
        }
    },
]


class DeriveAssetFamilyTests(unittest.TestCase):
    def test_strip_trailing_digits(self):
        from backend.tier1_copilot.intake.catalogs import derive_asset_family
        self.assertEqual(derive_asset_family("NY4-CORE-RTR-01"), "ny4-core-rtr")
        self.assertEqual(derive_asset_family("AUTH_SRV_07"), "auth_srv")

    def test_no_suffix(self):
        from backend.tier1_copilot.intake.catalogs import derive_asset_family
        self.assertEqual(
            derive_asset_family("V-Desktop Environment"),
            "v-desktop environment",
        )

    def test_blank(self):
        from backend.tier1_copilot.intake.catalogs import derive_asset_family
        self.assertIsNone(derive_asset_family(""))
        self.assertIsNone(derive_asset_family(None))


class CustomerNormalisationTests(unittest.TestCase):
    def test_strips_legal_suffix(self):
        from backend.tier1_copilot.intake.catalogs import normalise_customer
        self.assertEqual(normalise_customer("Aetheris Corp"), "aetheris")
        self.assertEqual(normalise_customer("Acme LLC."), "acme")

    def test_returns_none_for_empty(self):
        from backend.tier1_copilot.intake.catalogs import normalise_customer
        self.assertIsNone(normalise_customer(""))


class CatalogBuildTests(unittest.TestCase):
    def test_build_from_sample_rows(self):
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows(_SAMPLE_ROWS)
        self.assertTrue(cat.is_built())
        # Asset families
        self.assertIn("ny4-core-rtr", cat.asset_families)
        self.assertIn("v-desktop environment", cat.asset_families)
        # Customers (normalised)
        self.assertIn("aetheris", cat.customers)
        self.assertIn("phoenix industries", cat.customers)
        # Alert types (normalised)
        self.assertTrue(any("bgp peer flapping" in a for a in cat.alert_types))

    def test_top_n_returns_most_frequent_first(self):
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows(_SAMPLE_ROWS)
        # ny4-core-rtr appears twice → should top the list.
        top = cat.top_asset_families(5)
        self.assertEqual(top[0], "ny4-core-rtr")

    def test_alias_index_includes_raw_form(self):
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows(_SAMPLE_ROWS)
        # Raw asset name "NY4-CORE-RTR-01" → index → canonical family.
        self.assertEqual(
            cat.asset_family_index.get("ny4-core-rtr-01"), "ny4-core-rtr",
        )

    def test_health_snapshot_returns_sizes(self):
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows(_SAMPLE_ROWS)
        s, a, t, c = cat.health_snapshot()
        self.assertEqual(s, 4)
        self.assertGreater(a, 0)
        self.assertGreater(t, 0)
        self.assertGreater(c, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
