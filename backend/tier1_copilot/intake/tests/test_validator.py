"""Sprint 9 — validator (severity enum, fuzzy match, confidence)."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _build_catalog():
    from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
    cat = IntakeCatalogs()
    cat.rebuild_from_rows([
        {"metadata_json": {
            "Metadata": {
                "Affected_Assets": ["NY4-CORE-RTR-01"],
                "Target_Service": "Edge Routing",
                "customer_name": "Aetheris Corp",
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": "BGP peer flapping",
            },
        }},
        {"metadata_json": {
            "Metadata": {
                "Affected_Assets": ["V-Desktop Environment"],
                "customer_name": "Phoenix Industries",
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": "Desktop slowness",
            },
        }},
    ])
    return cat


class SeverityEnumTests(unittest.TestCase):
    def test_valid_severity_passes(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(severity="P2")
        out = validate_candidate(c, cat)
        self.assertEqual(out.severity, "P2")
        self.assertEqual(out.validation.severity_status, "valid")

    def test_invalid_severity_demoted(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(severity=None)
        c.severity = "P9"  # bypass pydantic for the demotion path
        out = validate_candidate(c, cat)
        self.assertIsNone(out.severity)
        self.assertEqual(out.validation.severity_status, "invalid")


class FuzzyAssetMatchTests(unittest.TestCase):
    def test_exact_alias_canonicalises(self):
        from backend.tier1_copilot.intake.validator import fuzzy_match_asset
        cat = _build_catalog()
        self.assertEqual(
            fuzzy_match_asset("NY4-CORE-RTR-12", cat), "ny4-core-rtr",
        )

    def test_above_threshold_match(self):
        from backend.tier1_copilot.intake.validator import fuzzy_match_asset
        cat = _build_catalog()
        # Slight typo — fuzzy match should still hit ny4-core-rtr.
        self.assertEqual(
            fuzzy_match_asset("NY4-CORE-RTR", cat), "ny4-core-rtr",
        )

    def test_below_threshold_returns_none(self):
        from backend.tier1_copilot.intake.validator import fuzzy_match_asset
        cat = _build_catalog()
        self.assertIsNone(fuzzy_match_asset("xyz123-totally-unrelated", cat))


class CustomerMatchTests(unittest.TestCase):
    def test_legal_suffix_stripped_in_match(self):
        from backend.tier1_copilot.intake.validator import fuzzy_match_customer
        cat = _build_catalog()
        self.assertEqual(
            fuzzy_match_customer("Aetheris Inc", cat), "aetheris",
        )

    def test_unknown_customer_returns_none(self):
        from backend.tier1_copilot.intake.validator import fuzzy_match_customer
        cat = _build_catalog()
        self.assertIsNone(fuzzy_match_customer("XYZ Holdings", cat))


class CandidateValidationTests(unittest.TestCase):
    def test_high_confidence_when_all_match(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(
            severity="P2",
            asset_name="NY4-CORE-RTR-12",
            alert_type="BGP peer flapping",
            customer="Aetheris Corp",
        )
        out = validate_candidate(c, cat)
        self.assertEqual(out.confidence, "High")
        self.assertEqual(out.validation.asset_status, "matched")
        self.assertEqual(out.validation.customer_status, "matched")
        # Sprint 9.2 — signatures are now lowercase + whitespace-
        # collapsed so the diversifier's case-insensitive dedup works.
        self.assertEqual(out.diversity_signature.split("|")[0], "p2")

    def test_medium_confidence_no_customer(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(
            severity="P2",
            asset_name="NY4-CORE-RTR-12",
            alert_type="BGP peer flapping",
            customer=None,
        )
        out = validate_candidate(c, cat)
        self.assertEqual(out.confidence, "Medium")

    def test_low_confidence_when_asset_unknown(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(
            severity="P3",
            asset_name="xyz-asset-not-in-corpus",
            alert_type="BGP peer flapping",
            customer="Aetheris Corp",
        )
        out = validate_candidate(c, cat)
        self.assertEqual(out.validation.asset_status, "unknown")
        self.assertEqual(out.confidence, "Low")


class SubstringGroundingTests(unittest.TestCase):
    """Sprint 9.2 — anti-hallucination via evidence substring grounding."""

    def test_evidence_must_be_substring_of_raw_text(self):
        """Field is nulled if `evidence` isn't a substring of raw_text.
        Catches the Sprint 9 hint-contamination bug where the LLM
        claimed an asset name not literally present in the email."""
        from backend.tier1_copilot.intake.schemas import (
            FieldEvidence, ValidatedCandidate,
        )
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        raw = "Email about NY4-CORE-RTR-01 going down."
        c = ValidatedCandidate(
            severity="P2",
            asset_name="some-other-asset",
            alert_type="bgp peer flapping",
            evidence=FieldEvidence(
                severity="critical",                # not in raw → grounded False
                asset_name="completely fabricated text",  # not a substring
                alert_type="bgp",                   # also not in raw
            ),
        )
        out = validate_candidate(c, cat, raw_text=raw)
        # Asset evidence wasn't a substring of raw → field nulled
        # before catalog matching ran.
        self.assertIsNone(out.asset_name)
        self.assertEqual(out.validation.asset_status, "absent")
        # Confidence drops to Low because evidence grounding failed.
        self.assertEqual(out.confidence, "Low")

    def test_evidence_substring_match_case_insensitive(self):
        """Lowercase evidence matches uppercase raw text + canonical
        form picks up the catalog customer."""
        from backend.tier1_copilot.intake.schemas import (
            FieldEvidence, ValidatedCandidate,
        )
        from backend.tier1_copilot.intake.validator import validate_candidate
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows([
            {"metadata_json": {
                "Metadata": {
                    "Affected_Assets": ["NY4-CORE-RTR-01"],
                    "Target_Service": "Edge Routing",
                    "customer_name": "Phoenix Quant Systems",
                },
                "Symptom_Solution_Mapping": {
                    "Detected_Symptom": "BGP session flap (BFD down)",
                },
            }},
        ])
        raw = (
            "P1 critical: Customer PHOENIX QUANT SYSTEMS reports "
            "BGP flap on NY4-CORE-RTR-01"
        )
        c = ValidatedCandidate(
            severity="P1",
            asset_name="NY4-CORE-RTR-01",
            alert_type="BGP flap",
            customer="Phoenix Quant Systems",
            evidence=FieldEvidence(
                severity="P1",
                asset_name="NY4-CORE-RTR-01",
                alert_type="BGP flap",
                customer="PHOENIX QUANT SYSTEMS",
            ),
        )
        out = validate_candidate(c, cat, raw_text=raw)
        self.assertEqual(out.canonical_form.customer, "phoenix quant systems")
        self.assertEqual(out.validation.customer_status, "matched")

    def test_hint_contamination_regression(self):
        """Sprint 9.0 bug: the LLM picked `v-bay-core-rtr` from the
        catalog hints when the email actually mentioned
        `NY4-CORE-RTR-01`. The bad value's evidence was the bad value
        itself (not in raw_text), so substring grounding nulls it
        before catalog lookup."""
        from backend.tier1_copilot.intake.schemas import (
            FieldEvidence, ValidatedCandidate,
        )
        from backend.tier1_copilot.intake.validator import validate_candidate
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows([
            {"metadata_json": {"Metadata": {
                "Affected_Assets": ["NY4-CORE-RTR-01"],
                "customer_name": "Phoenix Quant Systems",
            }, "Symptom_Solution_Mapping": {
                "Detected_Symptom": "bgp flap",
            }}},
            {"metadata_json": {"Metadata": {
                "Affected_Assets": ["v-bay-core-rtr-01"],
                "customer_name": "VBay Inc",
            }, "Symptom_Solution_Mapping": {
                "Detected_Symptom": "bgp flap",
            }}},
        ])
        raw = "BGP flap on NY4-CORE-RTR-01 router."
        c = ValidatedCandidate(
            severity=None,
            asset_name="v-bay-core-rtr",                  # bad hallucination
            alert_type="bgp flap",
            evidence=FieldEvidence(
                asset_name="v-bay-core-rtr",              # NOT in raw → rejected
                alert_type="BGP flap",                    # in raw — fine
            ),
        )
        out = validate_candidate(c, cat, raw_text=raw)
        # Hallucinated asset rejected before catalog lookup; status
        # collapses to "absent" rather than wrongly "matched".
        self.assertIsNone(out.asset_name)
        self.assertEqual(out.validation.asset_status, "absent")

    def test_short_asset_does_not_collapse_to_unrelated_family(self):
        """Threshold 0.85 prevents NY4-CORE-RTR-01 (the email's actual
        asset) from being downgraded to v-bay-core-rtr (a near-miss at
        ratio 0.69) just because they share core/rtr tokens."""
        from backend.tier1_copilot.intake.schemas import (
            FieldEvidence, ValidatedCandidate,
        )
        from backend.tier1_copilot.intake.validator import validate_candidate
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        cat = IntakeCatalogs()
        cat.rebuild_from_rows([
            {"metadata_json": {"Metadata": {
                "Affected_Assets": ["NY4-CORE-RTR-01"],
                "customer_name": "Phoenix",
            }, "Symptom_Solution_Mapping": {"Detected_Symptom": "bgp flap"}}},
            {"metadata_json": {"Metadata": {
                "Affected_Assets": ["v-bay-core-rtr-01"],
                "customer_name": "VBay",
            }, "Symptom_Solution_Mapping": {"Detected_Symptom": "bgp flap"}}},
        ])
        raw = "P1 critical: Issue on NY4-CORE-RTR-01."
        c = ValidatedCandidate(
            severity="P1",
            asset_name="NY4-CORE-RTR-01",
            evidence=FieldEvidence(
                severity="P1",
                asset_name="NY4-CORE-RTR-01",
            ),
        )
        out = validate_candidate(c, cat, raw_text=raw)
        canonical = (out.canonical_form.asset_name or "").lower()
        self.assertIn("ny4", canonical)
        self.assertNotIn("v-bay", canonical)


class CanonicalFormTests(unittest.TestCase):
    """Sprint 9.1 — `canonical_form` is the form-prefill payload that
    auto-fills the Tier-1 form on "Use this interpretation". It carries
    catalog-canonical values when matched, falls back to raw LLM
    strings when unknown, and severity is enum-strict (None on invalid
    LLM input)."""

    def test_canonical_form_uses_matched_canonical_when_asset_matches(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(
            severity="P2",
            asset_name="NY4-CORE-RTR-12",       # raw — fuzzy-matches family
            alert_type="BGP peer flapping",
            customer="Aetheris Corp",
        )
        out = validate_candidate(c, cat)
        # Canonical asset family wins — what gets pushed into the form.
        self.assertEqual(out.canonical_form.asset_name, "ny4-core-rtr")
        self.assertEqual(out.canonical_form.severity, "P2")
        self.assertEqual(out.canonical_form.customer, "aetheris")
        self.assertEqual(out.validation.asset_status, "matched")

    def test_canonical_form_falls_back_to_raw_when_asset_unknown(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(
            severity="P3",
            asset_name="Brand New Asset",       # not in catalog
            alert_type="BGP peer flapping",
            customer="Aetheris Corp",
        )
        out = validate_candidate(c, cat)
        # Engineer can still edit the raw value in the form — better
        # than blanking it.
        self.assertEqual(out.canonical_form.asset_name, "Brand New Asset")
        self.assertEqual(out.validation.asset_status, "unknown")

    def test_canonical_form_severity_uses_enum_only(self):
        from backend.tier1_copilot.intake.schemas import ValidatedCandidate
        from backend.tier1_copilot.intake.validator import validate_candidate
        cat = _build_catalog()
        c = ValidatedCandidate(severity=None, asset_name="A", alert_type="B")
        # Bypass pydantic to push a non-enum value through the demotion
        # path — same shape the extractor uses when the LLM returns
        # something like "urgent".
        c.severity = "urgent"
        out = validate_candidate(c, cat)
        self.assertIsNone(out.canonical_form.severity)
        self.assertEqual(out.validation.severity_status, "invalid")


if __name__ == "__main__":
    unittest.main(verbosity=2)
