"""Sprint 9 — extractor JSON parser + happy/retry path (offline)."""
from __future__ import annotations

import json
import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _empty_catalog():
    from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
    return IntakeCatalogs()


class ParseExtractionJsonTests(unittest.TestCase):
    def test_plain_array_parses(self):
        from backend.tier1_copilot.intake.extractor import parse_extraction_json
        out = parse_extraction_json('[{"severity":"P2"}]')
        self.assertEqual(out, [{"severity": "P2"}])

    def test_strips_code_fence(self):
        from backend.tier1_copilot.intake.extractor import parse_extraction_json
        out = parse_extraction_json("```json\n[{\"a\":1}]\n```")
        self.assertEqual(out, [{"a": 1}])

    def test_recovers_from_surrounding_prose(self):
        from backend.tier1_copilot.intake.extractor import parse_extraction_json
        out = parse_extraction_json("Here you go: [{\"a\":1}, {\"b\":2}] thanks")
        self.assertEqual(out, [{"a": 1}, {"b": 2}])

    def test_recovers_trailing_comma(self):
        from backend.tier1_copilot.intake.extractor import parse_extraction_json
        out = parse_extraction_json('[{"a":1},]')
        self.assertEqual(out, [{"a": 1}])

    def test_object_only_returns_none(self):
        from backend.tier1_copilot.intake.extractor import parse_extraction_json
        # spec demands an array — bare objects are rejected.
        self.assertIsNone(parse_extraction_json('{"a":1}'))

    def test_unrecoverable_returns_none(self):
        from backend.tier1_copilot.intake.extractor import parse_extraction_json
        self.assertIsNone(parse_extraction_json("absolutely not json"))


class ExtractCandidatesTests(unittest.TestCase):
    def test_happy_path_builds_candidates(self):
        from backend.tier1_copilot.intake.extractor import extract_candidates
        payload = json.dumps([
            {
                "severity": "P2",
                "asset_name": "VPN Service",
                "alert_type": "Login Failure",
                "customer": "Aetheris",
                "evidence": {
                    "severity": "20 users affected",
                    "asset_name": "VPN",
                    "alert_type": "cannot login",
                    "customer": "Aetheris",
                },
            },
            {
                "severity": "P3",
                "asset_name": "VPN Service",
                "alert_type": "Slow Login",
                "customer": None,
                "evidence": {
                    "severity": "minor",
                    "asset_name": "VPN",
                    "alert_type": "slow",
                    "customer": None,
                },
            },
        ])
        out = extract_candidates(
            raw_text="20 users in Chicago can't connect to VPN since 8 AM",
            source_type="email",
            catalogs=_empty_catalog(),
            n_candidates=4,
            llm_invoke=lambda _p: payload,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0].severity, "P2")
        self.assertEqual(out[1].severity, "P3")
        self.assertEqual(out[0].evidence.severity, "20 users affected")

    def test_retry_on_first_malformed(self):
        from backend.tier1_copilot.intake.extractor import extract_candidates
        good = '[{"severity":"P2"}]'

        calls = {"n": 0}

        def invoker(prompt):
            calls["n"] += 1
            return "garbage" if calls["n"] == 1 else good

        out = extract_candidates(
            raw_text="something",
            source_type="email",
            catalogs=_empty_catalog(),
            n_candidates=4,
            llm_invoke=invoker,
        )
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].severity, "P2")

    def test_double_failure_returns_empty(self):
        from backend.tier1_copilot.intake.extractor import extract_candidates
        out = extract_candidates(
            raw_text="x",
            source_type="email",
            catalogs=_empty_catalog(),
            n_candidates=4,
            llm_invoke=lambda _p: "garbage",
        )
        self.assertEqual(out, [])

    def test_invalid_severity_demoted(self):
        from backend.tier1_copilot.intake.extractor import extract_candidates
        out = extract_candidates(
            raw_text="x",
            source_type="email",
            catalogs=_empty_catalog(),
            n_candidates=4,
            llm_invoke=lambda _p: '[{"severity":"P9","asset_name":"x"}]',
        )
        self.assertEqual(len(out), 1)
        self.assertIsNone(out[0].severity)


if __name__ == "__main__":
    unittest.main(verbosity=2)
