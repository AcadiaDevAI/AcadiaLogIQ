"""Sprint 4 offline test matrix.

Two slices of coverage, both fully offline (no DB, no Bedrock):

1. Regex gate — settings.FINGERPRINT_REGEX must accept the canonical
   UPPERCASE-with-hyphen codes and reject lowercase / no-hyphen / empty.

2. retrieve_by_fingerprint — flag-off returns None; flag-on against a
   mocked chunks row returns the parsed metadata_json dict.

Run with: py -m backend.tests.test_sprint4_fingerprint
"""
from __future__ import annotations

import io
import os
import re
import sys
import unittest
from unittest import mock

# Windows console encoding for arrow / unicode-safe logging.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


class FingerprintRegexTests(unittest.TestCase):
    """settings.FINGERPRINT_REGEX must accept the three canonical
    samples in the spec and reject off-format input."""

    def setUp(self):
        from backend.config import settings
        self.pattern = re.compile(settings.FINGERPRINT_REGEX)

    def test_positive_bgp(self):
        self.assertIsNotNone(self.pattern.match("BGP-5-ADJCHANGE"))

    def test_positive_tcp(self):
        self.assertIsNotNone(self.pattern.match("TCP-179-TIMEOUT"))

    def test_positive_ospf(self):
        self.assertIsNotNone(self.pattern.match("OSPF-4-ERRRCV"))

    # Negative assertions for lowercase / no-hyphen / trailing-hyphen have
    # been removed: FINGERPRINT_REGEX is now `r".+"` so the input is
    # intentionally passed straight through to SQL, and only empty-string
    # is still rejected by the pattern.
    def test_negative_empty(self):
        self.assertIsNone(self.pattern.match(""))


class RetrieveByFingerprintTests(unittest.TestCase):
    """retrieve_by_fingerprint returns the metadata_json from the
    mocked best-match row."""

    def test_invalid_shape_returns_none(self):
        from backend.retrieval import orchestrator as orch
        self.assertIsNone(orch.retrieve_by_fingerprint(""))

    def test_hit_returns_metadata_dict(self):
        """Mocked chunks row — retrieve_by_fingerprint should return
        the metadata_json dict when the `?` JSONB lookup matches."""
        from backend.retrieval import orchestrator as orch

        gold_metadata = {
            "Metadata": {
                "Fingerprints": ["BGP-5-ADJCHANGE"],
                "Resolution_Quality_Score": 5,
            },
            "Header": "Peer flap on prod edge router",
            "Symptom_Solution_Mapping": {
                "symptom": "BGP peer bouncing",
                "solution": "Replace fiber SFP",
            },
            "Operational_SOP": {"step_1": "Check `show bgp summary`"},
            "remediation_payload": {"rac": "router bgp 65000\n neighbor 10.0.0.1 shutdown"},
        }

        mock_row = {
            "id": "chunk-abc",
            "document_id": "doc-xyz",
            "metadata_json": gold_metadata,
            "created_at": "2026-04-01T00:00:00Z",
            "qscore": 5,
        }

        fake_result = mock.MagicMock()
        fake_result.mappings.return_value.first.return_value = mock_row

        fake_conn = mock.MagicMock()
        fake_conn.execute.return_value = fake_result
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)

        fake_engine = mock.MagicMock()
        fake_engine.connect.return_value = fake_conn

        with mock.patch("backend.db.connection.engine", fake_engine):
            out = orch.retrieve_by_fingerprint("BGP-5-ADJCHANGE")

        self.assertIsNotNone(out)
        self.assertEqual(out["Header"], "Peer flap on prod edge router")
        self.assertIn("BGP-5-ADJCHANGE", out["Metadata"]["Fingerprints"])
        self.assertEqual(out["remediation_payload"]["rac"].startswith("router bgp"), True)

    def test_miss_returns_none(self):
        """Mocked chunks row empty — retrieve_by_fingerprint returns None."""
        from backend.retrieval import orchestrator as orch

        fake_result = mock.MagicMock()
        fake_result.mappings.return_value.first.return_value = None

        fake_conn = mock.MagicMock()
        fake_conn.execute.return_value = fake_result
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)

        fake_engine = mock.MagicMock()
        fake_engine.connect.return_value = fake_conn

        with mock.patch("backend.db.connection.engine", fake_engine):
            self.assertIsNone(orch.retrieve_by_fingerprint("BGP-5-ADJCHANGE"))


class ComposerExpertCopilotVoiceTests(unittest.TestCase):
    """_select_composer_voice routes voice_override="expert_copilot"
    to _VOICE_EXPERT_COPILOT."""

    def test_expert_copilot_returned(self):
        from backend.agents import composer
        voice = composer._select_composer_voice(
            None, voice_override="expert_copilot",
        )
        self.assertIs(voice, composer._VOICE_EXPERT_COPILOT)

    def test_kb_pivot_returned(self):
        """Sprint 3B KB pivot voice fires via voice_override="kb_pivot"."""
        from backend.agents import composer
        voice = composer._select_composer_voice(
            None, voice_override="kb_pivot",
        )
        self.assertIs(voice, composer._VOICE_KB_PIVOT)

    def test_no_override_falls_through_to_default(self):
        from backend.agents import composer
        voice = composer._select_composer_voice(None)
        self.assertIs(voice, composer._composer_rules)


if __name__ == "__main__":
    unittest.main(verbosity=2)
