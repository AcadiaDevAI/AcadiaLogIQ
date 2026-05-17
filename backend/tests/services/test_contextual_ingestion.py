"""Sprint 11 — Forward-fix tests for contextual_ingestion_service.

These tests verify the rich-parent merge inside `_ingest_gold_ticket_json`.
The four parents the journey readers need (Executive_Sharable_RCA,
Incident_Summary, Forensic_Performance_Audit, Key_Contributors) plus the
two bonus parents (QA_Auditor_Feedback, ITIL_5_Why) must land on the
chunk's metadata_json when the source ticket carries them, and must be
absent (not null) when it does not.

Run with: py -m unittest backend.tests.services.test_contextual_ingestion
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _no_duplicate(**_kwargs):
    return None


def _no_candidates(**_kwargs):
    return []


def _gold_ticket(**overrides):
    """Build one gold-ticket dict with the journey-required parents present."""
    base = {
        "Metadata": {
            "Incident_Number": "INC-0001",
            "customer_name": "Acme Corp",
            "priority": "P3",
            "component_category": "Wireless Handheld",
            "ticket_status": "Resolved",
            "resolved_date": "2026-01-01",
        },
        "Header": "Handheld connection drops on warehouse floor",
        "Executive_Sharable_RCA": {
            "Resolution_Quality_Score": 4,
            "SLA_Target_Met": True,
            "Root_Cause_Technical_High_Level": "AP roaming threshold misaligned",
            "Resolution_Steps": [
                {"step": 1, "action": "Confirm channel plan"},
            ],
        },
        "Incident_Summary": {
            "INCIDENT": "Wireless handheld disconnects every 90s",
        },
        "Forensic_Performance_Audit": [
            {
                "Critical_Intervention": "Lowered min RSSI threshold",
                "Key_Movements_Timeline": [
                    {"t": "00:01", "event": "first_drop"},
                ],
            },
        ],
        "Key_Contributors": {
            "Key_Impact_Players": [
                {"name": "Engineer A", "Hero_Action": "Replaced AP firmware"},
            ],
        },
        "QA_Auditor_Feedback": {"Rework_Detected": False, "comment": "clean"},
        "ITIL_5_Why": [
            {"why": 1, "answer": "AP roaming"},
        ],
        "Symptom_Solution_Mapping": {"sym": "drop", "sol": "raise threshold"},
        "Operational_SOP": {"sop": "page wireless oncall"},
        "Knowledge_Base": [{"title": "Roaming tuning"}],
        "remediation_payload": {"playbook": "wireless-roaming-001"},
    }
    base.update(overrides)
    return base


def _run_ingestion(tickets):
    """Invoke `_ingest_gold_ticket_json` with the given source tickets and
    return the chunk_rows list."""
    from backend.services import contextual_ingestion_service as cis

    file_bytes = json.dumps(tickets).encode("utf-8")
    with mock.patch.object(cis.settings, "ENABLE_DUPLICATE_CHECK", False), \
         mock.patch.object(cis.settings, "ENABLE_VERSION_DETECTION", False):
        result = cis._ingest_gold_ticket_json(
            file_bytes=file_bytes,
            filename="file1_txt.txt",
            file_type="kb",
            owner_id="test-owner",
            fingerprint="test-fp",
            exact_duplicate_lookup=_no_duplicate,
            version_candidate_lookup=_no_candidates,
        )
    return result["chunk_rows"]


class RichMergeForwardFixTests(unittest.TestCase):
    """§3.4 — verify the four required parents and two bonus parents land on
    metadata_json when the source ticket carries them."""

    def test_rich_merge_includes_executive_sharable_rca(self):
        rows = _run_ingestion([_gold_ticket()])
        self.assertEqual(len(rows), 1)
        meta = rows[0]["metadata_json"]
        self.assertIn("Executive_Sharable_RCA", meta)
        self.assertEqual(
            meta["Executive_Sharable_RCA"]["Root_Cause_Technical_High_Level"],
            "AP roaming threshold misaligned",
        )
        self.assertEqual(
            meta["Executive_Sharable_RCA"]["Resolution_Steps"][0]["action"],
            "Confirm channel plan",
        )

    def test_rich_merge_includes_incident_summary(self):
        rows = _run_ingestion([_gold_ticket()])
        meta = rows[0]["metadata_json"]
        self.assertIn("Incident_Summary", meta)
        self.assertEqual(
            meta["Incident_Summary"]["INCIDENT"],
            "Wireless handheld disconnects every 90s",
        )

    def test_rich_merge_includes_forensic_performance_audit(self):
        rows = _run_ingestion([_gold_ticket()])
        meta = rows[0]["metadata_json"]
        self.assertIn("Forensic_Performance_Audit", meta)
        # Real corpus shape is a list; verify list survives untouched.
        self.assertIsInstance(meta["Forensic_Performance_Audit"], list)
        self.assertEqual(
            meta["Forensic_Performance_Audit"][0]["Critical_Intervention"],
            "Lowered min RSSI threshold",
        )

    def test_rich_merge_includes_key_contributors(self):
        rows = _run_ingestion([_gold_ticket()])
        meta = rows[0]["metadata_json"]
        self.assertIn("Key_Contributors", meta)
        players = meta["Key_Contributors"]["Key_Impact_Players"]
        self.assertIsInstance(players, list)
        self.assertEqual(players[0]["Hero_Action"], "Replaced AP firmware")

    def test_rich_merge_skips_missing_parents_silently(self):
        ticket = _gold_ticket()
        del ticket["Forensic_Performance_Audit"]
        del ticket["ITIL_5_Why"]
        rows = _run_ingestion([ticket])
        meta = rows[0]["metadata_json"]
        # Absent keys must be missing, not present-as-null.
        self.assertNotIn("Forensic_Performance_Audit", meta)
        self.assertNotIn("ITIL_5_Why", meta)
        # Other rich parents are still present.
        self.assertIn("Executive_Sharable_RCA", meta)
        self.assertIn("Incident_Summary", meta)
        self.assertIn("Key_Contributors", meta)
        self.assertIn("QA_Auditor_Feedback", meta)



if __name__ == "__main__":
    unittest.main()
