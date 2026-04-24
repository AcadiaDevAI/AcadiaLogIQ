"""Sprint 7 — escalation package assembly (offline)."""
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


_TICKET = {
    "Metadata": {
        "Incident_Number": "INC-ALPHA-027",
        "priority": "P2",
        "customer_name": "Aetheris Corp",
        "Affected_Assets": ["V-Desktop Environment"],
        "Target_Service": "Virtual Desktop",
    },
    "Symptom_Solution_Mapping": {
        "Detected_Symptom": "Session launch >90s",
        "Primary_Fix": "Restart delivery controller",
    },
    "Engagement_Analysis": {
        "Team_Path": ["NOC-L1", "App Ops", "VDI Engineering"],
    },
}


class BuildPackageNoContactsTests(unittest.TestCase):
    def test_package_without_contacts(self):
        from backend.tier1_copilot.diagnostics.escalation_package import build_package
        pkg = build_package(
            ticket_metadata=_TICKET,
            alert_payload={
                "severity": "P2",
                "asset_name": "V-Desktop Environment",
                "alert_type": "Desktop Slowness",
                "customer": "Aetheris Corp",
            },
            session_what_tried=[
                {"step": "Ping gateway", "result": "normal"},
            ],
            client_what_tried=[
                {"step": "Host CPU check", "result": "skipped"},
            ],
            engine=None,          # no DB → no contacts
            related_incidents=["INC-ALPHA-031"],
        )
        self.assertEqual(pkg.priority, "P2")
        self.assertEqual(pkg.affected_customer, "Aetheris Corp")
        self.assertEqual(pkg.suggested_owner_team, "VDI Engineering")
        self.assertEqual(pkg.customer_contacts, [])
        self.assertEqual(len(pkg.what_was_tried), 2)
        self.assertIn("INC-ALPHA-027", pkg.relevant_tickets)
        self.assertIn("INC-ALPHA-031", pkg.relevant_tickets)
        self.assertIn("Recommended next action:", pkg.formatted_text)

    def test_deduplicates_what_tried(self):
        from backend.tier1_copilot.diagnostics.escalation_package import build_package
        pkg = build_package(
            ticket_metadata=_TICKET,
            alert_payload={"severity": "P2"},
            session_what_tried=[{"step": "ping", "result": "ok"}],
            client_what_tried=[{"step": "Ping", "result": "OK"}],
            engine=None,
        )
        self.assertEqual(len(pkg.what_was_tried), 1)


class BuildPackageWithContactsTests(unittest.TestCase):
    def test_contact_rows_queried_when_engine_provided(self):
        from backend.tier1_copilot.diagnostics import escalation_package as ep

        contact_row = {"metadata_json": {
            "doc_kind": "contact_customer",
            "organization": {"name": "Aetheris Corp"},
            "team": {"escalation_level": 1, "role": "NOC Manager"},
            "name": "Jane Doe",
            "phone": "555-0100",
            "email": "jane@aetheris",
        }}
        fake_result = mock.MagicMock()
        fake_result.all.return_value = [contact_row]
        fake_conn = mock.MagicMock()
        fake_conn.execute.return_value.mappings.return_value = fake_result
        fake_conn.__enter__ = mock.MagicMock(return_value=fake_conn)
        fake_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_engine = mock.MagicMock()
        fake_engine.connect.return_value = fake_conn

        pkg = ep.build_package(
            ticket_metadata=_TICKET,
            alert_payload={"severity": "P2", "customer": "Aetheris Corp"},
            engine=fake_engine,
        )
        self.assertEqual(len(pkg.customer_contacts), 1)
        self.assertEqual(pkg.customer_contacts[0].name, "Jane Doe")
        self.assertIn("Jane Doe", pkg.formatted_text)

    def test_engine_failure_degrades_gracefully(self):
        from backend.tier1_copilot.diagnostics import escalation_package as ep
        broken = mock.MagicMock()
        broken.connect.side_effect = RuntimeError("no db")
        pkg = ep.build_package(
            ticket_metadata=_TICKET,
            alert_payload={"severity": "P2", "customer": "Aetheris Corp"},
            engine=broken,
        )
        self.assertEqual(pkg.customer_contacts, [])
        self.assertEqual(pkg.affected_customer, "Aetheris Corp")


if __name__ == "__main__":
    unittest.main(verbosity=2)
