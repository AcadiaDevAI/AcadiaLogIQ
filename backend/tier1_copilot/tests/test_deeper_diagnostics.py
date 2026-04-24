"""Sprint 7 — deeper diagnostics skeleton + severity ordering (offline)."""
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


_SAMPLE_TICKET = {
    "Metadata": {"Incident_Number": "INC-1", "priority": "P3"},
    "Symptom_Solution_Mapping": {
        "Detected_Symptom": "Session launch >90s",
        "Validation_Metric": "Launch time < 20s",
    },
    "Operational_SOP": {
        "diagnostic_logic_chunks": [
            {
                "step_id": "D1",
                "action": "Check DDC health",
                "command": "Get-BrokerController",
                "branching_logic": "If unhealthy, restart DDC.",
            },
            {
                "step_id": "D2",
                "action": "Check license server",
                "command": "lsadmin -cli",
                "branching_logic": "If expired, renew license.",
            },
        ],
    },
    "Engagement_Analysis": {
        "Team_Path": ["NOC-L1", "App Ops", "VDI Engineering"],
    },
}


class SkeletonOrderingTests(unittest.TestCase):
    def test_p1_skips_scope_check(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_step_skeleton,
        )
        steps = build_step_skeleton(
            ticket_metadata=_SAMPLE_TICKET, severity="P1",
        )
        # P1 skips scope confirmation — first step is the SOP's D1.
        self.assertGreaterEqual(len(steps), 1)
        self.assertEqual(steps[0].title, "Check DDC health")

    def test_p2_starts_with_scope_confirmation(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_step_skeleton,
        )
        steps = build_step_skeleton(
            ticket_metadata=_SAMPLE_TICKET, severity="P2",
        )
        self.assertEqual(steps[0].title, "Confirm scope")
        self.assertIsNone(steps[0].command)  # scope step has no command

    def test_step_cap_at_five(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_step_skeleton,
        )
        big = dict(_SAMPLE_TICKET)
        big["Operational_SOP"] = {
            "diagnostic_logic_chunks": [
                {"step_id": f"D{i}", "action": f"step{i}", "command": "cmd"}
                for i in range(10)
            ]
        }
        steps = build_step_skeleton(ticket_metadata=big, severity="P3")
        self.assertLessEqual(len(steps), 5)

    def test_no_sop_returns_only_scope_for_non_p1(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_step_skeleton,
        )
        steps = build_step_skeleton(
            ticket_metadata={"Metadata": {}, "Operational_SOP": {}},
            severity="P3",
        )
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0].title, "Confirm scope")

    def test_no_sop_p1_returns_empty(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_step_skeleton,
        )
        steps = build_step_skeleton(
            ticket_metadata={"Metadata": {}, "Operational_SOP": {}},
            severity="P1",
        )
        self.assertEqual(steps, [])


class BuildResponseTests(unittest.TestCase):
    def test_skeleton_fallback_when_no_llm(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_deeper_diagnostics,
        )
        resp = build_deeper_diagnostics(
            ticket_metadata=_SAMPLE_TICKET,
            severity="P2",
            llm_formatter=None,
        )
        self.assertFalse(resp.llm_used)
        self.assertEqual(resp.severity, "P2")
        self.assertEqual(resp.escalation_path, ["NOC-L1", "App Ops", "VDI Engineering"])
        self.assertEqual(resp.validation, "Launch time < 20s")
        self.assertGreater(len(resp.steps), 0)

    def test_llm_json_round_trip(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_deeper_diagnostics,
        )
        fake_json = {
            "goal": "Isolate session launch latency",
            "validation": "Launch < 20s",
            "next_question": {"prompt": "Result?", "options": ["ok", "nope"]},
            "steps": [
                {
                    "step_number": 1,
                    "title": "Check DDC health",
                    "what_to_check": "DDC status",
                    "why": "bottleneck",
                    "command": "Get-BrokerController",
                    "expected_result": "Healthy",
                    "next_action_if_abnormal": "Restart DDC",
                    "next_action_if_normal": "Go to step 2",
                },
                {
                    "step_number": 2,
                    "title": "Check license",
                    "what_to_check": "License validity",
                    "why": "expired license blocks logons",
                    "command": "lsadmin -cli",
                    "expected_result": "Valid",
                    "next_action_if_abnormal": "Renew",
                    "next_action_if_normal": "Proceed",
                },
            ],
        }
        resp = build_deeper_diagnostics(
            ticket_metadata=_SAMPLE_TICKET,
            severity="P1",
            llm_formatter=lambda _: json.dumps(fake_json),
        )
        self.assertTrue(resp.llm_used)
        self.assertEqual(resp.goal, "Isolate session launch latency")
        self.assertEqual(len(resp.steps), 2)
        self.assertEqual(resp.steps[0].command, "Get-BrokerController")

    def test_llm_malformed_falls_back_to_skeleton(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import (
            build_deeper_diagnostics,
        )
        resp = build_deeper_diagnostics(
            ticket_metadata=_SAMPLE_TICKET,
            severity="P2",
            llm_formatter=lambda _: "not json at all",
        )
        self.assertFalse(resp.llm_used)
        # Skeleton still has the Confirm-scope step for P2.
        self.assertEqual(resp.steps[0].title, "Confirm scope")


class JsonParserTests(unittest.TestCase):
    def test_strips_code_fence(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import parse_llm_json
        fenced = "```json\n{\"a\":1}\n```"
        self.assertEqual(parse_llm_json(fenced), {"a": 1})

    def test_recovers_embedded_object(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import parse_llm_json
        noisy = "Here you go: {\"a\":1} cheers"
        self.assertEqual(parse_llm_json(noisy), {"a": 1})

    def test_none_on_unrecoverable(self):
        from backend.tier1_copilot.diagnostics.deeper_diagnostics import parse_llm_json
        self.assertIsNone(parse_llm_json("absolutely not json"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
