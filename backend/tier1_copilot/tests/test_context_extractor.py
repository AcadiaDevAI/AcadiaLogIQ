"""Sprint 6 — compact context extraction (offline)."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


_FULL = {
    "Header": "Desktop slowness on VDI pool A",
    "Metadata": {
        "Incident_Number": "INC-ALPHA-027",
        "priority": "P2",
        "customer_name": "Aetheris Corp",
        "component_category": "Citrix",
        "Target_Service": "V-Desktop Environment",
        "Resolution_Groups": ["Tier-2 EUC"],
        "Affected_Assets": ["VDI-Pool-A"],
    },
    "Symptom_Solution_Mapping": {
        "Detected_Symptom": "Session launch >90s",
        "Primary_Fix": "Restart Citrix delivery controller",
        "Validation_Metric": "Launch time < 20s",
    },
    "Operational_SOP": {
        "diagnostic_logic_chunks": [
            {"step_id": "D1", "action": "Check DDC health", "command": "Get-BrokerController"},
            {"step_id": "D2", "action": "Check license server", "command": "lsadmin -cli"},
        ],
    },
    "Executive_Sharable_RCA": {
        "Root_Cause": "Stale session profile lock",
        "Escalation_Path": "Tier-2 EUC on-call",
    },
}


class ExtractContextTests(unittest.TestCase):
    def test_full_context_populated(self):
        from backend.tier1_copilot.context_extractor import extract_compact_context
        ctx = extract_compact_context(_FULL)
        self.assertEqual(ctx["incident_number"], "INC-ALPHA-027")
        self.assertEqual(ctx["customer"], "Aetheris Corp")
        self.assertEqual(ctx["priority"], "P2")
        self.assertEqual(ctx["technology"], "Citrix")
        self.assertEqual(ctx["root_cause"], "Stale session profile lock")
        self.assertEqual(ctx["primary_fix"], "Restart Citrix delivery controller")
        self.assertIn("Get-BrokerController", ctx["recommended_checks"])

    def test_missing_fields_stay_none_not_na(self):
        from backend.tier1_copilot.context_extractor import extract_compact_context
        ctx = extract_compact_context({"Metadata": {}})
        # No "N/A" substring anywhere.
        self.assertNotIn("N/A", str(ctx))
        self.assertIsNone(ctx.get("incident_number"))
        self.assertIsNone(ctx.get("customer"))

    def test_prune_drops_empty_values(self):
        from backend.tier1_copilot.context_extractor import prune_none
        pruned = prune_none({
            "a": "kept", "b": None, "c": "", "d": [], "e": {}, "f": 0, "g": "x",
        })
        # 0 is retained (it's a legitimate value); None / "" / [] / {} stripped.
        self.assertIn("a", pruned)
        self.assertNotIn("b", pruned)
        self.assertNotIn("c", pruned)
        self.assertNotIn("d", pruned)
        self.assertNotIn("e", pruned)
        self.assertIn("f", pruned)

    def test_non_dict_input_returns_empty(self):
        from backend.tier1_copilot.context_extractor import extract_compact_context
        ctx = extract_compact_context(None)
        self.assertIsInstance(ctx, dict)
        self.assertIsNone(ctx["incident_number"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
