"""Sprint 5 offline test matrix.

Three slices of coverage, all fully offline (no DB, no Bedrock):

1. is_gold_schema_ticket — accepts the full schema, rejects partial /
   non-dict / missing-section / missing-fingerprints inputs.

2. Template renderers — Phase 2 / Phase 3 / header / KB citations
   render expected markdown substrings from the PHOENIX-402-like sample
   ticket in the sprint spec.

3. Cache roundtrip — get_cached_expert_answer / set_cached_expert_answer
   call the chunks.cached_expert_answer columns with the correct SQL
   against a mocked engine, and survive a DB error gracefully.

Run with: py -m backend.tests.test_sprint5_expert_template
"""
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


# A gold-schema sample modelled on PHOENIX-402 / BGP-5-ADJCHANGE. The
# values are what §4.2 asserts on. Kept inline so the test file is
# self-contained (no fixture dependency).
FULL_JSON = {
    "Header": "BGP adjacency flapping on prod edge router",
    "Metadata": {
        "Incident_Number": "INC-PHOENIX-402",
        "priority": "P1",
        "Target_Service": "Edge BGP",
        "Affected_Assets": ["edge-rtr-01"],
        "customer_name": "Phoenix Industries",
        "Fingerprints": ["BGP-5-ADJCHANGE", "TCP-179-TIMEOUT"],
        "Resolution_Quality_Score": 5,
    },
    "Symptom_Solution_Mapping": {
        "Detected_Symptom": "BGP peer bounces every ~90 seconds",
        "Origin_Event": "BFD session flap",
        "Primary_Fix": "Tune BFD timers to 100ms/100ms/3",
        "Primary_Fix_Confidence_Interval": "95%",
        "Validation_Metric": "BFD Up duration > 10 min",
    },
    "Operational_SOP": {
        "diagnostic_logic_chunks": [
            {
                "step_id": "D1",
                "action": "Verify if BFD is software or hardware",
                "command": "show bfd session detail",
                "branching_logic": "If software → proceed to D2; else escalate.",
            },
            {
                "step_id": "D2",
                "action": "Inspect timer config",
                "command": "show running-config | section bfd",
                "branching_logic": "If timers < 100ms → tune; else check link quality.",
            },
        ],
    },
    "remediation_payload": {
        "execution_steps": [
            {"task": "Apply BFD tuning", "action": "bfd interval 100 min_rx 100 multiplier 3"},
            {"task": "Validate", "action": "show bfd session"},
        ],
        "Remediation_As_Code": {
            "IaC_Language": "CiscoIOS",
            "Executable_Snippet": "interface Gi0/0/0\n bfd interval 100 min_rx 100 multiplier 3",
        },
    },
    "Knowledge_Base": [
        {"semantic_unit_educational": {"knowledge_id": "KB-BFD-001"}},
        {"semantic_unit_educational": {"knowledge_id": "KB-BGP-014"}},
    ],
}


class IsGoldSchemaTicketTests(unittest.TestCase):
    def test_accepts_full_schema(self):
        from backend.agents.expert_copilot_template import is_gold_schema_ticket
        self.assertTrue(is_gold_schema_ticket(FULL_JSON))

    def test_rejects_non_dict(self):
        from backend.agents.expert_copilot_template import is_gold_schema_ticket
        self.assertFalse(is_gold_schema_ticket(None))
        self.assertFalse(is_gold_schema_ticket("a string"))
        self.assertFalse(is_gold_schema_ticket([]))

    def test_rejects_empty_metadata(self):
        from backend.agents.expert_copilot_template import is_gold_schema_ticket
        self.assertFalse(is_gold_schema_ticket({"Metadata": {}}))

    def test_rejects_missing_sections(self):
        from backend.agents.expert_copilot_template import is_gold_schema_ticket
        # Fingerprints present but Symptom_Solution_Mapping / Operational_SOP
        # / remediation_payload are all missing.
        self.assertFalse(is_gold_schema_ticket(
            {"Metadata": {"Fingerprints": ["BGP-5-ADJCHANGE"]}}
        ))

    def test_rejects_missing_fingerprints(self):
        from backend.agents.expert_copilot_template import is_gold_schema_ticket
        payload = dict(FULL_JSON)
        payload["Metadata"] = dict(payload["Metadata"])
        payload["Metadata"].pop("Fingerprints", None)
        self.assertFalse(is_gold_schema_ticket(payload))

    def test_remediation_payload_nested_also_ok(self):
        """remediation_payload may live under Operational_SOP (legacy)."""
        from backend.agents.expert_copilot_template import is_gold_schema_ticket
        payload = {
            "Metadata": {"Fingerprints": ["X"]},
            "Symptom_Solution_Mapping": {"x": 1},
            "Operational_SOP": {
                "diagnostic_logic_chunks": [],
                "remediation_payload": {"execution_steps": []},
            },
        }
        self.assertTrue(is_gold_schema_ticket(payload))


class TemplateRenderTests(unittest.TestCase):
    def test_phase_2_contains_heading_and_action(self):
        from backend.agents.expert_copilot_template import render_phase_2_branching
        out = render_phase_2_branching(FULL_JSON)
        self.assertIn("## Phase 2: Branching Diagnostics", out)
        self.assertIn("Verify if BFD is software or hardware", out)
        self.assertIn("`show bfd session detail`", out)

    def test_phase_3_contains_primary_fix_and_rac(self):
        from backend.agents.expert_copilot_template import render_phase_3_remediation
        out = render_phase_3_remediation(FULL_JSON)
        self.assertIn("## Phase 3: Validated Fix", out)
        self.assertIn("Primary Fix", out)
        # Verbatim RaC snippet preserved inside a fenced block.
        self.assertIn("bfd interval 100 min_rx 100 multiplier 3", out)
        self.assertIn("```ciscoios", out)

    def test_header_block_has_fingerprints(self):
        from backend.agents.expert_copilot_template import render_header_and_fingerprints
        out = render_header_and_fingerprints(FULL_JSON)
        self.assertIn("Troubleshooting Guide:", out)
        self.assertIn("INC-PHOENIX-402", out)
        self.assertIn("`BGP-5-ADJCHANGE`", out)
        self.assertIn("`TCP-179-TIMEOUT`", out)

    def test_kb_citations(self):
        from backend.agents.expert_copilot_template import render_kb_citations
        out = render_kb_citations(FULL_JSON)
        self.assertIn("KB-BFD-001", out)
        self.assertIn("KB-BGP-014", out)

    def test_kb_citations_empty(self):
        from backend.agents.expert_copilot_template import render_kb_citations
        self.assertEqual(render_kb_citations({"Knowledge_Base": []}), "")


class CacheRoundtripTests(unittest.TestCase):
    def test_set_then_get_returns_answer(self):
        from backend import vector_store

        # Mock the set path.
        fake_set_conn = mock.MagicMock()
        fake_set_conn.__enter__ = mock.MagicMock(return_value=fake_set_conn)
        fake_set_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_set_engine = mock.MagicMock()
        fake_set_engine.begin.return_value = fake_set_conn

        with mock.patch("backend.db.connection.engine", fake_set_engine):
            self.assertTrue(
                vector_store.set_cached_expert_answer("chunk-id-1", "answer text")
            )

        # Mock the get path.
        fake_row = mock.MagicMock()
        fake_row.__getitem__ = lambda self, idx: "answer text"
        fake_get_result = mock.MagicMock()
        fake_get_result.first.return_value = fake_row
        fake_get_conn = mock.MagicMock()
        fake_get_conn.execute.return_value = fake_get_result
        fake_get_conn.__enter__ = mock.MagicMock(return_value=fake_get_conn)
        fake_get_conn.__exit__ = mock.MagicMock(return_value=False)
        fake_get_engine = mock.MagicMock()
        fake_get_engine.connect.return_value = fake_get_conn

        with mock.patch("backend.db.connection.engine", fake_get_engine):
            self.assertEqual(
                vector_store.get_cached_expert_answer("chunk-id-1"),
                "answer text",
            )

    def test_get_returns_none_on_db_error(self):
        from backend import vector_store
        broken_engine = mock.MagicMock()
        broken_engine.connect.side_effect = RuntimeError("no table")
        with mock.patch("backend.db.connection.engine", broken_engine):
            self.assertIsNone(vector_store.get_cached_expert_answer("x"))

    def test_set_returns_false_on_db_error(self):
        from backend import vector_store
        broken_engine = mock.MagicMock()
        broken_engine.begin.side_effect = RuntimeError("no table")
        with mock.patch("backend.db.connection.engine", broken_engine):
            self.assertFalse(vector_store.set_cached_expert_answer("x", "y"))

    def test_empty_inputs_short_circuit(self):
        from backend import vector_store
        self.assertIsNone(vector_store.get_cached_expert_answer(""))
        self.assertFalse(vector_store.set_cached_expert_answer("", "ans"))
        self.assertFalse(vector_store.set_cached_expert_answer("cid", ""))


class RetrieveByFingerprintReturnChunkIdTests(unittest.TestCase):
    """Additive kwarg: return_chunk_id=True yields a (dict, id) tuple;
    default False preserves the Sprint 4 Optional[Dict] contract."""

    def test_flag_off_returns_tuple_none(self):
        from backend.retrieval import orchestrator as orch
        with mock.patch.object(orch.settings, "LOGIQ_SPRINT4_BACKEND", False):
            out = orch.retrieve_by_fingerprint("BGP-5-ADJCHANGE", return_chunk_id=True)
            self.assertEqual(out, (None, None))

    def test_flag_off_default_kwarg_still_returns_none(self):
        from backend.retrieval import orchestrator as orch
        with mock.patch.object(orch.settings, "LOGIQ_SPRINT4_BACKEND", False):
            self.assertIsNone(orch.retrieve_by_fingerprint("BGP-5-ADJCHANGE"))

    def test_hit_returns_tuple_when_requested(self):
        from backend.retrieval import orchestrator as orch

        mock_row = {
            "id": "chunk-abc",
            "document_id": "doc-xyz",
            "metadata_json": FULL_JSON,
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

        with mock.patch.object(orch.settings, "LOGIQ_SPRINT4_BACKEND", True), \
             mock.patch("backend.db.connection.engine", fake_engine):
            out = orch.retrieve_by_fingerprint("BGP-5-ADJCHANGE", return_chunk_id=True)

        self.assertIsInstance(out, tuple)
        meta, cid = out
        self.assertEqual(cid, "chunk-abc")
        self.assertEqual(meta["Metadata"]["Incident_Number"], "INC-PHOENIX-402")


if __name__ == "__main__":
    unittest.main(verbosity=2)
