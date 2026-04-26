"""Sprint 9 — prompt builder substitution + hint capping."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _populated_catalog():
    from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
    cat = IntakeCatalogs()
    rows = []
    # Build 50 distinct asset families so we can test the 30-cap.
    for i in range(50):
        rows.append({"metadata_json": {
            "Metadata": {
                "Affected_Assets": [f"asset-fam-{i:02d}-XX"],
                "customer_name": f"Customer {i}",
            },
            "Symptom_Solution_Mapping": {
                "Detected_Symptom": f"alert type {i}",
            },
        }})
    cat.rebuild_from_rows(rows)
    return cat


class BuildPromptTests(unittest.TestCase):
    def test_includes_required_blocks(self):
        # Sprint 9.2 — section header names changed in the rewrite.
        # Severity guidance moved from "Severity rules" → "Severity
        # decoding"; the "Known asset families" / "Known customers"
        # hint sections were removed entirely (their contamination of
        # the LLM was the bug 9.2 fixes).
        from backend.tier1_copilot.intake.prompt_builder import (
            build_extraction_prompt,
        )
        cat = _populated_catalog()
        prompt = build_extraction_prompt(
            raw_text="users cannot connect since 8AM",
            source_type="email",
            catalogs=cat,
            n_candidates=4,
        )
        # Source label + raw text echoed.
        self.assertIn("email", prompt.lower())
        self.assertIn("users cannot connect since 8AM", prompt)
        # JSON schema and severity guidance still present (renamed).
        self.assertIn("Output schema", prompt)
        self.assertIn("Severity decoding", prompt)
        # Hint sections must be ABSENT (Sprint 9.2 invariant).
        self.assertNotIn("Known asset families", prompt)
        self.assertNotIn("Known customers", prompt)

    def test_hints_capped_at_30(self):
        # Sprint 9.2 — the hint section is gone entirely. This test
        # is repurposed as a regression guard that no per-term limit
        # exists because no hints exist. The marker that used to anchor
        # this test must NOT appear in the prompt.
        from backend.tier1_copilot.intake.prompt_builder import (
            build_extraction_prompt,
        )
        cat = _populated_catalog()
        prompt = build_extraction_prompt(
            raw_text="x",
            source_type="phone",
            catalogs=cat,
            n_candidates=4,
        )
        marker = "Known asset families"
        self.assertEqual(prompt.find(marker), -1)

    def test_empty_catalog_renders_placeholder(self):
        # Sprint 9.2: prompt no longer renders catalog hints, so the
        # placeholder phrase from Sprint 9 is gone. The new prompt has
        # NO catalog text whatsoever — verified explicitly below.
        from backend.tier1_copilot.intake.catalogs import IntakeCatalogs
        from backend.tier1_copilot.intake.prompt_builder import (
            build_extraction_prompt,
        )
        empty = IntakeCatalogs()
        prompt = build_extraction_prompt(
            raw_text="x",
            source_type="note",
            catalogs=empty,
            n_candidates=4,
        )
        # Sprint 9.2 — no catalog mention at all in the prompt.
        self.assertNotIn("Known asset families", prompt)
        self.assertNotIn("Known customers", prompt)


class Sprint92NoHintsContaminationTests(unittest.TestCase):
    """Sprint 9.2 — the prompt MUST NOT render catalog hints.

    Hint contamination caused the LLM to pick a near-looking value from
    the top-30 sample when its own extraction was uncertain (proven by
    reproduction tests). 9.2 removes the entire hint mechanism."""

    def test_no_asset_hints_in_prompt(self):
        from backend.tier1_copilot.intake.prompt_builder import (
            build_extraction_prompt,
        )
        cat = _populated_catalog()
        prompt = build_extraction_prompt(
            raw_text="P1 critical: BGP flap on NY4-CORE-RTR-01",
            source_type="email",
            catalogs=cat,
            n_candidates=4,
        )
        # The Sprint 9 hint sections must be absent.
        self.assertNotIn("Known asset families", prompt)
        self.assertNotIn("Known alert categories", prompt)
        self.assertNotIn("Known customers", prompt)
        # And no individual catalog term should leak into the prompt.
        for term in cat.top_asset_families(50):
            self.assertNotIn(term, prompt)
        # The new evidence-grounded instructions should be present.
        self.assertIn("evidence", prompt.lower())
        self.assertIn("LITERALLY PRESENT", prompt)


if __name__ == "__main__":
    unittest.main(verbosity=2)
