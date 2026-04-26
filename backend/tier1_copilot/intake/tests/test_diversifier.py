"""Sprint 9 — diversity reranker."""
from __future__ import annotations

import os
import sys
import unittest

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _cand(sig: str, **fields):
    from backend.tier1_copilot.intake.schemas import ValidatedCandidate
    c = ValidatedCandidate(diversity_signature=sig, **fields)
    return c


class DiversifierTests(unittest.TestCase):
    def test_drops_duplicate_signature(self):
        """When filtering already produced enough unique cards (>=
        max_cards) the duplicate is dropped. With max_cards=2, we get
        the first unique + the next unique signature, never the
        duplicate."""
        from backend.tier1_copilot.intake.diversifier import diversify
        a = _cand("P2|ny4-core-rtr|bgp peer flapping", asset_name="A")
        b = _cand("P2|ny4-core-rtr|bgp peer flapping", asset_name="B")
        c = _cand("P3|v-desktop|desktop slowness", asset_name="C")
        out = diversify([a, b, c], max_cards=2)
        sigs = [x.diversity_signature for x in out]
        self.assertEqual(len(out), 2)
        self.assertEqual(sigs.count("P2|ny4-core-rtr|bgp peer flapping"), 1)
        self.assertIn("P3|v-desktop|desktop slowness", sigs)

    def test_pads_back_duplicates_when_room(self):
        """Per spec §7: if dedupe leaves fewer than max_cards, pad with
        the suppressed candidates so the engineer always sees up to
        max_cards rows. Order preserved, the dup that got skipped earlier
        comes back at the tail."""
        from backend.tier1_copilot.intake.diversifier import diversify
        a = _cand("P2|asset|alert", asset_name="A")
        b = _cand("P2|asset|alert", asset_name="B")
        c = _cand("P3|asset|alert", asset_name="C")
        out = diversify([a, b, c], max_cards=4)
        # 2 unique sigs → padded to 3 (b joins back).
        self.assertEqual(len(out), 3)
        self.assertEqual([x.asset_name for x in out], ["A", "C", "B"])

    def test_pads_when_diversity_leaves_too_few(self):
        from backend.tier1_copilot.intake.diversifier import diversify
        a = _cand("P2|asset|alert", asset_name="A")
        b = _cand("P2|asset|alert", asset_name="B")
        c = _cand("P2|asset|alert", asset_name="C")
        # All three identical signature — diversifier keeps a, then pads
        # with b + c so the user sees 3 cards instead of 1.
        out = diversify([a, b, c], max_cards=4)
        self.assertEqual(len(out), 3)
        # First slot is the highest-ranked unique candidate.
        self.assertIs(out[0], a)

    def test_caps_at_max_cards(self):
        from backend.tier1_copilot.intake.diversifier import diversify
        cands = [_cand(f"P2|asset|alert-{i}", asset_name=f"A{i}") for i in range(8)]
        out = diversify(cands, max_cards=4)
        self.assertEqual(len(out), 4)

    def test_empty_input_returns_empty(self):
        from backend.tier1_copilot.intake.diversifier import diversify
        self.assertEqual(diversify([], max_cards=4), [])

    def test_preserves_input_order_for_first_unique(self):
        from backend.tier1_copilot.intake.diversifier import diversify
        a = _cand("S1", asset_name="A")
        b = _cand("S2", asset_name="B")
        c = _cand("S3", asset_name="C")
        out = diversify([a, b, c], max_cards=4)
        self.assertEqual([x.asset_name for x in out], ["A", "B", "C"])

    def test_case_variation_signatures_dedup_correctly(self):
        """Sprint 9.2 — signature comparison is case- and whitespace-
        insensitive. The same (severity, asset, alert) triple in
        different casing now collapses to a single card. Sprint 9.0
        leaked these as separate cards because the dedup compared the
        raw signature strings."""
        from backend.tier1_copilot.intake.diversifier import diversify
        a = _cand("P2|rtr-01|BGP Flap", asset_name="A1")
        b = _cand("P2|rtr-01|bgp flap", asset_name="A2")  # casing variant
        c = _cand("P2|rtr-02|MTU Mismatch", asset_name="B1")
        out = diversify([a, b, c], max_cards=2)
        self.assertEqual(len(out), 2)
        # First card preserved (highest-ranked unique signature),
        # casing variant suppressed, the genuinely-different rtr-02
        # takes the remaining slot.
        names = [x.asset_name for x in out]
        self.assertEqual(names[0], "A1")
        self.assertIn("B1", names)
        self.assertNotIn("A2", names)


if __name__ == "__main__":
    unittest.main(verbosity=2)
