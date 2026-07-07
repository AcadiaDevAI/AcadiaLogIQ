"""Tests for the per-org token-usage accounting service.

Covers the pure surface — pricing, universal usage extraction across model
families, and the in-process buffer → drain grouping — without touching the
DB (flush's persistence is exercised by the integration smoke test).

Run with: py -m unittest backend.tests.services.test_token_usage
"""
from __future__ import annotations

import os
import unittest

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("AWS_SECRETS_DISABLED", "true")

from backend.services import token_usage as tu  # noqa: E402


class TestPricing(unittest.TestCase):
    def test_resolves_by_substring_longest_match(self):
        self.assertEqual(tu.price_for("us.anthropic.claude-haiku-4-5-20251001-v1:0"), (1.00, 5.00))
        self.assertEqual(tu.price_for("us.anthropic.claude-sonnet-4-6"), (3.00, 15.00))
        self.assertEqual(tu.price_for("mistral.mistral-7b-instruct-v0:2"), (0.15, 0.20))
        # 'titan-embed' must win over the shorter 'titan' key.
        self.assertEqual(tu.price_for("amazon.titan-embed-text-v2:0"), (0.02, 0.0))

    def test_unknown_model_is_free_not_error(self):
        self.assertEqual(tu.price_for("some-unknown-model"), (0.0, 0.0))

    def test_cost_math(self):
        # 1M input @ $1 + 1M output @ $5 = $6.
        self.assertAlmostEqual(tu.cost_usd("haiku", 1_000_000, 1_000_000), 6.0, places=6)
        self.assertAlmostEqual(tu.cost_usd("titan-embed", 500_000, 0), 0.01, places=6)


class TestExtractUsage(unittest.TestCase):
    def test_claude_body_usage(self):
        self.assertEqual(
            tu.extract_bedrock_usage({"usage": {"input_tokens": 10, "output_tokens": 5}}),
            (10, 5),
        )

    def test_titan_body_input_only(self):
        self.assertEqual(tu.extract_bedrock_usage({"inputTextTokenCount": 7}), (7, 0))

    def test_header_fallback_for_mistral(self):
        resp = {
            "ResponseMetadata": {
                "HTTPHeaders": {
                    "x-amzn-bedrock-input-token-count": "20",
                    "x-amzn-bedrock-output-token-count": "8",
                }
            }
        }
        self.assertEqual(tu.extract_bedrock_usage({}, resp), (20, 8))

    def test_body_wins_over_headers(self):
        resp = {"ResponseMetadata": {"HTTPHeaders": {"x-amzn-bedrock-input-token-count": "999"}}}
        self.assertEqual(
            tu.extract_bedrock_usage({"usage": {"input_tokens": 3, "output_tokens": 2}}, resp),
            (3, 2),
        )

    def test_missing_everything_is_zero(self):
        self.assertEqual(tu.extract_bedrock_usage({}), (0, 0))
        self.assertEqual(tu.extract_bedrock_usage(None), (0, 0))


class TestBufferDrain(unittest.TestCase):
    def setUp(self):
        tu._drain()            # clear any buffered rows
        tu.reset_usage_totals()
        self._orig = tu._current_org

    def tearDown(self):
        tu._current_org = self._orig
        tu._drain()
        tu.reset_usage_totals()

    def test_same_key_accumulates_and_strips_retry_suffix(self):
        tu._current_org = lambda: "org-A"
        tu.record_token_usage("ingestion_metadata#a1", "haiku", 100, 50)
        tu.record_token_usage("ingestion_metadata#a2", "haiku", 200, 40)  # retry → same feature
        rows = tu._drain()
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["feature"], "ingestion_metadata")
        self.assertEqual(r["calls"], 2)
        self.assertEqual(r["in_toks"], 300)
        self.assertEqual(r["out_toks"], 90)
        self.assertEqual(r["org"], "org-A")

    def test_distinct_orgs_and_features_split(self):
        tu._current_org = lambda: "org-A"
        tu.record_token_usage("chat", "mistral", 10, 5)
        tu.record_token_usage("embeddings", "titan-embed", 7, 0)
        tu._current_org = lambda: "org-B"
        tu.record_token_usage("chat", "mistral", 1, 1)
        rows = tu._drain()
        by_org = {}
        for r in rows:
            by_org.setdefault(r["org"], []).append(r)
        self.assertEqual(set(by_org), {"org-A", "org-B"})
        self.assertEqual(len(by_org["org-A"]), 2)  # chat + embeddings
        self.assertEqual(len(by_org["org-B"]), 1)

    def test_totals_track_cost(self):
        tu._current_org = lambda: "org-A"
        tu.record_token_usage("chat", "haiku", 1_000_000, 0)  # $1.00 input
        totals = tu.get_usage_totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["input_tokens"], 1_000_000)
        self.assertAlmostEqual(totals["cost_usd"], 1.0, places=6)

    def test_drain_empties_buffer(self):
        tu._current_org = lambda: "org-A"
        tu.record_token_usage("chat", "haiku", 5, 5)
        self.assertEqual(len(tu._drain()), 1)
        self.assertEqual(len(tu._drain()), 0)  # already drained


if __name__ == "__main__":
    unittest.main()
