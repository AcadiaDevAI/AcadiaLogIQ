"""Sprint 11 — Layer 2 PII prompt tightening.

The prior `_LAYER2_SYSTEM` prompt told Haiku to flag ANY query that
mentioned PII as unsafe. That over-blocked legitimate telecom /
incident-troubleshooting queries that legitimately reference a
customer phone number, email, or address.

These tests lock in the new behaviour:
  1. The prompt itself contains explicit positive examples for
     phone-in-context, email-in-context, address-in-context.
  2. The prompt distinguishes "lookup INTENT" (unsafe) from
     "incidental mention" (safe).
  3. Layer 1 regex behaviour is unchanged: real SSN / Luhn-valid
     credit cards still scrub; phone numbers don't trigger Layer 1.
"""
from __future__ import annotations

import re
import unittest

from backend.services import input_guardrails as ig


class Layer2PromptContentTests(unittest.TestCase):
    """Direct assertions on the prompt string. Cheap, deterministic,
    and robust against API-cost regressions (tests don't call Haiku)."""

    def test_prompt_distinguishes_intent_from_incidental_mention(self):
        prompt = ig._LAYER2_SYSTEM
        # The new prompt must explicitly carry the "intent" framing —
        # this is the load-bearing distinction.
        self.assertIn("CORE INTENT", prompt)
        self.assertIn("INTENT to look someone up", prompt)

    def test_prompt_lists_phone_in_context_as_safe(self):
        prompt = ig._LAYER2_SYSTEM
        # Positive example matching the user's failing case.
        self.assertIn("+99 99 9999 99", prompt)
        self.assertIn("Phone numbers", prompt)

    def test_prompt_lists_email_and_address_in_context_as_safe(self):
        prompt = ig._LAYER2_SYSTEM
        self.assertIn("email addresses", prompt)
        self.assertIn("site addresses", prompt)

    def test_prompt_still_blocks_explicit_lookup_examples(self):
        prompt = ig._LAYER2_SYSTEM
        # Verifies we didn't go too far — explicit "find the user
        # whose phone is X" stays in the UNSAFE example list.
        self.assertIn("UNSAFE pii queries", prompt)
        self.assertIn("Find the user whose phone number", prompt)


class Layer1RegexUnchangedTests(unittest.TestCase):
    """Layer 1 regex tables must be untouched — phone numbers still
    don't match the credit_card pattern, real SSNs still match."""

    def test_phone_number_does_not_match_credit_card_regex(self):
        # The user's failing input — 11 digits with spaces.
        phone = "+99 99 9999 99"
        match = ig.PII_PATTERNS["credit_card"].search(phone)
        self.assertIsNone(match)

    def test_phone_number_does_not_match_ssn_regex(self):
        phone = "+99 99 9999 99"
        match = ig.PII_PATTERNS["ssn"].search(phone)
        self.assertIsNone(match)

    def test_real_ssn_still_matches(self):
        sample = "User SSN 123-45-6789 was on the account."
        match = ig.PII_PATTERNS["ssn"].search(sample)
        self.assertIsNotNone(match)


class Layer1WithBusinessContextPhoneTests(unittest.TestCase):
    """End-to-end Layer 1 scrub: a query that mentions a phone in
    business context should NOT trigger Layer 1's PII scrub-and-block
    path. Layer 2 may still be invoked separately if the query has
    non-ASCII or special chars; that's a Haiku decision and tested
    via the prompt contract above."""

    def setUp(self):
        # Layer 1 scrubbing is gated on settings.INPUT_GUARDRAIL_SCRUB_PII.
        # Tests assume the production default (True). If false, the
        # function returns None and the test is trivially satisfied.
        self.scrub_enabled = bool(
            getattr(ig.settings, "INPUT_GUARDRAIL_SCRUB_PII", True)
        )

    def test_phone_in_business_context_passes_layer1(self):
        if not self.scrub_enabled:
            self.skipTest("INPUT_GUARDRAIL_SCRUB_PII disabled")
        query = (
            "Severity P3 - Faxlinesunreachable on Telephony. "
            "Customer at +99 99 9999 99 reports fax dropping."
        )
        result = ig._layer1_regex(query)
        # Layer 1 returns None when nothing scrubbed/blocked.
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
