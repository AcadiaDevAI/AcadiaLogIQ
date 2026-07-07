"""Tests for the per-org profile registry.

Verifies slug resolution (incl. slug-format variants), the fail-safe default
for unknown/absent orgs, and the public_config shape the frontend consumes.

Run with: py -m unittest backend.tests.test_org_registry
"""
from __future__ import annotations

import os
import unittest
import uuid

os.environ.setdefault("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/test")
os.environ.setdefault("AWS_SECRETS_DISABLED", "true")

from backend.orgs.registry import get_profile  # noqa: E402
from backend.orgs.base import OrgProfile, DEFAULT_FLOWS  # noqa: E402
from backend.orgs.acadia import AcadiaProfile  # noqa: E402
from backend.orgs.uspharma import USPharmaProfile  # noqa: E402


class TestOrgRegistry(unittest.TestCase):
    def test_acadia_slug_resolves(self):
        p = get_profile(uuid.uuid4(), "acadia-consultants")
        self.assertIsInstance(p, AcadiaProfile)
        self.assertEqual(p.display_name, "Acadia LogIQ")

    def test_uspharma_slug_variants_resolve(self):
        for slug in (
            "uspharma",
            "us-pharma",
            "us_pharma",
            "US Pharma",
            # The REAL Clerk slug carries a long numeric org-id suffix.
            "us-pharma-1782167031742315025",
        ):
            p = get_profile(uuid.uuid4(), slug)
            self.assertIsInstance(p, USPharmaProfile, f"slug={slug!r}")
            self.assertEqual(p.display_name, "US Pharma")

    def test_acadia_clerk_variants_and_landing_site_are_acadia_default(self):
        # The Acadia "landing site" org and any Clerk-suffixed acadia slug
        # should behave as the shared default (not US Pharma).
        for slug in (
            "acadia-consultants",
            "acadia-consultants-landing-site-1782708801449057263",
        ):
            p = get_profile(uuid.uuid4(), slug)
            self.assertNotIsInstance(p, USPharmaProfile, f"slug={slug!r}")

    def test_unknown_slug_falls_back_to_base(self):
        p = get_profile(uuid.uuid4(), "some-brand-new-org")
        self.assertIs(type(p), OrgProfile)  # base, not a subclass
        self.assertEqual(p.enabled_flows, set(DEFAULT_FLOWS))

    def test_none_slug_is_safe(self):
        p = get_profile(None, None)
        self.assertIs(type(p), OrgProfile)
        self.assertEqual(p.display_name, "Acadia LogIQ")  # shared default

    def test_public_config_shape(self):
        cfg = get_profile(uuid.uuid4(), "uspharma").public_config()
        self.assertEqual(
            set(cfg),
            {"org_id", "slug", "display_name", "theme", "enabled_flows", "feature_flags"},
        )
        self.assertEqual(cfg["display_name"], "US Pharma")
        self.assertEqual(cfg["theme"], {"accent": "#0b7285"})
        self.assertIsInstance(cfg["enabled_flows"], list)

    def test_uspharma_tier1_requires_store_id(self):
        self.assertTrue(
            get_profile(uuid.uuid4(), "us-pharma-1782167031742315025").tier1_requires_store_id
        )
        self.assertFalse(
            get_profile(uuid.uuid4(), "acadia-consultants").tier1_requires_store_id
        )
        self.assertFalse(OrgProfile().tier1_requires_store_id)

    def test_uspharma_inherits_shared_flows(self):
        # Phase 1: US Pharma overrides nothing behavioral — same flows as base.
        self.assertEqual(
            get_profile(uuid.uuid4(), "uspharma").enabled_flows,
            get_profile(uuid.uuid4(), "acadia-consultants").enabled_flows,
        )


if __name__ == "__main__":
    unittest.main()
