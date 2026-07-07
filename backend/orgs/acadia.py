"""Acadia org profile — the default. No overrides: Acadia == shared core.

Kept as its own class (rather than using the bare base) so Acadia has an
explicit, named home for any future Acadia-specific config, and so the
registry has a concrete entry to resolve Acadia's slug to.
"""
from __future__ import annotations

from backend.orgs.base import OrgProfile


class AcadiaProfile(OrgProfile):
    # Actual Acadia slug from migration 055 (seed_acadia_org).
    slug = "acadia-consultants"
    display_name = "Acadia LogIQ"
    # theme / flows inherited from OrgProfile (shared defaults).
