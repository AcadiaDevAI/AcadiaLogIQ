"""US Pharma org profile.

Phase 1: inherits ALL shared/Acadia behavior — only the public identity
(display name + accent) differs, which is the visible proof that the org
switch works end-to-end. Phase 2 adds the real overrides here: the Store ID +
symptom intake, the Historic → KB → Escalate journey variant, and any
per-org prompt/matcher hooks. Each override is added to this one file; Acadia
stays untouched.
"""
from __future__ import annotations

from backend.orgs.base import OrgProfile


class USPharmaProfile(OrgProfile):
    slug = "uspharma"
    display_name = "US Pharma"
    # Distinct accent so the frontend can visibly confirm the module switched.
    theme = {"accent": "#0b7285"}

    # US Pharma's KB is a "search-everything" surface: store lookups live in
    # Excel/PDF/JSON, so on an identifier miss we always run the full hybrid
    # search rather than returning "not found". (Acadia keeps the default
    # natural-language-gated behavior.)
    kb_search_all_on_identifier_miss = True

    # US Pharma tier-1 intake is Store ID + symptom; historic matches are
    # hard-scoped to that store. Store ID is mandatory.
    tier1_requires_store_id = True
