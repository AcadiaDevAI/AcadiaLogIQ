"""Org-profile registry — resolve an org (id + slug) to its OrgProfile.

Keyed on a NORMALIZED slug so "us-pharma", "us_pharma", "US Pharma" and
"uspharma" all resolve to the same profile. An unknown/absent slug falls back
to the base OrgProfile (Acadia / shared-core behavior) — fail-safe: a
mis-typed or brand-new org never crashes, it just behaves like Acadia until
given its own profile.
"""
from __future__ import annotations

import logging
import re
import uuid
from typing import Dict, Optional, Type

from backend.orgs.base import OrgProfile
from backend.orgs.acadia import AcadiaProfile
from backend.orgs.uspharma import USPharmaProfile

logger = logging.getLogger("acadia-log-iq")


def _norm(slug: Optional[str]) -> str:
    """Collapse slug variants to a stable key.

    Clerk appends a long numeric org-id suffix to org slugs (e.g.
    "us-pharma-1782167031742315025"), so strip a trailing "-<6+ digits>"
    before removing separators. Then lowercase + keep alphanumerics:
        "us-pharma-1782167031742315025" -> "uspharma"
        "US Pharma"                     -> "uspharma"
        "acadia-consultants"            -> "acadiaconsultants"
    """
    if not slug:
        return ""
    base = re.sub(r"-\d{6,}$", "", slug.lower().strip())
    return "".join(ch for ch in base if ch.isalnum())


# Registered profiles, keyed by normalized slug.
_BY_SLUG: Dict[str, Type[OrgProfile]] = {}


def _register(cls: Type[OrgProfile]) -> None:
    _BY_SLUG[_norm(cls.slug)] = cls


for _cls in (AcadiaProfile, USPharmaProfile):
    _register(_cls)

# Optional ops override: if US Pharma's real Clerk slug differs from
# "uspharma", set settings.USPHARMA_ORG_SLUG and it maps here too — no code
# change needed to activate the profile for the real org.
try:
    from backend.config import settings as _settings

    _override = getattr(_settings, "USPHARMA_ORG_SLUG", None)
    if _override:
        _BY_SLUG[_norm(_override)] = USPharmaProfile
except Exception:  # pragma: no cover - config optional at import time
    pass


def get_profile(
    org_id: Optional[uuid.UUID], org_slug: Optional[str]
) -> OrgProfile:
    """Return the OrgProfile instance for this org. Unknown slug → base
    (Acadia/shared) profile. Never raises."""
    cls = _BY_SLUG.get(_norm(org_slug), OrgProfile)
    return cls(org_id=org_id, org_slug=org_slug)


def registered_slugs() -> Dict[str, str]:
    """Debug aid: normalized-slug → profile class name."""
    return {k: v.__name__ for k, v in _BY_SLUG.items()}
