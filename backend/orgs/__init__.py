"""Per-org module system (org-profile registry).

US Pharma and any future org get their own behavior/config WITHOUT scattering
`if org == "uspharma"` through shared code. Each org is an `OrgProfile`
subclass; the shared core asks the profile (resolved per request from the JWT
org) instead of branching. An org that overrides nothing inherits Acadia /
shared-core behavior verbatim — the sustainable form of "clone Acadia, then
diverge incrementally".

See [[project-uspharma-modular-decision]] in project memory for the rationale.

Public surface:
    from backend.orgs.context import get_org_profile   # FastAPI dependency
    from backend.orgs.base import OrgProfile
"""
from backend.orgs.base import OrgProfile  # noqa: F401
from backend.orgs.registry import get_profile  # noqa: F401
