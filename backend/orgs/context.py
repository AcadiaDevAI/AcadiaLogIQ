"""FastAPI dependency that resolves the current request's org to an OrgProfile.

Mirrors backend/tenancy/context.py::require_org_admin — it wraps
get_request_context and adds one resolution step. Any route can then do:

    @router.post("/ask")
    async def ask(profile: OrgProfile = Depends(get_org_profile)):
        ...

No org in scope → the base (Acadia/shared) profile, so shared routes keep
working for unauthenticated/no-org requests exactly as before (fail-safe).
"""
from __future__ import annotations

from fastapi import Depends

from backend.tenancy.context import get_request_context
from backend.tenancy.models import RequestContext
from backend.orgs.base import OrgProfile
from backend.orgs.registry import get_profile


async def get_org_profile(
    ctx: RequestContext = Depends(get_request_context),
) -> OrgProfile:
    return get_profile(ctx.org_id, ctx.org_slug)


def resolve_current_profile() -> OrgProfile:
    """Resolve the OrgProfile for the current request WITHOUT a FastAPI
    dependency — for call sites deep in the request (e.g. the /ask retrieval
    path) that only have the org ContextVar in scope. Reads the org id from
    ``current_org_id_var``, resolves its slug via the tenancy repository, and
    returns the matching profile. Falls back to the default (shared/Acadia)
    profile when no org is in scope or the lookup fails — never raises."""
    try:
        from backend.db.connection import current_org_id_var

        org_id = current_org_id_var.get()
        if not org_id:
            return get_profile(None, None)
        try:
            from backend.tenancy.repository import get_organization_by_id

            org = get_organization_by_id(org_id)
            slug = org.slug if org else None
        except Exception:
            slug = None
        return get_profile(org_id, slug)
    except Exception:  # pragma: no cover - never break the request path
        return get_profile(None, None)
