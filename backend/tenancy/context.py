"""
backend.tenancy.context
=======================

The bridge between Clerk authentication and our tenant-scoped data model.

`get_request_context` is the FastAPI dependency that:
    1. Validates the Clerk JWT.
    2. Reads the active org from the JWT claims.
    3. Resolves Clerk's user id → our internal users.id UUID.
    4. Resolves Clerk's external org id → our internal organizations.id UUID.
    5. Returns a frozen RequestContext carrying BOTH identifiers
       (UUID for joins, Clerk-id for logging / external API calls).
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import HTTPException, Request, status

from backend.clerk_auth import (
    extract_bearer_token,
    is_clerk_enabled,
    verify_clerk_token,
)
from backend.tenancy.constants import (
    PLATFORM_ROLE_SUPER_ADMIN,
    PLATFORM_ROLE_USER,
)
from backend.tenancy.models import RequestContext

logger = logging.getLogger("acadia-log-iq")


def _extract_org_from_claims(payload: dict) -> tuple[
    Optional[uuid.UUID], Optional[str], Optional[str]
]:
    """
    Read org_id / org_slug / org_role from Clerk's JWT claims and
    resolve Clerk's external org id → our internal organizations.id
    UUID.

    Clerk emits org claims in TWO different formats depending on the JWT
    template configuration and Clerk version:

      Compact format (Clerk's current default, observed in production):
          "o": {
              "id":  "org_2xyz...",
              "rol": "admin",
              "slg": "acadia-consultants"
          }

      Flat format (legacy / explicit template):
          "org_id":   "org_2xyz...",
          "org_role": "org:admin",
          "org_slug": "acadia-consultants"

    Notes:
      * Compact-format role does NOT carry the "org:" prefix.
      * Flat-format role DOES carry the "org:" prefix.
      * Clerk slugs sometimes carry a timestamp suffix (e.g.
        "acadia-consultants-178163992...") when the requested slug
        collides; resolution happens via clerk_org_id, not slug, so
        suffix mismatch is non-fatal.

    Returns (internal_org_uuid, slug, role) or (None, None, None) when
    the user has no active org or the org is suspended.
    """
    # --- Try compact format first (Clerk's current default) ---
    o = payload.get("o") if isinstance(payload.get("o"), dict) else None
    clerk_org_id: Optional[str] = None
    org_slug: Optional[str] = None
    org_role: Optional[str] = None

    if o:
        clerk_org_id = o.get("id")
        org_slug = o.get("slg")
        # Compact-format role is bare ("admin" / "member"). No prefix.
        org_role = o.get("rol") or None

    # --- Fall back to flat format if compact wasn't present ---
    if not clerk_org_id:
        clerk_org_id = payload.get("org_id")
        org_slug = org_slug or payload.get("org_slug")
        org_role_raw = payload.get("org_role") or ""
        # Flat-format role MAY carry the "org:" prefix — strip it.
        if org_role_raw.startswith("org:"):
            org_role = org_role_raw.split(":", 1)[1]
        elif org_role_raw and not org_role:
            org_role = org_role_raw

    if not clerk_org_id:
        return None, None, None

    # Lazy import to avoid a circular dependency (repository imports
    # constants; context imports repository).
    from backend.tenancy.repository import get_organization_by_clerk_id

    org = get_organization_by_clerk_id(clerk_org_id)
    if not org:
        logger.warning(
            "[tenancy] JWT references clerk_org_id=%s but not in DB yet "
            "(webhook sync lag?). Treating request as no-active-org.",
            clerk_org_id,
        )
        return None, None, None

    if not org.is_active:
        logger.warning(
            "[tenancy] JWT references suspended org=%s (clerk_org_id=%s). "
            "Refusing to use it as active context.",
            org.slug, clerk_org_id,
        )
        return None, None, None

    return org.id, org.slug, org_role


def _extract_platform_role(payload: dict) -> str:
    """
    Read platform_role from Clerk public_metadata claim. Defaults to
    PLATFORM_ROLE_USER when the claim is absent.
    """
    metadata = payload.get("public_metadata") or {}
    if not isinstance(metadata, dict):
        return PLATFORM_ROLE_USER
    role = metadata.get("platform_role")
    if role == PLATFORM_ROLE_SUPER_ADMIN:
        return PLATFORM_ROLE_SUPER_ADMIN
    return PLATFORM_ROLE_USER


async def get_request_context(request: Request) -> RequestContext:
    """
    FastAPI dependency: build a RequestContext for the current request.

    Raises:
        HTTPException 401 when Clerk is enabled and the request has no
        valid Clerk JWT.

    Does NOT raise when the user has no active organization — endpoints
    that require one should call `ctx.require_org()`.

    Behavior when `users.id` for the Clerk user can't be resolved:
        `ctx.user_id` is None. The user has authenticated via Clerk but
        their local users row hasn't been upserted yet. This is normal
        on the very first request after sign-up; `/auth/register-or-login`
        creates the row. Endpoints that need a stable `user_id` for FK
        joins should call `ctx.require_user_id()`.
    """
    # ── Path A: Clerk disabled (dev/testing without auth) ──
    if not is_clerk_enabled():
        return RequestContext(
            user_clerk_id="anonymous",
            user_id=None,
            org_id=None,
            org_slug=None,
            org_role=None,
            platform_role=PLATFORM_ROLE_USER,
        )

    # ── Path B: Clerk enabled — validate the JWT ──
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Wrap JWT decoding so malformed tokens (e.g. truncated, copy-paste
    # damage from PowerShell, expired tokens with corrupt cache) surface
    # as 401 rather than an uncaught exception that bubbles to 500. The
    # underlying PyJWT raises specific exceptions for bad header /
    # payload base64-decoding which we treat the same as "auth failed".
    try:
        payload = verify_clerk_token(token)
    except HTTPException:
        # verify_clerk_token already raises HTTPException for the
        # expected validation failures (expired, signature bad, etc.).
        # Let those through unchanged.
        raise
    except Exception as exc:  # noqa: BLE001 — broad on purpose
        logger.warning(
            "[tenancy] malformed/undecodable JWT (%s: %s) — returning 401",
            type(exc).__name__, exc,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed or invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    user_clerk_id = payload["sub"]

    # Resolve internal users.id UUID (may be None on very first request
    # before /auth/register-or-login has upserted the row).
    from backend.tenancy.repository import resolve_user_id
    user_id = resolve_user_id(user_clerk_id)

    org_id, org_slug, org_role = _extract_org_from_claims(payload)
    platform_role = _extract_platform_role(payload)

    request.state.clerk_user_id = user_clerk_id
    request.state.clerk_payload = payload

    return RequestContext(
        user_clerk_id=user_clerk_id,
        user_id=user_id,
        org_id=org_id,
        org_slug=org_slug,
        org_role=org_role,
        platform_role=platform_role,
    )
