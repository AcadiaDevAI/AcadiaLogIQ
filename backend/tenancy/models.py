"""
backend.tenancy.models
======================

Immutable data shapes for the tenancy layer. All models are FROZEN
dataclasses — once constructed, they cannot be mutated.

Architecture note on user identity:

    The existing `users` table has BOTH columns:
        users.id        UUID  (primary key, internal stable identifier)
        users.clerk_id  TEXT  (Clerk's external id, e.g. "user_2abc...")

    The existing `memberships` table joins via users.id (UUID).
    Clerk's JWT carries users.clerk_id (TEXT).

    Phase 0 reconciles these two worlds by carrying BOTH on every
    RequestContext: `user_id` (UUID for joins) and `user_clerk_id`
    (TEXT for Clerk Backend API calls and logging). The repository
    works in UUIDs internally; only the request boundary touches the
    Clerk-side identifier.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


# ─────────────────────────────────────────────────────────────────────
# RequestContext — the per-request tenant scope
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class RequestContext:
    """
    Carries the authenticated user identity AND the active organization
    for a single HTTP request.

    Built by `tenancy.context.get_request_context` and injected via
    FastAPI's Depends() into any endpoint that needs org-awareness.

    Fields:
        user_clerk_id — Clerk user id (e.g. "user_2abc..."). Comes
                        directly from JWT `sub`. Required.
        user_id       — Internal user UUID, resolved from clerk_id via
                        a `users` table lookup. Used for all FK joins
                        (memberships.user_id, org_access_requests.user_id).
                        May be None on first-ever request before
                        upsert_user has run.
        org_id        — Active organization UUID. May be None when:
                          * The user has no active org chosen — frontend
                            should send to picker.
                          * Clerk's JWT lacks the org_id claim.
        org_slug      — Active org's slug (the URL-safe identifier).
                        None when org_id is None.
        org_role      — User's role IN THE ACTIVE ORG ('admin' or
                        'member'). None when org_id is None.
        platform_role — Clerk public-metadata role. 'super_admin' or
                        'user'. Used for /admin/* endpoint gating.

    Design notes:
        * Frozen: cannot be mutated after construction.
        * slots: smaller memory footprint, slightly faster attribute access.
        * No methods that mutate state — pure value object.
    """

    user_clerk_id: str
    user_id: Optional[uuid.UUID] = None
    org_id: Optional[uuid.UUID] = None
    org_slug: Optional[str] = None
    org_role: Optional[str] = None
    platform_role: str = "user"

    def has_org(self) -> bool:
        """True when this request has a resolved active organization."""
        return self.org_id is not None

    def require_org(self) -> uuid.UUID:
        """
        Return org_id or raise HTTPException(403). Use in endpoints that
        cannot meaningfully proceed without org scope (e.g. /ask, /files).
        """
        if self.org_id is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error_code": "no_active_org",
                    "message": (
                        "This endpoint requires an active organization. "
                        "Pick one via the landing-page org picker or the "
                        "sidebar switcher."
                    ),
                },
            )
        return self.org_id

    def require_user_id(self) -> uuid.UUID:
        """Return user_id or raise 401 when the user row hasn't been
        upserted yet."""
        if self.user_id is None:
            from fastapi import HTTPException, status
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail={
                    "error_code": "unknown_user",
                    "message": (
                        "Your user record is not registered. Sign in "
                        "again to complete registration."
                    ),
                },
            )
        return self.user_id

    def is_admin(self) -> bool:
        """True when the user is admin of the active org."""
        return self.org_role == "admin"

    def is_super_admin(self) -> bool:
        """True when the user is a platform super-admin (cross-org)."""
        return self.platform_role == "super_admin"


# ─────────────────────────────────────────────────────────────────────
# Organization — snapshot of an organizations row
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Organization:
    """Read-side projection of an organizations row."""

    id: uuid.UUID
    slug: Optional[str]
    name: str
    clerk_org_id: Optional[str]
    logo_url: Optional[str]
    theme_json: Dict[str, Any] = field(default_factory=dict)
    settings_json: Dict[str, Any] = field(default_factory=dict)
    is_listed_publicly: bool = True
    force_org_picker: bool = False
    is_active: bool = True
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    deactivated_at: Optional[datetime] = None

    def to_public_dict(self) -> Dict[str, Any]:
        """Serialize for the public API surface."""
        return {
            "id": str(self.id),
            "slug": self.slug,
            "name": self.name,
            "logo_url": self.logo_url,
            "theme": self.theme_json or {},
            "is_active": self.is_active,
        }


# ─────────────────────────────────────────────────────────────────────
# Membership — snapshot of a memberships row
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class Membership:
    """
    Read-side projection of a memberships row.

    Note the JOIN convention: memberships.user_id is the internal
    users.id UUID, not Clerk's clerk_id. The repository surface accepts
    UUIDs; the Clerk-side clerk_id is only used at the request boundary
    (RequestContext) and in webhooks (resolved via users.clerk_id index).
    """

    user_id: uuid.UUID
    organization_id: uuid.UUID
    role: str
    is_active: bool = True
    granted_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None


# ─────────────────────────────────────────────────────────────────────
# AccessRequest — snapshot of an org_access_requests row
# ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True, slots=True)
class AccessRequest:
    """Read-side projection of an org_access_requests row."""

    id: uuid.UUID
    user_id: uuid.UUID
    organization_id: uuid.UUID
    status: str
    justification: Optional[str]
    responded_by_user_id: Optional[uuid.UUID]
    response_note: Optional[str]
    created_at: Optional[datetime] = None
    responded_at: Optional[datetime] = None
