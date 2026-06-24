"""
backend.tenancy.routes
======================

HTTP endpoints for the multi-tenant layer.

Routes mounted by api.py:

    GET    /organizations
        List orgs visible to the current user — their memberships AND
        every publicly-listable org. The landing-page "Your Orgs +
        Discover" feed.

    GET    /organizations/me/active
        Return the user's active org context. Frontend uses this on
        page load to render branding + role badges.

    PATCH  /users/me/active-org
        Set the user's active org (server-side persisted as
        users.last_active_org_id). Validates membership.

    POST   /organizations/{slug}/request-access
        Create a pending access request for a non-member.

    POST   /webhooks/clerk
        Receive Clerk webhook events. Signature-verified.

All routes return JSON. Error responses include a stable `error_code`
string so the frontend can switch on it instead of parsing English.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Header,
    Request,
    status,
)
from pydantic import BaseModel, Field

from backend.tenancy.constants import (
    ERR_DUPLICATE_REQUEST,
    ERR_NO_AUTH,
    ERR_NO_MEMBERSHIP,
    ERR_ORG_INACTIVE,
    ERR_ORG_NOT_FOUND,
    ERR_UNKNOWN_USER,
)
from backend.tenancy.context import get_request_context
from backend.tenancy.models import RequestContext
from backend.tenancy.repository import (
    create_access_request,
    get_membership,
    get_organization_by_id,
    get_organization_by_slug,
    is_member_of_org,
    list_listable_organizations,
    list_organizations_for_user,
    set_last_active_org,
)

logger = logging.getLogger("acadia-log-iq")

router = APIRouter(tags=["tenancy"])


# ─────────────────────────────────────────────────────────────────────
# Request / response shapes (Pydantic — public API surface only)
# ─────────────────────────────────────────────────────────────────────
class OrgListItem(BaseModel):
    """A single tile on the landing page."""

    id: str
    slug: Optional[str] = None
    name: str
    logo_url: Optional[str] = None
    theme: Dict[str, Any] = Field(default_factory=dict)
    is_member: bool
    role: Optional[str] = None
    is_listed_publicly: bool


class OrgListResponse(BaseModel):
    """Landing page payload — split into mine vs others for the UI."""

    your_organizations: List[OrgListItem]
    other_organizations: List[OrgListItem]


class ActiveOrgResponse(BaseModel):
    has_active_org: bool
    organization: Optional[OrgListItem] = None
    last_active_org_id: Optional[str] = None
    platform_role: str


class SetActiveOrgRequest(BaseModel):
    organization_id: str = Field(
        ..., description="Internal UUID of the org to make active.",
    )


class SetActiveOrgResponse(BaseModel):
    status: str = "ok"
    organization_id: str
    organization_slug: Optional[str] = None


class AccessRequestBody(BaseModel):
    justification: Optional[str] = Field(
        default=None,
        max_length=1000,
        description=(
            "Optional reason the user is requesting access. Shown to org "
            "admins on the review queue."
        ),
    )


class AccessRequestResponse(BaseModel):
    status: str
    request_id: Optional[str] = None
    organization_slug: Optional[str] = None
    organization_name: str


# ─────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────
def _require_auth(ctx: RequestContext) -> None:
    """Raise 401 unless the request has a real authenticated user."""
    if not ctx.user_clerk_id or ctx.user_clerk_id == "anonymous":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": ERR_NO_AUTH,
                "message": "Authentication required",
            },
        )


def _require_user_id(ctx: RequestContext) -> uuid.UUID:
    """
    Return the internal user UUID. Raises 401 if our users row hasn't
    been upserted yet (caller likely needs to invoke /auth/register-or-login
    first).
    """
    if ctx.user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": ERR_UNKNOWN_USER,
                "message": (
                    "Your user record is not registered. Sign in again "
                    "to complete registration."
                ),
            },
        )
    return ctx.user_id


def _safe_org_uuid(value: str) -> uuid.UUID:
    """Parse a UUID string or raise 400."""
    try:
        return uuid.UUID(value)
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error_code": "invalid_uuid",
                "message": f"Not a valid organization id: {value!r}",
            },
        )


def _build_list_item(
    org_id: uuid.UUID,
    slug: Optional[str],
    name: str,
    logo_url: Optional[str],
    theme: Dict[str, Any],
    is_member: bool,
    role: Optional[str],
    is_listed_publicly: bool,
) -> OrgListItem:
    return OrgListItem(
        id=str(org_id),
        slug=slug,
        name=name,
        logo_url=logo_url,
        theme=theme or {},
        is_member=is_member,
        role=role,
        is_listed_publicly=is_listed_publicly,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /organizations — the landing-page feed
# ─────────────────────────────────────────────────────────────────────
@router.get("/organizations", response_model=OrgListResponse)
async def list_orgs_for_landing(
    ctx: RequestContext = Depends(get_request_context),
) -> OrgListResponse:
    """
    Return both sections of the landing page in one call:

        your_organizations  — orgs the user is an active member of.
        other_organizations — publicly-listable orgs the user is NOT a
                              member of (shown as locked tiles).
    """
    _require_auth(ctx)
    user_id = _require_user_id(ctx)

    mine = list_organizations_for_user(user_id)
    mine_ids = {o.id for o in mine}

    listable = list_listable_organizations()
    others = [o for o in listable if o.id not in mine_ids]

    your_tiles: List[OrgListItem] = []
    for org in mine:
        m = get_membership(user_id, org.id)
        your_tiles.append(_build_list_item(
            org_id=org.id,
            slug=org.slug,
            name=org.name,
            logo_url=org.logo_url,
            theme=org.theme_json,
            is_member=True,
            role=m.role if m else None,
            is_listed_publicly=org.is_listed_publicly,
        ))

    other_tiles = [
        _build_list_item(
            org_id=o.id,
            slug=o.slug,
            name=o.name,
            logo_url=o.logo_url,
            theme=o.theme_json,
            is_member=False,
            role=None,
            is_listed_publicly=o.is_listed_publicly,
        )
        for o in others
    ]

    return OrgListResponse(
        your_organizations=your_tiles,
        other_organizations=other_tiles,
    )


# ─────────────────────────────────────────────────────────────────────
# GET /organizations/me/active — current active-org context
# ─────────────────────────────────────────────────────────────────────
@router.get("/organizations/me/active", response_model=ActiveOrgResponse)
async def get_active_org(
    ctx: RequestContext = Depends(get_request_context),
) -> ActiveOrgResponse:
    """
    Return the active organization for the current request. Frontend
    calls this on app shell mount to:
        * Render the correct logo + theme
        * Show role-gated UI controls
        * Decide whether to route to the picker
    """
    _require_auth(ctx)
    user_id = _require_user_id(ctx)

    from backend.tenancy.repository import get_last_active_org_id

    last_active = get_last_active_org_id(user_id)

    if ctx.org_id is None:
        return ActiveOrgResponse(
            has_active_org=False,
            organization=None,
            last_active_org_id=str(last_active) if last_active else None,
            platform_role=ctx.platform_role,
        )

    org = get_organization_by_id(ctx.org_id)
    if org is None:
        return ActiveOrgResponse(
            has_active_org=False,
            organization=None,
            last_active_org_id=str(last_active) if last_active else None,
            platform_role=ctx.platform_role,
        )

    return ActiveOrgResponse(
        has_active_org=True,
        organization=_build_list_item(
            org_id=org.id,
            slug=org.slug,
            name=org.name,
            logo_url=org.logo_url,
            theme=org.theme_json,
            is_member=True,
            role=ctx.org_role,
            is_listed_publicly=org.is_listed_publicly,
        ),
        last_active_org_id=str(last_active) if last_active else None,
        platform_role=ctx.platform_role,
    )


# ─────────────────────────────────────────────────────────────────────
# PATCH /users/me/active-org — persist active-org choice
# ─────────────────────────────────────────────────────────────────────
@router.patch("/users/me/active-org", response_model=SetActiveOrgResponse)
async def set_active_org(
    body: SetActiveOrgRequest = Body(...),
    ctx: RequestContext = Depends(get_request_context),
) -> SetActiveOrgResponse:
    """
    Persist the user's active-org choice. The frontend ALSO has to
    instruct Clerk to switch its own session's active org — Clerk
    re-issues a JWT with the new claim, and subsequent requests will
    arrive with `ctx.org_id` set correctly.
    """
    _require_auth(ctx)
    user_id = _require_user_id(ctx)
    target_org_id = _safe_org_uuid(body.organization_id)

    org = get_organization_by_id(target_org_id)
    if org is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": ERR_ORG_NOT_FOUND,
                "message": f"Organization {body.organization_id} not found.",
            },
        )
    if not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": ERR_ORG_INACTIVE,
                "message": f"Organization {org.slug or org.name} is suspended.",
            },
        )

    if not is_member_of_org(user_id, org.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error_code": ERR_NO_MEMBERSHIP,
                "message": (
                    f"You are not a member of {org.name}. Request access "
                    f"from the landing page."
                ),
            },
        )

    set_last_active_org(user_id=user_id, organization_id=org.id)

    logger.info(
        "[tenancy] user=%s set last_active_org=%s",
        ctx.user_clerk_id, org.slug or org.id,
    )

    return SetActiveOrgResponse(
        status="ok",
        organization_id=str(org.id),
        organization_slug=org.slug,
    )


# ─────────────────────────────────────────────────────────────────────
# POST /organizations/{slug}/request-access — request access
# ─────────────────────────────────────────────────────────────────────
@router.post(
    "/organizations/{slug}/request-access",
    response_model=AccessRequestResponse,
)
async def request_org_access(
    slug: str,
    body: AccessRequestBody = Body(default_factory=AccessRequestBody),
    ctx: RequestContext = Depends(get_request_context),
) -> AccessRequestResponse:
    """
    Create a pending request to join a non-member org. The org's admins
    receive a notification (email wiring in Phase 3) and can approve/
    deny via the admin UI.
    """
    _require_auth(ctx)
    user_id = _require_user_id(ctx)

    org = get_organization_by_slug(slug)
    if org is None or not org.is_active:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error_code": ERR_ORG_NOT_FOUND,
                "message": f"Organization {slug!r} not found.",
            },
        )

    # Already a member? Friendly no-op.
    if is_member_of_org(user_id, org.id):
        return AccessRequestResponse(
            status="already_member",
            request_id=None,
            organization_slug=org.slug,
            organization_name=org.name,
        )

    created = create_access_request(
        user_id=user_id,
        organization_id=org.id,
        justification=(body.justification or "").strip() or None,
    )
    if created is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error_code": ERR_DUPLICATE_REQUEST,
                "message": (
                    f"You already have a pending request to join "
                    f"{org.name}. An admin will respond shortly."
                ),
            },
        )

    logger.info(
        "[tenancy] access request created user=%s org=%s",
        ctx.user_clerk_id, org.slug,
    )

    return AccessRequestResponse(
        status="created",
        request_id=str(created.id),
        organization_slug=org.slug,
        organization_name=org.name,
    )


# ─────────────────────────────────────────────────────────────────────
# POST /webhooks/clerk — Clerk webhook receiver
# ─────────────────────────────────────────────────────────────────────
@router.post("/webhooks/clerk")
async def clerk_webhook(
    request: Request,
    svix_id: Optional[str] = Header(default=None, alias="svix-id"),
    svix_timestamp: Optional[str] = Header(default=None, alias="svix-timestamp"),
    svix_signature: Optional[str] = Header(default=None, alias="svix-signature"),
) -> Dict[str, Any]:
    """
    Receive Clerk webhook events for org / membership lifecycle.
    Always returns 200 EXCEPT on signature verification failure (401).
    """
    import json as _json

    body = await request.body()

    ok = False
    try:
        from backend.tenancy.clerk_sync import (
            handle_clerk_webhook_event,
            verify_webhook_signature,
        )
        ok = verify_webhook_signature(
            body=body,
            svix_id=svix_id,
            svix_timestamp=svix_timestamp,
            svix_signature=svix_signature,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("[tenancy.webhook] signature verification raised: %s", exc)
        ok = False

    if not ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={
                "error_code": "invalid_webhook_signature",
                "message": "Webhook signature verification failed.",
            },
        )

    try:
        event = _json.loads(body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tenancy.webhook] body is not valid JSON: %s", exc)
        return {"handled": False, "error": "invalid_json"}

    return handle_clerk_webhook_event(event)
