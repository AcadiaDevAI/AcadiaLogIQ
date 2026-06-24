"""
backend.tenancy.repository
==========================

All database access for the tenancy layer.

Schema reality check (the layer we ADAPTED to, not invented):
    * `organizations` — existing table extended in migration 051 with
      slug, clerk_org_id, theme_json, settings_json, etc. "Active"
      means `deactivated_at IS NULL`.
    * `memberships` — existing table; roles are `admin | member` after
      migration 055. "Active" means `revoked_at IS NULL`. Join via
      `memberships.user_id` (UUID) to `users.id` (UUID).
    * `org_access_requests` — new table, also keyed by `user_id` UUID.
    * `users.last_active_org_id` — UUID FK added in migration 053.

Connection management: every function opens its own short-lived session
via `SessionLocal()`. We never hold a session across function boundaries.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from backend.db.connection import SessionLocal
from backend.tenancy.constants import (
    ACCESS_REQUEST_STATUS_PENDING,
    ROLE_MEMBER,
    VALID_ROLES,
)
from backend.tenancy.models import (
    AccessRequest,
    Membership,
    Organization,
)

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────────────
# Row mappers — turn raw SQLAlchemy rows into our frozen models
# ─────────────────────────────────────────────────────────────────────
def _as_uuid(value: Any) -> uuid.UUID:
    """Coerce a value (UUID or str) into a UUID. Raises on garbage."""
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def _as_uuid_or_none(value: Any) -> Optional[uuid.UUID]:
    if value is None:
        return None
    return _as_uuid(value)


def _coerce_jsonb(value: Any) -> Dict[str, Any]:
    """JSONB can come back as dict (psycopg) or str (some drivers)."""
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value) or {}
        except (ValueError, TypeError):
            return {}
    return {}


def _row_to_organization(row: Dict[str, Any]) -> Organization:
    """Map a single organizations row dict to an Organization model."""
    deactivated_at = row.get("deactivated_at")
    return Organization(
        id=_as_uuid(row["id"]),
        slug=row.get("slug"),
        name=row["name"],
        clerk_org_id=row.get("clerk_org_id"),
        logo_url=row.get("logo_url"),
        theme_json=_coerce_jsonb(row.get("theme_json")),
        settings_json=_coerce_jsonb(row.get("settings_json")),
        is_listed_publicly=bool(row.get("is_listed_publicly", True)),
        force_org_picker=bool(row.get("force_org_picker", False)),
        is_active=deactivated_at is None,
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
        deactivated_at=deactivated_at,
    )


def _row_to_membership(row: Dict[str, Any]) -> Membership:
    revoked_at = row.get("revoked_at")
    return Membership(
        user_id=_as_uuid(row["user_id"]),
        organization_id=_as_uuid(row["organization_id"]),
        role=row["role"],
        is_active=revoked_at is None,
        granted_at=row.get("granted_at"),
        revoked_at=revoked_at,
    )


def _row_to_access_request(row: Dict[str, Any]) -> AccessRequest:
    return AccessRequest(
        id=_as_uuid(row["id"]),
        user_id=_as_uuid(row["user_id"]),
        organization_id=_as_uuid(row["organization_id"]),
        status=row["status"],
        justification=row.get("justification"),
        responded_by_user_id=_as_uuid_or_none(row.get("responded_by_user_id")),
        response_note=row.get("response_note"),
        created_at=row.get("created_at"),
        responded_at=row.get("responded_at"),
    )


# ─────────────────────────────────────────────────────────────────────
# User-identity resolution (clerk_id ↔ user_id)
# ─────────────────────────────────────────────────────────────────────
def resolve_user_id(clerk_id: str) -> Optional[uuid.UUID]:
    """
    Map Clerk's external user id (TEXT) to our internal users.id UUID.
    Returns None if no user row exists for that clerk_id — the caller
    decides whether to upsert or raise.
    """
    with SessionLocal() as db:
        row = db.execute(
            text("SELECT id FROM users WHERE clerk_id = :cid"),
            {"cid": clerk_id},
        ).mappings().first()
        return _as_uuid_or_none(row["id"]) if row else None


# ─────────────────────────────────────────────────────────────────────
# READ — organizations
# ─────────────────────────────────────────────────────────────────────
_ORG_SELECT = (
    "SELECT id, slug, name, clerk_org_id, logo_url, theme_json, "
    "       settings_json, is_listed_publicly, force_org_picker, "
    "       created_at, updated_at, deactivated_at "
    "FROM organizations"
)


def get_organization_by_id(org_id: uuid.UUID) -> Optional[Organization]:
    """Fetch a single organization by primary key. None if not found."""
    with SessionLocal() as db:
        row = db.execute(
            text(_ORG_SELECT + " WHERE id = :oid"),
            {"oid": str(org_id)},
        ).mappings().first()
        return _row_to_organization(dict(row)) if row else None


def get_organization_by_slug(slug: str) -> Optional[Organization]:
    """Fetch a single organization by slug. None if not found."""
    with SessionLocal() as db:
        row = db.execute(
            text(_ORG_SELECT + " WHERE slug = :slug"),
            {"slug": slug},
        ).mappings().first()
        return _row_to_organization(dict(row)) if row else None


def get_organization_by_clerk_id(clerk_org_id: str) -> Optional[Organization]:
    """Fetch by Clerk's external org id. Used by the webhook handler."""
    with SessionLocal() as db:
        row = db.execute(
            text(_ORG_SELECT + " WHERE clerk_org_id = :cid"),
            {"cid": clerk_org_id},
        ).mappings().first()
        return _row_to_organization(dict(row)) if row else None


def list_listable_organizations() -> List[Organization]:
    """
    All publicly-listable, active orgs. The "discover other organizations"
    feed on the landing page.
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                _ORG_SELECT
                + " WHERE deactivated_at IS NULL "
                  "   AND is_listed_publicly = TRUE "
                  " ORDER BY name ASC"
            ),
        ).mappings().all()
        return [_row_to_organization(dict(r)) for r in rows]


# ─────────────────────────────────────────────────────────────────────
# READ — memberships (UUID-keyed)
# ─────────────────────────────────────────────────────────────────────
_MEMBERSHIP_SELECT = (
    "SELECT user_id, organization_id, role, granted_at, revoked_at "
    "FROM memberships"
)


def list_organizations_for_user(user_id: uuid.UUID) -> List[Organization]:
    """
    All ACTIVE orgs this user is a member of, sorted alphabetically.
    "Active" on both sides:
        * users.is_active = TRUE  (implicit — caller already filtered)
        * memberships.revoked_at IS NULL
        * organizations.deactivated_at IS NULL
    """
    with SessionLocal() as db:
        rows = db.execute(
            text(
                "SELECT o.id, o.slug, o.name, o.clerk_org_id, o.logo_url, "
                "       o.theme_json, o.settings_json, o.is_listed_publicly, "
                "       o.force_org_picker, o.created_at, o.updated_at, "
                "       o.deactivated_at "
                "FROM organizations o "
                "JOIN memberships m ON m.organization_id = o.id "
                "WHERE m.user_id = :uid "
                "  AND m.revoked_at IS NULL "
                "  AND o.deactivated_at IS NULL "
                "ORDER BY o.name ASC"
            ),
            {"uid": str(user_id)},
        ).mappings().all()
        return [_row_to_organization(dict(r)) for r in rows]


def get_membership(user_id: uuid.UUID, organization_id: uuid.UUID) -> Optional[Membership]:
    """Single membership row by composite key. None if it doesn't exist."""
    with SessionLocal() as db:
        row = db.execute(
            text(
                _MEMBERSHIP_SELECT
                + " WHERE user_id = :uid AND organization_id = :oid"
            ),
            {"uid": str(user_id), "oid": str(organization_id)},
        ).mappings().first()
        return _row_to_membership(dict(row)) if row else None


def is_member_of_org(user_id: uuid.UUID, organization_id: uuid.UUID) -> bool:
    """Quick boolean check — does this user have an ACTIVE membership?"""
    m = get_membership(user_id, organization_id)
    return bool(m and m.is_active)


# ─────────────────────────────────────────────────────────────────────
# WRITE — memberships (used by webhook sync; not public API)
# ─────────────────────────────────────────────────────────────────────
def upsert_membership(
    *,
    user_id: uuid.UUID,
    organization_id: uuid.UUID,
    role: str = ROLE_MEMBER,
    is_active: bool = True,
) -> Membership:
    """
    Insert or update a membership. Idempotent. Called by:
      * The Clerk webhook handler when an `organizationMembership.created`
        event arrives.
      * Approval of an access request.

    Validates role against VALID_ROLES.
    """
    if role not in VALID_ROLES:
        raise ValueError(f"Invalid role: {role!r}. Allowed: {sorted(VALID_ROLES)}")

    with SessionLocal() as db:
        db.execute(
            text(
                "INSERT INTO memberships "
                "    (user_id, organization_id, role, granted_at, revoked_at) "
                "VALUES "
                "    (:uid, :oid, :role, NOW(), "
                "     CASE WHEN :active THEN NULL ELSE NOW() END) "
                "ON CONFLICT (user_id, organization_id) DO UPDATE SET "
                "    role = EXCLUDED.role, "
                "    revoked_at = CASE WHEN :active THEN NULL "
                "                 ELSE NOW() END"
            ),
            {
                "uid": str(user_id),
                "oid": str(organization_id),
                "role": role,
                "active": is_active,
            },
        )
        db.commit()
    m = get_membership(user_id, organization_id)
    assert m is not None, "upsert_membership succeeded but read returned None"
    return m


def deactivate_membership(*, user_id: uuid.UUID, organization_id: uuid.UUID) -> None:
    """Soft-remove a membership. Webhook calls this on member-removed event."""
    with SessionLocal() as db:
        db.execute(
            text(
                "UPDATE memberships "
                "   SET revoked_at = NOW() "
                " WHERE user_id = :uid "
                "   AND organization_id = :oid "
                "   AND revoked_at IS NULL"
            ),
            {"uid": str(user_id), "oid": str(organization_id)},
        )
        db.commit()


# ─────────────────────────────────────────────────────────────────────
# WRITE — last_active_org_id on users
# ─────────────────────────────────────────────────────────────────────
def set_last_active_org(*, user_id: uuid.UUID, organization_id: uuid.UUID) -> None:
    """
    Update the user's restore-on-next-login org. Called after every
    successful in-session switch.
    """
    with SessionLocal() as db:
        db.execute(
            text(
                "UPDATE users "
                "   SET last_active_org_id = :oid, "
                "       updated_at = CURRENT_TIMESTAMP "
                " WHERE id = :uid"
            ),
            {"uid": str(user_id), "oid": str(organization_id)},
        )
        db.commit()


def get_last_active_org_id(user_id: uuid.UUID) -> Optional[uuid.UUID]:
    """Read the user's last-active org. None if not set."""
    with SessionLocal() as db:
        row = db.execute(
            text("SELECT last_active_org_id FROM users WHERE id = :uid"),
            {"uid": str(user_id)},
        ).mappings().first()
        if not row:
            return None
        return _as_uuid_or_none(row.get("last_active_org_id"))


# ─────────────────────────────────────────────────────────────────────
# WRITE — organizations (used by webhook sync)
# ─────────────────────────────────────────────────────────────────────
def upsert_organization_from_clerk(
    *,
    clerk_org_id: str,
    slug: str,
    name: str,
) -> Organization:
    """
    Called by the Clerk webhook when an `organization.created` or
    `organization.updated` event arrives.

    Idempotent on `clerk_org_id`. The existing `organizations` table
    requires `org_type` (CHECK in 'acadia','customer') — we default
    new orgs created via Clerk to 'customer' since real customer orgs
    will be the common case. The Acadia row itself was seeded by
    migration 055 with org_type='acadia' (preserved from the pre-existing
    row).
    """
    with SessionLocal() as db:
        # Existing row by clerk_org_id?
        existing = db.execute(
            text(
                "SELECT id FROM organizations WHERE clerk_org_id = :cid"
            ),
            {"cid": clerk_org_id},
        ).mappings().first()

        if existing:
            db.execute(
                text(
                    "UPDATE organizations "
                    "   SET slug = :slug, "
                    "       name = :name, "
                    "       updated_at = NOW() "
                    " WHERE clerk_org_id = :cid"
                ),
                {"slug": slug, "name": name, "cid": clerk_org_id},
            )
        else:
            # New org from Clerk — insert with safe defaults. Note that
            # the existing schema's `org_type` CHECK constraint requires
            # one of {'acadia', 'customer'} — we default to 'customer'.
            db.execute(
                text(
                    "INSERT INTO organizations "
                    "    (slug, name, clerk_org_id, org_type) "
                    "VALUES (:slug, :name, :cid, 'customer')"
                ),
                {"slug": slug, "name": name, "cid": clerk_org_id},
            )
        db.commit()

    org = get_organization_by_clerk_id(clerk_org_id)
    assert org is not None, "upsert_organization_from_clerk read returned None"
    return org


# ─────────────────────────────────────────────────────────────────────
# READ + WRITE — access requests
# ─────────────────────────────────────────────────────────────────────
_ACCESS_REQUEST_SELECT = (
    "SELECT id, user_id, organization_id, status, justification, "
    "       responded_by_user_id, response_note, created_at, responded_at "
    "FROM org_access_requests"
)


def create_access_request(
    *,
    user_id: uuid.UUID,
    organization_id: uuid.UUID,
    justification: Optional[str] = None,
) -> Optional[AccessRequest]:
    """
    Create a pending access request. Returns None if a pending request
    already exists for this (user, org) pair — caller should treat that
    as ERR_DUPLICATE_REQUEST and return 409 to the client.
    """
    with SessionLocal() as db:
        existing = db.execute(
            text(
                "SELECT id FROM org_access_requests "
                " WHERE user_id = :uid "
                "   AND organization_id = :oid "
                "   AND status = :pending"
            ),
            {
                "uid": str(user_id),
                "oid": str(organization_id),
                "pending": ACCESS_REQUEST_STATUS_PENDING,
            },
        ).first()
        if existing:
            return None

        db.execute(
            text(
                "INSERT INTO org_access_requests "
                "    (user_id, organization_id, status, justification) "
                "VALUES (:uid, :oid, :status, :justification)"
            ),
            {
                "uid": str(user_id),
                "oid": str(organization_id),
                "status": ACCESS_REQUEST_STATUS_PENDING,
                "justification": justification,
            },
        )
        db.commit()

        row = db.execute(
            text(
                _ACCESS_REQUEST_SELECT
                + " WHERE user_id = :uid "
                  "   AND organization_id = :oid "
                  "   AND status = :pending "
                  " ORDER BY created_at DESC LIMIT 1"
            ),
            {
                "uid": str(user_id),
                "oid": str(organization_id),
                "pending": ACCESS_REQUEST_STATUS_PENDING,
            },
        ).mappings().first()
        return _row_to_access_request(dict(row)) if row else None


def list_pending_access_requests_for_org(
    organization_id: uuid.UUID,
) -> List[AccessRequest]:
    """All pending requests for one org — the admin review queue."""
    with SessionLocal() as db:
        rows = db.execute(
            text(
                _ACCESS_REQUEST_SELECT
                + " WHERE organization_id = :oid AND status = :pending "
                  " ORDER BY created_at ASC"
            ),
            {
                "oid": str(organization_id),
                "pending": ACCESS_REQUEST_STATUS_PENDING,
            },
        ).mappings().all()
        return [_row_to_access_request(dict(r)) for r in rows]
