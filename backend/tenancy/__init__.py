"""
backend.tenancy
===============

Phase 0 multi-tenant foundation. Adapted to the existing `organizations`
and `memberships` tables present in the database since April 2026.

Public surface (what other code should import):

    from backend.tenancy import (
        # Per-request context (active org for THIS request)
        RequestContext,
        get_request_context,

        # Read access to org / membership records
        get_organization_by_slug,
        list_organizations_for_user,
        is_member_of_org,
        resolve_user_id,

        # Constants
        ACADIA_ORG_ID,
        ACADIA_ORG_SLUG,
        ROLE_ADMIN,
        ROLE_MEMBER,
    )

Design notes are in README.md inside this directory.

Phase 0 is purely additive — nothing in this package is read by the
business hot path yet. Phase 2 wires the RequestContext into queries.
"""

from backend.tenancy.constants import (
    ACADIA_ORG_ID,
    ACADIA_ORG_SLUG,
    ACCESS_REQUEST_STATUS_APPROVED,
    ACCESS_REQUEST_STATUS_DENIED,
    ACCESS_REQUEST_STATUS_PENDING,
    MEMBERSHIP_STATUS_ACTIVE,
    ROLE_ADMIN,
    ROLE_MEMBER,
)
from backend.tenancy.context import get_request_context
from backend.tenancy.models import (
    AccessRequest,
    Membership,
    Organization,
    RequestContext,
)
from backend.tenancy.repository import (
    get_membership,
    get_organization_by_id,
    get_organization_by_slug,
    is_member_of_org,
    list_listable_organizations,
    list_organizations_for_user,
    resolve_user_id,
    set_last_active_org,
)

__all__ = [
    # Constants
    "ACADIA_ORG_ID",
    "ACADIA_ORG_SLUG",
    "ROLE_ADMIN",
    "ROLE_MEMBER",
    "MEMBERSHIP_STATUS_ACTIVE",
    "ACCESS_REQUEST_STATUS_PENDING",
    "ACCESS_REQUEST_STATUS_APPROVED",
    "ACCESS_REQUEST_STATUS_DENIED",
    # Models
    "RequestContext",
    "Organization",
    "Membership",
    "AccessRequest",
    # Context
    "get_request_context",
    # Repository (read-only public surface)
    "get_organization_by_id",
    "get_organization_by_slug",
    "list_organizations_for_user",
    "list_listable_organizations",
    "is_member_of_org",
    "get_membership",
    "set_last_active_org",
    "resolve_user_id",
]
