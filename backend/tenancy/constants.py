"""
backend.tenancy.constants
=========================

Centralized constants for the multi-tenant layer.

Why a separate module? Two reasons:
    1. These values are referenced from migrations, tests, route handlers,
       and the webhook sync. One place to change them.
    2. Importing constants from a small module is cheaper than pulling
       in the full tenancy.repository / tenancy.clerk_sync surface.
"""

from __future__ import annotations

import uuid

# ─────────────────────────────────────────────────────────────────────
# Seed organization — Acadia Consultants
# ─────────────────────────────────────────────────────────────────────
# This UUID is the SURVIVING id from the existing `organizations` table
# created by an April 2026 multi-tenant spike. Phase 0 ADOPTS this UUID
# rather than introducing a new one, because the existing row already
# has 98 + 17 + 1 FK references from tier1_sessions, intake_extractions,
# and chat_sessions. Replacing the id would break those FKs.
#
# Migration 055 normalizes the row's name/slug/branding without
# changing the id.
ACADIA_ORG_ID: uuid.UUID = uuid.UUID("76c36d23-b8c1-4b81-b121-bef5e1b10b3b")
ACADIA_ORG_SLUG: str = "acadia-consultants"

# ─────────────────────────────────────────────────────────────────────
# Role names (org-level)
# ─────────────────────────────────────────────────────────────────────
# Enforced by the CHECK constraint re-added in migration 055. If you add
# a value here, also update the constraint.
ROLE_ADMIN: str = "admin"
ROLE_MEMBER: str = "member"

VALID_ROLES: frozenset[str] = frozenset({ROLE_ADMIN, ROLE_MEMBER})

# ─────────────────────────────────────────────────────────────────────
# Access request lifecycle
# ─────────────────────────────────────────────────────────────────────
# Enforced by the CHECK constraint on org_access_requests.status.
ACCESS_REQUEST_STATUS_PENDING: str = "pending"
ACCESS_REQUEST_STATUS_APPROVED: str = "approved"
ACCESS_REQUEST_STATUS_DENIED: str = "denied"
ACCESS_REQUEST_STATUS_WITHDRAWN: str = "withdrawn"

VALID_ACCESS_REQUEST_STATUSES: frozenset[str] = frozenset({
    ACCESS_REQUEST_STATUS_PENDING,
    ACCESS_REQUEST_STATUS_APPROVED,
    ACCESS_REQUEST_STATUS_DENIED,
    ACCESS_REQUEST_STATUS_WITHDRAWN,
})

# Re-exports for back-compat with the older Phase 0 draft. Don't remove
# without grepping callers — these names are part of the public API.
MEMBERSHIP_STATUS_ACTIVE: str = "active"
MEMBERSHIP_STATUS_REMOVED: str = "removed"

# ─────────────────────────────────────────────────────────────────────
# Platform super-admin role
# ─────────────────────────────────────────────────────────────────────
# Stored in Clerk's public metadata as `platform_role: super_admin`. The
# Clerk JWT carries it; we read it from the JWT claims (not from the DB).
# Super-admins can manage all orgs and impersonate org members.
PLATFORM_ROLE_SUPER_ADMIN: str = "super_admin"
PLATFORM_ROLE_USER: str = "user"

# ─────────────────────────────────────────────────────────────────────
# HTTP error codes used by the tenancy layer
# ─────────────────────────────────────────────────────────────────────
# A small set of stable string codes the frontend can switch on without
# parsing English error messages.
ERR_NO_MEMBERSHIP: str = "no_membership"
ERR_ORG_NOT_FOUND: str = "org_not_found"
ERR_ORG_INACTIVE: str = "org_inactive"
ERR_DUPLICATE_REQUEST: str = "duplicate_request"
ERR_NO_AUTH: str = "no_auth"
ERR_UNKNOWN_USER: str = "unknown_user"
