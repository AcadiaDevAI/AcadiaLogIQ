"""
backend.tenancy.middleware
==========================

Phase 1 multi-tenant: HTTP middleware that resolves the current
organization id from the Clerk JWT and parks it in a ContextVar so
the SQLAlchemy ``after_begin`` hook in ``backend.db.connection`` can
stamp it onto every transaction's Postgres session var ``app.current_org``.

That session var is what the Row Level Security policies (migration 061)
compare against, so the chain is:

    Inbound request
        ↓ JWT decode (best-effort, in this middleware)
        ↓ ContextVar set to UUID
        ↓ Route handler opens SQLAlchemy session
        ↓ SQLAlchemy after_begin event hook reads ContextVar
        ↓ ``SELECT set_config('app.current_org', '<uuid>', true)``
        ↓ Postgres applies RLS policy on every SELECT/INSERT/UPDATE/DELETE
        ↓ Only this org's rows are visible

Failure-soft contract:
    * If the JWT is missing, malformed, or has no org claim, we DO NOT
      abort the request. The ContextVar stays unset, no SET LOCAL fires,
      and (post-RLS) tenant tables become invisible to the request.
      That's exactly what we want — an unauthenticated caller should not
      be able to read tenant data.
    * Routes that DON'T touch tenant tables (health checks, the Clerk
      webhook, SignedOut hits to /auth/register-or-login) keep working
      as before because they don't query RLS-enabled tables.
    * If the JWT decode itself raises (bad signature, expired, etc.) we
      swallow it here. The route-level ``Depends(get_request_context)``
      is where strict auth happens — this middleware is only for the
      "best effort" early stamping that lets RLS work.

Why a middleware instead of a Depends(): not every route uses
``get_request_context``. By binding the ContextVar in middleware, we
cover the entire app surface with one wire, including future routes
that forget the dependency. The dependency stays as the authoritative
authorization check.
"""

from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from backend.db.connection import current_org_id_var

logger = logging.getLogger("acadia-log-iq")


class TenancyContextMiddleware(BaseHTTPMiddleware):
    """
    Bind ``current_org_id_var`` for the lifetime of one HTTP exchange.

    The ContextVar token is reset in a ``finally`` so the next request
    handled by the same worker starts from a clean slate — no stale
    org id bleeds across.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        org_uuid = _try_resolve_org_uuid(request)

        if org_uuid is None:
            # No JWT or no org claim — let the request proceed without
            # an org-scoped DB context. RLS will hide tenant rows from
            # this request (intended for unauth + public routes).
            return await call_next(request)

        token = current_org_id_var.set(org_uuid)
        try:
            return await call_next(request)
        finally:
            current_org_id_var.reset(token)


# ─────────────────────────────────────────────────────────────────────
# JWT → org UUID resolution (best-effort, swallow-on-fail)
# ─────────────────────────────────────────────────────────────────────

def _try_resolve_org_uuid(request: Request) -> Optional[uuid.UUID]:
    """
    Extract the active org id from the Authorization header.

    Returns the org's internal UUID if everything lines up, ``None``
    otherwise. Never raises — any failure path returns ``None`` so the
    request continues unauthenticated (and RLS will hide tenant data
    from it, which is correct).
    """
    auth_header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth_header or not auth_header.lower().startswith("bearer "):
        return None

    token = auth_header[7:].strip()
    if not token:
        return None

    # Local imports keep middleware import cheap at boot — these
    # modules pull SQLAlchemy + httpx eagerly otherwise.
    try:
        from backend.clerk_auth import verify_clerk_token
        from backend.tenancy.context import _extract_org_from_claims
    except Exception as exc:  # pragma: no cover — import-time failure is fatal at startup
        logger.warning(
            "[tenancy.middleware] import failed; tenant scoping disabled this request: %s",
            exc,
        )
        return None

    try:
        claims = verify_clerk_token(token)
    except Exception:
        # Bad signature, expired, malformed — silently degrade. The
        # route-level dependency will return the proper 401 if the
        # route actually requires auth.
        return None

    try:
        # _extract_org_from_claims returns (internal_uuid, slug, role).
        # The repository lookup happens inside it, so we get the
        # database UUID directly — no extra session needed here.
        org_uuid, _slug, _role = _extract_org_from_claims(claims)
    except Exception as exc:
        logger.warning(
            "[tenancy.middleware] _extract_org_from_claims raised: %s",
            exc,
        )
        return None

    return org_uuid
