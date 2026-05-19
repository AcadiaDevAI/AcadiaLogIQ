"""
Shared rate-limiter — keyed on Clerk user_id with IP fallback.

Why a separate module
---------------------
The slowapi ``Limiter`` is a singleton that the main FastAPI app
constructs at import time. Sub-routers (RCA, Gap Analysis) need to
apply ``@limiter.limit(...)`` decorators on their own endpoints, but
importing from ``backend.api`` would create a circular dependency
because api.py imports the sub-routers. Factoring the Limiter here
breaks the cycle cleanly: api.py imports from observability, sub-
routers import from observability, no loops.

Why per-user (and not per-IP)
-----------------------------
Behind an ALB or a corporate NAT, many engineers share one IP. A
per-IP limit on the Generate endpoints would let one noisy engineer
brick the rest of the team. The Phase 4 design pivots to per-user
(Clerk user_id from the request) so each engineer gets their own
budget.

Fallback when no user_id is bound (e.g. unauthenticated routes,
boot diagnostics, internal pings) → fall back to the remote address.
Same key-function pattern slowapi has always supported.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from .request_context import get_user_id


def _per_user_key(request: Request) -> str:
    """Return a stable string identifying the caller for rate-limit
    purposes. Prefers the authenticated Clerk user_id (set by
    auth_dependency on every protected request); falls back to the
    remote IP for anonymous endpoints.

    The user_id ContextVar is populated by ``auth_dependency`` in
    ``backend/api.py`` AFTER the JWT has been validated, so a caller
    can't spoof another user's rate-limit bucket by sending a header.
    """
    uid = get_user_id()
    if uid:
        return f"user:{uid}"
    return f"ip:{get_remote_address(request)}"


# Single module-level Limiter. Importing this module is cheap (no
# DB / network) so api.py + every sub-router can import freely.
limiter = Limiter(
    key_func=_per_user_key,
    # Default limits — generous; per-endpoint overrides go via
    # ``@limiter.limit("5/minute")`` on the handler.
    default_limits=["1000/minute"],
)
