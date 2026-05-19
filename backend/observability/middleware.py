"""
FastAPI middleware — request_id minting + ContextVar binding.

This middleware runs **before** the existing ``log_requests``
middleware (defined in ``backend/api.py`` near the bottom). The
ordering matters: this layer establishes the request_id ContextVar
*before* the access log line is emitted, so the log line carries
the correlation ID.

Why a separate middleware (and not extending ``log_requests``)
--------------------------------------------------------------
* Single responsibility — this one only knows about IDs and context;
  the access log middleware only knows about timing. Either can be
  swapped out without disturbing the other.
* FastAPI's ``add_middleware(...)`` order is LIFO (last added runs
  outermost). Registering this one *after* ``log_requests`` puts it
  closer to the request edge, which is what we want — the
  ContextVar must be bound the moment the request enters Python.

X-Request-ID protocol
---------------------
* If the client sent ``X-Request-ID`` we accept it verbatim (length
  capped to 128 chars and ASCII-printable only — defensive against
  log-injection / header smuggling).
* Otherwise we mint a fresh 12-char hex slug. 12 chars × 4 bits =
  48 bits of entropy, ample for human-friendly request matching
  without bloating every log line.
* The chosen ID is echoed back to the client via the
  ``X-Request-ID`` response header so they can quote it when
  raising a support ticket.

User-ID binding
---------------
The middleware does **not** bind ``user_id``. That happens inside
``backend.clerk_auth.auth_dependency`` once the JWT is verified —
binding it here would require duplicating the verification logic
or trusting an unauthenticated header (which would let any caller
spoof the user_id in logs).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from .request_context import bind_request_id, request_id_var


logger = logging.getLogger("acadia-log-iq")


# Cap on caller-supplied request IDs. Long enough for any reasonable
# upstream tracing system (Datadog, Honeycomb, etc) to be passed
# through verbatim; short enough that a malicious caller can't spam
# CloudWatch with multi-kilobyte fake IDs.
_MAX_CLIENT_REQUEST_ID = 128


def _normalise_client_id(raw: Optional[str]) -> Optional[str]:
    """Accept a caller-supplied X-Request-ID iff it looks safe.

    Safety rules:
      * Strip leading/trailing whitespace.
      * Reject if empty after strip.
      * Reject if longer than _MAX_CLIENT_REQUEST_ID.
      * Reject non-printable / non-ASCII so the value can be safely
        echoed into log lines and HTTP headers.
    Returns the cleaned ID, or ``None`` if we should mint a fresh one.
    """
    if not raw:
        return None
    cleaned = raw.strip()
    if not cleaned:
        return None
    if len(cleaned) > _MAX_CLIENT_REQUEST_ID:
        return None
    # ``isascii`` + ``isprintable`` is a tight allow-list — matches
    # the character classes any sane tracing system actually uses.
    if not cleaned.isascii() or not cleaned.isprintable():
        return None
    return cleaned


def _mint_request_id() -> str:
    """Generate a 12-char hex slug, e.g. ``a8c7d2e14b3f``."""
    return uuid.uuid4().hex[:12]


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Bind a request_id for the lifetime of one HTTP exchange.

    The ContextVar token is reset in a ``finally`` so the worker
    that handles the next request starts from a clean slate — no
    stale ID bleeds across.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        rid = (
            _normalise_client_id(request.headers.get("X-Request-ID"))
            or _mint_request_id()
        )
        token = bind_request_id(rid)
        # The middleware does *not* emit a log line itself — the
        # existing ``log_requests`` access log handles that and now
        # sees the request_id via the ContextFilter on every record.
        try:
            response = await call_next(request)
        finally:
            # Reset BEFORE returning so any post-response callbacks
            # (uvicorn access log, sentry breadcrumbs) see the cleared
            # context rather than the previous request's ID.
            request_id_var.reset(token)
        # Echo the ID back so the client can quote it in support
        # requests. We set this AFTER call_next so user handlers
        # can't accidentally overwrite it.
        response.headers["X-Request-ID"] = rid
        return response
