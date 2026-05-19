"""
backend.observability
=====================

Cross-cutting telemetry primitives — request-scoped context, structured
log formatters, and (later) Sentry init. This package is intentionally
self-contained so the rest of the backend can keep using stdlib
``logging.getLogger("acadia-log-iq")`` and get the new behaviour for
free:

  * Every log record carries ``request_id`` + ``user_id`` (when present)
  * ``LOG_FORMAT=json``  → newline-delimited JSON to stdout (prod)
  * ``LOG_FORMAT=text``  → coloured human-readable output (laptop)

Why a package, not a single module
----------------------------------
Three concerns deserve their own files so each can be reviewed,
unit-tested, and replaced independently:

* ``request_context.py`` — ContextVar plumbing, no logging knowledge.
* ``log_formatters.py``  — pure formatting, no FastAPI knowledge.
* ``log_setup.py``       — wires the two together at process start.
* ``middleware.py``      — FastAPI glue (request_id minting + access log).

Importing this package has zero side effects. ``configure_logging()``
and ``RequestContextMiddleware`` must be called explicitly from
``backend.api`` (see the boot block there for the exact location).
"""

from .log_setup import configure_logging
from .middleware import RequestContextMiddleware
from .request_context import (
    bind_request_id,
    bind_user_id,
    get_request_id,
    get_user_id,
)
from .sentry_setup import init_sentry

__all__ = [
    "configure_logging",
    "RequestContextMiddleware",
    "bind_request_id",
    "bind_user_id",
    "get_request_id",
    "get_user_id",
    "init_sentry",
]
