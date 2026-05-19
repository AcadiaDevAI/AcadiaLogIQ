"""
Request-scoped context plumbing.

Two ``contextvars.ContextVar`` slots — ``request_id`` and ``user_id`` —
that follow a request through every ``await`` and ``asyncio.to_thread``
call without callers needing to plumb them by hand.

Why ContextVars (and not plain globals, threadlocal, or a parameter)
-------------------------------------------------------------------
* ``contextvars`` is the Python-native primitive for *implicit*
  request-scoped state. asyncio's event loop copies the current
  Context onto every scheduled coroutine, so a value bound in a
  middleware is visible inside ``await`` chains downstream.
* ``asyncio.to_thread`` *also* copies the calling Context onto the
  worker thread, so values bound here are visible inside the
  threadpool LLM / DB workers used by tier1 / RCA / Gap Analysis
  generators. (This is the bit that makes the request_id show up
  in Bedrock log lines without any extra wiring.)
* Threadlocal would work for sync code but break the async pipeline.
  Plain globals would be a data-race nightmare under concurrent
  requests. A function parameter would require touching every
  signature in the codebase — a non-starter given the size.

These are *not* a security boundary — anything that reads the user_id
for an authorization decision must still go through
``backend.clerk_auth`` and validate the JWT. The user_id stored here
is purely for log correlation.

Public API
----------
* ``bind_request_id(value)``  → token (call ``request_id_var.reset(token)``
  to undo); typically used by the middleware only.
* ``bind_user_id(value)``     → same shape; callers attach a user_id
  *after* auth has validated the JWT.
* ``get_request_id() -> str | None``
* ``get_user_id() -> str | None``
"""

from __future__ import annotations

from contextvars import ContextVar, Token
from typing import Optional


# Default = ``None`` so a log line emitted from an unrelated worker
# thread (boot logs, BM25 rebuild, scheduled jobs) carries an empty
# request_id rather than a misleading one from an earlier request.
request_id_var: ContextVar[Optional[str]] = ContextVar(
    "acadia_request_id", default=None,
)
user_id_var: ContextVar[Optional[str]] = ContextVar(
    "acadia_user_id", default=None,
)


def bind_request_id(value: Optional[str]) -> Token:
    """Attach ``value`` to the current asyncio Context.

    Returns a ``Token`` that callers should pass to
    ``request_id_var.reset(token)`` in a ``finally`` block. The
    middleware uses this to scrub the value once a response returns,
    so a worker that picks up the next request doesn't inherit the
    previous one's ID by accident.
    """
    return request_id_var.set(value)


def bind_user_id(value: Optional[str]) -> Token:
    """Attach ``value`` to the current Context. See ``bind_request_id``."""
    return user_id_var.set(value)


def get_request_id() -> Optional[str]:
    """Read the request_id bound to the current Context, or ``None``."""
    return request_id_var.get()


def get_user_id() -> Optional[str]:
    """Read the user_id bound to the current Context, or ``None``."""
    return user_id_var.get()
