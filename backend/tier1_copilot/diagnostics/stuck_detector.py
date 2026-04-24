"""Sprint 7 — stuck detection (pure logic, no side effects).

`should_nudge()` is called by the /tier1/session/{id}/status endpoint on
every poll. If it returns True, the session row gets its
stuck_nudge_shown flipped to True and the frontend renders the modal.
The flip happens in the caller so this module stays side-effect-free
(easy to unit-test, no DB mock required).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from backend.config import settings


def should_nudge(session: Any, now: datetime) -> bool:
    """Return True when the session qualifies for a stuck nudge.

    Criteria (first match wins):
      - already-nudged → False (never fire twice)
      - resolved / escalated → False
      - 2+ thumbs-down → True
      - elapsed >= TIER1_STUCK_THRESHOLD_SECONDS → True
      - otherwise → False

    `session` is duck-typed: any object with the attributes
    `stuck_nudge_shown`, `resolved`, `escalated`, `thumbs_down_count`,
    `created_at` satisfies the check. Works with both the ORM row and
    the Tier1SessionRecord dataclass.
    """
    if getattr(session, "stuck_nudge_shown", False):
        return False
    if getattr(session, "resolved", False) or getattr(session, "escalated", False):
        return False
    if int(getattr(session, "thumbs_down_count", 0) or 0) >= 2:
        return True

    created_at = getattr(session, "created_at", None)
    if not created_at:
        return False
    try:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    except AttributeError:
        return False

    elapsed = (now - created_at).total_seconds()
    threshold = int(getattr(settings, "TIER1_STUCK_THRESHOLD_SECONDS", 480))
    return elapsed >= threshold


def elapsed_seconds(session: Any, now: datetime) -> int:
    """Return integer seconds since session creation, clamped at 0."""
    created_at = getattr(session, "created_at", None)
    if not created_at:
        return 0
    try:
        if created_at.tzinfo is None:
            created_at = created_at.replace(tzinfo=timezone.utc)
    except AttributeError:
        return 0
    return max(0, int((now - created_at).total_seconds()))
