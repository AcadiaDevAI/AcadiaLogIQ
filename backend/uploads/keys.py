"""
S3 key generation for tenant-scoped uploads.

Layout (flat, human-browsable):

    {prefix}/{tenant_slug}/{safe_filename}
    └──┬──┘ └─────┬──────┘ └──────┬─────┘
       │         │                │
       │         │                └── original filename, sanitized
       │         │                    to S3-/FS-safe characters. Same
       │         │                    name twice = S3 overwrite (last
       │         │                    write wins) — matches what users
       │         │                    expect from a desktop folder.
       │         └── human-readable slug — today derived from the
       │             user's email (e.g. ``maruthi.phani``); falls back
       │             to the Clerk user_id when no email is available.
       │             Phase 2 will swap to a real tenants.slug FK; only
       │             ``derive_tenant_slug`` changes.
       └── env-wide root (``tenants`` by default)

Phase-2 medallion tiers (``silver/``, ``gold/``) will appear as sibling
prefixes under the SAME tenant slug, so the human-browse layout in S3
console grows naturally without breaking Phase-1 ``raw/``-less keys.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger("acadia-log-iq")

# Permissive but safe for S3 keys and filesystem use.
# Allowed: letters, digits, dot, underscore, hyphen.
# Everything else collapses to underscore.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")
_UNSAFE_SLUG_CHARS = re.compile(r"[^a-z0-9._-]+")


# ── Public helpers ───────────────────────────────────────────────────


def new_job_id() -> str:
    """Return a hex job identifier suitable for DB rows.

    Not used in the S3 key today (kept for ingestion_jobs row IDs).
    """
    return uuid.uuid4().hex


def sanitize_filename(name: Optional[str], *, max_len: int = 200) -> str:
    """Return a filesystem- and URL-safe version of ``name``.

    - Strips any directory components (defends against ``../`` etc.)
    - Replaces every char outside ``[A-Za-z0-9._-]`` with ``_``
    - Trims leading/trailing dots and underscores
    - Caps the result at ``max_len`` characters so keys stay short
    - Falls back to ``file`` if input is empty after sanitization
    """
    base = Path(name or "").name or "file"
    base = _UNSAFE_CHARS.sub("_", base).strip("._")
    if not base:
        base = "file"
    return base[:max_len]


def _slugify(value: str) -> str:
    """Lower-case, replace unsafe chars with underscore, trim, cap at 64.

    Returns empty string when nothing usable survives.
    """
    cleaned = _UNSAFE_SLUG_CHARS.sub("_", (value or "").strip().lower()).strip("._")
    return cleaned[:64] if cleaned else ""


def derive_tenant_slug(user_id: Optional[str]) -> str:
    """Return a human-readable, S3-safe tenant slug for path composition.

    Resolution order (best-effort, never raises). Each tier is tried
    only if the previous returned an unusable value:

    1. **Local users table** (``backend.vector_store.get_user_by_clerk_id``).
       Populated by the frontend's useAutoRegister on login. Holds the
       canonical email + full_name we already render in the UI. This is
       the fast and reliable source — no network call, no Clerk-side
       rate limits, no 403s from key mismatches.

       Slug source: first whitespace-delimited word of ``full_name``,
       else the local part of ``email``.

    2. **Clerk Backend API** (``get_clerk_user_display``). Used only
       when the local users row hasn't synced yet (a brand-new sign-in
       with no Auto-Register completion). Still cached for 1h.

    3. **Raw user_id**. Final fallback when both above are unavailable.

    4. ``anonymous`` when no ``user_id`` is supplied.

    Phase 2 (real tenants table) replaces this body with a tenants FK
    lookup. The S3 key shape stays the same so no object migration.
    """
    uid = (user_id or "").strip()
    if not uid:
        return "anonymous"

    # ── Tier 1: local users table (fast path) ───────────────────
    try:
        from backend.vector_store import get_user_by_clerk_id

        row = get_user_by_clerk_id(uid)
        if row:
            full_name = (row.get("full_name") or "").strip()
            email = (row.get("email") or "").strip().lower()

            # Prefer first name (first whitespace-delimited token).
            if full_name:
                first = full_name.split(maxsplit=1)[0]
                slug = _slugify(first)
                if slug:
                    logger.info(
                        "[upload.tenant_slug] resolved via local users "
                        "table user=%s slug=%s (from full_name)",
                        uid, slug,
                    )
                    return slug

            # Else fall back to email's local part.
            if email and "@" in email:
                slug = _slugify(email.split("@", 1)[0])
                if slug:
                    logger.info(
                        "[upload.tenant_slug] resolved via local users "
                        "table user=%s slug=%s (from email)",
                        uid, slug,
                    )
                    return slug
    except Exception as exc:
        logger.warning(
            "[upload.tenant_slug] local users lookup failed user=%s err=%s",
            uid, exc,
        )

    # ── Tier 2: Clerk Backend API (network call, cached 1h) ─────
    try:
        from backend.clerk_auth import get_clerk_user_display

        info = get_clerk_user_display(uid)
        name = (info.get("name") or "").strip()
        email = (info.get("email") or "").strip().lower()

        # ``get_clerk_user_display`` returns name=user_id when the API
        # call itself failed — detect and skip in that case.
        if name and name != uid:
            slug = _slugify(name.split(maxsplit=1)[0])
            if slug:
                return slug
        if email and email != uid and email != "anonymous" and "@" in email:
            slug = _slugify(email.split("@", 1)[0])
            if slug:
                return slug
    except Exception as exc:
        logger.warning(
            "[upload.tenant_slug] Clerk API lookup failed user=%s err=%s",
            uid, exc,
        )

    # ── Tier 3: raw user_id fallback ────────────────────────────
    logger.warning(
        "[upload.tenant_slug] falling back to user_id (no local row, no "
        "Clerk name/email); the user may not have completed auto-register: "
        "user_id=%s", uid,
    )
    return uid


# Backward-compat alias — some callers may still import this name.
derive_tenant_id = derive_tenant_slug


def build_upload_key(
    *,
    prefix: str,
    tenant_slug: str,
    filename: str,
) -> str:
    """Compose the canonical S3 key for an upload.

    Flat layout: ``{prefix}/{tenant_slug}/{safe_filename}``. Same
    filename uploaded twice overwrites in S3 (last write wins), which
    matches what end users expect from a desktop folder.
    """
    safe = sanitize_filename(filename)
    return f"{prefix.strip('/')}/{tenant_slug}/{safe}"
