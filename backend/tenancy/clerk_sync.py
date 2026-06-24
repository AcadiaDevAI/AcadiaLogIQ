"""
backend.tenancy.clerk_sync
==========================

Keeps our `organizations` and `memberships` tables in sync with Clerk
via webhook events.

Clerk is the SOURCE OF TRUTH. If Clerk and our DB ever disagree, Clerk
wins. The webhook is the one-way replication channel.

Schema reality check:
    * `memberships.user_id` is a UUID referencing `users.id` (not
      clerk_id). The webhook handler resolves Clerk's external user_xxx
      id through `users.clerk_id → users.id` before upserting.
    * If a webhook arrives for a Clerk user we don't have a row for
      (their /auth/register-or-login hasn't fired yet), we defensively
      skip the membership write and log a warning. The reconciliation
      helper `sync_user_memberships_from_clerk` catches up after the
      user row is created on next login.

Security:
    * Clerk signs webhook requests with Svix-format headers.
    * Verification happens BEFORE we trust any field in the body.
    * If `CLERK_WEBHOOK_SIGNING_SECRET` isn't configured the endpoint
      refuses everything — failing closed.

Idempotency:
    * Clerk can send the same event twice. All sync operations are
      idempotent (UPSERT on natural keys). Repeat delivery is harmless.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from base64 import b64decode, b64encode
from typing import Any, Dict, Optional

from backend.config import settings
from backend.tenancy.constants import (
    ROLE_ADMIN,
    ROLE_MEMBER,
    VALID_ROLES,
)
from backend.tenancy.repository import (
    deactivate_membership,
    get_organization_by_clerk_id,
    resolve_user_id,
    upsert_membership,
    upsert_organization_from_clerk,
)

logger = logging.getLogger("acadia-log-iq")


# ─────────────────────────────────────────────────────────────────────
# Webhook signature verification (Svix format used by Clerk)
# ─────────────────────────────────────────────────────────────────────
_WEBHOOK_MAX_TIMESTAMP_SKEW_SECONDS: int = 5 * 60   # 5 minutes
_WEBHOOK_VERSION_PREFIX: str = "v1,"


def _decode_signing_secret(secret: str) -> Optional[bytes]:
    """Clerk's secret is delivered as `whsec_<base64-of-raw-key>`."""
    if not secret:
        return None
    cleaned = secret.strip()
    if cleaned.startswith("whsec_"):
        cleaned = cleaned[len("whsec_"):]
    try:
        padding = 4 - (len(cleaned) % 4)
        if padding != 4:
            cleaned += "=" * padding
        return b64decode(cleaned)
    except Exception as exc:  # noqa: BLE001
        logger.warning("[tenancy.webhook] failed to decode signing secret: %s", exc)
        return None


def verify_webhook_signature(
    *,
    body: bytes,
    svix_id: Optional[str],
    svix_timestamp: Optional[str],
    svix_signature: Optional[str],
) -> bool:
    """
    True when ALL of these hold:
        * Signing secret is configured.
        * svix-id, svix-timestamp, svix-signature headers are present.
        * Timestamp is within the allowed skew window.
        * At least one v1 signature in svix-signature matches the
          computed HMAC of "{id}.{ts}.{body}".
    """
    if not svix_id or not svix_timestamp or not svix_signature:
        logger.warning("[tenancy.webhook] missing svix-* headers")
        return False

    raw_secret = getattr(settings, "CLERK_WEBHOOK_SIGNING_SECRET", None)
    if not raw_secret:
        logger.error(
            "[tenancy.webhook] CLERK_WEBHOOK_SIGNING_SECRET is not set — "
            "refusing to process webhook (failing closed)."
        )
        return False

    secret_bytes = _decode_signing_secret(raw_secret)
    if not secret_bytes:
        return False

    try:
        ts = int(svix_timestamp)
    except (TypeError, ValueError):
        logger.warning("[tenancy.webhook] invalid svix-timestamp: %r", svix_timestamp)
        return False
    skew = abs(int(time.time()) - ts)
    if skew > _WEBHOOK_MAX_TIMESTAMP_SKEW_SECONDS:
        logger.warning(
            "[tenancy.webhook] timestamp skew %ds exceeds limit %ds",
            skew, _WEBHOOK_MAX_TIMESTAMP_SKEW_SECONDS,
        )
        return False

    signed_payload = f"{svix_id}.{svix_timestamp}.".encode("utf-8") + body
    expected = b64encode(
        hmac.new(secret_bytes, signed_payload, hashlib.sha256).digest()
    ).decode("ascii")

    for entry in svix_signature.split(" "):
        entry = entry.strip()
        if not entry.startswith(_WEBHOOK_VERSION_PREFIX):
            continue
        candidate = entry[len(_WEBHOOK_VERSION_PREFIX):]
        if hmac.compare_digest(candidate, expected):
            return True

    logger.warning("[tenancy.webhook] no signature matched expected HMAC")
    return False


# ─────────────────────────────────────────────────────────────────────
# Event dispatch
# ─────────────────────────────────────────────────────────────────────
def handle_clerk_webhook_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single VERIFIED Clerk webhook event. Returns a dict with
    `handled` (bool) and `event_type` (str) for the HTTP response.
    """
    event_type = event.get("type") or ""
    data = event.get("data") or {}

    handlers = {
        "organization.created": _handle_org_upsert,
        "organization.updated": _handle_org_upsert,
        "organization.deleted": _handle_org_deleted,
        "organizationMembership.created": _handle_membership_upsert,
        "organizationMembership.updated": _handle_membership_upsert,
        "organizationMembership.deleted": _handle_membership_deleted,
    }

    handler = handlers.get(event_type)
    if handler is None:
        logger.info("[tenancy.webhook] ignoring event type=%s", event_type)
        return {"handled": False, "event_type": event_type, "reason": "no_handler"}

    try:
        handler(data)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "[tenancy.webhook] handler %s raised on event type=%s: %s",
            handler.__name__, event_type, exc,
        )
        return {"handled": False, "event_type": event_type, "reason": "handler_error"}

    return {"handled": True, "event_type": event_type}


# ─────────────────────────────────────────────────────────────────────
# Individual event handlers
# ─────────────────────────────────────────────────────────────────────
def _handle_org_upsert(data: Dict[str, Any]) -> None:
    """Clerk org created or updated — mirror into organizations table."""
    clerk_org_id = data.get("id")
    slug = data.get("slug") or ""
    name = data.get("name") or slug or "Untitled"
    if not clerk_org_id or not slug:
        logger.warning(
            "[tenancy.webhook] org event missing id or slug: %s", data,
        )
        return
    upsert_organization_from_clerk(
        clerk_org_id=clerk_org_id, slug=slug, name=name,
    )
    logger.info(
        "[tenancy.webhook] org upserted clerk_id=%s slug=%s",
        clerk_org_id, slug,
    )


def _handle_org_deleted(data: Dict[str, Any]) -> None:
    """Clerk org deleted — mark as inactive via deactivated_at."""
    from sqlalchemy import text
    from backend.db.connection import SessionLocal

    clerk_org_id = data.get("id")
    if not clerk_org_id:
        return
    with SessionLocal() as db:
        db.execute(
            text(
                "UPDATE organizations "
                "   SET deactivated_at = NOW(), updated_at = NOW() "
                " WHERE clerk_org_id = :cid "
                "   AND deactivated_at IS NULL"
            ),
            {"cid": clerk_org_id},
        )
        db.commit()
    logger.info(
        "[tenancy.webhook] org deactivated clerk_id=%s", clerk_org_id,
    )


def _handle_membership_upsert(data: Dict[str, Any]) -> None:
    """
    Clerk membership created or updated. Translates Clerk's payload to
    our shape and upserts.

    Resolves Clerk's user_xxx id to our internal users.id UUID via the
    users table. If the local users row doesn't exist yet (the user
    hasn't hit /auth/register-or-login), we log and skip — the
    register-or-login flow calls `sync_user_memberships_from_clerk`
    which will pick this membership up on the user's first login.
    """
    public_user_data = data.get("public_user_data") or {}
    organization = data.get("organization") or {}
    role_raw = data.get("role") or ""

    user_clerk_id = public_user_data.get("user_id")
    clerk_org_id = organization.get("id")

    if not user_clerk_id or not clerk_org_id:
        logger.warning(
            "[tenancy.webhook] membership event missing user_id or "
            "organization.id: %s", data,
        )
        return

    # Translate role vocabulary. Clerk arrives as "org:admin"/"org:member".
    role = role_raw.split(":", 1)[1] if role_raw.startswith("org:") else role_raw
    if role not in VALID_ROLES:
        # Defensive default — unknown roles become 'member'.
        logger.warning(
            "[tenancy.webhook] unknown role %r — defaulting to %s",
            role, ROLE_MEMBER,
        )
        role = ROLE_MEMBER

    # Resolve internal user UUID. Skip if missing — login flow catches up.
    user_id = resolve_user_id(user_clerk_id)
    if user_id is None:
        logger.info(
            "[tenancy.webhook] membership event for user=%s but no local "
            "users row yet — deferring until register-or-login",
            user_clerk_id,
        )
        return

    # Resolve internal org UUID; create placeholder if Clerk delivered
    # the membership event before the org event.
    org = get_organization_by_clerk_id(clerk_org_id)
    if not org:
        slug = organization.get("slug") or clerk_org_id
        name = organization.get("name") or slug
        org = upsert_organization_from_clerk(
            clerk_org_id=clerk_org_id, slug=slug, name=name,
        )
        logger.info(
            "[tenancy.webhook] out-of-order: created placeholder org for "
            "clerk_org_id=%s ahead of org event", clerk_org_id,
        )

    upsert_membership(
        user_id=user_id,
        organization_id=org.id,
        role=role,
        is_active=True,
    )
    logger.info(
        "[tenancy.webhook] membership upserted user=%s org=%s role=%s",
        user_clerk_id, org.slug, role,
    )


def _handle_membership_deleted(data: Dict[str, Any]) -> None:
    """Clerk membership removed — deactivate locally."""
    public_user_data = data.get("public_user_data") or {}
    organization = data.get("organization") or {}

    user_clerk_id = public_user_data.get("user_id")
    clerk_org_id = organization.get("id")

    if not user_clerk_id or not clerk_org_id:
        return

    user_id = resolve_user_id(user_clerk_id)
    org = get_organization_by_clerk_id(clerk_org_id)
    if user_id is None or org is None:
        return

    deactivate_membership(user_id=user_id, organization_id=org.id)
    logger.info(
        "[tenancy.webhook] membership deactivated user=%s org=%s",
        user_clerk_id, org.slug,
    )


# ─────────────────────────────────────────────────────────────────────
# One-shot sync helper for /auth/register-or-login
# ─────────────────────────────────────────────────────────────────────
def sync_user_memberships_from_clerk(user_clerk_id: str) -> int:
    """
    Pull the user's current org memberships from Clerk and reconcile our
    table. Belt-and-suspenders on top of the webhook. Never raises —
    sync failure must not break login.
    """
    if not getattr(settings, "CLERK_SECRET_KEY", None):
        return 0

    try:
        import urllib.request

        url = (
            f"https://api.clerk.com/v1/users/{user_clerk_id}"
            "/organization_memberships?limit=100"
        )
        # Cloudflare (which fronts api.clerk.com) 403s the default
        # "Python-urllib/x.y" User-Agent. Any explicit UA passes — without
        # this header every membership sync silently fails with HTTP 403.
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {settings.CLERK_SECRET_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "AcadiaLogIQ/1.0",
        })
        with urllib.request.urlopen(req, timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))

        items = payload.get("data") if isinstance(payload, dict) else payload
        if not isinstance(items, list):
            items = []

        processed = 0
        for item in items:
            try:
                _handle_membership_upsert(item)
                processed += 1
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "[tenancy.sync] per-membership upsert failed: %s", exc,
                )
        logger.info(
            "[tenancy.sync] reconciled %d Clerk membership(s) for user=%s",
            processed, user_clerk_id,
        )
        return processed
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "[tenancy.sync] sync_user_memberships_from_clerk failed for "
            "user=%s: %s", user_clerk_id, exc,
        )
        return 0
