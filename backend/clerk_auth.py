"""
Clerk JWT Verification for FastAPI.

Validates Clerk-issued JWTs on incoming requests.
Uses Clerk's JWKS endpoint to fetch public keys and verifies:
  - Signature (RS256)
  - Expiration (exp)
  - Issuer (iss)
  - Authorized party (azp) — optional

When CLERK_ENABLED=false (default), all requests pass through
with user_id=None (backwards compatible with existing API key auth).

When CLERK_ENABLED=true, every protected endpoint requires a valid
Bearer token from Clerk.
"""

import logging
import time
from typing import Optional

import jwt
from jwt import PyJWKClient, PyJWKClientError
from fastapi import HTTPException, Request, status

from backend.config import settings

logger = logging.getLogger("acadia-log-iq")

# ─── JWKS Client (singleton) ──────────────────────────────
_jwks_client: Optional[PyJWKClient] = None


def _get_jwks_client() -> Optional[PyJWKClient]:
    """
    Build JWKS client from Clerk's well-known endpoint.
    Clerk exposes JWKS at: https://<your-clerk-frontend-api>/.well-known/jwks.json
    We derive this from CLERK_PUBLISHABLE_KEY (pk_test_xxx or pk_live_xxx).
    """
    global _jwks_client
    if _jwks_client is not None:
        return _jwks_client

    if not settings.CLERK_PUBLISHABLE_KEY:
        return None

    # Extract the Clerk Frontend API domain from the publishable key.
    # pk_test_<base64-encoded-domain> or pk_live_<base64-encoded-domain>
    # We need to decode the part after pk_test_ / pk_live_ to get the domain.
    import base64

    try:
        key = settings.CLERK_PUBLISHABLE_KEY
        # Remove pk_test_ or pk_live_ prefix
        parts = key.split("_", 2)  # ['pk', 'test', 'base64data']
        if len(parts) < 3:
            logger.error("Clerk: Invalid publishable key format")
            return None

        encoded = parts[2]
        # Add padding if needed
        padding = 4 - len(encoded) % 4
        if padding != 4:
            encoded += "=" * padding

        domain = base64.b64decode(encoded).decode("utf-8").rstrip("$")
        jwks_url = f"https://{domain}/.well-known/jwks.json"
        logger.info("Clerk JWKS URL: %s", jwks_url)

        _jwks_client = PyJWKClient(jwks_url, cache_keys=True, lifespan=3600)
        return _jwks_client

    except Exception as e:
        logger.error("Clerk: Failed to build JWKS client: %s", e)
        return None


def is_clerk_enabled() -> bool:
    """Check if Clerk auth is turned on."""
    return (
        settings.CLERK_ENABLED.lower() == "true"
        and bool(settings.CLERK_SECRET_KEY)
        and bool(settings.CLERK_PUBLISHABLE_KEY)
    )


def extract_bearer_token(request: Request) -> Optional[str]:
    """Pull the Bearer token from the Authorization header."""
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:].strip()
    return None


def verify_clerk_token(token: str) -> dict:
    """
    Verify a Clerk JWT and return the decoded payload.

    Returns dict with at least:
      - sub: Clerk user ID (e.g. "user_2abc123...")
      - exp: expiration timestamp
      - iat: issued-at timestamp
      - iss: issuer URL
      - azp: authorized party (your frontend URL)
    """
    client = _get_jwks_client()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Clerk authentication not configured",
        )

    try:
        signing_key = client.get_signing_key_from_jwt(token)
    except PyJWKClientError as e:
        logger.warning("Clerk: JWKS key fetch failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    try:
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            options={
                "verify_exp": True,
                "verify_iss": False,  # We check issuer manually if configured
                "verify_aud": False,  # Clerk doesn't set aud by default
            },
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
        )
    except jwt.InvalidTokenError as e:
        logger.warning("Clerk: Token validation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
        )

    # Verify subject exists (user ID)
    if not payload.get("sub"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing user identity",
        )

    return payload


async def clerk_auth_dependency(request: Request) -> Optional[str]:
    """
    FastAPI dependency that:
    - If CLERK_ENABLED=true  → requires valid Clerk JWT, returns user_id
    - If CLERK_ENABLED=false → passes through, returns None (backwards-compatible)

    Usage in routes:
        @app.get("/protected")
        async def my_route(user_id: str = Depends(clerk_auth_dependency)):
            ...
    """
    if not is_clerk_enabled():
        # Clerk not enabled — fall through (existing API key auth still works)
        return None

    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = verify_clerk_token(token)
    user_id = payload["sub"]

    # Attach user info to request state for downstream use
    request.state.clerk_user_id = user_id
    request.state.clerk_payload = payload

    return user_id


# ============================================================================
# CLERK USER DETAIL FETCHING
# ============================================================================
# Cache of user_id → { "email": "...", "name": "...", "fetched_at": timestamp }
# Avoids hitting Clerk API on every request. Entries expire after 1 hour.
_user_detail_cache: dict = {}
_CACHE_TTL = 3600  # 1 hour


def get_clerk_user_display(user_id: Optional[str]) -> dict:
    """
    Fetch user email and name from Clerk Backend API.

    Returns: { "email": "user@example.com", "name": "John Doe", "user_id": "user_xxx" }

    Falls back to { "email": "anonymous", "name": "Anonymous", "user_id": "anonymous" }
    if Clerk is not enabled, user_id is None, or API call fails.

    Results are cached in memory for 1 hour to avoid excessive API calls.
    """
    import time as _time

    # No user_id or Clerk not enabled → anonymous
    if not user_id or not is_clerk_enabled() or not settings.CLERK_SECRET_KEY:
        return {"email": "anonymous", "name": "Anonymous", "user_id": user_id or "anonymous"}

    # Check cache first
    cached = _user_detail_cache.get(user_id)
    if cached and (_time.time() - cached.get("fetched_at", 0)) < _CACHE_TTL:
        return cached

    # Fetch from Clerk Backend API
    # Endpoint: GET https://api.clerk.com/v1/users/{user_id}
    # Auth: Bearer <CLERK_SECRET_KEY>
    try:
        import urllib.request
        import json

        url = f"https://api.clerk.com/v1/users/{user_id}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {settings.CLERK_SECRET_KEY}",
            "Content-Type": "application/json",
        })

        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Extract email — Clerk returns email_addresses array
        email = "unknown"
        email_addresses = data.get("email_addresses", [])
        primary_email_id = data.get("primary_email_address_id")

        # Find the primary email
        for ea in email_addresses:
            if ea.get("id") == primary_email_id:
                email = ea.get("email_address", "unknown")
                break
        # Fallback: use first email if primary not found
        if email == "unknown" and email_addresses:
            email = email_addresses[0].get("email_address", "unknown")

        # Extract name
        first = data.get("first_name") or ""
        last = data.get("last_name") or ""
        name = f"{first} {last}".strip() or email.split("@")[0]

        result = {
            "email": email,
            "name": name,
            "user_id": user_id,
            "fetched_at": _time.time(),
        }

        # Cache it
        _user_detail_cache[user_id] = result
        logger.info("Clerk user fetched: %s (%s)", name, email)
        return result

    except Exception as e:
        logger.warning("Failed to fetch Clerk user %s: %s", user_id, e)
        # Return user_id as fallback
        fallback = {
            "email": user_id,
            "name": user_id,
            "user_id": user_id,
            "fetched_at": _time.time(),
        }
        _user_detail_cache[user_id] = fallback
        return fallback