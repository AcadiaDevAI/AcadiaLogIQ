"""
AWS Secrets Manager integration — single source of truth for runtime config.

Why
---
The legacy backend read every key from ``backend/.env`` via pydantic's
``env_file=``. Production now stores those values in AWS Secrets Manager
under a single JSON object (one secret, many keys). This module fetches
that secret at process startup and pushes the key-value pairs into
``os.environ`` so the rest of the codebase keeps using
``os.getenv(...)`` / pydantic-settings unchanged — zero refactor.

Design
------
* Boto3 picks up credentials via the standard chain (EC2/ECS/EKS IAM
  role, AWS_PROFILE, AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY env
  vars, ``~/.aws/credentials``). **No credentials are ever read from
  application code.**
* The secret is fetched exactly once per process — at startup. Future
  reads go through ``os.environ`` which is already in memory.
* Failure modes never crash the app:
    - On network / permission / parse error → return False, log a
      warning, and (via :func:`bootstrap_environment`) fall back to
      the local ``backend/.env`` file so laptop dev keeps working.
* Disable entirely with ``AWS_SECRETS_DISABLED=true`` — useful on a
  laptop that has no AWS access at all.

Module-level imports are kept narrow on purpose: this file must NOT
trigger ``backend.config`` (which has its own pydantic ``Settings``
instantiation that snapshots ``os.environ`` on first import).
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger("acadia-log-iq")

# Defaults match the secret AWS created for dev. Overridable via env
# (e.g. point at acadialogiq/prod/backend/secrets in prod).
DEFAULT_SECRET_NAME = "acadialogiq/dev/backend/secrets"
DEFAULT_REGION = "us-east-1"


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def load_secrets_from_aws(
    secret_name: Optional[str] = None,
    region_name: Optional[str] = None,
    *,
    override: bool = False,
) -> bool:
    """Fetch a JSON secret from AWS Secrets Manager → ``os.environ``.

    Parameters
    ----------
    secret_name : Optional[str]
        SecretId. Defaults to env ``AWS_SECRETS_SECRET_NAME`` then
        ``acadialogiq/dev/backend/secrets``.
    region_name : Optional[str]
        AWS region. Defaults to env ``AWS_SECRETS_REGION`` then
        ``us-east-1``.
    override : bool
        When True, AWS values overwrite pre-existing ``os.environ``
        entries. When False (default), existing env vars win — this
        lets operators override a single key on the command line
        (``KEY=val docker run ...``) without rotating the secret.

    Returns
    -------
    bool
        True iff at least one key was loaded.

    Failure semantics
    -----------------
    Returns False, never raises. Caller (typically
    :func:`bootstrap_environment`) decides whether to fall back to a
    local ``.env`` file.
    """
    # Honor the kill switch FIRST so callers without AWS access on
    # their laptop don't trigger a 30s boto3 SDK timeout.
    if _truthy(os.environ.get("AWS_SECRETS_DISABLED")):
        logger.info("[secrets] AWS_SECRETS_DISABLED=true — skipping Secrets Manager")
        return False

    resolved_secret = (
        secret_name
        or os.environ.get("AWS_SECRETS_SECRET_NAME")
        or DEFAULT_SECRET_NAME
    )
    resolved_region = (
        region_name
        or os.environ.get("AWS_SECRETS_REGION")
        or os.environ.get("AWS_REGION")
        or DEFAULT_REGION
    )

    # Lazy import: boto3 is a heavy dependency tree — skip the cost
    # when the loader is disabled (the import-time check above).
    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError as exc:
        logger.warning("[secrets] boto3 not installed: %s — skipping AWS loader", exc)
        return False

    try:
        session = boto3.session.Session()
        client = session.client(service_name="secretsmanager", region_name=resolved_region)
        response = client.get_secret_value(SecretId=resolved_secret)
    except (ClientError, BotoCoreError) as exc:
        logger.warning(
            "[secrets] failed to fetch secret=%r region=%s: %s",
            resolved_secret, resolved_region, exc,
        )
        return False
    except Exception as exc:
        # Truly unexpected — never crash startup over secret retrieval.
        logger.warning(
            "[secrets] unexpected error fetching secret=%r: %s",
            resolved_secret, exc,
        )
        return False

    secret_string = response.get("SecretString")
    if not secret_string:
        logger.warning(
            "[secrets] secret=%r has no SecretString (binary?) — skipping",
            resolved_secret,
        )
        return False

    try:
        payload = json.loads(secret_string)
    except json.JSONDecodeError as exc:
        logger.warning("[secrets] secret=%r is not valid JSON: %s", resolved_secret, exc)
        return False

    if not isinstance(payload, dict):
        logger.warning(
            "[secrets] secret=%r is a JSON %s, expected object — skipping",
            resolved_secret, type(payload).__name__,
        )
        return False

    loaded_count = 0
    skipped_existing = 0
    for key, value in payload.items():
        if not isinstance(key, str) or not key:
            continue
        if not override and key in os.environ:
            skipped_existing += 1
            continue
        # All env vars are strings — coerce booleans / numbers / None
        # cleanly so the downstream parsers see canonical forms.
        os.environ[key] = "" if value is None else str(value)
        loaded_count += 1

    if loaded_count == 0:
        logger.info(
            "[secrets] secret=%r contained no new keys to load "
            "(all %d keys already in os.environ — set override=True to "
            "force-overwrite)",
            resolved_secret, skipped_existing,
        )
        return False

    logger.info(
        "Loaded secrets from AWS Secrets Manager "
        "(secret=%r region=%s loaded=%d skipped_existing=%d)",
        resolved_secret, resolved_region, loaded_count, skipped_existing,
    )
    return True


def bootstrap_environment() -> None:
    """Process-startup entry point — call this BEFORE anything that
    reads config.

    Resolution order (each tier fills only the gaps left by the
    previous one — pre-set env vars are never clobbered):

    1. Existing ``os.environ`` — operator-exported values
       (``KEY=val docker run ...``, CI overrides) ALWAYS win.
    2. AWS Secrets Manager — production source of truth.
    3. ``backend/.env`` — migration safety net, populates any keys
       that haven't been moved to AWS yet AND laptop-dev keys when
       AWS is unreachable / disabled.

    Safe to call multiple times.

    Why both AWS and .env?
    ----------------------
    During the migration from .env → AWS Secrets Manager, the AWS
    secret may not yet hold every key. Running both means a key
    can live in either store (or both) without breaking the app.
    Once all keys are in AWS, the .env file can be deleted and this
    function will silently no-op on the dotenv step.
    """
    # Configure root logger early so the success / fallback log lines
    # are visible. ``basicConfig`` is a no-op once handlers exist, so
    # any later config in ``backend/api.py`` is preserved when it
    # re-sets the level.
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

    # ── Tier 2: AWS Secrets Manager ──
    # By default ASM does NOT overwrite keys already in os.environ, so an
    # operator can override a single key on the command line. On EC2 the
    # container is started with docker-compose `env_file: ./backend/.env`,
    # which injects EVERY .env key into os.environ before this runs — that
    # would make ASM a no-op (it skips all already-present keys) and the
    # stale .env values would win. Setting AWS_SECRETS_OVERRIDE=true makes
    # ASM authoritative: it overwrites the env_file-injected values so the
    # secret store is the source of truth in production.
    #
    # IMPORTANT: with override on, any key present in BOTH the ASM secret
    # and the compose `environment:` block is won by ASM. Keep deployment-
    # specific config (APP_ROLE, DB_POOL_SIZE, GUNICORN_*, WORKER_KINDS,
    # STORAGE_TYPE) OUT of the ASM secret so the per-service environment
    # block keeps control.
    aws_override = _truthy(os.environ.get("AWS_SECRETS_OVERRIDE"))
    aws_loaded = load_secrets_from_aws(override=aws_override)

    # ── Tier 3: backend/.env (gap-fill + final fallback) ──
    # python-dotenv is a transitive dep of pydantic-settings so it's
    # already installed. Lazy import keeps this module importable
    # even if dotenv ever gets pruned.
    try:
        from dotenv import load_dotenv
    except ImportError:
        if not aws_loaded:
            logger.info("[secrets] python-dotenv not available; no .env fallback")
        return

    # backend/core/secrets.py → backend/core → backend
    candidate = Path(__file__).resolve().parent.parent / ".env"
    if not candidate.exists():
        if not aws_loaded:
            logger.info(
                "[secrets] no AWS secret loaded and no .env at %s — "
                "the app will start with whatever os.environ already holds",
                candidate,
            )
        return

    # override=False: AWS-loaded values (above) and pre-set env vars
    # both win over the .env file. .env only fills the genuine gaps.
    before = set(os.environ.keys())
    load_dotenv(candidate, override=False)
    filled = sorted(k for k in os.environ.keys() if k not in before)

    if aws_loaded:
        if filled:
            logger.info(
                "[secrets] .env filled %d keys missing from AWS: %s",
                len(filled), ", ".join(filled[:10]) + ("..." if len(filled) > 10 else ""),
            )
        else:
            logger.debug("[secrets] .env loaded — no gaps to fill (AWS had everything)")
    else:
        logger.info(
            "[secrets] fell back to local .env at %s (loaded %d keys)",
            candidate, len(filled),
        )
