"""
Database connection layer for PostgreSQL.

Single SQLAlchemy engine + session factory shared by every module.

Pool sizing is env-driven via ``backend.config.Settings``:
    DB_POOL_SIZE       — baseline connections kept open per worker
    DB_MAX_OVERFLOW    — burst capacity above pool_size
    DB_POOL_RECYCLE    — seconds before recycling a connection
    DB_POOL_PRE_PING   — connection-validity ping on checkout

Why env-driven:
* Laptop dev wants a generous pool (10 + 20) so manual scripts don't
  starve the API.
* Production Fargate tasks must size the pool tight so the total
  connection demand fits inside the RDS ``max_connections`` budget.
  See ``backend/config.py`` for the connection-budget math.

Boot logs the resolved pool config so an operator can confirm the
production override actually landed.
"""

import logging
import uuid
from contextvars import ContextVar
from typing import Optional

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from backend.config import settings


logger = logging.getLogger("acadia-log-iq")


engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_recycle=settings.DB_POOL_RECYCLE,
    future=True,
)


# Boot-time confirmation of resolved pool config. Helps catch the
# "I thought I set DB_POOL_SIZE=3 in the task definition" misconfig
# class — operators see the actual value in CloudWatch.
logger.info(
    "[db.pool] pool_size=%d max_overflow=%d pool_recycle=%ds pre_ping=%s",
    settings.DB_POOL_SIZE, settings.DB_MAX_OVERFLOW,
    settings.DB_POOL_RECYCLE, settings.DB_POOL_PRE_PING,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ─────────────────────────────────────────────────────────────────────
# Phase 1 multi-tenant: request-scoped organization id + RLS plumbing
# ─────────────────────────────────────────────────────────────────────
#
# This ContextVar is set by ``backend.tenancy.middleware.TenancyContextMiddleware``
# at the start of every authenticated HTTP request (read out of the
# Clerk JWT's org claim). It's read by the SQLAlchemy ``after_begin``
# event hook below.
#
# Why a ContextVar (not a global): asyncio Task isolation. Each request
# runs in its own task; ContextVars are per-task so two concurrent
# requests never see each other's org id.
#
# Default ``None`` — boot-time queries, ad-hoc scripts, scheduled jobs
# with no request context inherit the fallback behavior described
# below (see ``DEFAULT_ORG_ID_FOR_NO_CONTEXT``).
current_org_id_var: ContextVar[Optional[uuid.UUID]] = ContextVar(
    "current_org_id", default=None,
)


# ─────────────────────────────────────────────────────────────────────
# 🛑 SINGLE-TENANT FALLBACK — DELETE THIS BEFORE ONBOARDING TENANT #2 🛑
# ─────────────────────────────────────────────────────────────────────
#
# When the request-scoped ``current_org_id_var`` is unset (background
# workers, boot-time init, ad-hoc psql replacements like the BM25
# rebuild, the glossary refresher loop, etc.), the SQLAlchemy
# ``after_begin`` hook below falls back to THIS uuid.
#
# Why we need a fallback right now:
#   With ``ENABLE_ROW LEVEL SECURITY`` on (migration 061), any query
#   issued without a SET LOCAL would match zero rows. That would break
#   every non-HTTP code path that touches a tenant table — the BM25
#   index would boot empty, glossary would stay at zero acronyms,
#   ingestion workers would silently insert nothing.
#
# Why it's only temporary:
#   It SHORT-CIRCUITS the RLS guarantee for any code path that simply
#   forgot to set an org context. In a single-tenant DB that's fine —
#   all rows belong to Acadia anyway. But the moment a second tenant
#   exists, a forgetful background worker could read or write across
#   the boundary and we'd never notice. RLS becomes a paper tiger.
#
# ╔══════════════════════════════════════════════════════════════════╗
# ║  HARD PRE-FLIGHT CHECKLIST BEFORE PROVISIONING TENANT #2:        ║
# ║                                                                  ║
# ║  1. Set ``DEFAULT_ORG_ID_FOR_NO_CONTEXT = None`` below.          ║
# ║  2. Run the app — every background worker that previously        ║
# ║     touched tenant tables silently will now ERROR or return      ║
# ║     zero rows. Audit each failure:                              ║
# ║       a. BM25 rebuild  (backend/api.py lifespan)                ║
# ║       b. Glossary refresher loop (backend/api.py)               ║
# ║       c. Ingestion job workers (backend/jobs/queue.py)          ║
# ║       d. Report job workers (backend/jobs/, if any)             ║
# ║       e. Any new asyncio background task                        ║
# ║  3. For each: decide between                                     ║
# ║       - "runs PER ORG" → loop over orgs, set                    ║
# ║         current_org_id_var inside the loop, OR                  ║
# ║       - "runs CROSS-ORG by design" → use the ``logiq_admin``    ║
# ║         BYPASSRLS role (separate DATABASE_URL).                 ║
# ║  4. Run the smoke test suite — every Tier-1 journey, every      ║
# ║     /ask, every ingestion path. Anything that previously         ║
# ║     returned data must still return data.                        ║
# ║  5. ONLY THEN provision the second tenant.                      ║
# ║                                                                  ║
# ║  This module's ``_warn_if_multi_tenant_and_fallback_active``    ║
# ║  runs at boot — if any tenant beyond Acadia is found while       ║
# ║  this constant is still non-None, we emit an ERROR-level log    ║
# ║  on every boot. Do not ignore it.                               ║
# ╚══════════════════════════════════════════════════════════════════╝
# Disabled in Phase 2 (multi-tenant rollout). All known background
# paths now set ``current_org_id_var`` explicitly before touching
# tenant tables:
#   * HTTP requests        → TenancyContextMiddleware
#   * Worker dispatch      → backend/jobs/worker.py:run_one
#   * BM25 boot rebuild    → backend/vector_store.py:rebuild_bm25_from_postgres
#   * Glossary rebuild     → backend/retrieval/query_expansion.py
# Any path that forgot will now FAIL CLOSED via RLS (UUID cast on
# empty string raises) instead of silently mis-tagging US Pharma rows
# as Acadia.
#
# If a NEW cross-org admin operation appears (analytics rollup, ops
# CLI), point it at a separate ``logiq_admin`` DATABASE_URL with
# BYPASSRLS — don't reach for this fallback.
DEFAULT_ORG_ID_FOR_NO_CONTEXT: Optional[uuid.UUID] = None


def _resolve_effective_org_id() -> Optional[uuid.UUID]:
    """Resolution order shared by both transaction hooks below.

      1. ``current_org_id_var`` from the request / worker context.
      2. ``DEFAULT_ORG_ID_FOR_NO_CONTEXT`` fallback (background path).
      3. ``None`` → no SET LOCAL fires, RLS hides every tenant row.
    """
    org_id = current_org_id_var.get()
    if org_id is None:
        org_id = DEFAULT_ORG_ID_FOR_NO_CONTEXT
    return org_id


def _stamp_app_current_org(connection) -> None:
    """Issue the transaction-local SET on the given Connection.

    ``set_config(..., true)`` is preferred over a literal ``SET LOCAL``
    so the UUID flows through as a bind parameter (defense in depth
    even though UUIDs are validated upstream).

    Why we ALWAYS stamp something (never leave the GUC at its default):
        Postgres custom GUC variables behave oddly when never set in a
        session — ``current_setting('app.current_org', true)`` returns
        the empty string ``''`` instead of NULL. Our RLS policy then
        casts ``''::uuid`` and raises ``InvalidTextRepresentation``,
        which blows up any tenant-table query (most painfully visible
        in the orchestrator's ThreadPoolExecutor workers, which don't
        inherit the request's ContextVar).
        Stamping the zero-UUID sentinel when no real org is in scope
        keeps the cast valid and causes the policy to silently filter
        every tenant row out — fail-closed, not fail-crash.
    """
    org_id = _resolve_effective_org_id()
    value = str(org_id) if org_id is not None else "00000000-0000-0000-0000-000000000000"
    connection.execute(
        text("SELECT set_config('app.current_org', :v, true)"),
        {"v": value},
    )


@event.listens_for(Session, "after_begin")
def _set_tenant_on_session_begin(session, transaction, connection):
    """
    Stamp ``app.current_org`` on every Session-level transaction.

    Why we ALSO have an engine-level hook below: most of the codebase
    opens Sessions, but the queue helpers in ``backend/jobs/*`` and the
    stuck-job sweeper use raw ``engine.begin()`` / ``engine.connect()``
    — those don't trigger Session events. The engine-level hook
    covers them.
    """
    _stamp_app_current_org(connection)


@event.listens_for(engine, "begin")
def _set_tenant_on_engine_begin(connection):
    """
    Stamp ``app.current_org`` on every raw-connection transaction.

    Fires for ``engine.begin()`` / ``engine.connect().begin()`` paths
    used by ``backend/jobs/queue.py``, ``backend/jobs/ingestion_queue.py``,
    and ``backend/jobs/sweeper.py``. Without this, those paths run
    without an org context and either fail RLS' UUID cast or
    (if the cast happened to resolve) silently leak across tenants.
    """
    _stamp_app_current_org(connection)


def warn_if_multi_tenant_and_fallback_active() -> None:
    """
    Boot-time guardrail.

    Counts the active organizations in the database. If more than one
    exists AND ``DEFAULT_ORG_ID_FOR_NO_CONTEXT`` is still pointing at
    a real uuid, we log an ERROR every boot — because the single-tenant
    fallback is now a potential cross-tenant leak vector.

    Called from ``backend/api.py`` lifespan startup. Cheap (one COUNT
    on a tiny table), runs once per process.

    If you've intentionally onboarded a second tenant and are mid-
    audit of the background-worker contracts (see HARD PRE-FLIGHT
    CHECKLIST above), set ``DEFAULT_ORG_ID_FOR_NO_CONTEXT = None``
    BEFORE that second org appears in the DB and this warning will
    stop firing.
    """
    if DEFAULT_ORG_ID_FOR_NO_CONTEXT is None:
        # Operator already disabled the fallback — nothing to warn about.
        return

    try:
        with engine.connect() as conn:
            # "Active" on `organizations` is encoded by deactivated_at IS NULL
            # (see migration 051 header). There is no `status` column.
            row = conn.execute(
                text("SELECT COUNT(*) FROM organizations WHERE deactivated_at IS NULL")
            ).first()
            active_count = int(row[0]) if row else 0
    except Exception as exc:
        # Don't block startup over the warning probe itself. If the
        # organizations table isn't even queryable, larger problems exist.
        logger.warning(
            "[tenancy] could not probe organizations count for boot guardrail: %s",
            exc,
        )
        return

    if active_count > 1:
        logger.error(
            "[tenancy] 🛑 MULTI-TENANT DETECTED (%d active orgs) WHILE "
            "DEFAULT_ORG_ID_FOR_NO_CONTEXT IS STILL SET. Background "
            "workers without explicit org context will fall back to a "
            "hardcoded tenant — POTENTIAL CROSS-TENANT LEAK. See "
            "HARD PRE-FLIGHT CHECKLIST in backend/db/connection.py. "
            "Set DEFAULT_ORG_ID_FOR_NO_CONTEXT = None and audit each "
            "background worker before continuing.",
            active_count,
        )
    else:
        logger.info(
            "[tenancy] single-tenant fallback active (active_orgs=%d). "
            "Disable DEFAULT_ORG_ID_FOR_NO_CONTEXT before adding a "
            "second tenant — see HARD PRE-FLIGHT CHECKLIST in "
            "backend/db/connection.py.",
            active_count,
        )
