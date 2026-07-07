"""
Worker main loop — poll Postgres, claim a job, dispatch, mark done/failed.

Run as its own container in production::

    python -m backend.jobs.worker

Or programmatically in tests::

    from backend.jobs.worker import run_one
    run_one(worker_id="test")  # blocks until one job is processed (or no work)

Design contract
---------------
* The loop is deliberately simple — single-threaded inside each worker
  process. Horizontal scale = run more workers; vertical scale = N/A.
  This matches the LLM-bound latency profile: a single thread waiting
  on Bedrock doesn't block other workers because they're separate
  processes against the same DB queue.
* Every loop iteration is fail-safe — handler exceptions are caught and
  routed through ``mark_failed`` (retry or permanent based on the
  exception type). The loop NEVER dies on a job exception.
* The poll-when-idle backoff uses a simple constant sleep
  (``_IDLE_POLL_SECONDS``) rather than exponential — keeps wake-up
  latency tight for users who just queued a job.
* Graceful shutdown: SIGINT / SIGTERM flips ``_shutdown`` and the loop
  finishes the in-flight job before exiting. Critical for blue/green
  deploys — the new worker takes over without dropping work.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import time
import traceback
import uuid
from typing import Optional

from backend.services.token_usage import (
    flush as flush_token_usage,
    get_usage_totals,
    reset_usage_totals,
)
from .handlers import HANDLERS, NonRetryableJobError
from .queue import claim_next as claim_next_report
from .queue import mark_done as mark_done_report
from .queue import mark_failed as mark_failed_report
from .ingestion_queue import (
    JOB_KIND_INGEST_DOCUMENT,
    claim_next as claim_next_ingestion,
    mark_done as mark_done_ingestion,
    mark_failed as mark_failed_ingestion,
)

logger = logging.getLogger("acadia-log-iq")


# How long to sleep when the queue is empty. Short enough that user-
# facing latency stays acceptable, long enough that idle workers don't
# burn DB connections in a tight loop.
_IDLE_POLL_SECONDS = 2.0


# Job kinds that live in the ingestion_jobs table. Everything else
# lives in report_jobs. The worker uses this to route mark_done /
# mark_failed back to the right table after a handler returns.
_INGESTION_KINDS = frozenset({JOB_KIND_INGEST_DOCUMENT})


def _resolve_worker_kinds() -> Optional[set]:
    """Read the ``WORKER_KINDS`` env var. Returns a set of kinds the
    worker is allowed to claim, or None to mean "any kind".

    Examples
    --------
    ``WORKER_KINDS=ingest_document``
        Specialised ingest-only worker.
    ``WORKER_KINDS=rca_customer,rca_internal,gap_analysis_gap,gap_analysis_pm``
        Specialised report-only worker.
    Unset / empty / ``*``
        Mixed worker — claims any kind. Used by docker-compose and
        single-machine deployments.
    """
    raw = (os.environ.get("WORKER_KINDS", "") or "").strip()
    if not raw or raw == "*":
        return None
    return {k.strip() for k in raw.split(",") if k.strip()}


# Graceful-shutdown signal — set by SIGINT / SIGTERM handlers.
_shutdown = False


def _install_signal_handlers() -> None:
    """Register SIGINT + SIGTERM handlers that flip ``_shutdown``.

    Skipped silently on Windows or in non-main threads (where
    signal.signal raises) — laptop test runs use ``run_one``
    directly so they don't need signals anyway.
    """
    def _handler(signum, _frame):
        global _shutdown
        logger.info("[worker] shutdown signal %s received — draining", signum)
        _shutdown = True

    try:
        signal.signal(signal.SIGINT, _handler)
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, AttributeError):
        # Non-main thread (tests) or platform without these signals.
        # Not fatal — the loop's run_one paths don't depend on signals.
        pass


def _worker_id() -> str:
    """Stable per-process identifier for log correlation."""
    return f"{socket.gethostname()}/{os.getpid()}"


def _claim_any(*, worker_id: str, kinds: Optional[set]):
    """Try report_jobs first, then ingestion_jobs.

    Returns a tuple ``(table_tag, job_obj)`` so the caller knows which
    mark_done / mark_failed pair to use. ``table_tag`` is the string
    ``"report"`` or ``"ingestion"``; ``job_obj`` has the same
    duck-typed surface either way (``.id``, ``.kind``, ``.attempts``,
    ``.max_attempts``, ``.payload``, plus a key field — ``.incident_number``
    for report jobs, ``.file_id`` for ingestion jobs).

    ``kinds`` filters per-table — workers specialised by env var
    (``WORKER_KINDS=ingest_document``) skip the table that has no
    intersecting kinds, which is the cheap way to keep poll volume
    proportional to the worker's actual job set.
    """
    # Report table — only poll if the worker is allowed to handle at
    # least one report kind. The set intersection is recomputed each
    # tick because it's tiny; a startup-cached copy would save
    # nanoseconds per poll at the cost of readability.
    report_kinds_allowed = (
        None if kinds is None else (kinds - _INGESTION_KINDS)
    )
    if report_kinds_allowed is None or report_kinds_allowed:
        job = claim_next_report(worker_id=worker_id)
        if job is not None:
            # Defense in depth: if the worker has a kinds filter and
            # the claimed job's kind isn't in it, push the row back.
            # claim_next_report doesn't yet take a kinds arg (RCA / Gap
            # are the only kinds in report_jobs today and every report
            # worker is authorised for all of them), so this branch is
            # cheap insurance against a future kind being added.
            if kinds is not None and job.kind not in kinds:
                mark_failed_report(
                    job_id=job.id,
                    error_message=f"kind {job.kind!r} not in WORKER_KINDS — re-queueing",
                    permanent=False,
                )
                return None
            return ("report", job)

    # Ingestion table — only poll if the worker is allowed to handle
    # at least one ingestion kind.
    ingestion_kinds_allowed = (
        None if kinds is None else (kinds & _INGESTION_KINDS)
    )
    if ingestion_kinds_allowed is None or ingestion_kinds_allowed:
        job = claim_next_ingestion(
            worker_id=worker_id,
            kinds=ingestion_kinds_allowed,
        )
        if job is not None:
            return ("ingestion", job)

    return None


def run_one(*, worker_id: Optional[str] = None, kinds: Optional[set] = None) -> bool:
    """Process exactly one claimable job, if any. Returns True if work
    was processed (success OR failure), False if both queues were empty.

    Used by tests and as the inner step of ``run`` so the loop logic
    stays trivially testable.
    """
    wid = worker_id or _worker_id()
    claimed = _claim_any(worker_id=wid, kinds=kinds)
    if claimed is None:
        return False
    table_tag, job = claimed

    # Route mark_done / mark_failed back to the right table.
    if table_tag == "ingestion":
        _mark_done = lambda **kw: mark_done_ingestion(job_id=kw["job_id"])
        _mark_failed = mark_failed_ingestion
        key_value = getattr(job, "file_id", "")
    else:
        _mark_done = mark_done_report
        _mark_failed = mark_failed_report
        key_value = getattr(job, "incident_number", "")

    handler = HANDLERS.get(job.kind)
    if handler is None:
        logger.warning(
            "[worker] unknown job kind=%s id=%s table=%s — marking permanently failed",
            job.kind, job.id, table_tag,
        )
        _mark_failed(
            job_id=job.id,
            error_message=f"unknown_job_kind: {job.kind}",
            permanent=True,
        )
        return True

    # ── Multi-tenant: stamp the job's organization onto the ContextVar
    # so every SQLAlchemy session inside the handler picks up the right
    # ``app.current_org`` (see backend/db/connection.py after_begin hook).
    # Without this the SQLAlchemy hook falls through to
    # DEFAULT_ORG_ID_FOR_NO_CONTEXT — fine for single-tenant, a
    # cross-tenant leak the moment a second org exists.
    #
    # If a job row is missing organization_id (shouldn't happen after
    # migrations 058/059 — NOT NULL columns — but defend in depth),
    # we permanently fail it rather than silently process it under
    # the wrong tenant.
    from backend.db.connection import current_org_id_var
    job_org_raw = job.organization_id
    if not job_org_raw:
        logger.error(
            "[worker] job missing organization_id id=%s kind=%s table=%s — "
            "marking permanently failed (cannot dispatch without tenant scope)",
            job.id, job.kind, table_tag,
        )
        _mark_failed(
            job_id=job.id,
            error_message="job row has no organization_id — refusing to dispatch",
            permanent=True,
        )
        return True
    try:
        job_org_uuid = uuid.UUID(str(job_org_raw))
    except (TypeError, ValueError) as exc:
        logger.error(
            "[worker] job has malformed organization_id=%r id=%s (%s) — "
            "marking permanently failed",
            job_org_raw, job.id, exc,
        )
        _mark_failed(
            job_id=job.id,
            error_message=f"malformed organization_id: {job_org_raw!r}",
            permanent=True,
        )
        return True

    ctx_token = current_org_id_var.set(job_org_uuid)
    try:
        started = time.perf_counter()
        # Zero the Haiku token counters so the summary below reflects THIS
        # job's Bedrock spend only. (Per-process, so accurate as long as one
        # worker processes one job at a time — which this loop guarantees.)
        reset_usage_totals()
        try:
            result_kind, result_incident = handler(key_value, job.payload)
        except NonRetryableJobError as exc:
            elapsed = time.perf_counter() - started
            logger.warning(
                "[worker] permanent failure id=%s kind=%s key=%s org=%s table=%s after=%.1fs err=%s",
                job.id, job.kind, key_value, job_org_uuid, table_tag, elapsed, exc,
            )
            _mark_failed(
                job_id=job.id,
                error_message=str(exc),
                permanent=True,
            )
            return True
        except Exception as exc:  # noqa: BLE001 — retry path
            elapsed = time.perf_counter() - started
            logger.warning(
                "[worker] soft failure id=%s kind=%s key=%s org=%s table=%s attempt=%d/%d after=%.1fs err=%s",
                job.id, job.kind, key_value, job_org_uuid, table_tag,
                job.attempts, job.max_attempts, elapsed, exc,
            )
            tb_short = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            _mark_failed(
                job_id=job.id,
                error_message=tb_short,
                permanent=False,
            )
            return True

        elapsed = time.perf_counter() - started
        _u = get_usage_totals()
        logger.info(
            "[worker] done id=%s kind=%s key=%s org=%s table=%s elapsed=%.1fs result=%s/%s "
            "| haiku_calls=%d in_toks=%d out_toks=%d cost=$%.4f",
            job.id, job.kind, key_value, job_org_uuid, table_tag, elapsed,
            result_kind, result_incident,
            _u["calls"], _u["input_tokens"], _u["output_tokens"], _u["cost_usd"],
        )
        # ingestion-side mark_done doesn't take result_* pointers — the
        # work product lives in chunks/embeddings, not report_cache.
        if table_tag == "ingestion":
            _mark_done(job_id=job.id)
        else:
            _mark_done(
                job_id=job.id,
                result_kind=result_kind,
                result_incident=result_incident,
            )
        return True
    finally:
        # Persist this job's buffered Bedrock token usage while the org
        # ContextVar is still set (flush stamps app.current_org from it so
        # RLS accepts the rows). Must run BEFORE the reset below.
        try:
            flush_token_usage()
        except Exception:  # noqa: BLE001 — accounting must never fail a job
            logger.debug("[worker] token-usage flush failed", exc_info=True)
        # Always restore — even if mark_done / mark_failed raise. The
        # ContextVar leaking to the next loop iteration would silently
        # mis-tag every subsequent job until shutdown.
        current_org_id_var.reset(ctx_token)


def run() -> None:
    """Main entry point — install signals + loop until shutdown.

    Logs each transition (idle → busy, busy → idle) so an operator
    tailing CloudWatch can immediately see whether the worker is
    healthy and what it's doing.

    Reads ``WORKER_KINDS`` at start: an unset / ``*`` value means
    "claim any kind from either queue"; a comma-separated list
    specialises this worker so the ECS service auto-scaler can
    treat ingest-workers and report-workers as independent fleets.
    """
    _install_signal_handlers()
    wid = _worker_id()
    allowed_kinds = _resolve_worker_kinds()
    logger.info(
        "[worker] start id=%s allowed_kinds=%s registered_handlers=%s",
        wid,
        sorted(allowed_kinds) if allowed_kinds else "*",
        sorted(HANDLERS.keys()),
    )

    consecutive_idle = 0
    while not _shutdown:
        try:
            had_work = run_one(worker_id=wid, kinds=allowed_kinds)
        except Exception as exc:
            # Anything that escapes run_one is a queue-side bug
            # (DB unreachable, etc.). Log + sleep so we don't hot-loop.
            logger.error("[worker] loop iteration crashed (%s) — sleeping", exc)
            time.sleep(_IDLE_POLL_SECONDS)
            continue

        if had_work:
            consecutive_idle = 0
        else:
            # Occasional idle-tick log so an operator knows the worker
            # is alive even when there's no traffic. Throttled to
            # avoid CloudWatch noise.
            consecutive_idle += 1
            if consecutive_idle in (1, 30, 300):
                logger.info("[worker] idle tick=%d", consecutive_idle)
            time.sleep(_IDLE_POLL_SECONDS)

    logger.info("[worker] shutdown complete id=%s", wid)


if __name__ == "__main__":
    # MUST run before ``backend.config`` is imported. In production
    # the Fargate task is provisioned with ONLY the Secrets Manager
    # ARN attached — DATABASE_URL / BEDROCK_* / CLERK_* all live in
    # the secret blob. Without this call, ``Settings()`` raises at
    # boot because os.environ is empty.
    from backend.core.secrets import bootstrap_environment
    bootstrap_environment()

    # Defer configure_logging until we know we're the main entry —
    # avoids tweaking handlers when run_one is imported into tests.
    from backend.observability import configure_logging
    from backend.config import settings
    configure_logging(level=settings.LOG_LEVEL, fmt=settings.LOG_FORMAT)
    run()
