"""
Postgres-backed async job queue.

Operations
----------
* ``enqueue(...)``      — INSERT a new pending row, returning its id.
                           Idempotent on ``(kind, incident_number)`` —
                           a duplicate while a job is open returns the
                           existing id instead of creating a duplicate.
* ``get_job(job_id)``   — fetch the current row by id, used by the
                           ``GET /jobs/{id}`` endpoint.
* ``claim_next(...)``   — atomically claim the oldest runnable row via
                           ``SELECT ... FOR UPDATE SKIP LOCKED``.
                           Multiple workers polling concurrently never
                           collide on the same row.
* ``mark_running``      — bumps ``attempts`` + sets ``started_at``.
* ``mark_done``         — terminal success; stores ``result_kind`` +
                           ``result_incident`` pointers.
* ``mark_failed``       — terminal-or-retry depending on
                           ``attempts vs max_attempts``. Sets the
                           ``not_before`` backoff gate.

Design contract
---------------
* Every public function uses a fresh connection (``with engine.begin()``)
  so each call is its own transaction. No state leaks across calls.
* ``claim_next`` MUST hold the row lock from SELECT through to the
  ``mark_running`` UPDATE inside the same transaction — that's the
  whole point of FOR UPDATE SKIP LOCKED. The function returns a dict
  AFTER it has committed the running-status flip, so the caller can
  release the connection and proceed without an open transaction.

Failure mode philosophy
-----------------------
* Queue functions that the API path calls (``enqueue``, ``get_job``)
  raise on DB error — the user needs to know we couldn't accept the
  request.
* Queue functions that the worker calls (``mark_*``) log + raise; the
  worker's retry loop will catch and re-queue.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# Whitelist of job kinds. Mirrors the report_cache constants so a
# job's ``kind`` is exactly the same string used to key the cached
# result row.
JOB_KIND_RCA_CUSTOMER = "rca_customer"
JOB_KIND_RCA_INTERNAL = "rca_internal"
JOB_KIND_GAP_ANALYSIS = "gap_analysis_gap"
JOB_KIND_POST_MORTEM = "gap_analysis_pm"

_VALID_KINDS = frozenset({
    JOB_KIND_RCA_CUSTOMER,
    JOB_KIND_RCA_INTERNAL,
    JOB_KIND_GAP_ANALYSIS,
    JOB_KIND_POST_MORTEM,
})


class JobStatus:
    """String constants — matches the CHECK constraint in migration 047."""
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Backoff schedule used by ``mark_failed`` when re-queueing. Capped
# at the last entry; we don't want a job to sleep forever.
_BACKOFF_SECONDS = (5, 30, 300, 1800)  # 5 s, 30 s, 5 min, 30 min


def _engine():
    """Lazy import so this module is import-safe without a live DB."""
    from backend.db.connection import engine  # type: ignore
    return engine


@dataclass
class _Job:
    """Convenience type for the worker dispatch loop.

    Internal — the API surface returns plain dicts so the wire
    contract stays explicit. ``_Job`` exists so ``claim_next``'s
    return type is statically documented.
    """
    id: str
    kind: str
    incident_number: str
    payload: Dict[str, Any]
    attempts: int
    max_attempts: int
    requested_by: Optional[str]


# ─────────────────────────────────────────────────────────────────
# Enqueue
# ─────────────────────────────────────────────────────────────────


def enqueue(
    *,
    kind: str,
    incident_number: str,
    requested_by: Optional[str],
    payload: Optional[Dict[str, Any]] = None,
    max_attempts: int = 3,
) -> str:
    """Insert a new pending job, returning its id.

    Idempotent on ``(kind, incident_number)`` while a row is open
    (status IN pending|running) — duplicates return the existing id.
    This protects against the React.StrictMode / rapid-double-click
    pattern we already saw in the Gap Analysis flow.
    """
    kind = (kind or "").strip()
    inc = (incident_number or "").strip()
    if kind not in _VALID_KINDS:
        raise ValueError(f"unknown job kind: {kind!r}")
    if not inc:
        raise ValueError("incident_number is required")

    payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)

    # The ON CONFLICT path can't reference the partial-unique-index
    # directly (Postgres lacks a syntax for "conflict on this partial
    # index"), so we do this in two steps inside one transaction:
    #   1. SELECT an open row.
    #   2. INSERT only if none was found.
    # SERIALIZABLE isolation isn't needed — the partial-unique index
    # itself prevents the race; a concurrent INSERT will get a unique
    # violation and we retry once.
    with _engine().begin() as conn:
        existing = conn.execute(
            text(
                """
                SELECT id::text
                  FROM report_jobs
                 WHERE kind = :kind
                   AND incident_number = :inc
                   AND status IN ('pending', 'running')
                 LIMIT 1
                """
            ),
            {"kind": kind, "inc": inc},
        ).scalar()
        if existing:
            logger.info(
                "[jobs.enqueue] idempotent-hit kind=%s inc=%s existing_id=%s",
                kind, inc, existing,
            )
            return str(existing)

        try:
            row = conn.execute(
                text(
                    """
                    INSERT INTO report_jobs
                        (kind, incident_number, requested_by, payload, max_attempts)
                    VALUES
                        (:kind, :inc, :user, CAST(:payload AS JSONB), :max_attempts)
                    RETURNING id::text
                    """
                ),
                {
                    "kind": kind,
                    "inc": inc,
                    "user": (requested_by or None),
                    "payload": payload_json,
                    "max_attempts": int(max_attempts),
                },
            ).first()
        except Exception as exc:
            # Almost always a unique-violation racing with a concurrent
            # enqueue from the same user. Re-select and return the
            # winner's id — caller never sees a 5xx for a duplicate.
            logger.info(
                "[jobs.enqueue] insert raced (%s) — re-reading existing row", exc,
            )
            existing = conn.execute(
                text(
                    """
                    SELECT id::text
                      FROM report_jobs
                     WHERE kind = :kind
                       AND incident_number = :inc
                       AND status IN ('pending', 'running')
                     LIMIT 1
                    """
                ),
                {"kind": kind, "inc": inc},
            ).scalar()
            if existing:
                return str(existing)
            raise

    job_id = str(row[0])
    logger.info(
        "[jobs.enqueue] new id=%s kind=%s inc=%s by=%s",
        job_id, kind, inc, requested_by or "(anon)",
    )
    return job_id


# ─────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the row as a dict, or None if the id is unknown."""
    jid = (job_id or "").strip()
    if not jid:
        return None
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT id::text       AS id,
                       kind, incident_number, status,
                       requested_by, payload, attempts, max_attempts,
                       not_before, created_at, started_at, finished_at,
                       result_kind, result_incident, error_message
                  FROM report_jobs
                 WHERE id = CAST(:id AS UUID)
                """
            ),
            {"id": jid},
        ).mappings().first()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────
# Claim — worker side
# ─────────────────────────────────────────────────────────────────


def claim_next(*, worker_id: str) -> Optional[_Job]:
    """Atomically claim the oldest runnable job.

    Implementation: ``SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1`` —
    canonical Postgres pattern. Two workers polling concurrently will
    SKIP each other's locked rows so they never collide; rows past
    ``not_before`` are filtered out for the exponential-backoff gate.

    Inside the same transaction we UPDATE the row to
    ``status='running'`` and bump ``attempts`` so a third worker
    polling a microsecond later sees the row as taken even after
    we release the lock by committing.
    """
    sql_claim = text(
        """
        SELECT id::text AS id, kind, incident_number, payload,
               attempts, max_attempts, requested_by
          FROM report_jobs
         WHERE status = 'pending'
           AND not_before <= NOW()
         ORDER BY created_at ASC
         FOR UPDATE SKIP LOCKED
         LIMIT 1
        """
    )
    sql_mark_running = text(
        """
        UPDATE report_jobs
           SET status      = 'running',
               attempts    = attempts + 1,
               started_at  = NOW()
         WHERE id = CAST(:id AS UUID)
        """
    )

    with _engine().begin() as conn:
        row = conn.execute(sql_claim).mappings().first()
        if not row:
            return None
        conn.execute(sql_mark_running, {"id": row["id"]})

    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}

    job = _Job(
        id=row["id"],
        kind=row["kind"],
        incident_number=row["incident_number"],
        payload=payload or {},
        attempts=int(row["attempts"]) + 1,  # we just bumped it
        max_attempts=int(row["max_attempts"]),
        requested_by=row["requested_by"],
    )
    logger.info(
        "[jobs.claim] worker=%s id=%s kind=%s inc=%s attempt=%d/%d",
        worker_id, job.id, job.kind, job.incident_number,
        job.attempts, job.max_attempts,
    )
    return job


# ─────────────────────────────────────────────────────────────────
# Status transitions — worker side
# ─────────────────────────────────────────────────────────────────


def mark_running(job_id: str) -> None:
    """Re-affirm the running status — used when a worker resumes
    a previously claimed row (defensive, currently unused).
    """
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE report_jobs
                   SET status     = 'running',
                       started_at = COALESCE(started_at, NOW())
                 WHERE id = CAST(:id AS UUID)
                """
            ),
            {"id": job_id},
        )


def mark_done(
    *,
    job_id: str,
    result_kind: str,
    result_incident: str,
) -> None:
    """Terminal success — set status=done + result pointers.

    ``result_kind`` and ``result_incident`` are the lookup key into
    ``report_cache``. The frontend, on receiving status=done from the
    polling endpoint, hits the existing ``/rca/{inc}`` or
    ``/gap-analysis/{inc}`` GET path and gets the cached row instantly.
    """
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE report_jobs
                   SET status          = 'done',
                       finished_at     = NOW(),
                       result_kind     = :rk,
                       result_incident = :ri,
                       error_message   = NULL
                 WHERE id = CAST(:id AS UUID)
                """
            ),
            {"id": job_id, "rk": result_kind, "ri": result_incident},
        )
    logger.info(
        "[jobs.done] id=%s result=%s/%s", job_id, result_kind, result_incident,
    )


def mark_failed(
    *,
    job_id: str,
    error_message: str,
    permanent: bool = False,
) -> None:
    """Either move to status=failed (permanent) or re-queue with backoff.

    ``permanent=True`` skips the retry counter — used when the worker
    knows the error is non-retryable (ticket not found, ValidationException
    on invalid model id, etc.).

    Otherwise: if ``attempts >= max_attempts`` we mark failed; else we
    flip status back to pending and set ``not_before = NOW() + backoff``
    so the worker pool won't immediately re-pick it.
    """
    err = (error_message or "")[:4000]  # safety cap

    with _engine().begin() as conn:
        if permanent:
            conn.execute(
                text(
                    """
                    UPDATE report_jobs
                       SET status        = 'failed',
                           finished_at   = NOW(),
                           error_message = :err
                     WHERE id = CAST(:id AS UUID)
                    """
                ),
                {"id": job_id, "err": err},
            )
            logger.warning("[jobs.failed.permanent] id=%s err=%s", job_id, err[:160])
            return

        # Soft fail — consult attempts vs max_attempts.
        row = conn.execute(
            text(
                """
                SELECT attempts, max_attempts
                  FROM report_jobs
                 WHERE id = CAST(:id AS UUID)
                """
            ),
            {"id": job_id},
        ).mappings().first()
        if not row:
            return
        attempts = int(row["attempts"])
        max_attempts = int(row["max_attempts"])

        if attempts >= max_attempts:
            conn.execute(
                text(
                    """
                    UPDATE report_jobs
                       SET status        = 'failed',
                           finished_at   = NOW(),
                           error_message = :err
                     WHERE id = CAST(:id AS UUID)
                    """
                ),
                {"id": job_id, "err": err},
            )
            logger.warning(
                "[jobs.failed.exhausted] id=%s attempts=%d/%d err=%s",
                job_id, attempts, max_attempts, err[:160],
            )
            return

        # Re-queue with exponential backoff. Backoff index = attempts
        # already-elapsed (the just-failed attempt counts).
        backoff_index = min(attempts - 1, len(_BACKOFF_SECONDS) - 1)
        backoff_seconds = _BACKOFF_SECONDS[backoff_index]
        next_run = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)

        conn.execute(
            text(
                """
                UPDATE report_jobs
                   SET status        = 'pending',
                       not_before    = :nb,
                       error_message = :err
                 WHERE id = CAST(:id AS UUID)
                """
            ),
            {"id": job_id, "nb": next_run, "err": err},
        )
        logger.info(
            "[jobs.failed.retry] id=%s attempts=%d/%d backoff=%ds err=%s",
            job_id, attempts, max_attempts, backoff_seconds, err[:160],
        )
