"""
Ingestion job queue — Postgres-backed, mirrors ``backend.jobs.queue``.

Scope
-----
Operates on the ``ingestion_jobs`` table (existing application table,
queue columns added by migration 048). Separate module from the
report-job queue because:

  * ``ingestion_jobs`` carries application-specific fields
    (``processed_chunks``, ``file_size_mb``, ``ingestion_status``)
    that the sidebar UI + admin endpoints already depend on. We
    keep that schema intact instead of force-merging it into
    ``report_jobs`` — see migration 048 commentary.
  * Two purpose-separated queues let us scale ingest-workers and
    report-workers independently (ingest peaks at ~2 GB RAM on big
    PDFs; report workers stay tight at ~1 GB).

Surface
-------
* ``enqueue_ingestion_job(...)`` — INSERT a new pending row carrying
  the storage URI + filename metadata the worker will need.
* ``get_job(job_id)``           — fetch by id; powers GET /ingestion_jobs/{id}.
* ``claim_next(worker_id, kinds)`` — atomic SELECT FOR UPDATE SKIP
  LOCKED claim. Same semantics as the report-queue version.
* ``mark_done`` / ``mark_failed`` — terminal-state transitions with
  the same backoff schedule the report queue uses.

Failure mode contract
---------------------
Same as ``backend.jobs.queue``:
  * API-side calls (``enqueue_ingestion_job``) raise on DB error so
    the upload route returns 5xx and the user retries.
  * Worker-side calls (``mark_*``) log + raise; the worker's outer
    loop catches and the row remains visible to the next claim.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import text

logger = logging.getLogger("acadia-log-iq")


# Only ingestion kind we ship today. The kind column exists so future
# kinds (re-ingest with new prompts, glossary-refresh, etc.) can share
# the table without another migration.
JOB_KIND_INGEST_DOCUMENT = "ingest_document"


# Backoff schedule on soft failure. Identical to the report queue
# (backend/jobs/queue.py:_BACKOFF_SECONDS) — failure modes are
# isomorphic (Bedrock throttle, S3 hiccup, network blip).
_BACKOFF_SECONDS = (5, 30, 300, 1800)


def _engine():
    """Lazy-import the SQLAlchemy engine so this module stays import-safe
    in unit tests / static analysis that have no live DB."""
    from backend.db.connection import engine  # type: ignore
    return engine


@dataclass
class _IngestJob:
    """Worker-side dispatch tuple. Internal only — the API surface
    returns plain dicts so the wire contract stays explicit.
    """
    id: str            # this is the ``job_id`` TEXT, not a UUID
    kind: str
    file_id: str       # ingestion-side stable key
    owner_id: Optional[str]
    payload: Dict[str, Any]
    attempts: int
    max_attempts: int


# ─────────────────────────────────────────────────────────────────
# Enqueue — API side
# ─────────────────────────────────────────────────────────────────


def enqueue_ingestion_job(
    *,
    job_id: str,
    file_id: str,
    owner_id: str,
    file_name: str,
    file_type: str,
    file_hash: Optional[str],
    payload: Dict[str, Any],
    kind: str = JOB_KIND_INGEST_DOCUMENT,
    max_attempts: int = 3,
) -> None:
    """INSERT a new pending ingestion job.

    Idempotent on ``(kind, file_id)`` via the partial UNIQUE index
    added in migration 048 — a duplicate enqueue (React.StrictMode
    double-fire, rapid double-click) raises a unique violation that
    the caller can catch and ignore.

    ``payload`` carries everything the worker needs to run the
    existing ``index_file_job`` logic:

        {
          "job_id":       job_id,        # convenience — also the row PK
          "storage_uri":  "s3://..." | "local:///app/uploads/...",
          "filename":     "<original name>",
          "file_type":    "kb",
          "file_id":      "<uuid>",
          "owner_id":     "<clerk user>",
          "file_size_mb": 3.2,
          "doc_kind":     "ticket",
        }
    """
    payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO ingestion_jobs (
                    job_id, file_id, owner_id, file_name, file_type,
                    file_hash, status, processed_chunks, total_chunks,
                    successful_chunks, created_at,
                    kind, payload, attempts, max_attempts, not_before
                )
                VALUES (
                    :job_id, :file_id, :owner_id, :file_name, :file_type,
                    :file_hash, 'pending', '0', '0',
                    '0', NOW(),
                    :kind, CAST(:payload AS JSONB), 0, :max_attempts, NOW()
                )
                """
            ),
            {
                "job_id": job_id,
                "file_id": file_id,
                "owner_id": owner_id,
                "file_name": file_name,
                "file_type": file_type,
                "file_hash": file_hash,
                "kind": kind,
                "payload": payload_json,
                "max_attempts": int(max_attempts),
            },
        )
    logger.info(
        "[ingestion.enqueue] job_id=%s file_id=%s kind=%s file=%r",
        job_id, file_id, kind, file_name,
    )


def attach_payload_and_enqueue(
    *,
    job_id: str,
    payload: Dict[str, Any],
    kind: str = JOB_KIND_INGEST_DOCUMENT,
    max_attempts: int = 3,
) -> None:
    """Populate the queue columns on a row that already exists.

    Used by ``POST /upload/finalize`` — at presign time the row is
    created with empty payload and ``status='pending'`` (defaults
    from migration 048). At finalize time we know the storage URI
    + file size, so we attach the payload and mark the row ready
    to claim.

    The status flip from ``pending → pending`` is a no-op; we leave
    it alone so a worker that happens to be polling between presign
    and finalize doesn't claim a half-filled row (it would, because
    pending is the default — but with empty payload the handler
    would correctly raise NonRetryableJobError). Setting
    ``not_before = NOW()`` here ensures the row becomes claimable
    only AFTER finalize commits.
    """
    payload_json = json.dumps(payload or {}, ensure_ascii=False, default=str)
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE ingestion_jobs
                   SET kind         = :kind,
                       payload      = CAST(:payload AS JSONB),
                       max_attempts = :max_attempts,
                       attempts     = 0,
                       not_before   = NOW(),
                       status       = 'pending'
                 WHERE job_id = :id
                """
            ),
            {
                "id": job_id,
                "kind": kind,
                "payload": payload_json,
                "max_attempts": int(max_attempts),
            },
        )
    logger.info(
        "[ingestion.attach_payload] job_id=%s kind=%s payload_keys=%s",
        job_id, kind, sorted((payload or {}).keys()),
    )


# ─────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    """Return the row as a dict, or None if the id is unknown.

    Wire-compatible with ``backend.jobs.queue.get_job`` so the
    ``GET /jobs/{id}`` endpoint can paper over both queues if we
    ever extend it to read from either.
    """
    jid = (job_id or "").strip()
    if not jid:
        return None
    with _engine().connect() as conn:
        row = conn.execute(
            text(
                """
                SELECT job_id        AS id,
                       kind,
                       file_id::text AS file_id,
                       file_name,
                       owner_id,
                       status,
                       attempts,
                       max_attempts,
                       payload,
                       processed_chunks,
                       total_chunks,
                       successful_chunks,
                       error,
                       not_before,
                       created_at,
                       started_at,
                       completed_at
                  FROM ingestion_jobs
                 WHERE job_id = :id
                """
            ),
            {"id": jid},
        ).mappings().first()
    return dict(row) if row else None


# ─────────────────────────────────────────────────────────────────
# Claim — worker side
# ─────────────────────────────────────────────────────────────────


def claim_next(
    *,
    worker_id: str,
    kinds: Optional[Iterable[str]] = None,
) -> Optional[_IngestJob]:
    """Atomically claim the oldest runnable ingestion job.

    ``kinds`` lets a specialised worker (``WORKER_KINDS=ingest_document``)
    skip rows it isn't authorised to run. When ``kinds`` is None or
    empty, every kind in the table is fair game — used by the
    single-fleet dev/docker-compose deployment.
    """
    if kinds is not None:
        kinds_list = sorted({k.strip() for k in kinds if k and k.strip()})
    else:
        kinds_list = []

    if kinds_list:
        sql_claim = text(
            """
            SELECT job_id        AS id,
                   kind,
                   file_id::text AS file_id,
                   owner_id,
                   payload,
                   attempts,
                   max_attempts
              FROM ingestion_jobs
             WHERE status = 'pending'
               AND not_before <= NOW()
               AND kind = ANY(:kinds)
             ORDER BY created_at ASC
             FOR UPDATE SKIP LOCKED
             LIMIT 1
            """
        )
        params: Dict[str, Any] = {"kinds": kinds_list}
    else:
        sql_claim = text(
            """
            SELECT job_id        AS id,
                   kind,
                   file_id::text AS file_id,
                   owner_id,
                   payload,
                   attempts,
                   max_attempts
              FROM ingestion_jobs
             WHERE status = 'pending'
               AND not_before <= NOW()
             ORDER BY created_at ASC
             FOR UPDATE SKIP LOCKED
             LIMIT 1
            """
        )
        params = {}

    sql_mark_running = text(
        """
        UPDATE ingestion_jobs
           SET status     = 'running',
               attempts   = attempts + 1,
               started_at = NOW()
         WHERE job_id = :id
        """
    )

    with _engine().begin() as conn:
        row = conn.execute(sql_claim, params).mappings().first()
        if not row:
            return None
        conn.execute(sql_mark_running, {"id": row["id"]})

    payload = row["payload"]
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            payload = {}

    job = _IngestJob(
        id=row["id"],
        kind=row["kind"],
        file_id=row["file_id"],
        owner_id=row["owner_id"],
        payload=payload or {},
        attempts=int(row["attempts"]) + 1,  # we just bumped it
        max_attempts=int(row["max_attempts"]),
    )
    logger.info(
        "[ingestion.claim] worker=%s id=%s file_id=%s attempt=%d/%d",
        worker_id, job.id, job.file_id, job.attempts, job.max_attempts,
    )
    return job


# ─────────────────────────────────────────────────────────────────
# Status transitions — worker side
# ─────────────────────────────────────────────────────────────────


def mark_done(*, job_id: str) -> None:
    """Terminal success — flip to done and stamp the completion time.

    Per-panel result pointers are not needed for ingestion (the work
    product is the chunk rows themselves, materialised by the worker's
    handler directly into ``chunks`` + ``embeddings``).
    """
    with _engine().begin() as conn:
        conn.execute(
            text(
                """
                UPDATE ingestion_jobs
                   SET status       = 'done',
                       completed_at = NOW(),
                       error        = NULL
                 WHERE job_id = :id
                """
            ),
            {"id": job_id},
        )
    logger.info("[ingestion.done] id=%s", job_id)


def mark_failed(
    *,
    job_id: str,
    error_message: str,
    permanent: bool = False,
) -> None:
    """Permanent-failed (``permanent=True``) or soft-failed with retry.

    Mirrors ``backend.jobs.queue.mark_failed`` semantics so the worker
    dispatch loop can treat both queues identically.
    """
    err = (error_message or "")[:4000]

    with _engine().begin() as conn:
        if permanent:
            conn.execute(
                text(
                    """
                    UPDATE ingestion_jobs
                       SET status       = 'failed',
                           completed_at = NOW(),
                           error        = :err
                     WHERE job_id = :id
                    """
                ),
                {"id": job_id, "err": err},
            )
            logger.warning(
                "[ingestion.failed.permanent] id=%s err=%s", job_id, err[:160],
            )
            return

        # Soft failure — consult attempts vs max_attempts.
        row = conn.execute(
            text(
                """
                SELECT attempts, max_attempts
                  FROM ingestion_jobs
                 WHERE job_id = :id
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
                    UPDATE ingestion_jobs
                       SET status       = 'failed',
                           completed_at = NOW(),
                           error        = :err
                     WHERE job_id = :id
                    """
                ),
                {"id": job_id, "err": err},
            )
            logger.warning(
                "[ingestion.failed.exhausted] id=%s attempts=%d/%d err=%s",
                job_id, attempts, max_attempts, err[:160],
            )
            return

        # Re-queue with exponential backoff.
        backoff_index = min(attempts - 1, len(_BACKOFF_SECONDS) - 1)
        backoff_seconds = _BACKOFF_SECONDS[backoff_index]
        next_run = datetime.now(timezone.utc) + timedelta(seconds=backoff_seconds)

        conn.execute(
            text(
                """
                UPDATE ingestion_jobs
                   SET status     = 'pending',
                       not_before = :nb,
                       error      = :err
                 WHERE job_id = :id
                """
            ),
            {"id": job_id, "nb": next_run, "err": err},
        )
        logger.info(
            "[ingestion.failed.retry] id=%s attempts=%d/%d backoff=%ds err=%s",
            job_id, attempts, max_attempts, backoff_seconds, err[:160],
        )
