-- Migration 047 — Async report-generation job queue.
--
-- Phase 1: RCA + Gap Analysis generations move out of the request
-- thread and into a worker pool. The API ``POST /rca/{inc}`` and
-- ``POST /gap-analysis/{inc}`` endpoints insert a row here and
-- return 202 + job_id immediately. A separate worker container
-- claims rows via ``SELECT ... FOR UPDATE SKIP LOCKED``, runs the
-- LLM, writes the result into ``report_cache`` (migration 043),
-- and updates this row's status to ``done``.
--
-- Why Postgres-only (no SQS):
--   * Single transactional store for status + result.
--   * SELECT FOR UPDATE SKIP LOCKED is the canonical
--     production-grade pattern (GitLab, Stripe River, oban,
--     pg_boss all use it). Battle-tested.
--   * No AWS-specific surface. Workers on a laptop, in dev, in
--     staging, in prod all work identically.
--   * SQS can be wired in front later as a publisher-side
--     optimisation if we ever cross the scale where polling
--     becomes meaningful — the job table stays unchanged.
--
-- Idempotency contract:
--   ``(report_kind, incident_number)`` is UNIQUE WHERE status IN
--   ('pending', 'running'). A duplicate POST while a job is in
--   flight returns the existing job_id instead of creating a new
--   row. After the job hits a terminal status (done/failed/cancelled)
--   the partial-unique releases so the next POST can enqueue a
--   fresh attempt (e.g. user re-clicks Regenerate).
--
-- Retry model:
--   ``attempts`` counter incremented on each claim. Worker reads
--   ``max_attempts`` (3 by default) — past that the row is moved
--   to ``failed`` permanently and a Sentry event fires. Retry
--   backoff is enforced via the ``not_before`` column: the worker's
--   claim query filters ``not_before <= NOW()``.
--
-- Retention:
--   ``report_jobs`` rows are deleted after 30 days (ops cron — see
--   docs/runbooks/job-retention.md). The actual cached result lives
--   in ``report_cache`` forever (it's the product).
--
-- No BEGIN/COMMIT — migrate.py wraps each migration in engine.begin().

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS report_jobs (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- ``rca_customer`` | ``rca_internal`` | ``gap_analysis_gap`` |
    -- ``gap_analysis_pm`` — same whitelist as report_cache.
    -- Free-form text so we don't need a new migration to add a
    -- new report kind.
    kind            VARCHAR(64) NOT NULL,
    incident_number VARCHAR(128) NOT NULL,
    -- Status state machine:
    --   pending → running → done
    --   pending → running → failed (after max_attempts)
    --   pending → cancelled (admin action)
    status          VARCHAR(16) NOT NULL DEFAULT 'pending',
    CONSTRAINT report_jobs_status_chk CHECK (status IN (
        'pending', 'running', 'done', 'failed', 'cancelled'
    )),

    -- Caller identity — anchored to whoever pressed Generate.
    -- Recorded but not used for cache scoping (cache is global per
    -- the Phase-13 design decision).
    requested_by    VARCHAR(256),

    -- Optional structured payload (e.g. ``regenerate=true``).
    -- Worker reads this to decide whether to honour the report_cache
    -- on dispatch or force a fresh LLM call.
    payload         JSONB NOT NULL DEFAULT '{}'::JSONB,

    -- After max_attempts the worker stops retrying.
    attempts        INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 3,
    -- Exponential-backoff gate: worker only claims rows where
    -- ``not_before <= NOW()``. Set by the worker on a failed attempt.
    not_before      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Bookkeeping. ``started_at`` is set when a worker claims the
    -- row; ``finished_at`` is set on terminal status.
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    started_at      TIMESTAMPTZ,
    finished_at     TIMESTAMPTZ,

    -- Result locator. We don't store the markdown here — that lives
    -- in ``report_cache`` keyed on (kind, incident_number). Storing
    -- the cache row id (or simply the (kind, inc) key — the cache
    -- table already has UNIQUE on that pair) lets the frontend
    -- poll ``/jobs/{id}`` and follow the pointer to the rendered
    -- markdown.
    result_kind     VARCHAR(64),
    result_incident VARCHAR(128),

    -- Failure narrative for the UI when status='failed'.
    error_message   TEXT
);

-- Workers claim by polling ``status='pending' AND not_before<=NOW()``
-- in created_at order. The partial index keeps this hot scan cheap
-- even when terminal rows accumulate.
CREATE INDEX IF NOT EXISTS idx_report_jobs_runnable
    ON report_jobs (created_at)
 WHERE status = 'pending';

-- Idempotency: only one open job per (kind, incident_number) at a
-- time. Terminal rows are excluded from the constraint so a user
-- can re-submit after a failure or a previous done.
CREATE UNIQUE INDEX IF NOT EXISTS idx_report_jobs_open_unique
    ON report_jobs (kind, incident_number)
 WHERE status IN ('pending', 'running');

-- API polling lookup. The frontend calls ``GET /jobs/{id}`` so the
-- primary key already covers this. Index on (requested_by, created_at)
-- supports the future "my recent jobs" admin view.
CREATE INDEX IF NOT EXISTS idx_report_jobs_by_user
    ON report_jobs (requested_by, created_at DESC);
