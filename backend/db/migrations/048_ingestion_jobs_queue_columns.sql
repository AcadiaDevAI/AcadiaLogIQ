-- Migration 048 — Promote ingestion_jobs into a Postgres queue.
--
-- Phase 1 introduced report_jobs as a SELECT-FOR-UPDATE-SKIP-LOCKED
-- queue for async LLM report generation. This migration brings the
-- pre-existing ingestion_jobs table up to the same surface so the
-- worker container can claim ingestion rows the same way it claims
-- report rows.
--
-- Why not merge into report_jobs:
--   * ingestion_jobs already carries application-specific columns
--     (file_size_mb, processed_chunks, ingestion_status, file_hash)
--     that don't belong in a generic job table.
--   * The sidebar UI + GET /ingestion_jobs/{id} both depend on
--     ingestion_jobs' existing schema — disturbing it for a refactor
--     would breach the "don't disturb core code" rule.
--   * Keeping the queues split also lets us scale ingest workers
--     and report workers independently (different CPU/memory needs).
--
-- Added columns (all NULL-safe defaults so existing rows are valid):
--   kind          — queue dispatch key. Default 'ingest_document' for
--                   the only kind we ship today; the column exists so
--                   future kinds (re-ingest, glossary-refresh, etc.)
--                   can share the table without another migration.
--   payload       — JSONB the worker reads to know what to do.
--                   Carries storage_uri, filename, file_type, file_id,
--                   owner_id, file_size_mb, doc_kind — i.e. everything
--                   the legacy ``background_tasks.add_task(index_file_job, …)``
--                   call used to pass as positional args.
--   attempts      — incremented by the worker on each claim.
--   max_attempts  — soft-fail re-queue cap (default 3, same as report_jobs).
--   not_before    — exponential-backoff gate; worker filters claims by it.
--   started_at    — set when the worker claims the row; used by the
--                   stuck-job sweeper to detect runaway ingests.
--
-- The status column already exists (pending/running/done/failed) — we
-- reuse it. Same state machine as report_jobs.
--
-- Indexing:
--   * idx_ingestion_jobs_runnable — partial on status='pending' for
--     the hot claim scan.
--   * idx_ingestion_jobs_open_unique — partial UNIQUE on (kind, file_id)
--     where status IN ('pending','running'), so the React.StrictMode
--     double-fire on the upload widget can't enqueue duplicates.
--
-- No BEGIN/COMMIT — migrate.py wraps each file in engine.begin() and
-- splits on ';'.

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS kind         VARCHAR(64) NOT NULL DEFAULT 'ingest_document',
    ADD COLUMN IF NOT EXISTS payload      JSONB       NOT NULL DEFAULT '{}'::JSONB,
    ADD COLUMN IF NOT EXISTS attempts     INTEGER     NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS max_attempts INTEGER     NOT NULL DEFAULT 3,
    ADD COLUMN IF NOT EXISTS not_before   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS started_at   TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_runnable
    ON ingestion_jobs (created_at)
 WHERE status = 'pending';

CREATE UNIQUE INDEX IF NOT EXISTS idx_ingestion_jobs_open_unique
    ON ingestion_jobs (kind, file_id)
 WHERE status IN ('pending', 'running');
