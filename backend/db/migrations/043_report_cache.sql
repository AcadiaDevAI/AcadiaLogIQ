-- Migration 043 — Report cache + report feedback.
--
-- Backs the cached RCA / Gap Analysis report-generation pipeline.
-- Two purpose-separated tables:
--
--   report_cache     — current-best cached Markdown per incident
--                      per report_kind. Read on every generate
--                      request. WRITTEN once per (incident, kind)
--                      lifetime; DELETED when a 👎 fires so the
--                      next generate re-runs the LLM and replaces
--                      the row.
--
--   report_feedback  — append-only audit log of every 👍 / 👎.
--                      Survives cache deletion (the analytics use
--                      case keeps the history intact even after a
--                      cached row is invalidated). Future admin
--                      dashboard queries this table to surface
--                      "most disliked incidents", "feedback by
--                      report_kind", etc.
--
-- Cache scope: per (report_kind, incident_number) — global across
-- users. Two engineers viewing the same incident see the same
-- cached report unless someone dislikes it.
--
-- Recognised report_kind values today:
--   rca_customer       — Customer-Facing External RCA
--   rca_internal       — Internal Incident RCA
--   gap_analysis_gap   — LogIQ Gap Analysis Report
--   gap_analysis_pm    — Blameless SRE Post-Mortem
--
-- Keep the column a free-form TEXT (not an enum) so new report
-- kinds don't require another migration — the application layer
-- owns the whitelist.
--
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py
-- wraps each migration in engine.begin() and splits on ';', so
-- explicit transaction control would double-wrap and misfire the
-- semicolon split (matches migration 042's pattern).
-- pgcrypto is needed for gen_random_uuid(); CREATE EXTENSION is
-- idempotent in modern Postgres.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS report_cache (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_kind     VARCHAR(64) NOT NULL,
    incident_number VARCHAR(128) NOT NULL,
    markdown        TEXT NOT NULL,
    model_id        VARCHAR(128),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT report_cache_unique UNIQUE (report_kind, incident_number)
);

CREATE INDEX IF NOT EXISTS idx_report_cache_incident
    ON report_cache (incident_number);

CREATE TABLE IF NOT EXISTS report_feedback (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_kind     VARCHAR(64) NOT NULL,
    incident_number VARCHAR(128) NOT NULL,
    feedback_type   VARCHAR(16) NOT NULL,
    feedback_by     VARCHAR(256),
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT report_feedback_type_chk
        CHECK (feedback_type IN ('like', 'dislike'))
);

CREATE INDEX IF NOT EXISTS idx_report_feedback_incident
    ON report_feedback (incident_number, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_report_feedback_kind
    ON report_feedback (report_kind, feedback_type, created_at DESC);
