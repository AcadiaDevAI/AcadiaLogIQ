-- Sprint 9 — Universal Intake (Email/Phone/Portal/Chat/Note)
--
-- Single new audit table. No schema changes to any existing object.
-- Idempotent (CREATE TABLE IF NOT EXISTS). Safe to apply with
-- LOGIQ_UNIVERSAL_INTAKE_BACKEND=False — the table sits unused.
--
-- Why log full candidates_json (rejected ones included): future
-- fine-tuning / eval datasets need both picks AND rejections to
-- preserve the ranking signal. Logging only the pick loses
-- information (per spec §10).

BEGIN;

CREATE TABLE IF NOT EXISTS intake_extractions (
    id              TEXT PRIMARY KEY,
    created_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    session_id      TEXT,
    source          TEXT NOT NULL,           -- email | phone | portal | chat | note
    raw_text        TEXT NOT NULL,
    raw_text_hash   TEXT NOT NULL,           -- sha1, for dedup queries
    candidates_json JSONB NOT NULL,          -- full candidate list (validation included)
    picked_index    INTEGER,                 -- which card the engineer picked, NULL = none
    edits_json      JSONB,                   -- engineer's manual edits after picking
    was_rejected    BOOLEAN NOT NULL DEFAULT FALSE,
    feedback_at     TIMESTAMP                -- when picked_index/edits were recorded
);

CREATE INDEX IF NOT EXISTS idx_intake_extractions_created_at
    ON intake_extractions(created_at);

CREATE INDEX IF NOT EXISTS idx_intake_extractions_source
    ON intake_extractions(source);

CREATE INDEX IF NOT EXISTS idx_intake_extractions_session
    ON intake_extractions(session_id);

CREATE INDEX IF NOT EXISTS idx_intake_extractions_raw_hash
    ON intake_extractions(raw_text_hash);

COMMIT;
