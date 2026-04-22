-- Migration 031 — Dynamic vocabulary learned from ingested documents.
-- Preserves customer-specific tokens (ticket IDs, field names, enum values)
-- during query normalization so retrieval matches stored content exactly.
-- Safe to re-run; all statements are IF NOT EXISTS.
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py wraps
-- each migration in engine.begin() and splits on ';'.

CREATE TABLE IF NOT EXISTS learned_vocabulary (
    token            TEXT PRIMARY KEY,
    token_type       TEXT NOT NULL CHECK (token_type IN ('identifier', 'field_name', 'enum_value')),
    occurrence_count INTEGER NOT NULL DEFAULT 1,
    first_seen_file  TEXT,
    first_seen_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_vocab_token_type ON learned_vocabulary (token_type);
CREATE INDEX IF NOT EXISTS idx_vocab_last_seen  ON learned_vocabulary (last_seen_at DESC);
