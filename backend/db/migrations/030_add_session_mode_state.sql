-- Migration 030 — Add session mode state fields for guided workflow.
-- See: User Journey - LogIQ Landing Page Guidance (PRD Section 13).
-- Safe to re-run; all ADD COLUMN statements are IF NOT EXISTS.
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py wraps
-- each migration in engine.begin() and splits on ';', so explicit
-- transaction control would double-wrap and misfire the semicolon split.

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS selected_mode            TEXT,
    ADD COLUMN IF NOT EXISTS sub_mode                 TEXT,
    ADD COLUMN IF NOT EXISTS conversation_context_active BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS customer_name            TEXT,
    ADD COLUMN IF NOT EXISTS technology_domain        TEXT,
    ADD COLUMN IF NOT EXISTS ticket_id                TEXT,
    ADD COLUMN IF NOT EXISTS issue_summary            TEXT,
    ADD COLUMN IF NOT EXISTS last_recommendation      JSONB,
    ADD COLUMN IF NOT EXISTS form_data                JSONB,
    ADD COLUMN IF NOT EXISTS mode_set_at              TIMESTAMPTZ;

-- Index for fast session-mode lookups on /ask hot path.
CREATE INDEX IF NOT EXISTS idx_chat_sessions_mode
    ON chat_sessions (selected_mode)
    WHERE selected_mode IS NOT NULL;
