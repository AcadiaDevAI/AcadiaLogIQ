-- Migration: Create LogIQ sessions table
-- Safe to run multiple times (IF NOT EXISTS guards).
-- No impact on existing tables, no foreign keys into existing schema.
--
-- Rollback: DROP TABLE IF EXISTS logiq_sessions;

CREATE TABLE IF NOT EXISTS logiq_sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id VARCHAR(255) NOT NULL,
    org_id VARCHAR(255) NOT NULL DEFAULT 'acadia_internal',

    current_mode VARCHAR(50),
    mode_history JSONB NOT NULL DEFAULT '[]'::jsonb,

    troubleshoot_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    process_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    kb_context JSONB NOT NULL DEFAULT '{}'::jsonb,
    escalate_context JSONB NOT NULL DEFAULT '{}'::jsonb,

    root_query TEXT,
    root_mode VARCHAR(50),

    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL DEFAULT 'active',

    thumbs_history JSONB NOT NULL DEFAULT '[]'::jsonb,
    resolution_path JSONB NOT NULL DEFAULT '[]'::jsonb,

    CONSTRAINT logiq_sessions_status_check CHECK (
        status IN ('active', 'resolved', 'escalated', 'abandoned')
    )
);

CREATE INDEX IF NOT EXISTS idx_logiq_sessions_user_id
    ON logiq_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_logiq_sessions_org_id
    ON logiq_sessions(org_id);
CREATE INDEX IF NOT EXISTS idx_logiq_sessions_status
    ON logiq_sessions(status);
CREATE INDEX IF NOT EXISTS idx_logiq_sessions_updated_at
    ON logiq_sessions(updated_at DESC)
