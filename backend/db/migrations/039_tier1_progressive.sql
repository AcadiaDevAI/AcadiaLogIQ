-- Sprint 7 — Tier-1 Progressive Workflow
--
-- Three idempotent changes, safe to apply while
-- LOGIQ_TIER1_PROGRESSIVE_BACKEND=False (the new objects sit unused):
--
-- 1. Add chunks.asset_family (denormalized prefix-family column)
--    + backfill it from the first element of
--    metadata_json->'Metadata'->'Affected_Assets' (JSONB array) with the
--    trailing numeric suffix stripped. The spec's original backfill used
--    `metadata_json->'Metadata'->>'Affected_Assets'` which treats the
--    array as a single JSON text (`["edge-rtr-01"]`) — the resulting
--    asset_family contains the brackets and quotes. Corrected below to
--    pick the first array element and normalize it.
-- 2. Create tier1_sessions — per-alert session state that feeds arrow
--    pagination + stuck detection + the escalation package's
--    "what_tried" log.
-- 3. Extend tier1_feedback with event_type + session_elapsed_seconds so
--    stuck-nudge + action-log rows can be recorded in the same table
--    without breaking the Sprint 6 UNIQUE telemetry shape.
--
-- Rollback: each ADD COLUMN / CREATE is IF NOT EXISTS; dropping requires
-- an explicit migration (out of scope for this sprint).

BEGIN;

-- ────────────────────────────────────────────────
-- 1. Asset-family column + backfill
-- ────────────────────────────────────────────────
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS asset_family TEXT;

CREATE INDEX IF NOT EXISTS idx_chunks_asset_family
    ON chunks(asset_family);

-- Pick the first element of the Affected_Assets JSONB array (if any)
-- and strip its trailing -NN / _NN numeric suffix. If Affected_Assets
-- is absent or empty, fall back to Target_Service (string). Idempotent:
-- only touches rows where asset_family is still NULL.
UPDATE chunks
SET asset_family = LOWER(
    REGEXP_REPLACE(
        COALESCE(
            (
                SELECT value
                FROM jsonb_array_elements_text(
                    CASE
                        WHEN jsonb_typeof(metadata_json->'Metadata'->'Affected_Assets') = 'array'
                            THEN metadata_json->'Metadata'->'Affected_Assets'
                        ELSE '[]'::jsonb
                    END
                ) value
                LIMIT 1
            ),
            metadata_json->'Metadata'->>'Target_Service',
            ''
        ),
        '[-_]?\d+$', ''
    )
)
WHERE asset_family IS NULL
  AND metadata_json->'Metadata' IS NOT NULL;

-- ────────────────────────────────────────────────
-- 2. tier1_sessions
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tier1_sessions (
    id                    TEXT PRIMARY KEY,
    created_at            TIMESTAMP NOT NULL DEFAULT NOW(),
    last_activity_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    alert_signature       TEXT NOT NULL,
    alert_payload         JSONB NOT NULL,
    top_5_match_ids       TEXT[] NOT NULL DEFAULT '{}',
    current_match_index   INTEGER NOT NULL DEFAULT 0,
    thumbs_down_count     INTEGER NOT NULL DEFAULT 0,
    what_tried            JSONB NOT NULL DEFAULT '[]'::jsonb,
    stuck_nudge_shown     BOOLEAN NOT NULL DEFAULT FALSE,
    resolved              BOOLEAN NOT NULL DEFAULT FALSE,
    escalated             BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_tier1_sessions_last_activity
    ON tier1_sessions(last_activity_at);

-- ────────────────────────────────────────────────
-- 3. Feedback table extension
-- ────────────────────────────────────────────────
ALTER TABLE tier1_feedback
    ADD COLUMN IF NOT EXISTS event_type TEXT DEFAULT 'feedback',
    ADD COLUMN IF NOT EXISTS session_elapsed_seconds INTEGER;

COMMIT;
