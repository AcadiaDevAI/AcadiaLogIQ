-- Sprint 10 — Tier-1 Resolution Journey telemetry table.
--
-- Per spec §4: append-only event log capturing every stage's render,
-- helpful click, next-stage click, and abandoned signal. Insertion is
-- fire-and-forget from `tier1_copilot/journey/telemetry.py`; failures
-- are logged but never propagated. Read by `stage5_escalation.py` to
-- compose the "stages traversed" log appended to the escalation
-- package's what_was_tried.
--
-- Idempotent — safe to re-apply on a fresh checkout or snapshot DB.

CREATE TABLE IF NOT EXISTS tier1_journey_events (
    id              BIGSERIAL PRIMARY KEY,
    session_id      TEXT NOT NULL,
    user_id         TEXT NULL,
    stage           TEXT NOT NULL,
        -- 'stage_0' | 'stage_1a' | 'stage_1b'
        -- 'stage_2' | 'stage_3'  | 'stage_4'  | 'stage_5'
    event_type      TEXT NOT NULL,
        -- 'stage_rendered' | 'helpful_clicked'
        -- 'next_stage_clicked' | 'abandoned'
    payload_json    JSONB NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tier1_journey_events_session
    ON tier1_journey_events(session_id);

CREATE INDEX IF NOT EXISTS idx_tier1_journey_events_stage
    ON tier1_journey_events(stage);

CREATE INDEX IF NOT EXISTS idx_tier1_journey_events_type
    ON tier1_journey_events(event_type);

COMMENT ON TABLE tier1_journey_events IS
    'Sprint 10 — append-only telemetry for the Tier-1 Resolution Journey. '
    'Captures stage_rendered / helpful_clicked / next_stage_clicked / abandoned '
    'per stage so we can measure which stage actually solved each ticket.';
