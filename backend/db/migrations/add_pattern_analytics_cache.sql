-- Pattern analytics cache — topic-level computed statistics for Layer 3
-- selective dynamic pattern engine. TTL-expiring entries let multiple
-- users asking about the same topic reuse freshly computed stats.
--
-- Safe to run multiple times: all DDL uses IF NOT EXISTS guards.

CREATE TABLE IF NOT EXISTS pattern_analytics_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id TEXT NOT NULL,
    topic_key TEXT NOT NULL,          -- normalized topic hash (e.g. router_failure)
    query_signature TEXT,             -- optional full-query hash for precise cache key

    -- Structure:
    -- {
    --   "occurrence_count": 7,
    --   "timeframe_months": 4,
    --   "top_actions": [{"action": "...", "count": 5}, ...],
    --   "most_successful": "...",
    --   "success_rate": 0.71,
    --   "success_count": 5,
    --   "total_count": 7,
    --   "recent_count_30d": 3,
    --   "confidence_score": 0.82,
    --   "matched_ticket_ids": ["INC-10005", ...]
    -- }
    pattern_data JSONB NOT NULL,

    confidence_score FLOAT DEFAULT 0.0,
    matched_ticket_count INT DEFAULT 0,
    computed_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL,

    UNIQUE (organization_id, topic_key)
);

CREATE INDEX IF NOT EXISTS idx_pattern_cache_lookup
    ON pattern_analytics_cache (organization_id, topic_key, expires_at);

CREATE INDEX IF NOT EXISTS idx_pattern_cache_expiry
    ON pattern_analytics_cache (expires_at);
