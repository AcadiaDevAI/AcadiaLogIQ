-- Migration 044 — Exact-match answer cache (Postgres-backed).
--
-- Replaces the in-process LRU in ``backend/routing/answer_cache.py``.
-- Same role: catch rapid duplicate ``/ask`` queries (exact string
-- match after normalisation) so we don't pay the embedding+LLM cost
-- to answer the same question twice.
--
-- Why a separate table instead of reusing ``semantic_answer_cache``:
--   * ``semantic_answer_cache`` requires an embedding vector on every
--     entry — embedding cost is ~50–100 ms via Titan and is precisely
--     what this layer is meant to skip on a hot exact match.
--   * Different invalidation rules: semantic entries are cleared on
--     a user dislike; exact entries TTL out on their own.
--   * Different scan patterns — exact lookup is a single ``=`` query
--     on a sha key (≤1 ms). Semantic is a vector kNN.
-- Keeping the two purpose-separated keeps each schema simple.
--
-- Cache scope: ``(normalized_query, owner_id, file_ids_fingerprint)``
-- — global across replicas. Two engineers running the same query
-- against the same file set hit the same cache row.
--
-- TTL is enforced at READ time (``expires_at > NOW()``) rather than
-- via a cron sweeper — keeps the migration minimal and avoids a
-- background job. Old rows do accumulate; the read filter ignores
-- them and a nightly ``VACUUM`` (RDS automatic) reclaims space.
-- If table size becomes a concern, add ``DELETE FROM
-- answer_cache_exact WHERE expires_at < NOW()`` to a cron.
--
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py
-- wraps each migration in engine.begin() and splits on ';'.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS answer_cache_exact (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    -- sha1 of (normalized_query || owner_id || sorted_file_ids).
    -- 40-char hex; the application layer computes the key.
    cache_key        VARCHAR(40) NOT NULL UNIQUE,
    -- Full payload (sources + answer + metadata). The application
    -- layer serialises whatever the legacy LRU used to store.
    payload_json     JSONB NOT NULL,
    -- TTL — the read path ignores rows where expires_at < NOW().
    expires_at       TIMESTAMPTZ NOT NULL,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- The UNIQUE (cache_key) constraint above already creates a btree
-- index on cache_key, so hot exact-match lookups (.get) are O(log n).
-- We don't add a partial index filtered by expires_at because
-- Postgres rejects ``WHERE expires_at > NOW()`` in an index predicate
-- (NOW() is not IMMUTABLE — the index "alive" set would drift over
-- time, breaking the planner's correctness contract). The runtime
-- filter on the read path is plenty fast against a single-row index
-- probe.

-- Time-ordered scan for the optional sweeper / monitoring queries.
CREATE INDEX IF NOT EXISTS idx_answer_cache_exact_expires
    ON answer_cache_exact (expires_at);
