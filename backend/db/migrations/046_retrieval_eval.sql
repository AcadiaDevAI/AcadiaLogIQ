-- Migration 046 — Shadow-mode retrieval diff log.
--
-- During the BM25 → FTS migration, every retrieval call runs BOTH
-- channels (BM25 authoritative, FTS shadow) and logs the diff here.
-- After ~2 weeks of clean diffs (or a green golden-query eval run),
-- ``RETRIEVAL_BM25_ENABLED`` flips to false, the BM25 channel goes
-- silent, and 30 days later the BM25 module is deleted from the
-- codebase.
--
-- Why a dedicated table:
--   * Volume — every /ask call writes a row. CloudWatch JSON logs
--     work too, but a structured table makes ad-hoc analysis
--     ("which queries did the two channels disagree on most?") a
--     single SQL away.
--   * Retention — 30 days is plenty for the migration window. A
--     scheduled delete keeps growth bounded.
--   * Cheap — single INSERT, no joins, fail-safe (never blocks the
--     request that triggered it).
--
-- The harness reads this table to produce the migration verdict
-- ("FTS recall ≥ 0.90 of BM25 recall over the last 14 days?").

CREATE TABLE IF NOT EXISTS retrieval_eval (
    id              BIGSERIAL PRIMARY KEY,
    -- The query text the user actually sent. Useful for "what kind
    -- of queries does FTS struggle with?". We DO NOT store the
    -- caller's user_id here — anonymous comparison only.
    query_text      TEXT NOT NULL,
    -- Number of files in the active set when the query ran. Lets us
    -- bucket diffs by tenancy / scoping.
    file_set_size   INTEGER NOT NULL DEFAULT 0,
    -- Ordered top-K chunk IDs from each channel (jsonb array).
    bm25_top_k      JSONB NOT NULL,
    fts_top_k       JSONB NOT NULL,
    -- Cheap pre-aggregated metrics so dashboards don't have to
    -- re-compute set ops at query time.
    overlap         INTEGER NOT NULL DEFAULT 0,
    bm25_size       INTEGER NOT NULL DEFAULT 0,
    fts_size        INTEGER NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Time-range scans dominate analysis ("show me last 7 days").
CREATE INDEX IF NOT EXISTS idx_retrieval_eval_created
    ON retrieval_eval (created_at DESC);

-- Bucketed analysis ("show me low-overlap queries").
CREATE INDEX IF NOT EXISTS idx_retrieval_eval_overlap
    ON retrieval_eval (overlap, created_at DESC);
