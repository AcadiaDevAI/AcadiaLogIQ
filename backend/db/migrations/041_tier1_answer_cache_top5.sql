-- Sprint 10.3 — cache must carry the cohort so cache hits don't lose
-- downstream state needed by the Resolution Journey. Without this,
-- /tier1/journey/{sid}/initial returns cohort=0 on every cache hit
-- because the new tier1_sessions row has no top_5_match_ids to read.
--
-- This is the structural fix per spec §3.3.1 — extending the cache
-- schema to carry the cohort. NOT a workaround like skip-cache or
-- re-run-retrieval-on-hit.

ALTER TABLE tier1_answer_cache
    ADD COLUMN IF NOT EXISTS top_5_match_ids TEXT[] NOT NULL DEFAULT '{}';

-- No backfill. Existing rows have empty arrays; the next analyze on
-- those signatures repopulates. This is fine because:
--   (a) cached rows pre-Sprint 10.3 were never used by the journey
--       anyway (the journey wasn't reading cache top_5_match_ids), and
--   (b) the next call regenerates the cohort during the cache-miss
--       path before the new write.

COMMENT ON COLUMN tier1_answer_cache.top_5_match_ids IS
    'Sprint 10.3 — the cohort chunk_ids this answer was derived from. '
    'Repopulated on every fresh /analyze call. Empty array for rows '
    'cached pre-Sprint 10.3 (they will refresh on next signature hit).';
