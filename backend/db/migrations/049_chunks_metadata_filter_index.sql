-- Migration 049 — Composite expression index on chunks for the
-- Ticket Filter feature.
--
-- The Ticket Filter route (POST /ticket-filter) lets engineers
-- pull historical tickets matching a (SLA_Target_Met,
-- Resolution_Quality_Score) pair. Both values live inside
-- ``metadata_json->'Metadata'`` per the ticket JSON schema.
--
-- Without this index the route is a full sequential scan of the
-- chunks table. At 31 chunks today it's invisible; at 1 M+ chunks
-- it would be several seconds per query. The composite btree on
-- the two JSON path expressions turns it into an O(log n) lookup.
--
-- Why a btree, not a GIN:
--   * Equality on two scalar string fields → btree is the right
--     tool. GIN is for full-text / array containment.
--   * Composite ordering lets the planner use index-only scans
--     when both columns are restricted (the only query shape we
--     have today).
--
-- Why no DESC on Timestamp inside the index:
--   * The route pulls candidates first, then sorts the
--     deduplicated result by Timestamp in the application layer.
--     Adding Timestamp to the index would help large result sets
--     but waste storage on every other read pattern.
--
-- ``IF NOT EXISTS`` keeps this migration idempotent — safe to re-run.
-- No BEGIN/COMMIT; migrate.py wraps each migration in engine.begin().

CREATE INDEX IF NOT EXISTS idx_chunks_sla_score
    ON chunks (
        (metadata_json->'Metadata'->>'SLA_Target_Met'),
        (metadata_json->'Metadata'->>'Resolution_Quality_Score')
    )
 WHERE metadata_json->'Metadata'->>'Incident_Number' IS NOT NULL;
