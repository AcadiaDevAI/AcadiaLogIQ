-- Sprint 4 — Fingerprint-First Expert Copilot
-- GIN indexes to support millisecond exact-match lookups on gold-ticket
-- JSON payloads. Fingerprints are stored at
--   chunks.metadata_json -> 'Metadata' -> 'Fingerprints'
-- as a JSONB array of strings (e.g., ["BGP-5-ADJCHANGE", "TCP-179-TIMEOUT"]).
--
-- Correction note: the Sprint 4 spec §3.2 indexes the `documents` table,
-- but gold-ticket JSON is one-chunk-per-ticket and its rich metadata lives
-- on chunks.metadata_json (see _ingest_gold_ticket_json in
-- contextual_ingestion_service.py). `idx_chunks_metadata_json_gin`
-- already exists (migration 002) as a general-purpose GIN on the whole
-- JSONB; the indexes below are NARROW keypath indexes that the `?`
-- operator on `Fingerprints` and the `@>` operator on Domain_Type
-- targeting can use directly — those two lookups are what Sprint 4
-- retrieve_by_fingerprint() and future domain-based routing need.
--
-- All three statements are idempotent (IF NOT EXISTS). Safe to apply
-- with LOGIQ_SPRINT4_BACKEND=False — the indexes just sit unused.

BEGIN;

-- Narrow GIN on the Fingerprints array. Enables `?` / `?|` / `?&`
-- operators with index support. Sprint 4 retrieval uses `?` (single
-- fingerprint exact match).
CREATE INDEX IF NOT EXISTS idx_chunks_fingerprints
    ON chunks USING GIN ((metadata_json -> 'Metadata' -> 'Fingerprints'));

-- Narrow GIN on Domain_Type (future-proof for WAN/LAN/Datacenter
-- domain-based routing). Lives under Dynamic_Domain_Payload per the
-- Sprint 4 schema sample.
CREATE INDEX IF NOT EXISTS idx_chunks_domain_type
    ON chunks USING GIN ((metadata_json -> 'Metadata' -> 'Dynamic_Domain_Payload' -> 'Domain_Type'));

-- jsonb_path_ops GIN on the full metadata_json — optimized for `@>`
-- containment queries which are smaller and faster than the default
-- jsonb_ops for that single operator. Useful for ad-hoc filters like
--   WHERE metadata_json @> '{"Metadata": {"Priority": "P1"}}'::jsonb
-- that might emerge in future sprints.
CREATE INDEX IF NOT EXISTS idx_chunks_metadata_path_ops
    ON chunks USING GIN (metadata_json jsonb_path_ops);

COMMIT;
