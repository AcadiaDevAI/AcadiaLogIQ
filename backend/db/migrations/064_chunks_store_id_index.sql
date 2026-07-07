-- Migration 064 — index for US Pharma store-scoped tier-1 matching.
--
-- US Pharma's tier-1 intake hard-filters historic tickets to a single store
-- via metadata_json -> 'Metadata' ->> 'store_id' (see
-- backend/tier1_copilot/retrieval.py). This btree expression index makes that
-- exact-equality filter fast.
--
-- Additive + org-agnostic + safe: Acadia tickets have no store_id, so they are
-- simply absent from the index and unaffected. Idempotent (IF NOT EXISTS).
-- Standalone statement (no DO block) so the semicolon-splitting migrate.py
-- runner applies it.
CREATE INDEX IF NOT EXISTS idx_chunks_store_id
    ON chunks ((metadata_json -> 'Metadata' ->> 'store_id'));
