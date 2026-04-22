-- Sprint 2.9 — JSON Structure Validator
-- Adds ingestion_status + ingestion_error columns so malformed-JSON uploads
-- can be surfaced in the admin UI with a persistent red-dot badge.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS ingestion_status VARCHAR(40) NOT NULL DEFAULT 'ok';

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS ingestion_error TEXT;
