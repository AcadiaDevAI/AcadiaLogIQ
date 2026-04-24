-- Sprint 6 — Tier-1 Alert Copilot (form-driven triage module)
--
-- Three buckets of changes, all idempotent and safe to apply with
-- LOGIQ_TIER1_COPILOT_BACKEND=False (the columns / tables sit unused
-- until the router mount is enabled at app boot):
--
-- 1. Three denormalized searchable columns on chunks
--    (alert_signature, fingerprints_text, component_category) + their
--    indexes. These power the Tier-1 "Stage 1 — exact signature"
--    SQL path. The columns are derivable from metadata_json but are
--    cheaper to index and grep as plain text.
-- 2. A new tier1_answer_cache table keyed by a SHA-1 of the
--    normalized alert signature. Separate table because the cache
--    lifecycle is INDEPENDENT of chunk lifecycle — a ticket re-upload
--    should NOT auto-bust Tier-1 answers (§5 of the sprint spec).
-- 3. A new tier1_feedback table — append-only telemetry of 👍/👎 +
--    follow-up action selections.
--
-- Correction vs spec §2 migration block:
--   chunks.id is TEXT (see migration 001_phase1_foundation.sql:75),
--   NOT UUID. The spec's
--     matched_chunk_id UUID REFERENCES chunks(id)
--   would fail with a type mismatch. Corrected to TEXT below. The FK
--   semantics (ON DELETE CASCADE) are preserved.
--
-- Correction vs spec backfill predicate:
--   doc_kind lives on the documents table (migration 034), not on
--   chunks — but ingestion ALSO denormalizes it into
--   metadata_json->>'doc_kind' (contextual_ingestion_service.py:822).
--   The spec's "WHERE doc_kind = 'ticket'" predicate would resolve
--   against a non-existent chunks.doc_kind column. Using the
--   metadata_json path keeps the backfill local to chunks and
--   correct.

BEGIN;

-- ────────────────────────────────────────────────
-- 1. Denormalized searchable columns on chunks
-- ────────────────────────────────────────────────
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS alert_signature TEXT,
    ADD COLUMN IF NOT EXISTS fingerprints_text TEXT,
    ADD COLUMN IF NOT EXISTS component_category TEXT;

CREATE INDEX IF NOT EXISTS idx_chunks_alert_signature
    ON chunks(alert_signature);

CREATE INDEX IF NOT EXISTS idx_chunks_component_category
    ON chunks(component_category);

CREATE INDEX IF NOT EXISTS idx_chunks_fingerprints_text_gin
    ON chunks USING GIN (to_tsvector('english', fingerprints_text));

-- ────────────────────────────────────────────────
-- 2. Signature-keyed Tier-1 answer cache
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tier1_answer_cache (
    signature_hash      VARCHAR(40) PRIMARY KEY,
    alert_signature     TEXT NOT NULL,
    answer_json         JSONB NOT NULL,
    matched_chunk_id    TEXT REFERENCES chunks(id) ON DELETE CASCADE,
    confidence          VARCHAR(10) NOT NULL,
    created_at          TIMESTAMP NOT NULL DEFAULT NOW(),
    expires_at          TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tier1_cache_expires
    ON tier1_answer_cache(expires_at);

-- ────────────────────────────────────────────────
-- 3. Tier-1 feedback telemetry (append-only)
-- ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tier1_feedback (
    id                  SERIAL PRIMARY KEY,
    response_id         VARCHAR(40) NOT NULL,
    session_id          VARCHAR(64) NOT NULL,
    helpful             BOOLEAN NOT NULL,
    follow_up_action    VARCHAR(40),
    created_at          TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tier1_feedback_response
    ON tier1_feedback(response_id);

-- ────────────────────────────────────────────────
-- 4. Backfill (idempotent — only rows that are ticket gold-schema AND
--    don't yet have alert_signature populated)
-- ────────────────────────────────────────────────
UPDATE chunks
SET
    alert_signature = LOWER(
        COALESCE(metadata_json->'Metadata'->>'priority','') || ' | ' ||
        COALESCE(metadata_json->'Metadata'->>'Target_Service','') || ' | ' ||
        COALESCE(metadata_json->'Symptom_Solution_Mapping'->>'Detected_Symptom','')
    ),
    fingerprints_text = COALESCE(
        (
            SELECT string_agg(value::text, ' ')
            FROM jsonb_array_elements_text(metadata_json->'Metadata'->'Fingerprints')
        ),
        ''
    ),
    component_category = metadata_json->'Metadata'->>'component_category'
WHERE metadata_json->'Metadata' IS NOT NULL
  AND (metadata_json->>'doc_kind' IS NULL OR metadata_json->>'doc_kind' = 'ticket')
  AND alert_signature IS NULL;

COMMIT;
