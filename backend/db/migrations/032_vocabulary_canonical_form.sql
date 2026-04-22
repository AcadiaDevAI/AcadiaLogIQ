-- Migration 032 — Canonical-form alias matching on learned vocabulary.
-- Collapses casing/underscore variants (RAGPotencyMetadata,
-- RAG_Potency_Metadata, rag_potency_metadata) to a shared upper snake-case
-- canonical_form so queries match any alias regardless of source casing.
-- Safe to re-run; all statements are idempotent.
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py wraps
-- each migration in engine.begin() and splits on ';'.

ALTER TABLE learned_vocabulary
    ADD COLUMN IF NOT EXISTS canonical_form TEXT;

CREATE INDEX IF NOT EXISTS idx_vocab_canonical_form
    ON learned_vocabulary (canonical_form);
