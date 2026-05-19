-- Migration 045 — FTS column + GIN index on chunks.
--
-- The retrieval ``fulltext_search`` channel currently computes
-- ``to_tsvector('english', ...)`` inline on every query, which forces
-- a sequential scan + per-row tsvector materialisation against the
-- chunks table. At 31 chunks today this is invisible; at 100 K
-- chunks it's seconds-per-query.
--
-- Fix: add a STORED generated column that materialises the tsvector
-- at write time, and an inverted GIN index over it. Reads then become
-- a single index probe with no per-row tokenisation.
--
-- Source: ``COALESCE(contextualized_content, '') || ' ' || COALESCE(content, '')``
-- — preserves the existing weighting where the LLM-enriched
-- ``contextualized_content`` (when present) is preferred over the raw
-- chunk text. Matches the SQL inside ``backend/retrieval/keyword_search.py``
-- so the index encodes exactly the same string the query function used to
-- tokenise inline.
--
-- ``IMMUTABLE`` only — Postgres requires the generation expression to
-- be immutable, which means we can't reference NOW() / random() / etc.
-- Plain text concatenation and ``to_tsvector`` are both immutable, so
-- this expression is fine.
--
-- Backfill: STORED generated columns are populated automatically on
-- INSERT and on column ADD. Postgres locks the table briefly to
-- materialise existing rows. At the current corpus size this is
-- imperceptible; on a >1 M-row corpus, run during a quiet window.
--
-- No BEGIN/COMMIT — migrate.py wraps each migration in engine.begin().

-- pg_trgm — needed when we add the trigram fallback for very short
-- FTS queries (≤ 2 tokens, where ``to_tsquery`` returns nothing
-- useful and similarity matching pays off). Installing here keeps
-- staging/prod migrations identical to dev with no separate "run
-- this CREATE EXTENSION" step. IF NOT EXISTS is idempotent so
-- re-runs are safe.
CREATE EXTENSION IF NOT EXISTS pg_trgm;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (
        to_tsvector(
            'english',
            COALESCE(contextualized_content, '') || ' ' || COALESCE(content, '')
        )
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv
    ON chunks USING GIN (content_tsv);
