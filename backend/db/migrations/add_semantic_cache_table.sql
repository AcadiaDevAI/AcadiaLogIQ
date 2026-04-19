-- Brief 5 / Part 1 — Semantic Answer Cache
-- Cross-user, document-scoped semantic cache. Requires pgvector + pgcrypto
-- (both already provisioned for the chunks table).

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS semantic_answer_cache (
    id                     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query_text             TEXT         NOT NULL,
    query_embedding        VECTOR(1024) NOT NULL,
    answer_text            TEXT         NOT NULL,
    answer_metadata_json   JSONB        NOT NULL,
    file_ids_fingerprint   TEXT         NOT NULL,
    source_file_ids        TEXT[]       NOT NULL,
    identifiers_extracted  TEXT[]       NOT NULL,
    negation_tokens        TEXT[]       NOT NULL,
    confidence             REAL         NOT NULL,
    grounding_score        REAL         NOT NULL,
    model_used             TEXT         NOT NULL,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    last_hit_at            TIMESTAMPTZ,
    hit_count              INTEGER      NOT NULL DEFAULT 0,
    origin_owner_id        TEXT         NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_semantic_cache_fingerprint
    ON semantic_answer_cache (file_ids_fingerprint);

CREATE INDEX IF NOT EXISTS ix_semantic_cache_created_at
    ON semantic_answer_cache (created_at);

CREATE INDEX IF NOT EXISTS ix_semantic_cache_embedding_hnsw
    ON semantic_answer_cache
    USING hnsw (query_embedding vector_cosine_ops);
