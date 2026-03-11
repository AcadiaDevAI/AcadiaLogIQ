-- Phase 1 structured migration
-- Creates the remaining PostgreSQL objects needed to finish Phase 1.
-- This version is SAFE for an existing older documents/chunks schema.

CREATE EXTENSION IF NOT EXISTS vector;

-- Optional but helpful for UUID defaults in PostgreSQL
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================================================
-- DOCUMENTS
-- If documents table already exists from old Phase 1 work,
-- add the new columns we now need.
-- =========================================================
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    fingerprint TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'anonymous',
    ADD COLUMN IF NOT EXISTS file_type TEXT NOT NULL DEFAULT 'kb',
    ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'active',
    ADD COLUMN IF NOT EXISTS current_version_id UUID NULL,
    ADD COLUMN IF NOT EXISTS notes TEXT NULL,
    ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;

-- Keep old fingerprint column if it already exists.
-- If this DB started from a newer schema and fingerprint is missing, add it.
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS fingerprint TEXT NULL;

-- =========================================================
-- DOCUMENT VERSIONS
-- =========================================================
CREATE TABLE IF NOT EXISTS document_versions (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    version_number INTEGER NOT NULL,
    fingerprint TEXT NULL,
    storage_uri TEXT NULL,
    mime_type TEXT NULL,
    file_size_mb TEXT NULL,
    is_active BOOLEAN NOT NULL DEFAULT TRUE,
    uploaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    superseded_at TIMESTAMP NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_current_version_id_fkey;

ALTER TABLE documents
    ADD CONSTRAINT documents_current_version_id_fkey
    FOREIGN KEY (current_version_id) REFERENCES document_versions(id);

-- =========================================================
-- DOCUMENT METADATA
-- =========================================================
CREATE TABLE IF NOT EXISTS document_metadata (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL UNIQUE REFERENCES documents(id) ON DELETE CASCADE,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =========================================================
-- CHUNKS
-- If chunks table already exists, add missing columns.
-- =========================================================
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS document_version_id UUID NULL,
    ADD COLUMN IF NOT EXISTS chunk_index INTEGER NULL,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP;

ALTER TABLE chunks
    DROP CONSTRAINT IF EXISTS chunks_document_version_id_fkey;

ALTER TABLE chunks
    ADD CONSTRAINT chunks_document_version_id_fkey
    FOREIGN KEY (document_version_id) REFERENCES document_versions(id) ON DELETE CASCADE;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_name = 'chunks'
          AND column_name = 'embedding'
    ) THEN
        ALTER TABLE chunks
            ALTER COLUMN embedding DROP NOT NULL;
    END IF;
END $$;
-- =========================================================
-- EMBEDDINGS
-- Separate vector table
-- =========================================================
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector(1024) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =========================================================
-- INGESTION JOBS
-- =========================================================
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id TEXT PRIMARY KEY,
    file_id UUID NOT NULL,
    owner_id TEXT NOT NULL,
    file_name TEXT NOT NULL,
    file_type TEXT NOT NULL DEFAULT 'kb',
    file_hash TEXT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    processed_chunks TEXT NOT NULL DEFAULT '0',
    total_chunks TEXT NOT NULL DEFAULT '0',
    successful_chunks TEXT NOT NULL DEFAULT '0',
    error TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL
);

-- =========================================================
-- CHAT PERSISTENCE
-- =========================================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id TEXT PRIMARY KEY,
    owner_id TEXT NOT NULL,
    title TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_json JSONB NULL,
    feedback TEXT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- =========================================================
-- INDEXES
-- =========================================================
CREATE INDEX IF NOT EXISTS idx_documents_owner_status
    ON documents(owner_id, status);

CREATE INDEX IF NOT EXISTS idx_document_versions_doc_active
    ON document_versions(document_id, is_active);

CREATE INDEX IF NOT EXISTS idx_chunks_doc_version
    ON chunks(document_version_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_owner_status
    ON ingestion_jobs(owner_id, status);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_updated
    ON chat_sessions(owner_id, updated_at DESC);

-- One active document with the same owner + name at a time.
CREATE UNIQUE INDEX IF NOT EXISTS uq_documents_owner_name_active
    ON documents(owner_id, name)
    WHERE status = 'active';