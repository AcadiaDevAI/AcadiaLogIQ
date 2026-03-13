CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS normalized_name TEXT NULL,
    ADD COLUMN IF NOT EXISTS source_type TEXT NULL DEFAULT 'file',
    ADD COLUMN IF NOT EXISTS version_family_key TEXT NULL,
    ADD COLUMN IF NOT EXISTS duplicate_status TEXT NULL DEFAULT 'unique',
    ADD COLUMN IF NOT EXISTS superseded_by_document_id UUID NULL,
    ADD COLUMN IF NOT EXISTS latest_effective_at TIMESTAMP NULL;

ALTER TABLE documents
    DROP CONSTRAINT IF EXISTS documents_superseded_by_document_id_fkey;

ALTER TABLE documents
    ADD CONSTRAINT documents_superseded_by_document_id_fkey
    FOREIGN KEY (superseded_by_document_id) REFERENCES documents(id);

ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS version_label TEXT NULL,
    ADD COLUMN IF NOT EXISTS version_rank NUMERIC(12,4) NULL,
    ADD COLUMN IF NOT EXISTS document_date TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS effective_date TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS created_date TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS extraction_model TEXT NULL,
    ADD COLUMN IF NOT EXISTS contextualization_model TEXT NULL,
    ADD COLUMN IF NOT EXISTS parse_strategy TEXT NULL,
    ADD COLUMN IF NOT EXISTS duplicate_decision_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS enrichment_json JSONB NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE document_metadata
    ADD COLUMN IF NOT EXISTS title TEXT NULL,
    ADD COLUMN IF NOT EXISTS section_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS chunk_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS metadata_version TEXT NULL DEFAULT 'phase2',
    ADD COLUMN IF NOT EXISTS extracted_at TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS vendor TEXT NULL,
    ADD COLUMN IF NOT EXISTS product TEXT NULL,
    ADD COLUMN IF NOT EXISTS domain TEXT NULL,
    ADD COLUMN IF NOT EXISTS document_type TEXT NULL,
    ADD COLUMN IF NOT EXISTS version_label TEXT NULL,
    ADD COLUMN IF NOT EXISTS document_date TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS effective_date TIMESTAMP NULL,
    ADD COLUMN IF NOT EXISTS created_date TIMESTAMP NULL;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS chunk_type TEXT NULL,
    ADD COLUMN IF NOT EXISTS section_heading TEXT NULL,
    ADD COLUMN IF NOT EXISTS page_number INTEGER NULL,
    ADD COLUMN IF NOT EXISTS token_estimate INTEGER NULL,
    ADD COLUMN IF NOT EXISTS summary TEXT NULL,
    ADD COLUMN IF NOT EXISTS contextualized_content TEXT NULL,
    ADD COLUMN IF NOT EXISTS labels_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS source_order INTEGER NULL;

CREATE INDEX IF NOT EXISTS idx_documents_owner_normalized_name
    ON documents(owner_id, normalized_name);

CREATE INDEX IF NOT EXISTS idx_documents_owner_version_family
    ON documents(owner_id, version_family_key);

CREATE INDEX IF NOT EXISTS idx_document_versions_fingerprint
    ON document_versions(fingerprint);

CREATE INDEX IF NOT EXISTS idx_chunks_section_heading
    ON chunks(document_id, section_heading);

CREATE INDEX IF NOT EXISTS idx_chunks_chunk_type
    ON chunks(document_id, chunk_type);

CREATE INDEX IF NOT EXISTS idx_chunks_labels_json_gin
    ON chunks USING GIN (labels_json);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata_json_gin
    ON chunks USING GIN (metadata_json);