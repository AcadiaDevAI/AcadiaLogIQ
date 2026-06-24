-- Migration 056 — Phase 1 multi-tenant: organization_id on the RAG corpus.
--
-- Adds organization_id UUID (FK → organizations(id)) to every table in
-- the document → chunk → embedding chain plus ingestion_jobs. The column
-- is populated with the Acadia UUID for every existing row (single-tenant
-- baseline), then locked NOT NULL with a temporary DEFAULT as a Phase-1
-- safety net.
--
-- Why each table gets its own organization_id even though FK chains
-- already exist (chunks → documents, embeddings → chunks): Postgres
-- Row Level Security policies are most efficient (and least error-prone)
-- when each row carries the org column directly. RLS policies written
-- against an FK chain require an EXISTS subquery on every read — fine
-- for one or two joins, lethal on hot retrieval paths that scan chunks.
-- Denormalizing organization_id keeps the policy a single equality test.
--
-- The DEFAULT '76c36d23-...' is intentional for Phase 1: it lets newly
-- inserted rows (which Phase 1 app code does NOT yet stamp explicitly)
-- still satisfy NOT NULL. Phase 2 removes the DEFAULT — by then every
-- INSERT call will set organization_id from request context.
--
-- Idempotency: every statement is guarded with IF NOT EXISTS or a DO
-- block that probes information_schema first. Safe to re-apply.

-- ─────────────────────────────────────────────────────────────
-- documents
-- ─────────────────────────────────────────────────────────────
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE documents
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE documents
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'documents_organization_id_fkey'
           AND table_name = 'documents'
    ) THEN
        ALTER TABLE documents
            ADD CONSTRAINT documents_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_documents_org
    ON documents(organization_id);

-- ─────────────────────────────────────────────────────────────
-- document_versions
-- ─────────────────────────────────────────────────────────────
ALTER TABLE document_versions
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE document_versions
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE document_versions
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'document_versions_organization_id_fkey'
           AND table_name = 'document_versions'
    ) THEN
        ALTER TABLE document_versions
            ADD CONSTRAINT document_versions_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_document_versions_org
    ON document_versions(organization_id);

-- ─────────────────────────────────────────────────────────────
-- document_metadata
-- ─────────────────────────────────────────────────────────────
ALTER TABLE document_metadata
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE document_metadata
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE document_metadata
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'document_metadata_organization_id_fkey'
           AND table_name = 'document_metadata'
    ) THEN
        ALTER TABLE document_metadata
            ADD CONSTRAINT document_metadata_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_document_metadata_org
    ON document_metadata(organization_id);

-- ─────────────────────────────────────────────────────────────
-- chunks  (CRITICAL — the retrieval primitive; hottest read path)
-- ─────────────────────────────────────────────────────────────
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE chunks
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE chunks
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'chunks_organization_id_fkey'
           AND table_name = 'chunks'
    ) THEN
        ALTER TABLE chunks
            ADD CONSTRAINT chunks_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

-- Hot retrieval read path: org_id is always the first WHERE clause once
-- enforcement lands, so a btree on organization_id alone (and a composite
-- with document_id for the per-doc fetch) cover the dominant queries.
CREATE INDEX IF NOT EXISTS idx_chunks_org
    ON chunks(organization_id);
CREATE INDEX IF NOT EXISTS idx_chunks_org_doc
    ON chunks(organization_id, document_id);

-- ─────────────────────────────────────────────────────────────
-- embeddings  (CRITICAL — vector search; would otherwise leak across orgs)
-- ─────────────────────────────────────────────────────────────
ALTER TABLE embeddings
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE embeddings
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE embeddings
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'embeddings_organization_id_fkey'
           AND table_name = 'embeddings'
    ) THEN
        ALTER TABLE embeddings
            ADD CONSTRAINT embeddings_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_embeddings_org
    ON embeddings(organization_id);

-- ─────────────────────────────────────────────────────────────
-- ingestion_jobs
-- ─────────────────────────────────────────────────────────────
ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE ingestion_jobs
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE ingestion_jobs
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'ingestion_jobs_organization_id_fkey'
           AND table_name = 'ingestion_jobs'
    ) THEN
        ALTER TABLE ingestion_jobs
            ADD CONSTRAINT ingestion_jobs_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_org
    ON ingestion_jobs(organization_id);
