-- Sprint 3-PREP-A — doc_kind multi-corpus tagging
-- Adds a classification column so every ingested document is tagged as
-- one of: ticket, sop, kb, contact_customer, contact_vendor, vendor_case.
-- Mode-specific sprints (3B–3E) will use this column to filter the
-- retrieval corpus per user mode without scattering conditionals.
--
-- Existing rows are all gold tickets — backfilled explicitly to 'ticket'
-- so even rows inserted before the DEFAULT took effect are consistent.
-- A single-column index is required for fast ANY() filtering at
-- retrieval time as the corpus grows past a few thousand rows.

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS doc_kind VARCHAR(40) NOT NULL DEFAULT 'ticket';

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS doc_kind_confidence FLOAT DEFAULT 1.0;

UPDATE documents
    SET doc_kind = 'ticket'
    WHERE doc_kind IS NULL OR doc_kind = '';

CREATE INDEX IF NOT EXISTS idx_documents_doc_kind ON documents(doc_kind);
