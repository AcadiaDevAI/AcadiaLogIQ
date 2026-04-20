-- Organization-specific schema mappings for data ingestion.
-- Stores per-org mappings of canonical field names to actual column/key
-- names found in their uploaded CSV/JSON/XLSX files.
--
-- Safe to run multiple times: uses IF NOT EXISTS guards throughout.

CREATE TABLE IF NOT EXISTS organization_schemas (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id TEXT NOT NULL,
    source_type TEXT NOT NULL,          -- 'csv', 'json', 'xlsx'
    file_pattern TEXT DEFAULT '',       -- optional glob pattern for file name matching
    schema_mapping JSONB NOT NULL,      -- {canonical_field: actual_column_name}
    identifier_pattern TEXT,            -- regex for identifier extraction, e.g. 'INC-\d+'
    confidence_score FLOAT DEFAULT 0.0, -- auto-inferred vs admin-confirmed
    confirmed_by TEXT,                  -- admin user_id who confirmed (if applicable)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (organization_id, source_type, file_pattern)
);

CREATE INDEX IF NOT EXISTS idx_org_schemas_lookup
    ON organization_schemas (organization_id, source_type);
