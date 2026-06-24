-- Migration 059 — Phase 1 multi-tenant: organization_id on report tables.
--
-- Covers report_cache, report_feedback, report_jobs.
--
-- CRITICAL fix to the UNIQUE constraint on report_cache:
--   Pre-Phase-1:  UNIQUE (report_kind, incident_number)
--   Post-Phase-1: UNIQUE (organization_id, report_kind, incident_number)
--
-- Without this change, US Pharma uploading an incident "INC-LAN-88902"
-- (a number Acadia already has) would collide on the unique constraint
-- and either (a) write its report on top of Acadia's cached row, or
-- (b) fail to insert and read Acadia's cached row back to a US Pharma
-- user. Either outcome is a data-isolation breach.
--
-- See migration 056 header for the column / NOT NULL / DEFAULT pattern
-- rationale and the idempotency contract.

-- ─────────────────────────────────────────────────────────────
-- report_cache
-- ─────────────────────────────────────────────────────────────
ALTER TABLE report_cache
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE report_cache
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE report_cache
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'report_cache_organization_id_fkey'
           AND table_name = 'report_cache'
    ) THEN
        ALTER TABLE report_cache
            ADD CONSTRAINT report_cache_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

-- Replace the old (kind, incident) UNIQUE with the org-scoped variant.
ALTER TABLE report_cache
    DROP CONSTRAINT IF EXISTS report_cache_unique;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'report_cache_org_kind_inc_unique'
           AND table_name = 'report_cache'
    ) THEN
        ALTER TABLE report_cache
            ADD CONSTRAINT report_cache_org_kind_inc_unique
            UNIQUE (organization_id, report_kind, incident_number);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_report_cache_org
    ON report_cache(organization_id);

-- ─────────────────────────────────────────────────────────────
-- report_feedback
-- ─────────────────────────────────────────────────────────────
ALTER TABLE report_feedback
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE report_feedback
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE report_feedback
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'report_feedback_organization_id_fkey'
           AND table_name = 'report_feedback'
    ) THEN
        ALTER TABLE report_feedback
            ADD CONSTRAINT report_feedback_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_report_feedback_org
    ON report_feedback(organization_id);

-- ─────────────────────────────────────────────────────────────
-- report_jobs
-- ─────────────────────────────────────────────────────────────
ALTER TABLE report_jobs
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE report_jobs
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE report_jobs
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'report_jobs_organization_id_fkey'
           AND table_name = 'report_jobs'
    ) THEN
        ALTER TABLE report_jobs
            ADD CONSTRAINT report_jobs_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

-- Also widen the open-job uniqueness so two orgs can have a pending
-- job on the same (kind, incident_number).
DROP INDEX IF EXISTS idx_report_jobs_open_unique;
CREATE UNIQUE INDEX IF NOT EXISTS idx_report_jobs_open_unique
    ON report_jobs (organization_id, kind, incident_number)
 WHERE status IN ('pending', 'running');

CREATE INDEX IF NOT EXISTS idx_report_jobs_org
    ON report_jobs(organization_id);
