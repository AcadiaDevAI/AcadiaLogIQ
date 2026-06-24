-- Migration 058 — Phase 1 multi-tenant: organization_id on Tier-1 Copilot + intake.
--
-- Covers tier1_sessions, tier1_journey_events, tier1_feedback,
-- tier1_answer_cache, intake_extractions.
--
-- tier1_answer_cache is keyed on a SHA-1 of the alert signature
-- (priority + target service + symptom). Two orgs running the same
-- alert signature would, without an org filter, hit each other's
-- cached resolution path. Phase 1 closes this at the row level;
-- the app-layer fix to include organization_id in the signature
-- hash is Phase 2.
--
-- See migration 056 header for the column / NOT NULL / DEFAULT pattern
-- rationale and the idempotency contract.

-- ─────────────────────────────────────────────────────────────
-- tier1_sessions
-- ─────────────────────────────────────────────────────────────
ALTER TABLE tier1_sessions
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE tier1_sessions
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE tier1_sessions
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'tier1_sessions_organization_id_fkey'
           AND table_name = 'tier1_sessions'
    ) THEN
        ALTER TABLE tier1_sessions
            ADD CONSTRAINT tier1_sessions_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_tier1_sessions_org
    ON tier1_sessions(organization_id);

-- ─────────────────────────────────────────────────────────────
-- tier1_journey_events
-- ─────────────────────────────────────────────────────────────
ALTER TABLE tier1_journey_events
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE tier1_journey_events
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE tier1_journey_events
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'tier1_journey_events_organization_id_fkey'
           AND table_name = 'tier1_journey_events'
    ) THEN
        ALTER TABLE tier1_journey_events
            ADD CONSTRAINT tier1_journey_events_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_tier1_journey_events_org
    ON tier1_journey_events(organization_id);

-- ─────────────────────────────────────────────────────────────
-- tier1_feedback
-- ─────────────────────────────────────────────────────────────
ALTER TABLE tier1_feedback
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE tier1_feedback
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE tier1_feedback
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'tier1_feedback_organization_id_fkey'
           AND table_name = 'tier1_feedback'
    ) THEN
        ALTER TABLE tier1_feedback
            ADD CONSTRAINT tier1_feedback_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_tier1_feedback_org
    ON tier1_feedback(organization_id);

-- ─────────────────────────────────────────────────────────────
-- tier1_answer_cache  (HIGH-RISK leak vector — see header)
-- ─────────────────────────────────────────────────────────────
ALTER TABLE tier1_answer_cache
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE tier1_answer_cache
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE tier1_answer_cache
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'tier1_answer_cache_organization_id_fkey'
           AND table_name = 'tier1_answer_cache'
    ) THEN
        ALTER TABLE tier1_answer_cache
            ADD CONSTRAINT tier1_answer_cache_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_tier1_answer_cache_org
    ON tier1_answer_cache(organization_id);

-- ─────────────────────────────────────────────────────────────
-- intake_extractions
-- ─────────────────────────────────────────────────────────────
ALTER TABLE intake_extractions
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE intake_extractions
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE intake_extractions
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'intake_extractions_organization_id_fkey'
           AND table_name = 'intake_extractions'
    ) THEN
        ALTER TABLE intake_extractions
            ADD CONSTRAINT intake_extractions_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_intake_extractions_org
    ON intake_extractions(organization_id);
