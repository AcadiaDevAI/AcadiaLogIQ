-- Migration 057 — Phase 1 multi-tenant: organization_id on chat + Q&A caches.
--
-- Covers four tables: chat_sessions, chat_messages, semantic_answer_cache,
-- answer_cache_exact. Two of these (the caches) are HIGH-RISK leak
-- vectors — a US Pharma user issuing the same query that Acadia previously
-- cached would, without isolation, receive Acadia's cached answer. Phase 1
-- closes this at the row level; the app-layer key-composition fix is
-- tracked separately in Phase 2.
--
-- See migration 056 header for the column / NOT NULL / DEFAULT pattern
-- rationale and the idempotency contract.

-- ─────────────────────────────────────────────────────────────
-- chat_sessions
-- ─────────────────────────────────────────────────────────────
ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE chat_sessions
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE chat_sessions
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'chat_sessions_organization_id_fkey'
           AND table_name = 'chat_sessions'
    ) THEN
        ALTER TABLE chat_sessions
            ADD CONSTRAINT chat_sessions_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chat_sessions_org
    ON chat_sessions(organization_id);

-- ─────────────────────────────────────────────────────────────
-- chat_messages
-- ─────────────────────────────────────────────────────────────
ALTER TABLE chat_messages
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE chat_messages
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE chat_messages
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'chat_messages_organization_id_fkey'
           AND table_name = 'chat_messages'
    ) THEN
        ALTER TABLE chat_messages
            ADD CONSTRAINT chat_messages_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_chat_messages_org
    ON chat_messages(organization_id);
CREATE INDEX IF NOT EXISTS idx_chat_messages_org_session
    ON chat_messages(organization_id, session_id);

-- ─────────────────────────────────────────────────────────────
-- semantic_answer_cache  (HIGH-RISK leak vector)
--
-- Pre-Phase-1: a single global table keyed on (query_text + file_ids
-- fingerprint). A US Pharma user asking the same question Acadia
-- previously cached would, without an org filter, receive Acadia's
-- cached answer back. RLS in migration 061 closes this at the row
-- level. The app-layer fix to include organization_id in the
-- file_ids_fingerprint computation is Phase 2.
-- ─────────────────────────────────────────────────────────────
ALTER TABLE semantic_answer_cache
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE semantic_answer_cache
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE semantic_answer_cache
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'semantic_answer_cache_organization_id_fkey'
           AND table_name = 'semantic_answer_cache'
    ) THEN
        ALTER TABLE semantic_answer_cache
            ADD CONSTRAINT semantic_answer_cache_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_semantic_cache_org
    ON semantic_answer_cache(organization_id);

-- ─────────────────────────────────────────────────────────────
-- answer_cache_exact  (HIGH-RISK leak vector — same shape as above)
-- ─────────────────────────────────────────────────────────────
ALTER TABLE answer_cache_exact
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE answer_cache_exact
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE answer_cache_exact
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'answer_cache_exact_organization_id_fkey'
           AND table_name = 'answer_cache_exact'
    ) THEN
        ALTER TABLE answer_cache_exact
            ADD CONSTRAINT answer_cache_exact_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_answer_cache_exact_org
    ON answer_cache_exact(organization_id);
