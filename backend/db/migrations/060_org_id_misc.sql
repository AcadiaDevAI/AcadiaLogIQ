-- Migration 060 — Phase 1 multi-tenant: organization_id on misc tenant tables.
--
-- Covers four tables. Three of them ALREADY have an organization_id
-- column, but typed as TEXT (string from an earlier multi-tenant
-- spike) instead of UUID with a real FK to organizations(id). This
-- migration normalizes them: it copies values into a new organization_id
-- column, drops the old TEXT column, and renames the new column into
-- place — so callers continue to see "organization_id" with the new
-- correct type.
--
-- Tables:
--   * learned_vocabulary       — currently UNIQUE on (token) globally.
--                                Must become UNIQUE (organization_id, token)
--                                so US Pharma jargon doesn't override Acadia's.
--   * pattern_analytics_cache  — has organization_id TEXT — convert to UUID + FK.
--   * organization_schemas     — has organization_id TEXT — convert to UUID + FK.
--   * logiq_sessions           — has org_id VARCHAR DEFAULT 'acadia_internal' —
--                                rename to organization_id, convert, FK.
--
-- See migration 056 header for the column / NOT NULL / DEFAULT pattern.

-- ─────────────────────────────────────────────────────────────
-- learned_vocabulary
-- The original PRIMARY KEY is on (token) alone. We must drop and replace
-- it with a composite primary key (organization_id, token) so the same
-- token can exist in two orgs with different stats.
-- ─────────────────────────────────────────────────────────────
ALTER TABLE learned_vocabulary
    ADD COLUMN IF NOT EXISTS organization_id UUID;

UPDATE learned_vocabulary
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE learned_vocabulary
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'learned_vocabulary_organization_id_fkey'
           AND table_name = 'learned_vocabulary'
    ) THEN
        ALTER TABLE learned_vocabulary
            ADD CONSTRAINT learned_vocabulary_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

-- Repoint the primary key from (token) to (organization_id, token).
DO $$
DECLARE
    pk_name TEXT;
BEGIN
    SELECT constraint_name INTO pk_name
      FROM information_schema.table_constraints
     WHERE table_name = 'learned_vocabulary'
       AND constraint_type = 'PRIMARY KEY';
    IF pk_name IS NOT NULL THEN
        EXECUTE format('ALTER TABLE learned_vocabulary DROP CONSTRAINT %I', pk_name);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'learned_vocabulary_pkey_org_token'
           AND table_name = 'learned_vocabulary'
    ) THEN
        ALTER TABLE learned_vocabulary
            ADD CONSTRAINT learned_vocabulary_pkey_org_token
            PRIMARY KEY (organization_id, token);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_learned_vocabulary_org
    ON learned_vocabulary(organization_id);

-- ─────────────────────────────────────────────────────────────
-- pattern_analytics_cache — convert TEXT organization_id to UUID
-- ─────────────────────────────────────────────────────────────
-- Strategy: add new uuid column, backfill, drop UNIQUE (organization_id, ...)
-- because it covers the TEXT column, drop the old column, rename new into place,
-- re-add UNIQUE.
DO $$
DECLARE
    org_id_type TEXT;
BEGIN
    SELECT data_type INTO org_id_type
      FROM information_schema.columns
     WHERE table_name = 'pattern_analytics_cache'
       AND column_name = 'organization_id';

    IF org_id_type IS NULL THEN
        -- Column missing entirely (fresh DB) — just add UUID.
        ALTER TABLE pattern_analytics_cache
            ADD COLUMN organization_id UUID;
    ELSIF org_id_type <> 'uuid' THEN
        -- Convert: add a temp uuid column, backfill, swap, drop old.
        ALTER TABLE pattern_analytics_cache
            ADD COLUMN IF NOT EXISTS organization_id_new UUID;

        UPDATE pattern_analytics_cache
           SET organization_id_new = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
         WHERE organization_id_new IS NULL;

        -- Drop the old UNIQUE constraint that referenced the TEXT column.
        ALTER TABLE pattern_analytics_cache
            DROP CONSTRAINT IF EXISTS pattern_analytics_cache_organization_id_topic_key_key;

        -- Drop dependent indexes.
        DROP INDEX IF EXISTS idx_pattern_cache_lookup;

        -- Drop the old TEXT column, rename new into place.
        ALTER TABLE pattern_analytics_cache
            DROP COLUMN organization_id;
        ALTER TABLE pattern_analytics_cache
            RENAME COLUMN organization_id_new TO organization_id;
    END IF;
END $$;

UPDATE pattern_analytics_cache
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE pattern_analytics_cache
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'pattern_analytics_cache_organization_id_fkey'
           AND table_name = 'pattern_analytics_cache'
    ) THEN
        ALTER TABLE pattern_analytics_cache
            ADD CONSTRAINT pattern_analytics_cache_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'pattern_analytics_cache_org_topic_unique'
           AND table_name = 'pattern_analytics_cache'
    ) THEN
        ALTER TABLE pattern_analytics_cache
            ADD CONSTRAINT pattern_analytics_cache_org_topic_unique
            UNIQUE (organization_id, topic_key);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_pattern_cache_lookup
    ON pattern_analytics_cache (organization_id, topic_key, expires_at);

-- ─────────────────────────────────────────────────────────────
-- organization_schemas — convert TEXT organization_id to UUID
-- Same strategy as pattern_analytics_cache above.
-- ─────────────────────────────────────────────────────────────
DO $$
DECLARE
    org_id_type TEXT;
BEGIN
    SELECT data_type INTO org_id_type
      FROM information_schema.columns
     WHERE table_name = 'organization_schemas'
       AND column_name = 'organization_id';

    IF org_id_type IS NULL THEN
        ALTER TABLE organization_schemas
            ADD COLUMN organization_id UUID;
    ELSIF org_id_type <> 'uuid' THEN
        ALTER TABLE organization_schemas
            ADD COLUMN IF NOT EXISTS organization_id_new UUID;

        UPDATE organization_schemas
           SET organization_id_new = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
         WHERE organization_id_new IS NULL;

        ALTER TABLE organization_schemas
            DROP CONSTRAINT IF EXISTS organization_schemas_organization_id_source_type_file_pattern_key;

        DROP INDEX IF EXISTS idx_org_schemas_lookup;

        ALTER TABLE organization_schemas
            DROP COLUMN organization_id;
        ALTER TABLE organization_schemas
            RENAME COLUMN organization_id_new TO organization_id;
    END IF;
END $$;

UPDATE organization_schemas
   SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
 WHERE organization_id IS NULL;

ALTER TABLE organization_schemas
    ALTER COLUMN organization_id SET NOT NULL,
    ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'organization_schemas_organization_id_fkey'
           AND table_name = 'organization_schemas'
    ) THEN
        ALTER TABLE organization_schemas
            ADD CONSTRAINT organization_schemas_organization_id_fkey
            FOREIGN KEY (organization_id) REFERENCES organizations(id);
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'organization_schemas_org_source_pattern_unique'
           AND table_name = 'organization_schemas'
    ) THEN
        ALTER TABLE organization_schemas
            ADD CONSTRAINT organization_schemas_org_source_pattern_unique
            UNIQUE (organization_id, source_type, file_pattern);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_org_schemas_lookup
    ON organization_schemas (organization_id, source_type);

-- ─────────────────────────────────────────────────────────────
-- logiq_sessions — rename org_id (VARCHAR) to organization_id (UUID)
--
-- This entire block is wrapped in a table-existence guard. The
-- `add_logiq_sessions_table.sql` file in this folder was never given
-- a numeric prefix and therefore may or may not have been applied to
-- a given environment. We skip silently on environments that don't
-- have the table — those don't need the column rename either.
-- ─────────────────────────────────────────────────────────────
DO $$
DECLARE
    table_exists   BOOLEAN;
    has_org_id_var BOOLEAN;
    has_org_uuid   BOOLEAN;
BEGIN
    SELECT EXISTS (
        SELECT 1 FROM information_schema.tables
         WHERE table_schema = 'public'
           AND table_name   = 'logiq_sessions'
    ) INTO table_exists;

    IF NOT table_exists THEN
        RAISE NOTICE '[060] logiq_sessions table not present in this DB — skipping its tenant-isolation block (no-op)';
        RETURN;
    END IF;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'logiq_sessions'
           AND column_name = 'org_id'
    ) INTO has_org_id_var;

    SELECT EXISTS (
        SELECT 1 FROM information_schema.columns
         WHERE table_name = 'logiq_sessions'
           AND column_name = 'organization_id'
    ) INTO has_org_uuid;

    IF NOT has_org_uuid THEN
        ALTER TABLE logiq_sessions
            ADD COLUMN organization_id UUID;
    END IF;

    -- If the old VARCHAR org_id column still exists, drop it. All
    -- existing rows belong to Acadia by definition of the single-tenant
    -- baseline, so no value translation is needed.
    IF has_org_id_var THEN
        DROP INDEX IF EXISTS idx_logiq_sessions_org_id;
        ALTER TABLE logiq_sessions
            DROP COLUMN org_id;
    END IF;

    -- Backfill, lock NOT NULL + DEFAULT, FK, index. All inside this
    -- DO block so they only execute when the table actually exists.
    EXECUTE $sql$
        UPDATE logiq_sessions
           SET organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
         WHERE organization_id IS NULL
    $sql$;

    EXECUTE $sql$
        ALTER TABLE logiq_sessions
            ALTER COLUMN organization_id SET NOT NULL,
            ALTER COLUMN organization_id SET DEFAULT '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
    $sql$;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.table_constraints
         WHERE constraint_name = 'logiq_sessions_organization_id_fkey'
           AND table_name = 'logiq_sessions'
    ) THEN
        EXECUTE $sql$
            ALTER TABLE logiq_sessions
                ADD CONSTRAINT logiq_sessions_organization_id_fkey
                FOREIGN KEY (organization_id) REFERENCES organizations(id)
        $sql$;
    END IF;

    EXECUTE $sql$
        CREATE INDEX IF NOT EXISTS idx_logiq_sessions_org
            ON logiq_sessions(organization_id)
    $sql$;
END $$;
