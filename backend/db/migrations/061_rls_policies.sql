-- Migration 061 — Phase 1 multi-tenant: Row Level Security policies.
--
-- This is the actual data-isolation guarantee. Migrations 056-060 added
-- the organization_id column to every tenant table; this migration
-- attaches a Postgres policy that physically restricts SELECT/INSERT/
-- UPDATE/DELETE to rows where organization_id matches the current
-- request's session variable `app.current_org`.
--
-- How it works at runtime:
--   1. FastAPI middleware (Phase 1 step "SET LOCAL") issues
--        SET LOCAL app.current_org = '<uuid>';
--      at the start of every request, taken from request_context.
--   2. Every query the app runs against a tenant table is silently
--      filtered by Postgres — if app.current_org doesn't match
--      organization_id, the row is invisible (not 403 — actually
--      absent).
--   3. Migration scripts + ops use a separate role (logiq_admin) with
--      BYPASSRLS so they can still see all rows.
--
-- ENABLE vs FORCE:
--   * ENABLE turns RLS on for non-owners (the regular app role).
--   * FORCE additionally applies RLS to the table OWNER. Without FORCE,
--     a superuser-owned table would be readable without policy checks
--     by the owner — defeating the security model. We FORCE everywhere.
--
-- Defense in depth: this is the hard guarantee. Even if a developer
-- forgets `WHERE organization_id = :ctx_org` in a query, Postgres
-- physically refuses to leak rows.
--
-- Trade-offs we're accepting:
--   * Any code path that genuinely needs cross-org reads (admin
--     dashboards, ops tooling) must use the BYPASSRLS role, OR call
--     `SET LOCAL app.current_org = '<other-uuid>'` with explicit
--     authorization. There are currently no such paths.
--   * `current_setting('app.current_org', true)` returns NULL if the
--     session hasn't set it. Policies that compare NULL = anything
--     return false, so a request that forgets to set the var gets
--     ZERO rows back. This is FAIL-CLOSED — the right default.
--
-- Idempotency:
--   ENABLE/FORCE ROW LEVEL SECURITY is idempotent.
--   CREATE POLICY is NOT — we DROP POLICY IF EXISTS first.

-- ─────────────────────────────────────────────────────────────
-- Helper: a single DO block emitting one ENABLE + FORCE + policy
-- per tenant table. Repeating raw DDL would be 22*4=88 lines.
-- ─────────────────────────────────────────────────────────────
DO $$
DECLARE
    tbl TEXT;
    tenant_tables TEXT[] := ARRAY[
        -- Group A — RAG corpus (migration 056)
        'documents',
        'document_versions',
        'document_metadata',
        'chunks',
        'embeddings',
        'ingestion_jobs',
        -- Group B — Chat & Q&A (migration 057)
        'chat_sessions',
        'chat_messages',
        'semantic_answer_cache',
        'answer_cache_exact',
        -- Group C — Tier 1 Copilot (migration 058)
        'tier1_sessions',
        'tier1_journey_events',
        'tier1_feedback',
        'tier1_answer_cache',
        'intake_extractions',
        -- Group D — Reports (migration 059)
        'report_cache',
        'report_feedback',
        'report_jobs',
        -- Group E — Misc (migration 060)
        'learned_vocabulary',
        'pattern_analytics_cache',
        'organization_schemas',
        'logiq_sessions'
    ];
BEGIN
    FOREACH tbl IN ARRAY tenant_tables LOOP
        -- Skip tables that don't exist in this environment. The
        -- `logiq_sessions` table in particular comes from an
        -- unnumbered migration file (add_logiq_sessions_table.sql)
        -- that may or may not have been applied. We don't want a
        -- missing-table to abort the whole RLS rollout.
        IF NOT EXISTS (
            SELECT 1 FROM information_schema.tables
             WHERE table_schema = 'public'
               AND table_name   = tbl
        ) THEN
            RAISE NOTICE '[061] table % not present — skipping RLS for it', tbl;
            CONTINUE;
        END IF;

        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', tbl);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', tbl);

        -- DROP-then-CREATE so re-running this migration is safe.
        EXECUTE format('DROP POLICY IF EXISTS org_isolation ON %I', tbl);
        EXECUTE format(
            'CREATE POLICY org_isolation ON %I
                USING (organization_id = current_setting(''app.current_org'', true)::uuid)
                WITH CHECK (organization_id = current_setting(''app.current_org'', true)::uuid)',
            tbl
        );
    END LOOP;
END $$;

-- ─────────────────────────────────────────────────────────────
-- BYPASSRLS role for migrations and ops scripts.
--
-- Why a separate role: the application connection pool runs as a
-- regular user. RLS treats that user as a tenant — exactly what we
-- want. But the migration runner, the report worker, ad-hoc psql
-- sessions from ops, etc. need to read across orgs. We model that
-- by attaching BYPASSRLS to a dedicated role.
--
-- Wiring (done outside this migration):
--   * The app's DATABASE_URL points at the regular role (no BYPASSRLS).
--   * The ops/admin DATABASE_URL points at logiq_admin.
--   * Migrations run with the admin role.
--
-- Idempotent — only creates the role if it doesn't already exist.
-- ─────────────────────────────────────────────────────────────
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_roles WHERE rolname = 'logiq_admin'
    ) THEN
        -- NOLOGIN by default; ops grants login separately when handing
        -- the role to a human or a scheduled job. Password is also set
        -- out-of-band (we don't bake credentials into a migration).
        CREATE ROLE logiq_admin BYPASSRLS NOLOGIN;
    ELSE
        -- Defensive: an existing role with the same name might predate
        -- the BYPASSRLS requirement.
        ALTER ROLE logiq_admin BYPASSRLS;
    END IF;
END $$;
