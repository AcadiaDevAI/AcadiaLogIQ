-- Migration 062 — organization_id DEFAULT follows the session's active org.
--
-- Migrations 056-060 added organization_id to every tenant table with a
-- DEFAULT of the Acadia org UUID ('76c36d23-...') — correct for the
-- single-tenant baseline. In Phase 2 (multi-tenant + RLS, migration 061)
-- that default is WRONG: any INSERT that omits organization_id lands in
-- Acadia, and the RLS WITH CHECK policy
--     organization_id = current_setting('app.current_org')::uuid
-- then REJECTS the row whenever the session's org is not Acadia (a US
-- Pharma user creating a chat session, tier1 session, report, etc.) with
-- "new row violates row-level security policy for table ...".
--
-- Fix: repoint each column DEFAULT at the SAME session GUC the RLS policy
-- reads. The app stamps app.current_org on every transaction (after_begin
-- / engine-begin hooks in backend/db/connection.py, zero-UUID fallback
-- when no org is in scope), so the cast is always valid. After this every
-- insert into every tenant table is org-correct by default.
--
-- Safety:
--   * When the session org IS Acadia, the default still resolves to
--     Acadia — unchanged for the existing tenant.
--   * Explicit organization_id values always override the default.
--   * Migrations/ops run as BYPASSRLS and set organization_id explicitly.
--
-- Idempotent: ALTER COLUMN ... SET DEFAULT is naturally idempotent.
-- `ALTER TABLE IF EXISTS` skips tables not present in this environment
-- (e.g. logiq_sessions). Each statement is standalone (NO dollar-quoted
-- DO block) so the semicolon-splitting migrate.py runner applies them
-- correctly — one ALTER per statement.
--
-- Note: BEGIN/COMMIT omitted — migrate.py wraps each migration in
-- engine.begin().

-- Group A — RAG corpus (migration 056)
ALTER TABLE IF EXISTS documents          ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS document_versions  ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS document_metadata  ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS chunks             ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS embeddings         ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS ingestion_jobs     ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;

-- Group B — Chat & Q&A (migration 057)
ALTER TABLE IF EXISTS chat_sessions          ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS chat_messages          ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS semantic_answer_cache  ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS answer_cache_exact     ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;

-- Group C — Tier 1 Copilot (migration 058)
ALTER TABLE IF EXISTS tier1_sessions        ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS tier1_journey_events  ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS tier1_feedback        ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS tier1_answer_cache    ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS intake_extractions    ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;

-- Group D — Reports (migration 059)
ALTER TABLE IF EXISTS report_cache     ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS report_feedback  ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS report_jobs      ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;

-- Group E — Misc (migration 060)
ALTER TABLE IF EXISTS learned_vocabulary       ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS pattern_analytics_cache  ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS organization_schemas     ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
ALTER TABLE IF EXISTS logiq_sessions           ALTER COLUMN organization_id SET DEFAULT current_setting('app.current_org', true)::uuid;
