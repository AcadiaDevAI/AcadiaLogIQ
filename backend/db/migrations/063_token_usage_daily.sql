-- Migration 063 — per-org Bedrock token-usage rollup.
--
-- Records exact token consumption from EVERY Bedrock call (chat/Mistral,
-- embeddings/Titan, all Haiku/Sonnet paths, gap analysis, RCA, escalation)
-- aggregated per organization, per day, per feature, per model. The app
-- buffers usage in-process and flushes increments into this table at
-- request/job boundaries, so row count stays bounded
-- (orgs x features x models x days) and there is no per-call row churn.
--
-- Tenancy: organization_id defaults to the session GUC (same pattern as
-- migration 062) and the table is RLS-protected with the same org_isolation
-- policy migration 061 applies to every other tenant table. Because 061's
-- policy creation is a fixed DO block over a known table list, this NEW table
-- must declare its own policy here.
--
-- Idempotent: IF NOT EXISTS on the table/index; the policy is dropped-then-
-- created so a re-run is safe. Standalone statements (NO dollar-quoted DO
-- block) so the semicolon-splitting migrate.py runner applies them cleanly.
--
-- Note: BEGIN/COMMIT omitted — migrate.py wraps each migration in
-- engine.begin().

CREATE TABLE IF NOT EXISTS token_usage_daily (
    id              BIGSERIAL PRIMARY KEY,
    organization_id UUID NOT NULL
        DEFAULT current_setting('app.current_org', true)::uuid
        REFERENCES organizations(id),
    usage_date      DATE NOT NULL DEFAULT (now() AT TIME ZONE 'UTC')::date,
    feature         TEXT NOT NULL,
    model_id        TEXT NOT NULL,
    call_count      BIGINT NOT NULL DEFAULT 0,
    input_tokens    BIGINT NOT NULL DEFAULT 0,
    output_tokens   BIGINT NOT NULL DEFAULT 0,
    cost_usd        NUMERIC(14, 6) NOT NULL DEFAULT 0
);

-- Upsert key: one row per (org, day, feature, model).
CREATE UNIQUE INDEX IF NOT EXISTS uq_token_usage_daily_key
    ON token_usage_daily (organization_id, usage_date, feature, model_id);

-- Dashboard read path: filter by org + date range.
CREATE INDEX IF NOT EXISTS idx_token_usage_daily_org_date
    ON token_usage_daily (organization_id, usage_date);

-- Row-level security — identical org_isolation policy to migration 061.
ALTER TABLE token_usage_daily ENABLE ROW LEVEL SECURITY;
ALTER TABLE token_usage_daily FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS org_isolation ON token_usage_daily;
CREATE POLICY org_isolation ON token_usage_daily
    USING (organization_id = current_setting('app.current_org', true)::uuid)
    WITH CHECK (organization_id = current_setting('app.current_org', true)::uuid);
