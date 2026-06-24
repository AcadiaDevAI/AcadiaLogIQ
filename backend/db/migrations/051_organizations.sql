-- Migration 051 — Phase 0 / Multi-tenant foundation: EXTEND organizations.
--
-- HISTORY NOTE — this file replaces an earlier draft of migration 051
-- that attempted to CREATE TABLE organizations from scratch. We
-- discovered the database ALREADY has an `organizations` table created
-- in April 2026 by a previous multi-tenant spike, plus four memberships
-- and FK constraints into `chat_sessions`, `documents`, `tier1_sessions`,
-- and `intake_extractions`. The existing table is the survivor; we
-- ADAPT it rather than replace it.
--
-- Existing schema (untouched by this migration):
--     id              UUID         PK, default gen_random_uuid()
--     name            TEXT         NOT NULL, UNIQUE
--     org_type        TEXT         NOT NULL, CHECK in ('acadia', 'customer')
--     plan            TEXT         NULL
--     metadata        JSONB        NOT NULL, default '{}'
--     created_at      TIMESTAMPTZ  NOT NULL, default now()
--     updated_at      TIMESTAMPTZ  NOT NULL, default now()
--     deactivated_at  TIMESTAMPTZ  NULL
--
-- Columns this migration ADDS:
--     slug                TEXT         UNIQUE, URL-safe identifier
--     clerk_org_id        TEXT         UNIQUE, mirror of Clerk's org id
--     logo_url            TEXT         per-org branding image URL
--     theme_json          JSONB        per-org branding theme
--     settings_json       JSONB        per-org config (model prefs, etc.)
--     is_listed_publicly  BOOLEAN      landing-page discoverability
--     force_org_picker    BOOLEAN      compliance flag for explicit login
--
-- Semantics we adopt from the existing table:
--     * "Active" means `deactivated_at IS NULL`. We do NOT add a parallel
--       `is_active` column — adopting the existing convention keeps the
--       indexes (idx_organizations_active) useful and avoids drift.
--     * `name` remains the human-readable display name; `slug` is the
--       URL identifier (frontend never displays slug).
--
-- Backward compatibility:
--     * All new columns are NULL or have safe defaults. Existing app
--       paths reading `organizations` continue to work unchanged.
--     * Existing FKs on chat_sessions / documents / tier1_sessions /
--       intake_extractions remain pointing at organizations(id) — no
--       cascade risk from this migration.
--
-- See: backend/tenancy/repository.py
--      backend/tenancy/README.md
--
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py wraps
-- each migration in engine.begin() and splits on ';'.

ALTER TABLE organizations
    ADD COLUMN IF NOT EXISTS slug                TEXT         NULL,
    ADD COLUMN IF NOT EXISTS clerk_org_id        TEXT         NULL,
    ADD COLUMN IF NOT EXISTS logo_url            TEXT         NULL,
    ADD COLUMN IF NOT EXISTS theme_json          JSONB        NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS settings_json       JSONB        NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS is_listed_publicly  BOOLEAN      NOT NULL DEFAULT TRUE,
    ADD COLUMN IF NOT EXISTS force_org_picker    BOOLEAN      NOT NULL DEFAULT FALSE;

-- Slug must be unique once populated. We allow NULL during the gap
-- between this migration and migration 055 (which backfills the
-- existing Acadia row's slug).
CREATE UNIQUE INDEX IF NOT EXISTS uq_organizations_slug
    ON organizations (slug)
    WHERE slug IS NOT NULL;

CREATE UNIQUE INDEX IF NOT EXISTS uq_organizations_clerk_org_id
    ON organizations (clerk_org_id)
    WHERE clerk_org_id IS NOT NULL;

-- Landing-page discovery query: "list every active publicly-listable org."
CREATE INDEX IF NOT EXISTS idx_organizations_listable
    ON organizations (is_listed_publicly)
    WHERE deactivated_at IS NULL AND is_listed_publicly = TRUE;

COMMENT ON COLUMN organizations.slug IS
    'URL-safe identifier (lowercase, hyphenated). Used in URLs and as '
    'the S3 prefix for per-org uploads. Backfilled for existing rows '
    'by migration 055.';

COMMENT ON COLUMN organizations.clerk_org_id IS
    'Clerk''s external org identifier (org_xxx). Sync target for the '
    'webhook handler. Nullable to allow seeding the row before linking '
    'to Clerk.';

COMMENT ON COLUMN organizations.theme_json IS
    'Per-org branding: {logo_url, primary_color, accent_color, ...}. '
    'Read by the frontend to swap visual identity on org switch.';

COMMENT ON COLUMN organizations.settings_json IS
    'Per-org configuration: default doc_kinds, model preferences, '
    'validation thresholds, retention policy. Evolves freely without '
    'schema migrations.';

COMMENT ON COLUMN organizations.is_listed_publicly IS
    'When TRUE, the org appears in the landing-page discovery list to '
    'non-members. When FALSE, only members see the org exists.';

COMMENT ON COLUMN organizations.force_org_picker IS
    'Compliance flag — when TRUE, members must explicitly pick this org '
    'on every login (no last-active-org auto-restore).';
