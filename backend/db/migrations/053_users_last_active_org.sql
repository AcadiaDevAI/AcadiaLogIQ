-- Migration 053 — Phase 0 / Multi-tenant: users.last_active_org_id.
--
-- Stores each user's MOST RECENT active organization so we can restore
-- them to the right context on next login.
--
-- This migration is compatible with the existing `users` table shape:
--     users.id (uuid, PK), users.clerk_id (text, unique), etc.
-- We add one new column. Migration 055 backfills the existing Acadia
-- users to point at the existing Acadia row.
--
-- UX intent:
--   * First-ever login OR no last-active set → landing page picker.
--   * Returning login with last-active set → straight into that org.
--   * In-session switch → updates this column → restored next login.
--
-- Design decisions worth knowing:
--
--   * NULL is meaningful — it means "no preference / show the picker".
--     Backfill in migration 055 sets it for the 5 existing active users.
--
--   * FK with ON DELETE SET NULL — if an org is ever hard-deleted,
--     users who pointed at it lose their preference and see the picker
--     instead of erroring out.
--
-- Backward compatibility:
--   * Column is nullable and ignored by all existing code paths.
--   * No behavior change until the landing-page picker UX (Phase 3)
--     reads this column.

ALTER TABLE users
    ADD COLUMN IF NOT EXISTS last_active_org_id UUID NULL
    REFERENCES organizations(id) ON DELETE SET NULL;

COMMENT ON COLUMN users.last_active_org_id IS
    'Phase 0 multi-tenant — most recent org this user switched into. '
    'On next login the frontend restores the user to this org if it '
    'is still valid. NULL means show the picker on next login.';
