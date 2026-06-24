-- Migration 052 — Phase 0 / Multi-tenant foundation: ADAPT memberships.
--
-- HISTORY NOTE — this file replaces an earlier draft of migration 052
-- that attempted to CREATE TABLE org_memberships from scratch. The
-- database already has a `memberships` table (created in April 2026
-- alongside the original organizations spike). We adapt the existing
-- table rather than create a parallel one.
--
-- Existing schema (untouched by this migration):
--     id                  UUID         PK
--     user_id             UUID         NOT NULL, FK → users(id) CASCADE
--     organization_id     UUID         NOT NULL, FK → organizations(id) CASCADE
--     role                TEXT         NOT NULL,
--                                       CHECK in ('acadia_admin',
--                                                 'customer_admin', 'user')
--     granted_by_user_id  UUID         NULL, FK → users(id)
--     granted_at          TIMESTAMPTZ  NOT NULL
--     revoked_at          TIMESTAMPTZ  NULL    ← "active" means this IS NULL
--     revoked_by_user_id  UUID         NULL, FK → users(id)
--     metadata            JSONB        NOT NULL, default '{}'
--
--     UNIQUE (user_id, organization_id)
--     Indexes on org+active, role+active, user+active
--
-- What this migration does:
--
--   1. DROPs the existing CHECK constraint `chk_membership_role`.
--      The existing allowlist is `acadia_admin | customer_admin | user`,
--      which we're replacing with `admin | member` for cleaner vocabulary
--      across the rest of the multi-tenant code. The data migration
--      (UPDATE acadia_admin → admin, customer_admin → admin, user →
--      member) happens in migration 055, so we drop the constraint here
--      to allow the values to coexist during the transition. Migration
--      055 re-adds a tightened constraint afterwards.
--
-- What this migration does NOT do:
--
--   * Does NOT rename `memberships` → `org_memberships`. The existing
--     name stays; my code maps to it. Renaming would require updating
--     all FK names and dependent indexes for no real benefit.
--
--   * Does NOT change the `user_id` join shape. memberships.user_id is
--     a UUID referencing users(id). My Phase 0 Python code adapts to
--     this rather than the other way around.
--
--   * Does NOT touch the `revoked_at` semantics. "Active" means
--     `revoked_at IS NULL`. We adopt that convention everywhere.
--
-- Backward compatibility:
--   * All existing data is preserved.
--   * No existing code references this table outside the abandoned
--     spike, so no app behavior change.
--
-- Note: BEGIN/COMMIT intentionally omitted (see migration 051 note).

ALTER TABLE memberships
    DROP CONSTRAINT IF EXISTS chk_membership_role;

COMMENT ON TABLE memberships IS
    'Multi-tenant — who can access which org and in what role. Created '
    'April 2026 by a multi-tenant spike; extended in Phase 0 to align '
    'role vocabulary and to serve the production tenancy layer. '
    '"Active" membership means revoked_at IS NULL.';

COMMENT ON COLUMN memberships.role IS
    'Org-level role. Values after migration 055: admin, member. '
    'CHECK constraint is tightened in migration 055 after data migration.';
