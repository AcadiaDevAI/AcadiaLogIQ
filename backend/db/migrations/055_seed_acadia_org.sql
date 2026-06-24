-- Migration 055 — Phase 0 / Multi-tenant: seed existing Acadia + role migration.
--
-- HISTORY NOTE — this file replaces the original draft that INSERTed a
-- new Acadia row. We discovered the database already has an Acadia row
-- (id 76c36d23-b8c1-4b81-b121-bef5e1b10b3b, name 'Acadia Internal',
-- org_type 'acadia') created in April 2026 by a previous spike. We
-- ADAPT that row rather than insert a duplicate.
--
-- Five operations, all idempotent:
--
--   1. Normalize the existing Acadia row:
--        * Set name = 'Acadia Consultants' (was 'Acadia Internal')
--        * Set slug = 'acadia-consultants' (new column from mig 051)
--        * is_listed_publicly = TRUE
--        * force_org_picker = FALSE
--        * theme_json / settings_json defaulted by migration 051
--
--   2. Translate existing membership roles to the canonical vocabulary:
--        acadia_admin   → admin
--        customer_admin → admin
--        user           → member
--      The 4 existing rows are all `acadia_admin`, becoming `admin`.
--
--   3. Backfill memberships for any active users WITHOUT a current
--      membership. The investigation showed 5 active users but only 4
--      memberships — backfill the 5th as `member` (not admin, since we
--      don't know who they are without explicit promotion).
--
--   4. Set users.last_active_org_id to the existing Acadia UUID for
--      every active user (so next-login restores to Acadia).
--
--   5. Re-add the tightened CHECK constraint on memberships.role,
--      now allowing only the canonical values { admin, member }.
--      (Was dropped in migration 052 to allow the data migration.)
--
-- The Acadia organization UUID is hardcoded here as the SURVIVING UUID
-- from the existing schema. Code references it via the
-- backend/tenancy/constants.py ACADIA_ORG_ID constant.

-- ── Step 1: Normalize the existing Acadia row ────────────────────
UPDATE organizations
   SET name                = 'Acadia Consultants',
       slug                = 'acadia-consultants',
       is_listed_publicly  = TRUE,
       force_org_picker    = FALSE,
       updated_at          = NOW()
 WHERE id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid;

-- ── Step 2: Translate role vocabulary on existing memberships ────
UPDATE memberships
   SET role = 'admin'
 WHERE role IN ('acadia_admin', 'customer_admin')
   AND revoked_at IS NULL;

UPDATE memberships
   SET role = 'member'
 WHERE role = 'user'
   AND revoked_at IS NULL;

-- ── Step 3: Backfill memberships for active users without one ────
-- Insert a default `member` row for every active user who isn't
-- already a member of Acadia. Existing admins remain admins (Step 2
-- already promoted them).
INSERT INTO memberships (
    user_id,
    organization_id,
    role,
    granted_at
)
SELECT
    u.id,
    '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid,
    'member',
    NOW()
FROM users u
WHERE u.is_active = TRUE
  AND NOT EXISTS (
    SELECT 1
      FROM memberships m
     WHERE m.user_id = u.id
       AND m.organization_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid
  )
ON CONFLICT (user_id, organization_id) DO NOTHING;

-- ── Step 4: Restore-on-login points to Acadia for existing users ─
UPDATE users
   SET last_active_org_id = '76c36d23-b8c1-4b81-b121-bef5e1b10b3b'::uuid,
       updated_at         = CURRENT_TIMESTAMP
 WHERE is_active = TRUE
   AND last_active_org_id IS NULL;

-- ── Step 5: Re-add tightened CHECK constraint on memberships.role ─
-- Drop again (idempotent — IF EXISTS catches the case where this
-- migration is re-run) then add the new constraint allowing only
-- the canonical { admin, member } vocabulary.
ALTER TABLE memberships
    DROP CONSTRAINT IF EXISTS chk_membership_role;

ALTER TABLE memberships
    ADD CONSTRAINT chk_membership_role
    CHECK (role IN ('admin', 'member'));
