-- Migration 054 — Phase 0 / Multi-tenant: org_access_requests table.
--
-- When a user sees an org they don't belong to on the landing page and
-- clicks "Request Access," a row lands here. Org admins review pending
-- rows and approve/deny. Approval creates a corresponding row in
-- `memberships` (and in Clerk via the backend API).
--
-- HISTORY NOTE — this file replaces an earlier draft that used
-- `user_clerk_id TEXT`. We've now adopted the existing pattern of
-- joining through `users.id (UUID)`, so this version uses `user_id UUID`
-- with a FK to `users(id)` for consistency with the rest of the layer.
--
-- Design decisions worth knowing:
--
--   * `(user_id, organization_id)` is NOT a unique constraint — a user
--     can re-request after a denial (with new justification). We use a
--     PARTIAL unique index that prevents only DUPLICATE PENDING
--     requests for the same user+org pair, which is the actual UX rule.
--
--   * `status` is a string-enum with a CHECK allowlist. Values:
--     pending, approved, denied, withdrawn.
--
--   * `justification` is TEXT, optional. Some orgs require a reason
--     before approving; others approve based on email domain alone.
--
--   * `responded_by_user_id` references users(id) of the admin who acted
--     on the request. NULL until someone responds.
--
-- Backward compatibility:
--   * New table — nothing existing references it.
--   * Endpoints in backend/tenancy/routes.py work the moment this
--     migration is applied.

CREATE TABLE IF NOT EXISTS org_access_requests (
    id                     UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                UUID         NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    organization_id        UUID         NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    status                 TEXT         NOT NULL DEFAULT 'pending',
    justification          TEXT         NULL,
    responded_by_user_id   UUID         NULL REFERENCES users(id),
    response_note          TEXT         NULL,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    responded_at           TIMESTAMPTZ  NULL,
    CONSTRAINT chk_org_access_requests_status
        CHECK (status IN ('pending', 'approved', 'denied', 'withdrawn'))
);

-- Block duplicate PENDING requests for the same user+org pair. A user
-- can re-request after a denial / approval / withdrawal.
CREATE UNIQUE INDEX IF NOT EXISTS uq_org_access_requests_pending
    ON org_access_requests (user_id, organization_id)
    WHERE status = 'pending';

-- Admin UI query: "all pending requests for this org, oldest first."
CREATE INDEX IF NOT EXISTS idx_org_access_requests_org_pending
    ON org_access_requests (organization_id, created_at)
    WHERE status = 'pending';

-- User-side query: "did I already request this?"
CREATE INDEX IF NOT EXISTS idx_org_access_requests_user
    ON org_access_requests (user_id, created_at DESC);

COMMENT ON TABLE org_access_requests IS
    'Phase 0 multi-tenant — pending and historical requests from users '
    'to join orgs they''re not yet members of. Created when a user clicks '
    'a locked org tile on the landing page; processed by org admins via '
    'the admin UI.';

COMMENT ON COLUMN org_access_requests.status IS
    'Lifecycle: pending → approved | denied | withdrawn. Only `pending` '
    'rows are subject to the duplicate-prevention unique index.';
