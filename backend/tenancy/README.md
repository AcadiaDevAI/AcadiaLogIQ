# backend.tenancy — Multi-Tenant Foundation (Phase 0 + Phase 1)

This package introduces the multi-tenant data and HTTP layer for Acadia
LogIQ. Phase 0 is **purely additive** — Phase 1 adds an `organization_id`
column to every tenant table (migrations 056–060) plus Postgres Row
Level Security policies (migration 061). Phase 2 will flip the
`ENABLE_ORG_SCOPING` flag and make the business hot path consume
`RequestContext` for explicit query filtering + perf.

---

## 🛑 HARD PRE-FLIGHT CHECKLIST — read before onboarding a 2nd tenant 🛑

Phase 1 ships a **single-tenant fallback** in `backend/db/connection.py`
(constant `DEFAULT_ORG_ID_FOR_NO_CONTEXT`) that bypasses RLS for any
SQLAlchemy session whose `current_org_id_var` ContextVar is unset —
background workers, boot-time init, ad-hoc scripts. This is necessary
today so the BM25 init / glossary refresher / ingestion workers don't
silently break when RLS is enabled. **In a single-tenant DB this is
safe** because all rows belong to Acadia anyway.

**The moment a second tenant exists, this fallback becomes a leak vector.**
A forgetful background worker would cross the boundary undetected. Before
provisioning ANY second org (US Pharma, XYZ Manufacturing, anything),
complete this checklist in order:

1. In `backend/db/connection.py`, set `DEFAULT_ORG_ID_FOR_NO_CONTEXT = None`.
2. Boot the app. The startup log will switch from
   `single-tenant fallback active` to silent. Any background worker
   that previously touched tenant tables will now ERROR or return
   zero rows. Triage each failure:
   - **BM25 rebuild** (`backend/api.py` lifespan) — typically per-org
     in multi-tenant; loop over orgs and set `current_org_id_var`
     inside the loop, OR move BM25 to a single index keyed by `(org, doc)`.
   - **Glossary refresher loop** (`backend/api.py`) — same pattern.
   - **Ingestion job workers** (`backend/jobs/queue.py`) — the worker
     pulls a job row; the row already carries `organization_id`, so
     set the ContextVar from it before processing.
   - **Report job workers** (if any) — same as ingestion.
   - **Any new asyncio background task** — apply the same rule.
3. For workers that are LEGITIMATELY cross-org (admin dashboards, ops
   tooling), point them at the `logiq_admin` BYPASSRLS role via a
   separate `DATABASE_URL`. Don't use the var-set workaround for
   genuinely global queries.
4. Run the full smoke test (Tier-1 journey end-to-end, `/ask` against
   a known incident, document upload + ingest). Everything that
   previously returned data must still return data.
5. ONLY THEN insert the second org row into `organizations`.

The boot-time guardrail `warn_if_multi_tenant_and_fallback_active()`
in `connection.py` emits an ERROR log on every startup if it detects
`organizations.status='active' COUNT > 1` AND
`DEFAULT_ORG_ID_FOR_NO_CONTEXT` is still set. Treat that log line as
production-down severity — it means a cross-tenant leak is live.

---

If you're picking this up cold, read in this order:
1. `constants.py` — value definitions
2. `models.py` — what shapes the package returns
3. `repository.py` — DB access
4. `context.py` — how a request becomes a `RequestContext`
5. `clerk_sync.py` — how Clerk keeps us in sync
6. `routes.py` — HTTP surface

## What ships in Phase 0

**Tables (migrations 051–055):**
- `organizations` — tenant container, mirrored from Clerk
- `org_memberships` — who belongs to which org, in what role
- `users.last_active_org_id` — restore-on-login hint
- `org_access_requests` — pending join requests
- Acadia Consultants seed row + backfill all existing users into it

**Endpoints (mounted at root):**
- `GET /organizations` — landing-page feed (yours + others)
- `GET /organizations/me/active` — current active-org context
- `PATCH /users/me/active-org` — persist active-org choice
- `POST /organizations/{slug}/request-access` — join request
- `POST /webhooks/clerk` — webhook receiver

**Dependency:**
- `get_request_context` — FastAPI dependency that returns a frozen
  `RequestContext` for tenant-aware endpoints.

## What does NOT ship in Phase 0

- Org creation API (`POST /admin/organizations`) — Phase 5
- Branding upload / theme application — Phase 4
- Access-request approval flow — Phase 3
- `organization_id` columns on business tables — Phase 1
- `WHERE organization_id = :org` filters on queries — Phase 2

## Architectural principles

### Clerk is the source of truth

Our `organizations` and `org_memberships` tables MIRROR Clerk's data.
When Clerk and our DB disagree, **Clerk wins**. The webhook handler is
the one-way replication channel. The reconciliation helper
`sync_user_memberships_from_clerk` in `clerk_sync.py` is a belt-and-
suspenders layer for webhook delivery lag — call it from
`/auth/register-or-login` to guarantee a freshly-logged-in user's
membership view is correct.

### `RequestContext` is immutable

Frozen dataclass. No code path can mutate it mid-request. If you need
a different context (e.g. super-admin impersonating an org), construct
a new `RequestContext` for that scope and pass it explicitly.

### Failure-CLOSED for identity

When Clerk is enabled, missing/invalid JWT → 401. We never silently
treat a missing identity as anonymous. The only "anonymous" path is
when Clerk is disabled entirely (dev/testing without auth).

### Failure-OPEN for org scope (Phase 0 only)

`get_request_context` does NOT raise when there's no active org.
Some pages (the landing-page picker) need to render without an active
org. Endpoints that REQUIRE an org should call `ctx.require_org()`,
which raises 403.

In Phase 2 we'll add explicit guards on every business endpoint.

### Webhook security

`POST /webhooks/clerk` verifies the Svix signature on every request
BEFORE trusting any field in the body. If
`CLERK_WEBHOOK_SIGNING_SECRET` is not configured, the endpoint refuses
to process anything — failing closed is the only safe behavior.

The signature verification is vendored (no `svix` PyPI dependency) —
~30 lines in `clerk_sync.verify_webhook_signature`. Trade-off: we own
the verification correctness; we lose the package's future bug fixes.
Worth it for the dependency reduction.

## Settings additions

Add these to your `.env` / AWS Secrets Manager:

- `CLERK_WEBHOOK_SIGNING_SECRET` — from Clerk dashboard → Webhooks →
  Your endpoint → Signing Secret. Format: `whsec_...`. REQUIRED for
  webhook processing.
- `ENABLE_ORG_SCOPING` — defaults `false`. Stays false through Phase 0
  and Phase 1. Flipped to `true` in Phase 2.

## Wiring required outside this package

Two surgical edits to `backend/api.py`:

1. **Mount the router** (alongside existing routers around line 575):

   ```python
   try:
       from backend.tenancy.routes import router as _tenancy_router
       app.include_router(_tenancy_router)
       logger.info("[tenancy] router mounted")
   except Exception as exc:
       logger.warning("[tenancy] failed to mount router: %s", exc)
   ```

2. **Sync memberships on login** in `/auth/register-or-login` (around
   line 2161, after `upsert_user(...)`):

   ```python
   try:
       from backend.tenancy.clerk_sync import sync_user_memberships_from_clerk
       sync_user_memberships_from_clerk(user_id)
   except Exception as exc:
       logger.warning("[tenancy] post-login sync failed: %s", exc)
   ```

Both edits are FAIL-OPEN (login keeps working even if tenancy is
broken) and BACKWARDS-COMPATIBLE (zero impact on existing behavior).

## Clerk dashboard setup

Before deploying:

1. Enable Organizations on the Clerk dashboard.
2. Create the "Acadia Consultants" org with `slug=acadia-consultants`.
3. Note its `org_xxx` id and either:
   - Run a one-time SQL update to set `organizations.clerk_org_id` on
     the Acadia row, OR
   - Let the next webhook event populate it automatically.
4. Configure the JWT template to include:
   - `org_id`, `org_slug`, `org_role` (default for Clerk Orgs)
   - `public_metadata` (for `platform_role` claim)
5. Create the webhook endpoint pointed at
   `https://your-domain/webhooks/clerk` and enable these events:
   - `organization.created`, `organization.updated`, `organization.deleted`
   - `organizationMembership.created`, `organizationMembership.updated`,
     `organizationMembership.deleted`
6. Copy the signing secret into `CLERK_WEBHOOK_SIGNING_SECRET`.

## Testing locally without Clerk

When `CLERK_ENABLED=false`, `get_request_context` returns:
```python
RequestContext(
    user_id="anonymous",
    org_id=None,
    org_slug=None,
    org_role=None,
    platform_role="user",
)
```

That's enough for unit tests that exercise the repository or sync
functions without HTTP. Webhook signature verification can be
exercised by setting `CLERK_WEBHOOK_SIGNING_SECRET` and computing the
HMAC manually — see `tests/test_tenancy_webhook.py` (TODO).

## Phase 1 preview

Phase 1 adds `organization_id` columns to:
- `documents`, `document_chunks`, `document_versions`
- `chat_sessions`, `chat_messages`
- `tier1_sessions`, `tier1_journey_events`
- `semantic_answer_cache`, `answer_cache_exact`
- `report_jobs`, `reports`, `pattern_analytics_cache`
- Anything else carrying `owner_id` today

All columns added as NULL, backfilled to Acadia's id, then constrained
NOT NULL. No code change in Phase 1 — that's Phase 2.

## Common pitfalls

- **JWT lag.** Switching orgs in Clerk's UI re-issues the JWT, but the
  current page may still hold the OLD one. The frontend must invalidate
  its in-memory token after a switch event.
- **Webhook delivery is best-effort.** Don't rely on it alone for
  freshly-logged-in users — that's why
  `sync_user_memberships_from_clerk` exists.
- **Out-of-order events.** Membership can arrive before its org event.
  `_handle_membership_upsert` handles this by creating a placeholder
  org row.
- **Clerk's role format.** Roles arrive as `"org:admin"` /
  `"org:member"`. Strip the `"org:"` prefix before storing.
