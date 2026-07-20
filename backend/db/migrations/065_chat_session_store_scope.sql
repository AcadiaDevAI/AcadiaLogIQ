-- Migration 065 — Add per-chat-session STORE scope (US Pharma).
-- Companion to migration 042 (scope_incident_id). When the engineer
-- clicks "KB SOP" on the US Pharma journey, the Stage 4 handoff stamps
-- the intake Store ID onto the new chat session. The /ask retrieval path
-- reads this column and, when set, restricts retrieval to chunks whose
--   metadata_json->'Metadata'->>'store_id' = scope_store_id
-- so the whole KB conversation answers strictly from that one store's
-- indexed content (its JSON ticket corpus).
--
-- When NULL (the default — Acadia, per-bullet incident scope, regular
-- chat) the retrieval path is unchanged.
--
-- Only US Pharma sets this: it is the only org whose intake carries a
-- Store ID (OrgProfile.tier1_requires_store_id). Acadia rows stay NULL.
--
-- Safe to re-run. Single nullable TEXT column, no backfill required.
-- Existing chat_sessions rows continue to behave exactly as before.
-- See: backend/tier1_copilot/journey/stage4_search_kb_handoff.py
--      backend/retrieval/scoped_retrieval.py (retrieve_within_store())
--      backend/api.py (/ask store-scope branch)
--
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py wraps
-- each migration in engine.begin() and splits on ';' (matches 042/030).

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS scope_store_id TEXT;

COMMENT ON COLUMN chat_sessions.scope_store_id IS
    'Migration 065 — When non-NULL, restrict /ask retrieval for this '
    'chat session to chunks whose metadata_json->''Metadata''->>''store_id'' '
    'equals this Store ID. Set by the US Pharma Stage 4 "KB SOP" handoff. '
    'NULL preserves global / incident-scoped behavior.';
