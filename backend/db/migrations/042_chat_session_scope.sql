-- Migration 042 — Add per-chat-session ticket scope.
-- Sprint 12.1 — Per-bullet "Ask in Chat" handoff carries a source
-- Incident_Number (the Stage 0 bullet's tagged source ticket). The
-- chat /ask retrieval path reads this column and, when set, filters
-- pgvector + BM25 + full-text + identifier-exact channels by
--   metadata_json->>'primary_id' = scope_incident_id
-- so the conversation answers strictly from that one ticket's chunks.
--
-- When NULL (the default — Search-in-KB path, regular chat openings)
-- the orchestrator path is unchanged: full hybrid retrieval across
-- the engineer's authorized corpus.
--
-- Safe to re-run. Single nullable TEXT column, no backfill required.
-- Existing chat_sessions rows continue to behave exactly as before.
-- See: backend/tier1_copilot/journey/stage4_search_kb_handoff.py
--      backend/retrieval/orchestrator.py (retrieve())
--
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py
-- wraps each migration in engine.begin() and splits on ';', so
-- explicit transaction control would double-wrap and misfire the
-- semicolon split (matches migration 030's pattern).

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS scope_incident_id TEXT;

COMMENT ON COLUMN chat_sessions.scope_incident_id IS
    'Sprint 12.1 — When non-NULL, restrict /ask retrieval for this '
    'chat session to chunks whose metadata_json->>''primary_id'' '
    'equals this Incident_Number. Set by the Stage 0 per-bullet '
    'Ask-in-Chat handoff. NULL preserves global Search-in-KB '
    'behavior.';
