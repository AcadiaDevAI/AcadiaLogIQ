-- Migration 050 — Persist allowed_doc_kinds on chat sessions.
--
-- Sprint 10 introduced an `allowed_doc_kinds` request parameter on
-- /ask so the Stage 4 Search KB / SOP handoff could restrict retrieval
-- to KB / SOP content for the prefilled first message. The frontend
-- (Stage4SearchKBHandoff.js, useChatHandoff.js) sets that parameter
-- ONLY on the auto-fired first message. Follow-up messages typed by
-- the user in the same chat session went through the regular ChatArea
-- path without the parameter, so the backend silently lost KB-search
-- context after turn 1 — composer voice fell back to "default" and
-- the KB-Search system prompt (backend/routing/kb_search_prompt.py)
-- never activated.
--
-- This column persists the KB-search context ON THE CHAT SESSION
-- ROW. When create_chat_session_with_handoff opens a non-scoped
-- (Search-in-KB) chat, it writes ['sop','kb'] here. The /ask handler
-- defaults `_effective_doc_kinds` to this value whenever the request
-- body doesn't carry an explicit override.
--
-- Safe to re-run. Single nullable JSONB column, no backfill required.
-- Existing chat_sessions rows continue to behave exactly as before
-- (NULL = no per-session doc_kind preference).
--
-- See: backend/tier1_copilot/journey/stage4_search_kb_handoff.py
--      backend/api.py (/ask `_effective_doc_kinds` resolution)
--      backend/routing/kb_search_prompt.py (the actual prompt)
--
-- Note: BEGIN/COMMIT intentionally omitted — backend/db/migrate.py
-- wraps each migration in engine.begin() and splits on ';', so
-- explicit transaction control would double-wrap and misfire the
-- semicolon split (matches migration 030's / 042's pattern).

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS allowed_doc_kinds JSONB;

COMMENT ON COLUMN chat_sessions.allowed_doc_kinds IS
    'Sprint 10 follow-up — When non-NULL, this JSONB array (e.g. '
    '["sop","kb"]) is used as the default `allowed_doc_kinds` for '
    'every /ask call in this chat session whenever the request body '
    'does not override it. Set by create_chat_session_with_handoff '
    'when a Stage 4 Search-in-KB chat is opened without a '
    'scope_incident_id. NULL preserves global all-corpora behavior.';
