-- Sprint 4 — Fingerprint-First Expert Copilot
-- Two audit/provenance columns on chat_sessions so the session row
-- records how the conversation was entered.
--
-- entered_via: one of 'fingerprint' | 'skip' | NULL
--   NULL  — pre-Sprint-4 session OR Sprint 4 disabled at session start
--   'fingerprint' — user submitted a fingerprint (hit or miss)
--   'skip'        — user clicked Skip on the fingerprint screen
--
-- original_fingerprint: the UPPERCASE fingerprint code submitted, if any.
--   NULL when entered_via != 'fingerprint'.
--
-- Both columns are nullable with no default so existing rows (and
-- flag-off sessions) preserve their current behavior. Idempotent.

BEGIN;

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS entered_via VARCHAR(20) DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS original_fingerprint VARCHAR(100) DEFAULT NULL;

COMMIT;
