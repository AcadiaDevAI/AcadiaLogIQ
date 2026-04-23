-- Sprint 5 — Template-First Expert Copilot + Answer Cache
-- Two columns on chunks that hold the fully-rendered Expert Copilot
-- answer for a gold-schema JSON ticket chunk, plus a timestamp so
-- future TTL / invalidation policy has a cheap index to scan.
--
-- cached_expert_answer     TEXT      — the stitched markdown answer
--                                      (header + Phase 1/Expert Pivot from
--                                      LLM + Phase 2/Phase 3 from template).
-- cached_expert_answer_at  TIMESTAMP — NOW() at cache write time.
--
-- Cache invalidation is automatic: Sprint 2.9 _ingest_gold_ticket_json
-- deletes and re-inserts chunk rows on re-upload, so the new row's
-- cached columns start NULL. No explicit DELETE needed.
--
-- All statements are idempotent (IF NOT EXISTS). Safe to apply with
-- LOGIQ_SPRINT5_BACKEND=False — the columns sit unused.

BEGIN;

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS cached_expert_answer TEXT DEFAULT NULL,
    ADD COLUMN IF NOT EXISTS cached_expert_answer_at TIMESTAMP DEFAULT NULL;

CREATE INDEX IF NOT EXISTS idx_chunks_cached_expert_answer_at
    ON chunks(cached_expert_answer_at);

COMMIT;
