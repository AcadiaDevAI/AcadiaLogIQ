-- =========================================================
-- Phase 3: Multi-User Isolation & User Registration
-- 
-- PURPOSE:
--   1. Create a `users` table to store user profiles on first login
--   2. Add indexes for owner-scoped query performance
--   3. Ensure complete data isolation between users
--   4. Support hard-delete of all user data (GDPR-friendly)
--
-- SAFE to run on existing Phase 1 + Phase 2 schemas.
-- Run AFTER 001_phase1_foundation.sql and 002_phase2_contextual_ingestion.sql
-- =========================================================

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- =========================================================
-- USERS TABLE
-- Stores user profile on first Clerk login.
-- owner_id throughout the app maps to users.clerk_id
-- =========================================================
CREATE TABLE IF NOT EXISTS users (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    clerk_id      TEXT NOT NULL UNIQUE,          -- Clerk user ID (e.g., "user_2x...")
    email         TEXT NULL,                      -- from Clerk JWT or API
    full_name     TEXT NULL,                      -- display name
    avatar_url    TEXT NULL,                      -- profile image URL
    auth_provider TEXT NOT NULL DEFAULT 'clerk',  -- 'clerk', 'api_key', 'anonymous'
    is_active     BOOLEAN NOT NULL DEFAULT TRUE,
    last_login_at TIMESTAMP NULL,
    created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for fast lookup by clerk_id (used on every request)
CREATE INDEX IF NOT EXISTS idx_users_clerk_id
    ON users(clerk_id);

-- Index for admin queries by email
CREATE INDEX IF NOT EXISTS idx_users_email
    ON users(email);

-- =========================================================
-- ENSURE OWNER_ID COLUMNS EXIST (idempotent)
-- These should already exist from Phase 1, but let's be safe.
-- =========================================================
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'anonymous';

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'anonymous';

ALTER TABLE ingestion_jobs
    ADD COLUMN IF NOT EXISTS owner_id TEXT NOT NULL DEFAULT 'anonymous';

-- =========================================================
-- INDEXES FOR OWNER-SCOPED QUERIES
-- These ensure every query filtered by owner_id is fast.
-- =========================================================

-- Documents: list files for a user
CREATE INDEX IF NOT EXISTS idx_documents_owner_id
    ON documents(owner_id);

-- Chat sessions: list chats for a user
CREATE INDEX IF NOT EXISTS idx_chat_sessions_owner_id
    ON chat_sessions(owner_id);

-- Chat messages: cascade delete needs this
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id
    ON chat_messages(session_id);

-- Ingestion jobs: list jobs for a user
CREATE INDEX IF NOT EXISTS idx_ingestion_jobs_owner_id
    ON ingestion_jobs(owner_id);

-- Chunks: owner-scoped search requires document_id lookup
CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

-- =========================================================
-- HELPER: Delete ALL data for a specific user
-- Call this when a user requests full account data removal.
-- Cascades handle chunks, embeddings, versions, metadata.
-- =========================================================
CREATE OR REPLACE FUNCTION delete_user_data(p_owner_id TEXT)
RETURNS TABLE(
    deleted_documents  BIGINT,
    deleted_sessions   BIGINT,
    deleted_jobs       BIGINT,
    deleted_user       BOOLEAN
) AS $$
DECLARE
    v_docs    BIGINT;
    v_sess    BIGINT;
    v_jobs    BIGINT;
    v_user    BOOLEAN := FALSE;
BEGIN
    -- Delete documents (cascades to versions, chunks, embeddings, metadata)
    DELETE FROM documents WHERE owner_id = p_owner_id;
    GET DIAGNOSTICS v_docs = ROW_COUNT;

    -- Delete chat sessions (cascades to chat_messages)
    DELETE FROM chat_sessions WHERE owner_id = p_owner_id;
    GET DIAGNOSTICS v_sess = ROW_COUNT;

    -- Delete ingestion jobs
    DELETE FROM ingestion_jobs WHERE owner_id = p_owner_id;
    GET DIAGNOSTICS v_jobs = ROW_COUNT;

    -- Delete user record
    DELETE FROM users WHERE clerk_id = p_owner_id;
    IF FOUND THEN v_user := TRUE; END IF;

    RETURN QUERY SELECT v_docs, v_sess, v_jobs, v_user;
END;
$$ LANGUAGE plpgsql;
