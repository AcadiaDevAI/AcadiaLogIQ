// Sprint 10 — Tier-1 Resolution Journey API client.
//
// Sprint 10.6.1 hotfix — route every call through the authenticated
// `api` instance from services/api.js. The previous local
// `axios.create(...)` client had no Clerk request interceptor, so no
// call ever carried Authorization: Bearer .... That was tolerable
// while every journey endpoint was flag-gated only — the requests
// happened to work because the backend wasn't checking auth — but
// Sprint 10.6 §4 added Depends(_lazy_auth_dependency) on
// /search-kb-handoff, and the missing token surfaced as a 401 in
// 4.86ms (Clerk rejecting before the handler ran).
//
// Switching to the shared `api` instance means:
//   - Every journey call now goes through the same request
//     interceptor that attaches the Clerk JWT (services/api.js:50-83).
//   - The same 401-retry-with-fresh-token response interceptor
//     applies for free.
//   - Single auth source of truth — when Clerk's behaviour changes,
//     one fix updates the entire app.

import { api } from "../../../services/api";


// GET /tier1/journey/{session_id}/initial → JourneyInitial
// (Stage 0 + Stage 1A + Stage 1B in one payload)
export async function fetchInitial(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/initial`,
  );
  return data;
}

// GET /tier1/journey/{session_id}/stage-2 → Stage2HistoricalMatches
export async function fetchStage2(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/stage-2`,
  );
  return data;
}

// GET /tier1/journey/{session_id}/stage-3 → Stage3TroubleshootingApproach
export async function fetchStage3(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/stage-3`,
  );
  return data;
}

// GET /tier1/journey/{session_id}/stage-4 → Stage4SearchKB
export async function fetchStage4(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/stage-4`,
  );
  return data;
}

// GET /tier1/journey/{session_id}/stage-5 → Tier1EscalationPackage
export async function fetchStage5(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/stage-5`,
  );
  return data;
}

// Sprint 10.7 §4.4 — GET /tier1/journey/{session_id}/resume-state
// → {session_id, current_stage, last_event_at}
//
// ResolutionJourney calls this on mount to seed which stages are
// already "revealed" (every stage up to and including current_stage
// is mounted; later stages stay hidden until the engineer clicks
// next-stage). On any error the journey falls back to stage_0 — same
// graceful-degrade contract the backend uses.
export async function getResumeState(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/resume-state`,
  );
  return data;
}

// Sprint 10.2 — POST /tier1/journey/{session_id}/search-kb-handoff
// → {chat_session_id, redirect_url}
//
// Creates a real chat session, auto-submits the prefilled question,
// and invokes /ask the same way regular chat does (Sprint 10.6 §3 —
// no doc-kind gate, no filter). Sprint 10.6 §4 added Clerk auth on
// this endpoint; the request MUST carry Authorization: Bearer ...
// (provided automatically by the shared `api` instance's request
// interceptor).
//
// Sprint 11 — optional `prefilledMessageOverride` lets per-step
// "Ask in chat" links carry an arbitrary step text into the chat
// session instead of the journey's Stage 4 default. Empty / nullish
// values fall through to default behaviour on the backend.
export async function searchKbHandoff(sessionId, prefilledMessageOverride) {
  const body = prefilledMessageOverride
    ? { prefilled_message_override: prefilledMessageOverride }
    : undefined;
  const { data } = await api.post(
    `/tier1/journey/${encodeURIComponent(sessionId)}/search-kb-handoff`,
    body,
  );
  return data;
}


// POST /tier1/journey/{session_id}/event → {ok}
//
// Telemetry write — `stage` and `event_type` validated server-side.
// Errors are caller's responsibility to swallow; we don't catch here
// because some callers may want to retry. HelpfulButton + NextStageButton
// already wrap in try/catch.
export async function postJourneyEvent(sessionId, stage, eventType, payload) {
  const { data } = await api.post(
    `/tier1/journey/${encodeURIComponent(sessionId)}/event`,
    {
      stage,
      event_type: eventType,
      ...(payload ? { payload } : {}),
    },
  );
  return data;
}
