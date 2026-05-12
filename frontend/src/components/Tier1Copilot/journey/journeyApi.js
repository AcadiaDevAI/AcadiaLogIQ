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

// Sprint 12.7 — GET /tier1/journey/{session_id}/escalation-routing
// → EscalationRouting (deduped Resolution_Groups + Team_Paths +
// recommended Tier-2 entry candidates + forward-compat vendor data).
// Fetched alongside /stage-5 when the engineer reveals Operational
// Handoff. Fails gracefully — caller treats network error as "no
// routing data, hide section" rather than blocking the page.
export async function fetchEscalationRouting(sessionId) {
  const { data } = await api.get(
    `/tier1/journey/${encodeURIComponent(sessionId)}/escalation-routing`,
  );
  return data;
}

// Sprint 13.19 — POST /tier1/journey/{session_id}/escalation-handoff-note
// → {note, used_fallback}. Body now carries the Stage 3 checkbox
// state ({attempted_step_numbers: [1, 3]}) so the deterministic
// note backend can include only the steps the engineer actually
// ticked. Empty array (or missing body) → no diagnostic bullets;
// the note honestly states the engineer didn't mark any steps.
//
// Sprint 13.24 PERF — `force` flag bypasses the session-keyed
// consolidated-ledger cache. Auto-fetch on mount uses cache (fast);
// the Regenerate button passes force=true for a fresh LLM call.
export async function generateEscalationHandoffNote(
  sessionId,
  attemptedStepNumbers,
  { force = false } = {},
) {
  const { data } = await api.post(
    `/tier1/journey/${encodeURIComponent(sessionId)}/escalation-handoff-note`,
    {
      attempted_step_numbers: Array.isArray(attemptedStepNumbers)
        ? attemptedStepNumbers
        : [],
      force: !!force,
    },
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

// ─────────────────────────────────────────────────────────────
// Sprint 13.30 — Journey activity-version counter.
//
// A monotonically-increasing client-side counter that ticks any
// time the engineer's journey state mutates (events posted via
// `postJourneyEvent`, chat sessions opened via `searchKbHandoff`).
// Stage 5's Operational Handoff snapshots the counter at note-fetch
// time and re-reads it on every render to detect "the engineer
// went back and did something after the note was generated" —
// powering the stale-note UX (Acadia-themed Regenerate button +
// pulsing dot + Copy guardrail) without any polling or extra
// backend endpoint.
//
// Why client-side: every action that mutates journey state already
// flows through one of the two functions below, so a single bump
// in each is sufficient. Cheap, no network cost, no extra DB load.
// ─────────────────────────────────────────────────────────────
let _activityVersion = 0;
const _activityListeners = new Set();

export function getActivityVersion() {
  return _activityVersion;
}

export function bumpActivityVersion() {
  _activityVersion += 1;
  _activityListeners.forEach((cb) => {
    try { cb(_activityVersion); } catch { /* listener errors are non-fatal */ }
  });
}

export function subscribeActivityVersion(cb) {
  _activityListeners.add(cb);
  return () => { _activityListeners.delete(cb); };
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
//
// Sprint 12.1 — optional `scopeIncidentId` carries the bullet's source
// Incident_Number (extracted by the caller from the bullet's trailing
// " - INC-XXX" suffix). When present, the backend persists it on the
// new chat_sessions row so every subsequent /ask in that chat session
// is filtered to chunks belonging to that one ticket. NULL/omitted =
// global Search-in-KB behaviour preserved (Stage 4 default handoff).
export async function searchKbHandoff(
  sessionId,
  prefilledMessageOverride,
  scopeIncidentId,
) {
  const body =
    prefilledMessageOverride || scopeIncidentId
      ? {
          ...(prefilledMessageOverride
            ? { prefilled_message_override: prefilledMessageOverride }
            : {}),
          ...(scopeIncidentId ? { scope_incident_id: scopeIncidentId } : {}),
        }
      : undefined;
  const { data } = await api.post(
    `/tier1/journey/${encodeURIComponent(sessionId)}/search-kb-handoff`,
    body,
  );
  // Sprint 13.30 — opening a chat (Search-KB or per-bullet Discuss
  // with LogIQ) is journey-state mutation. Bump so Stage 5's stale
  // detector picks it up.
  bumpActivityVersion();
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
  // Sprint 13.30 — every journey event (stage_rendered,
  // next_stage_clicked, helpful_clicked, dislike_clicked,
  // kb_chat_engaged, ...) is a state mutation. Bump so Stage 5's
  // stale detector knows the engineer has done something since
  // the last handoff-note generation.
  bumpActivityVersion();
  return data;
}
