// Sprint 6 — Tier-1 Copilot API client.
//
// Uses the SHARED axios instance from services/api.js so the Clerk
// Bearer-token interceptor attaches `Authorization: Bearer ...` on
// every request. Earlier this file constructed its own axios client
// (no interceptors) which meant Tier-1 calls bypassed Clerk — fine
// while the backend tolerated anonymous, broken now that
// auth_dependency is strict and returns 401 without a JWT.

import { api as client } from "../../services/api";

export async function analyzeAlert(payload) {
  // payload = { severity, asset_name, alert_type, customer?, location?,
  //   technology?, ip_or_device_id?, error_code?, notes?, session_id }
  const { data } = await client.post("/tier1/analyze", payload);
  return data;
}

export async function sendFeedback(payload) {
  // payload = { response_id, helpful, follow_up_action?, session_id }
  const { data } = await client.post("/tier1/feedback", payload);
  return data;
}

export async function healthCheck() {
  const { data } = await client.get("/tier1/health");
  return data;
}

// ── Sprint 7 — progressive workflow clients ─────────────────

export async function createSession(payload) {
  // payload = { alert_signature, alert_payload, top_5_match_ids }
  const { data } = await client.post("/tier1/session", payload);
  return data;
}

export async function sessionStatus(sessionId) {
  const { data } = await client.get(`/tier1/session/${encodeURIComponent(sessionId)}/status`);
  return data;
}

export async function swapMatchIndex(sessionId, matchIndex) {
  const { data } = await client.post(
    `/tier1/session/${encodeURIComponent(sessionId)}/match-index`,
    { match_index: matchIndex },
  );
  return data;
}

export async function logAction(sessionId, entry) {
  // entry = { step, result, note? }
  const { data } = await client.post(
    `/tier1/session/${encodeURIComponent(sessionId)}/action`,
    entry,
  );
  return data;
}

export async function fetchDeeperDiagnostics(sessionId, matchedIncident) {
  const { data } = await client.post("/tier1/deeper-diagnostics", {
    session_id: sessionId,
    matched_incident: matchedIncident || null,
  });
  return data;
}

export async function fetchEscalationPackage(
  sessionId,
  matchedIncident,
  whatTried,
) {
  const { data } = await client.post("/tier1/escalation-package", {
    session_id: sessionId,
    matched_incident: matchedIncident || null,
    what_tried: whatTried || [],
  });
  return data;
}

export async function fetchExplain(sessionId, matchedIncident) {
  const { data } = await client.post("/tier1/explain", {
    session_id: sessionId,
    matched_incident: matchedIncident || null,
  });
  return data;
}

// ── Sprint 8 — rank-N match endpoint ────────────────────────

export async function fetchMatchByIndex(sessionId, matchIndex) {
  const url = `/tier1/session/${encodeURIComponent(sessionId)}/match/${encodeURIComponent(
    String(matchIndex),
  )}`;
  const { data } = await client.get(url);
  return data;
}
