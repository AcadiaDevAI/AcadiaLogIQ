// Sprint 9 — Universal Intake API client.
//
// Uses the SHARED axios instance from services/api.js so the Clerk
// Bearer-token interceptor attaches `Authorization: Bearer ...` on
// every request. Earlier this file constructed its own axios client
// (no interceptors), which worked while the backend tolerated
// anonymous requests and broke when auth_dependency became strict.

import { api as client } from "../../../services/api";

export async function extractIntake(payload) {
  // payload = { source, raw_text, session_id?, context_hints? }
  const { data } = await client.post("/intake/extract", payload);
  return data;
}

export async function sendExtractionFeedback(extractionId, payload) {
  // payload = { picked_index?, edits?, was_rejected }
  const { data } = await client.post(
    `/intake/extraction/${encodeURIComponent(extractionId)}/feedback`,
    payload,
  );
  return data;
}

export async function intakeHealth() {
  const { data } = await client.get("/intake/health");
  return data;
}
