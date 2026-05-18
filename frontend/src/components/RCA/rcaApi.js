// Sprint 13.32 — RCA API client.
//
// Single endpoint: POST /rca/{incident_number} → both panel payloads.
// Routed through the shared authenticated `api` instance (Clerk JWT
// + 401-retry interceptors) so the new flow inherits the same auth
// guarantees as the rest of the journey calls.

import { api } from "../../services/api";


// Look up a ticket by Incident_Number and run both LLM calls
// server-side. Returns:
//   {
//     incident_number: string,
//     customer_facing_md: string,
//     internal_md: string,
//     customer_facing_error: string | null,
//     internal_error: string | null,
//   }
// Throws on 4xx / 5xx; callers should distinguish 404 ("ticket not
// found") from 500 ("LLM / DB failure") by inspecting err.response.
// `opts` is optional. Recognised keys:
//   regenerateCustomer : boolean  — force-refresh customer-facing panel
//   regenerateInternal : boolean  — force-refresh internal panel
// Both default to false → cache lookup is honoured. Backward
// compatible with callers that pass no argument or just `(inc)`.
export async function generateRCA(incidentNumber, opts = {}) {
  const inc = (incidentNumber || "").trim();
  if (!inc) {
    const err = new Error("incident_number_required");
    err.code = "EMPTY_INPUT";
    throw err;
  }
  const body = {
    regenerate_customer: !!opts.regenerateCustomer,
    regenerate_internal: !!opts.regenerateInternal,
  };
  // Two LLM calls run server-side in parallel. RCA outputs are
  // typically smaller than Gap Analysis but a cache-miss fresh
  // generation can still exceed the shared api default of 180s,
  // so override per-call to a generous ceiling. Cached hits return
  // in milliseconds, so this only matters on first-time generation.
  const { data } = await api.post(
    `/rca/${encodeURIComponent(inc)}`,
    body,
    { timeout: 600000 },  // 10 min — covers worst-case fresh LLM run
  );
  return data;
}


// Record a 👍 / 👎 against a specific RCA panel.
//   panel         : "customer_facing" | "internal"
//   feedbackType  : "like" | "dislike"
// 👎 also invalidates that panel's cached row on the backend; the
// next generateRCA() call will run a fresh LLM for that panel.
export async function recordRCAFeedback(incidentNumber, panel, feedbackType) {
  const inc = (incidentNumber || "").trim();
  if (!inc) throw new Error("incident_number_required");
  const { data } = await api.post(
    `/rca/${encodeURIComponent(inc)}/feedback`,
    { panel, feedback_type: feedbackType },
  );
  return data;
}
