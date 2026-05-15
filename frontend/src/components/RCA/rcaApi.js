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
export async function generateRCA(incidentNumber) {
  const inc = (incidentNumber || "").trim();
  if (!inc) {
    const err = new Error("incident_number_required");
    err.code = "EMPTY_INPUT";
    throw err;
  }
  const { data } = await api.post(
    `/rca/${encodeURIComponent(inc)}`,
    // No request body required — incident number is in the path.
    // Empty object keeps the POST well-formed for proxies that
    // refuse zero-length bodies.
    {},
  );
  return data;
}
