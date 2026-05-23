// ServiceNow — API client.
//
// One read-only endpoint: GET /ticket-filter/servicenow. Backend reads
// credentials from backend/env.bvk (or AWS Secrets Manager once those
// keys are migrated), calls the ServiceNow Table REST API, requests
// XML, and converts to JSON. We just consume the JSON.
//
// Routes through the shared authenticated `api` instance so the Clerk
// JWT interceptor is inherited — same auth contract as every other
// authenticated client in the app.
//
// This module deliberately shares NO code with ticketFilterApi.js —
// ServiceNow is its own optional integration and must not couple to
// the historical filter feature.

import { api } from "../../services/api";


// Fetch priority-1 incidents from the configured ServiceNow instance.
//
// Response envelope (always this shape, regardless of count):
//   {
//     source:       "servicenow",
//     instance_url: "https://dev....service-now.com",
//     query:        "priority=1",
//     count:        <int>,
//     incidents:    [ {...}, ... ],
//     raw:          {...},   // full nested dict for debugging
//   }
//
// Throws on 4xx / 5xx. Caller distinguishes:
//   - 503 → credentials not configured (operator action)
//   - 502 → ServiceNow unreachable / bad XML / non-2xx
//   - 429 → per-user rate limit exceeded
export async function fetchServiceNowIncidents() {
  const { data } = await api.get(
    "/ticket-filter/servicenow",
    // External HTTP call inside the backend — give it room to breathe.
    { timeout: 30000 },
  );
  return data;
}
