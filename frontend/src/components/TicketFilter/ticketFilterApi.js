// Ticket Filter — API client.
//
// Single endpoint: POST /ticket-filter. Returns paginated ticket
// summaries matching the (SLA_Target_Met, Resolution_Quality_Score)
// pair. Both filter fields are REQUIRED — caller validates before
// calling so the backend's 422 only fires on actual misuse.
//
// Routes through the shared authenticated `api` instance so Clerk
// JWT + 401-retry interceptors are inherited — same auth contract
// as every other authenticated client (RCA, Gap Analysis, journey).
//
// This module deliberately shares NO code with rcaApi.js or
// gapAnalysisApi.js — the Ticket Filter feature is independent
// per the design brief.

import { api } from "../../services/api";


// Filter historical tickets. Response shape:
//   {
//     tickets: [
//       {
//         incident_number, customer_name, priority, ticket_status,
//         timestamp, sla_target_met, resolution_quality_score,
//       }, ...
//     ],
//     total:     <distinct-incident count across all pages>,
//     page,
//     page_size,
//     has_more:  bool,
//   }
//
// Throws on 4xx / 5xx. The UI distinguishes empty-results
// (HTTP 200 + total=0) from errors by checking the status code.
export async function filterTickets({
  slaTargetMet,
  resolutionQualityScore,
  page = 1,
  pageSize = 20,
} = {}) {
  if (!slaTargetMet || !resolutionQualityScore) {
    const err = new Error("Both SLA and Resolution Quality Score are required.");
    err.code = "MISSING_REQUIRED";
    throw err;
  }
  const { data } = await api.post(
    "/ticket-filter",
    {
      sla_target_met: slaTargetMet,
      resolution_quality_score: resolutionQualityScore,
      page,
      page_size: pageSize,
    },
    // Read-only query — small response, fast — short timeout is fine.
    { timeout: 30000 },
  );
  return data;
}
