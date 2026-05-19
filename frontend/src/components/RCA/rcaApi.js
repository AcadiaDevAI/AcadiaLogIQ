// Sprint 13.32 — RCA API client.
//
// Phase 1 rewrite — long-running LLM calls now run on the worker
// container instead of inside the request thread. The HTTP contract
// from this client's point of view:
//
//   * POST /rca/{inc} → 200 + payload      (everything cached)
//   * POST /rca/{inc} → 202 + job IDs      (LLM work pending)
//   * GET  /jobs/{id} → status polling
//
// We hide that distinction from RCAFlow.js so the existing
// `generateRCA(inc, opts)` signature keeps working unchanged:
// callers get back the same `{customer_facing_md, internal_md, ...}`
// shape regardless of whether the result was cached or had to be
// generated. The polling loop happens inside this module.
//
// Auth + 401-retry still flow through the shared `api` instance.

import { api } from "../../services/api";
import { pollJobs } from "../../services/jobPolling";


// Look up a ticket by Incident_Number and run both LLM calls
// server-side. Returns:
//   {
//     incident_number: string,
//     customer_facing_md: string,
//     internal_md: string,
//     customer_facing_error: string | null,
//     internal_error: string | null,
//     customer_facing_cached: bool,
//     internal_cached: bool,
//   }
//
// Throws on 4xx/5xx; callers distinguish 404 (ticket not found)
// from 500/503 (job queue or LLM failure) by inspecting err.response.
//
// `opts.regenerateCustomer` / `opts.regenerateInternal` force-refresh
// the matching panel — same flags as before.
// `opts.signal`  — optional AbortSignal; if it fires while polling,
//                  the promise rejects with code="ABORTED" so the
//                  UI can stop spinning when the user navigates away.
// `opts.onProgress` — optional callback invoked with each poll
//                     response (`{status, attempts, ...}`). Used by
//                     RCAFlow.js to show "attempt 2 of 3" hints.
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

  // First POST. Cached responses come back as 200 + payload (~50 ms);
  // misses come back as 202 + job IDs (also fast — no LLM in this hop).
  // The 600 s axios timeout from earlier is preserved on the off-chance
  // we ever fall back to a synchronous path, but at this point both
  // 200 and 202 paths are sub-second.
  const initial = await api.post(
    `/rca/${encodeURIComponent(inc)}`,
    body,
    { timeout: 600000 },
  );

  // Fast path — everything was cached, just return.
  if (initial.status === 200) {
    return initial.data;
  }

  // Slow path — at least one panel was enqueued. Collect job IDs and
  // poll them in parallel.
  const { customer_job_id, internal_job_id } = initial.data || {};
  const jobIds = [customer_job_id, internal_job_id].filter(Boolean);

  await pollJobs(jobIds, {
    signal: opts.signal,
    onProgress: opts.onProgress,
  });

  // All jobs done — re-POST to get the cached payload back as 200.
  // We deliberately do NOT pass the regenerate flags on this hop
  // (otherwise we'd enqueue a fresh job again).
  const finalResp = await api.post(
    `/rca/${encodeURIComponent(inc)}`,
    { regenerate_customer: false, regenerate_internal: false },
    { timeout: 60000 },  // pure cache read — short timeout is safe
  );

  if (finalResp.status !== 200) {
    const err = new Error("unexpected_status_after_jobs");
    err.response = finalResp;
    throw err;
  }
  return finalResp.data;
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
