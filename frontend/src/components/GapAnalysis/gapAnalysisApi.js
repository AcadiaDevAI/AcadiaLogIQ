// Gap Analysis — API client.
//
// Phase 1 rewrite — long-running LLM calls now run on the worker
// container. The HTTP contract from this client's point of view:
//
//   * POST /gap-analysis/{inc} → 200 + payload      (both panels cached)
//   * POST /gap-analysis/{inc} → 202 + job IDs      (LLM work pending)
//   * GET  /jobs/{id}          → status polling
//
// We hide that distinction from GapAnalysisFlow.js so the existing
// `generateGapAnalysis(inc, opts)` signature keeps working unchanged
// — callers get back the same `{gap_analysis_md, post_mortem_md, ...}`
// shape regardless of cache hit / generation.
//
// Routes through the shared authenticated `api` instance so Clerk JWT
// + 401-retry interceptors are inherited.

import { api } from "../../services/api";
import { pollJobs } from "../../services/jobPolling";


// Look up a ticket by Incident_Number and run both LLM calls
// server-side in parallel. Response shape:
//   {
//     incident_number: string,
//     gap_analysis_md: string,
//     post_mortem_md: string,
//     gap_analysis_error: string | null,
//     post_mortem_error: string | null,
//     gap_analysis_cached: bool,
//     post_mortem_cached: bool,
//   }
//
// `opts.regenerateGapAnalysis` / `opts.regeneratePostMortem` force-
// refresh the matching panel.
// `opts.signal` — optional AbortSignal so a user navigating away cancels polling.
// `opts.onProgress` — optional callback invoked with each poll
//                     response; lets the UI render attempt counters.
export async function generateGapAnalysis(incidentNumber, opts = {}) {
  const inc = (incidentNumber || "").trim();
  if (!inc) {
    const err = new Error("incident_number_required");
    err.code = "EMPTY_INPUT";
    throw err;
  }
  const body = {
    regenerate_gap_analysis: !!opts.regenerateGapAnalysis,
    regenerate_post_mortem: !!opts.regeneratePostMortem,
  };

  // Initial POST. 200 = both panels cached; 202 = enqueued.
  const initial = await api.post(
    `/gap-analysis/${encodeURIComponent(inc)}`,
    body,
    { timeout: 600000 },
  );

  if (initial.status === 200) {
    return initial.data;
  }

  // Slow path — poll the returned job IDs.
  const { gap_analysis_job_id, post_mortem_job_id } = initial.data || {};
  const jobIds = [gap_analysis_job_id, post_mortem_job_id].filter(Boolean);

  await pollJobs(jobIds, {
    signal: opts.signal,
    onProgress: opts.onProgress,
  });

  // Re-POST without regenerate flags to retrieve the cached panels.
  const finalResp = await api.post(
    `/gap-analysis/${encodeURIComponent(inc)}`,
    { regenerate_gap_analysis: false, regenerate_post_mortem: false },
    { timeout: 60000 },
  );

  if (finalResp.status !== 200) {
    const err = new Error("unexpected_status_after_jobs");
    err.response = finalResp;
    throw err;
  }
  return finalResp.data;
}


// Record a 👍 / 👎 against a specific Gap Analysis panel.
//   panel         : "gap_analysis" | "post_mortem"
//   feedbackType  : "like" | "dislike"
// 👎 also invalidates that panel's cached row on the backend.
export async function recordGapAnalysisFeedback(incidentNumber, panel, feedbackType) {
  const inc = (incidentNumber || "").trim();
  if (!inc) throw new Error("incident_number_required");
  const { data } = await api.post(
    `/gap-analysis/${encodeURIComponent(inc)}/feedback`,
    { panel, feedback_type: feedbackType },
  );
  return data;
}
