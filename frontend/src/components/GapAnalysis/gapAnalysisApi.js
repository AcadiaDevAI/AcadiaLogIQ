// Gap Analysis — API client.
//
// Single endpoint: POST /gap-analysis/{incident_number}
// Routes through the shared authenticated `api` instance from
// services/api.js so Clerk JWT + 401-retry interceptors are
// inherited — same auth contract as RCA / journey / intake clients.

import { api } from "../../services/api";


// Look up a ticket by Incident_Number and generate two reports
// server-side in parallel. Response shape:
//   {
//     incident_number: string,
//     gap_analysis_md: string,
//     post_mortem_md: string,
//     gap_analysis_error: string | null,
//     post_mortem_error: string | null,
//   }
//
// Throws on 4xx/5xx; callers distinguish 404 ("ticket not found")
// from 500 ("LLM / DB failure") by inspecting err.response.
// `opts` is optional. Recognised keys:
//   regenerateGapAnalysis : boolean  — force-refresh Gap Analysis panel
//   regeneratePostMortem  : boolean  — force-refresh Post-Mortem panel
// Both default to false → cache lookup honoured.
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
  // The two LLM calls run server-side in parallel and the Gap
  // Analysis panel routinely produces 12k+ output tokens at the
  // current max_tokens budget — observed wall-clock ~4 min on a
  // fresh (cache-miss) generation. The shared `api` instance
  // defaults to 180s which is too tight, so override per-call
  // here. Cached hits return in milliseconds, so the long timeout
  // only matters on first-time generation.
  const { data } = await api.post(
    `/gap-analysis/${encodeURIComponent(inc)}`,
    body,
    { timeout: 600000 },  // 10 min — covers worst-case fresh LLM run
  );
  return data;
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
