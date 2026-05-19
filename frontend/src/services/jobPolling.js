// Generic async-job polling helper.
//
// Phase 1 backend rewrite: long-running endpoints (POST /rca/{inc},
// POST /gap-analysis/{inc}) now return HTTP 202 + job IDs when one
// or more panels need a fresh LLM call. The frontend polls
// `GET /jobs/{id}` until status becomes terminal, then re-POSTs the
// original endpoint (which now returns 200 + cached markdown).
//
// This helper centralises the polling loop so each API client
// (rcaApi.js, gapAnalysisApi.js) doesn't re-implement timeouts,
// backoff, and failure handling.
//
// Design:
//   * Constant 2 s poll interval. Tight enough that the UX feels
//     "live", loose enough that we don't hammer the API.
//   * Wall-clock deadline (default 8 min) covers worst-case
//     Gap Analysis generations even with one retry.
//   * Cooperative cancellation via `AbortSignal` — if the user
//     leaves the panel or clicks Return, we stop polling.
//   * Per-attempt callback (`onProgress`) lets the UI render
//     "attempt 2 of 3" hints.
//
// Failure model:
//   * status === "done"     → resolve with the row dict
//   * status === "failed"   → throw with error_message
//   * status === "cancelled"→ throw "job_cancelled"
//   * timeout / abort       → throw a tagged Error
//   * network errors        → retried silently (until deadline)

import { api } from "./api";


// Poll one job until it reaches a terminal status.
// Returns the final job row on success; throws on terminal failure
// or timeout.
export async function pollJob(
  jobId,
  {
    intervalMs = 2000,
    maxWaitMs = 480_000,   // 8 min — covers worst-case Gap Analysis
    signal,
    onProgress,
  } = {},
) {
  if (!jobId) {
    throw new Error("pollJob: jobId is required");
  }
  const deadline = Date.now() + maxWaitMs;

  while (true) {
    if (signal && signal.aborted) {
      const err = new Error("polling_aborted");
      err.code = "ABORTED";
      throw err;
    }
    if (Date.now() > deadline) {
      const err = new Error("polling_timeout");
      err.code = "TIMEOUT";
      throw err;
    }

    let row;
    try {
      const { data } = await api.get(`/jobs/${encodeURIComponent(jobId)}`);
      row = data;
    } catch (e) {
      // Network blip — sleep and retry, the deadline above is the
      // ultimate fence. 404 means the row was deleted (e.g. the
      // 30-day retention sweeper); surface that as a hard failure.
      if (e?.response?.status === 404) {
        const err = new Error("job_not_found");
        err.code = "NOT_FOUND";
        throw err;
      }
      // Other transients — back off and retry below.
      await _sleep(intervalMs);
      continue;
    }

    if (typeof onProgress === "function") {
      try {
        onProgress(row);
      } catch (_cbErr) {
        // Caller's progress callback must NEVER break the polling loop.
      }
    }

    if (row.status === "done") {
      return row;
    }
    if (row.status === "failed") {
      const err = new Error(row.error_message || "job_failed");
      err.code = "JOB_FAILED";
      err.job = row;
      throw err;
    }
    if (row.status === "cancelled") {
      const err = new Error("job_cancelled");
      err.code = "JOB_CANCELLED";
      err.job = row;
      throw err;
    }

    // Otherwise pending / running — sleep and re-poll.
    await _sleep(intervalMs);
  }
}


// Poll multiple jobs in parallel. Resolves when ALL succeed; rejects
// on the first failure (after letting the others finish, so partial
// progress isn't lost to a single bad panel — the caller can inspect
// e.partial when surfaced).
export async function pollJobs(jobIds, opts = {}) {
  const ids = (jobIds || []).filter((x) => !!x);
  if (ids.length === 0) return [];
  const results = await Promise.allSettled(ids.map((id) => pollJob(id, opts)));
  const rejected = results.find((r) => r.status === "rejected");
  if (rejected) {
    // Propagate the first failure, but attach the partial results so
    // the UI can render whatever succeeded.
    const err = rejected.reason instanceof Error
      ? rejected.reason
      : new Error(String(rejected.reason));
    err.partial = results.map((r) =>
      r.status === "fulfilled" ? r.value : { status: "failed", error: String(r.reason) },
    );
    throw err;
  }
  return results.map((r) => r.value);
}


function _sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
