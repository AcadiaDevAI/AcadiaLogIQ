import axios from "axios";

// ── Config ────────────────────────────────────────────────
//const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
const API_BASE = window.location.origin.replace(":8501", ":8000");
const API_KEY = process.env.REACT_APP_API_KEY || "";
const CLERK_ENABLED = !!process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

// Sprint 10.6.1 — exported so other API modules (e.g.
// journey/journeyApi.js) can route their calls through this single
// authenticated instance. The Clerk request/response interceptors
// below attach the Bearer token + retry on 401 — every consumer that
// imports `api` inherits both behaviours.
export const api = axios.create({
  baseURL: API_BASE,
  timeout: 180000,
  headers: { ...(API_KEY ? { "X-API-Key": API_KEY } : {}) },
});

// ── Token management ──────────────────────────────────────
let _getToken = null;

// ── Auth-ready gate ───────────────────────────────────────
let _authReadyResolve = null;
let _authReady = CLERK_ENABLED
  ? new Promise((resolve) => {
      _authReadyResolve = resolve;
    })
  : Promise.resolve();

let _gateOpen = false;

export function setTokenGetter(fn) {
  _getToken = fn;

  if (fn && CLERK_ENABLED && !_gateOpen) {
    _gateOpen = true;
    if (_authReadyResolve) {
      _authReadyResolve();
      _authReadyResolve = null;
    }
    console.log("[API] Auth gate opened — requests will now include tokens");
  }

  if (!fn && CLERK_ENABLED && _gateOpen) {
    _gateOpen = false;
    _authReady = new Promise((resolve) => {
      _authReadyResolve = resolve;
    });
    console.log("[API] Auth gate closed — user signed out");
  }
}

// ── Request interceptor: wait for auth, then attach JWT ───
api.interceptors.request.use(
  async (config) => {
    // If this is a 401-retry, the response interceptor already set
    // a fresh token on the headers — don't overwrite it.
    if (config._retry) {
      return config;
    }

    if (CLERK_ENABLED) {
      const timeout = new Promise((_, reject) =>
        setTimeout(() => reject(new Error("Auth initialization timed out (8s)")), 8000)
      );
      try {
        await Promise.race([_authReady, timeout]);
      } catch (err) {
        console.error("[API]", err.message);
        return config;
      }
    }

    if (_getToken) {
      try {
        const token = await _getToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
      } catch (err) {
        console.warn("[API] Failed to get token for request:", err);
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response interceptor: handle 401 with retry ───────────
// KEY FIX: Adds a small delay before retry to let Clerk's internal
// session refresh complete, and marks the retry so the request
// interceptor won't overwrite the fresh token.
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;

    if (
      error.response?.status === 401 &&
      !originalRequest._retry &&
      _getToken
    ) {
      originalRequest._retry = true;

      try {
        // Small delay — gives Clerk time to complete any in-flight
        // session refresh triggered by the expired token
        await new Promise((r) => setTimeout(r, 500));

        const freshToken = await _getToken();
        if (freshToken) {
          originalRequest.headers.Authorization = `Bearer ${freshToken}`;
          return api(originalRequest);
        } else {
          console.warn("[API] 401 retry: getToken returned null — session may have expired");
        }
      } catch (retryErr) {
        console.error("[API] Token refresh failed on retry:", retryErr);
      }
    }

    return Promise.reject(error);
  }
);

// ── API functions ─────────────────────────────────────────
export const healthCheck = () => api.get("/health");
export const getCurrentUser = () => api.get("/me");

export const uploadFile = (file, fileType, onProgress, docKind) => {
  const form = new FormData();
  form.append("file", file);
  // Sprint 3-PREP-B — optional doc_kind form field. Backend silently
  // coerces missing / unknown values to 'ticket' so pre-PREP-B callers
  // (which don't pass the 4th arg) keep identical behavior.
  if (docKind) {
    form.append("doc_kind", docKind);
  }
  return api.post(`/upload?file_type=${fileType}`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
};
export const getUploadStatus = (jobId) => api.get(`/upload_status/${jobId}`, { timeout: 30000 });

// ─────────────────────────────────────────────────────────────
// Phase 1 — S3 direct upload helpers (presigned PUT).
//
// Three-step flow (browser-friendly):
//   1) presignUpload({ filename, content_type, size, doc_kind, file_type })
//        → backend returns { upload_url, key, job_id, file_id, required_headers }
//   2) putToS3(upload_url, file, content_type, onProgress)
//        → browser PUTs bytes DIRECTLY to S3 (bypasses FastAPI)
//   3) finalizeUpload({ job_id, key })
//        → backend HEADs the object and schedules ingestion
//
// uploadFileV2 wraps all three in a single async call with the SAME
// signature as the legacy uploadFile, so it's a drop-in replacement.
// ─────────────────────────────────────────────────────────────

export const presignUpload = ({ filename, content_type, size_bytes, doc_kind, file_type }) =>
  api.post(
    "/upload/presign",
    {
      filename,
      content_type: content_type || "application/octet-stream",
      size_bytes: typeof size_bytes === "number" ? size_bytes : null,
      doc_kind: doc_kind || "ticket",
      file_type: file_type || "kb",
    },
    { timeout: 30000 },
  );

export const finalizeUpload = ({ job_id, key, sha256 }) =>
  api.post("/upload/finalize", { job_id, key, sha256 }, { timeout: 60000 });

// Direct browser → S3 PUT. We use XHR (not fetch) because fetch still
// does not expose upload-progress events. The Content-Type header MUST
// match what the backend signed into the presigned URL or S3 rejects
// the request as SignatureDoesNotMatch.
export const putToS3 = (uploadUrl, file, contentType, onProgress) =>
  new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("PUT", uploadUrl, true);
    xhr.setRequestHeader("Content-Type", contentType || "application/octet-stream");
    xhr.upload.onprogress = (e) => {
      if (onProgress && e.lengthComputable) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        const etag = (xhr.getResponseHeader("ETag") || "").replace(/"/g, "");
        resolve({ status: xhr.status, etag });
      } else {
        reject(new Error(`S3 PUT failed: ${xhr.status} ${xhr.statusText} ${xhr.responseText || ""}`));
      }
    };
    xhr.onerror = () => reject(new Error("S3 PUT network error"));
    xhr.onabort = () => reject(new Error("S3 PUT aborted"));
    xhr.send(file);
  });

// Drop-in replacement for uploadFile() — same signature, same return
// shape ({ data: { job_id, file_id, ... } }) so call sites can swap
// behind a single boolean flag.
export const uploadFileV2 = async (file, fileType, onProgress, docKind) => {
  const presign = await presignUpload({
    filename: file.name,
    content_type: file.type || "application/octet-stream",
    size_bytes: file.size,
    doc_kind: docKind,
    file_type: fileType,
  });
  const { upload_url, key, job_id, file_id, storage_uri } = presign.data;

  await putToS3(upload_url, file, file.type || "application/octet-stream", onProgress);
  // Snap the bar to 100% — finalize is server-side and finishes in <1s.
  if (onProgress) onProgress(100);

  const fin = await finalizeUpload({ job_id, key });
  return {
    data: {
      job_id,
      file_id,
      storage_uri,
      message: "Uploaded to S3. Processing started.",
      file_hash: "",            // legacy compatibility shape
      ...fin.data,
    },
  };
};

export const listFiles = () => api.get("/files");
export const deleteFile = (fileId) => api.delete(`/files/${fileId}`);

// askQuestion(question, sessionId, clarificationResponse, options)
//
// `options` (added in the content-aware doc_kind iteration) carries
// optional retrieval scoping flags that ride on top of the existing
// /ask contract:
//
//   options.allowedDocKinds  — Array<string> of doc_kind values to
//                              restrict retrieval to. Used by the
//                              "Search-in-KB → open new chat" handoff
//                              to pin retrieval to KB chunks ("kb"),
//                              keeping the result set off ticket data.
//                              When omitted/empty, retrieval falls
//                              through to backend's mode-derived filter
//                              exactly as before — no behavioural
//                              change for existing callers.
//
// Backward-compatible: 4th positional arg, defaults to empty options.
export const askQuestion = (
  question,
  sessionId,
  clarificationResponse = null,
  options = {},
) => {
  const payload = { q: question, session_id: sessionId || null };
  if (clarificationResponse) {
    payload.clarification_response = {
      clarification_id: clarificationResponse.clarificationId,
      selected_option_id: clarificationResponse.selectedOptionId,
      free_text: clarificationResponse.freeText || null,
    };
  }
  if (Array.isArray(options.allowedDocKinds) && options.allowedDocKinds.length) {
    payload.allowed_doc_kinds = options.allowedDocKinds;
  }
  return api.post("/ask", payload);
};

export const listSessions = () => api.get("/chat/sessions");
export const getSession = (sessionId) => api.get(`/chat/sessions/${sessionId}`);
export const deleteSession = (sessionId) => api.delete(`/chat/sessions/${sessionId}`);
export const deleteAllSessions = () => api.delete("/chat/sessions");

export const resetAll = () => api.post("/reset", {}, { timeout: 30000 });

// ─── Feedback ─────────────────────────────────────────────
// Sprint 3B — `sessionMode` and `originalQuery` are optional; when a user
// 👎s a troubleshooting answer, passing them lets the backend run the
// KB/runbook pivot and return a `pivot` block the caller can render.
// Backward-compatible: any existing 4-arg callsite works unchanged.
export const saveFeedbackState = (
  sessionId,
  messageIndex,
  feedbackType,
  semanticCacheId = null,
  sessionMode = null,
  originalQuery = null,
) =>
  api.post("/feedback/state", {
    session_id: sessionId,
    message_index: messageIndex,
    feedback_type: feedbackType,
    semantic_cache_id: semanticCacheId,
    session_mode: sessionMode,
    original_query: originalQuery,
  });

export const submitFeedback = (data) => api.post("/feedback/submit", data);

// ─── Multi-User Auth ─────────────────────────────────────
export const registerOrLogin = (data) =>
  api.post("/auth/register-or-login", data);

export const getProfile = () => api.get("/auth/profile");

export const deleteAccount = () => api.delete("/auth/delete-account");

// ─── Guided Workflow (Sprint 1) ───────────────────────────
// Endpoints land the session in one of the 4 PRD modes.
// See backend/services/session_mode_state.py for the canonical
// list of mode / sub_mode values.

export const getSessionMode = (sessionId) =>
  api.get(`/chat/sessions/${sessionId}/mode`);

export const setSessionMode = (sessionId, { selectedMode, subMode, formData }) =>
  api.post(`/chat/sessions/${sessionId}/mode`, {
    selected_mode: selectedMode,
    sub_mode: subMode || null,
    form_data: formData || null,
  });

export const resetSessionContext = (sessionId) =>
  api.post(`/chat/sessions/${sessionId}/context/reset`);

// ─── Guided Workflow (Sprint 2) ───────────────────────────
// Partial patch to mode-state. Server returns 403 when the
// Sprint 2 flag is off — callers must treat that as a feature
// disabled signal (not an error to surface to the user).
export const patchSessionForm = (sessionId, fields) =>
  api.post(`/chat/sessions/${sessionId}/mode/form`, fields || {});

// ─── Sprint 4 — Fingerprint-First Expert Copilot ──────────
// Both endpoints return 404 when LOGIQ_SPRINT4_BACKEND is off.
// Callers must treat 404 as "feature disabled" and fall through
// to the classic LandingPage, not surface as an error.
export const fingerprintLookup = (sessionId, fingerprint) =>
  api.post("/fingerprint/lookup", {
    session_id: sessionId,
    fingerprint,
  });

export const fingerprintSkip = (sessionId) =>
  api.post("/fingerprint/skip", { session_id: sessionId });

// ─────────────────────────────────────────────────────────────
// Tenancy (multi-tenant Phase 0) — backend endpoints in
// backend/tenancy/routes.py
//
// All four functions return raw axios promises so callers can use
// .then / await + read response.data exactly like every other API
// wrapper in this file. They inherit the auth-ready gate + JWT
// attachment via the shared `api` instance.
// ─────────────────────────────────────────────────────────────

// GET /organizations
// Returns the landing-page payload — { your_organizations, other_organizations }.
// `your_organizations` = orgs the current user is an ACTIVE member of.
// `other_organizations` = publicly-listable orgs they're NOT in (locked tiles).
export const listOrganizations = () => api.get("/organizations");

// GET /organizations/me/active
// Returns current active-org context derived from the JWT claims:
//   { has_active_org, organization?, last_active_org_id?, platform_role }
// `organization` is null when the user has no active org chosen.
// Frontend's OrgContext calls this on mount and after every Clerk
// org-switch event.
export const getActiveOrganization = () => api.get("/organizations/me/active");

// PATCH /users/me/active-org
// Persist the user's restore-on-next-login org choice. The frontend
// MUST also call Clerk's setActive({ organization }) so the JWT
// re-issues with the new org claim — this endpoint is just the
// server-side persistence half.
export const setActiveOrganization = (organizationId) =>
  api.patch("/users/me/active-org", { organization_id: organizationId });

// POST /organizations/{slug}/request-access
// Create a pending access request for a non-member org. Optional
// `justification` shown to org admins in the review queue.
// Server returns 409 if the same user already has a pending request
// for the same org.
export const requestOrganizationAccess = (slug, justification) =>
  api.post(`/organizations/${encodeURIComponent(slug)}/request-access`, {
    justification: justification || null,
  });

export default api;