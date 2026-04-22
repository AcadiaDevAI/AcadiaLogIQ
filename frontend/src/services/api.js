import axios from "axios";

// ── Config ────────────────────────────────────────────────
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
//const API_BASE = window.location.origin.replace(":8501", ":8000");
const API_KEY = process.env.REACT_APP_API_KEY || "";
const CLERK_ENABLED = !!process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

const api = axios.create({
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

export const uploadFile = (file, fileType, onProgress) => {
  const form = new FormData();
  form.append("file", file);
  return api.post(`/upload?file_type=${fileType}`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
    onUploadProgress: (e) => {
      if (onProgress && e.total) onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
};
export const getUploadStatus = (jobId) => api.get(`/upload_status/${jobId}`, { timeout: 30000 });

export const listFiles = () => api.get("/files");
export const deleteFile = (fileId) => api.delete(`/files/${fileId}`);

export const askQuestion = (question, sessionId, clarificationResponse = null) => {
  const payload = { q: question, session_id: sessionId || null };
  if (clarificationResponse) {
    payload.clarification_response = {
      clarification_id: clarificationResponse.clarificationId,
      selected_option_id: clarificationResponse.selectedOptionId,
      free_text: clarificationResponse.freeText || null,
    };
  }
  return api.post("/ask", payload);
};

export const listSessions = () => api.get("/chat/sessions");
export const getSession = (sessionId) => api.get(`/chat/sessions/${sessionId}`);
export const deleteSession = (sessionId) => api.delete(`/chat/sessions/${sessionId}`);
export const deleteAllSessions = () => api.delete("/chat/sessions");

export const resetAll = () => api.post("/reset", {}, { timeout: 30000 });

// ─── Feedback ─────────────────────────────────────────────
export const saveFeedbackState = (sessionId, messageIndex, feedbackType, semanticCacheId = null) =>
  api.post("/feedback/state", {
    session_id: sessionId,
    message_index: messageIndex,
    feedback_type: feedbackType,
    semantic_cache_id: semanticCacheId,
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

export default api;