import axios from "axios";

// const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
// const API_BASE = process.env.REACT_APP_API_BASE || "http://18.233.93.19:8000";
const API_BASE = window.location.origin.replace(":8501", ":8000");
const API_KEY = process.env.REACT_APP_API_KEY || "";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 180000,
  headers: {
    ...(API_KEY ? { "X-API-Key": API_KEY } : {}),
  },
});

// ─── Clerk Token Injection ───────────────────────────────
// This gets set by useAuthInterceptor() hook in the React app.
// When Clerk is active, every API call automatically gets
// Authorization: Bearer <clerk-session-token>
let _getToken = null;

/**
 * Call this once from a React component that has access to
 * Clerk's useAuth() hook. It registers a token-getter function
 * that the axios interceptor uses.
 */
export function setTokenGetter(fn) {
  _getToken = fn;
}

api.interceptors.request.use(
  async (config) => {
    if (_getToken) {
      try {
        const token = await _getToken();
        if (token) {
          config.headers.Authorization = `Bearer ${token}`;
        }
      } catch {
        // Token fetch failed — request will proceed without auth
        // and backend will return 401 if Clerk is required
      }
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ─── Health ───────────────────────────────────────────────
export const healthCheck = () => api.get("/health");

// ─── Auth ─────────────────────────────────────────────────
export const getCurrentUser = () => api.get("/me");

// ─── Upload ───────────────────────────────────────────────
export const uploadFile = (file, fileType, onProgress) => {
  const form = new FormData();
  form.append("file", file);
  return api.post(`/upload?file_type=${fileType}`, form, {
    headers: { "Content-Type": "multipart/form-data" },
    timeout: 300000,
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    },
  });
};

export const getUploadStatus = (jobId) =>
  api.get(`/upload_status/${jobId}`, { timeout: 30000 });

// ─── Files ────────────────────────────────────────────────
export const listFiles = () => api.get("/files");

export const deleteFile = (fileId) => api.delete(`/files/${fileId}`);

// ─── Chat ─────────────────────────────────────────────────
export const askQuestion = (question, sessionId) =>
  api.post("/ask", { q: question, session_id: sessionId || null });

// ─── Chat History ─────────────────────────────────────────
export const listSessions = () => api.get("/chat/sessions");

export const getSession = (sessionId) =>
  api.get(`/chat/sessions/${sessionId}`);

export const deleteSession = (sessionId) =>
  api.delete(`/chat/sessions/${sessionId}`);

export const deleteAllSessions = () => api.delete("/chat/sessions");

// ─── Reset ────────────────────────────────────────────────
export const resetAll = () => api.post("/reset", {}, { timeout: 30000 });

export default api;
