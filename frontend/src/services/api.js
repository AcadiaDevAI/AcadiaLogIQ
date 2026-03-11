import axios from "axios";

//const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
//const API_BASE = process.env.REACT_APP_API_BASE || "http://18.233.93.19:8000";
//const API_BASE = window.location.origin.replace(":8501", ":8000");
// import axios from "axios";

//for local development
const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

 //const API_BASE = window.location.origin.replace(":8501", ":8000");
const API_KEY = process.env.REACT_APP_API_KEY || "";

const api = axios.create({
  baseURL: API_BASE,
  timeout: 180000,
  headers: { ...(API_KEY ? { "X-API-Key": API_KEY } : {}) },
});

let _getToken = null;
export function setTokenGetter(fn) { _getToken = fn; }

api.interceptors.request.use(
  async (config) => {
    if (_getToken) {
      try {
        const token = await _getToken();
        if (token) config.headers.Authorization = `Bearer ${token}`;
      } catch {}
    }
    return config;
  },
  (error) => Promise.reject(error)
);

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

export const askQuestion = (question, sessionId) =>
  api.post("/ask", { q: question, session_id: sessionId || null });

export const listSessions = () => api.get("/chat/sessions");
export const getSession = (sessionId) => api.get(`/chat/sessions/${sessionId}`);
export const deleteSession = (sessionId) => api.delete(`/chat/sessions/${sessionId}`);
export const deleteAllSessions = () => api.delete("/chat/sessions");

export const resetAll = () => api.post("/reset", {}, { timeout: 30000 });

// ─── Feedback ─────────────────────────────────────────────
// Persist like/dislike icon state on a message (survives refresh/sign-out)
export const saveFeedbackState = (sessionId, messageIndex, feedbackType) =>
  api.post("/feedback/state", { session_id: sessionId, message_index: messageIndex, feedback_type: feedbackType });

// Submit feedback with message → sends SES email with user name/email
export const submitFeedback = (data) => api.post("/feedback/submit", data);

export default api;