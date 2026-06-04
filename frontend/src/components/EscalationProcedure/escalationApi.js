// Thin wrappers over the Escalation Procedures KB endpoints.
// Reuses the auth-aware axios instance from services/api.js.

import { api } from "../../services/api";

export async function getEscalationStatus() {
  const { data } = await api.get("/escalation/status");
  return data;
}

export async function uploadEscalationPdf(file, { onUploadProgress } = {}) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post("/escalation/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress,
    timeout: 600000,
  });
  return data;
}

export async function askEscalation({ section, question, history }) {
  const { data } = await api.post("/escalation/ask", {
    section,
    question,
    history: history || [],
  });
  return data;
}

export async function deleteEscalationKb() {
  const { data } = await api.delete("/escalation/kb");
  return data;
}
