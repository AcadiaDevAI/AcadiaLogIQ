// Sprint 9 — Universal Intake API client.
// Reuses the same axios pattern as tier1Api.js so Clerk + X-API-Key
// headers propagate identically.

import axios from "axios";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
//const API_BASE = window.location.origin.replace(":8501", ":8000");
const API_KEY = process.env.REACT_APP_API_KEY || "";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: { ...(API_KEY ? { "X-API-Key": API_KEY } : {}) },
});

export async function extractIntake(payload) {
  // payload = { source, raw_text, session_id?, context_hints? }
  const { data } = await client.post("/intake/extract", payload);
  return data;
}

export async function sendExtractionFeedback(extractionId, payload) {
  // payload = { picked_index?, edits?, was_rejected }
  const { data } = await client.post(
    `/intake/extraction/${encodeURIComponent(extractionId)}/feedback`,
    payload,
  );
  return data;
}

export async function intakeHealth() {
  const { data } = await client.get("/intake/health");
  return data;
}
