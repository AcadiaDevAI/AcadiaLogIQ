import { useCallback, useEffect, useRef, useState } from "react";
import {
  createSession as apiCreateSession,
  fetchMatchByIndex as apiFetchMatchByIndex,
  sessionStatus as apiSessionStatus,
  swapMatchIndex as apiSwapMatchIndex,
} from "../components/Tier1Copilot/tier1Api";

/**
 * Sprint 7 — Tier-1 session state hook.
 *
 * Owns: sessionId, matchIndex, startedAt, elapsed, stuck flag.
 * Polls /tier1/session/{id}/status every 60s so the stuck-detection
 * modal can fire without requiring user interaction.
 *
 * The parent Tier1Workspace passes the initial analyze response; this
 * hook takes over from there. If `initial.session_id` is null (Sprint 7
 * backend flag off), the hook no-ops and returns null-ish values — the
 * Sprint 6 rendering path handles that case.
 */
const POLL_MS = 60_000;

export default function useTier1Session(initial) {
  const [sessionId, setSessionId] = useState(
    initial && initial.session_id ? initial.session_id : null,
  );
  const [matchIndex, setMatchIndex] = useState(0);
  const [stuckNudge, setStuckNudge] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [thumbsDownCount, setThumbsDownCount] = useState(0);
  const [escalated, setEscalated] = useState(false);
  const startedAtRef = useRef(
    initial && initial.started_at ? new Date(initial.started_at) : new Date(),
  );

  // Sync sessionId when the parent receives a new /analyze response.
  useEffect(() => {
    if (initial && initial.session_id) {
      setSessionId(initial.session_id);
    }
    if (initial && initial.started_at) {
      const d = new Date(initial.started_at);
      if (!isNaN(d.getTime())) {
        startedAtRef.current = d;
      }
    }
  }, [initial && initial.session_id, initial && initial.started_at]);

  // Poll backend every POLL_MS.
  useEffect(() => {
    if (!sessionId) return undefined;
    let cancelled = false;
    const tick = async () => {
      try {
        const s = await apiSessionStatus(sessionId);
        if (cancelled || !s) return;
        setElapsed(Number(s.elapsed_seconds || 0));
        setStuckNudge(!!s.stuck_nudge);
        setThumbsDownCount(Number(s.thumbs_down_count || 0));
        setMatchIndex(Number(s.current_match_index || 0));
        setEscalated(!!s.escalated);
      } catch {
        /* transient failure — next tick will retry */
      }
    };
    tick();
    const id = setInterval(tick, POLL_MS);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, [sessionId]);

  // Local 1Hz ticker so the timer feels live between polls.
  useEffect(() => {
    const id = setInterval(() => {
      setElapsed((prev) => prev + 1);
    }, 1000);
    return () => clearInterval(id);
  }, []);

  const createLocal = useCallback(async (alertSignature, alertPayload, matchIds) => {
    const s = await apiCreateSession({
      alert_signature: alertSignature,
      alert_payload: alertPayload,
      top_5_match_ids: matchIds || [],
    });
    if (s && s.session_id) setSessionId(s.session_id);
    return s;
  }, []);

  const swapMatch = useCallback(
    async (newIndex) => {
      if (!sessionId) return null;
      const out = await apiSwapMatchIndex(sessionId, newIndex);
      if (out && typeof out.current_match_index === "number") {
        setMatchIndex(out.current_match_index);
      }
      return out;
    },
    [sessionId],
  );

  // Sprint 8 — rank-N match fetch + paginating flag. Returns the full
  // Tier1AnalyzeResponse shape for the requested match. The workspace
  // uses this to replace the answer-card content in place. Parent also
  // consumes `paginating` to flash the skeleton card.
  const [paginating, setPaginating] = useState(false);
  const fetchMatch = useCallback(
    async (newIndex) => {
      if (!sessionId) return null;
      setPaginating(true);
      try {
        const next = await apiFetchMatchByIndex(sessionId, newIndex);
        if (next && typeof next.match_index === "number") {
          setMatchIndex(next.match_index);
        }
        return next;
      } catch (err) {
        return null;
      } finally {
        setPaginating(false);
      }
    },
    [sessionId],
  );

  return {
    sessionId,
    matchIndex,
    setMatchIndex,
    stuckNudge,
    setStuckNudge,
    elapsed,
    thumbsDownCount,
    escalated,
    startedAt: startedAtRef.current,
    createLocal,
    swapMatch,
    // Sprint 8 additions
    paginating,
    fetchMatch,
  };
}
