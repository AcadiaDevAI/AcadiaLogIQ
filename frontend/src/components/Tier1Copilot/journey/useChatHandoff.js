// Sprint 11 — useChatHandoff
//
// Shared hook that opens a new chat session with a prefilled question
// and auto-fires /ask, mirroring Stage 4's existing handleOpenChat
// flow (Stage4SearchKBHandoff.js:43-116). Lifted out so Stage 0's
// "How they did it" step links and Stage 3's per-ticket-detail
// Resolution Steps can share the exact same plumbing without
// duplicating the 60-line handler in three places.
//
// Inputs:
//   journeySessionId — the Tier-1 journey session id (sid)
// Returns:
//   { busy, askInChat(prefilledMessage) }
//
// Behaviour of askInChat(text):
//   1. POST /tier1/journey/<sid>/search-kb-handoff with the override
//      → backend creates a chat session + persists the user turn
//   2. SET_MODE("troubleshooting") so AppLayout swaps in ChatArea
//   3. SET_SESSION with the hydrated chat session so sidebar +
//      sessionMetadata are correct (and JourneyMessageActions can
//      offer the chat-back buttons we wired in Sprint 11)
//   4. /ask the prefilled text and dispatch the assistant turn
//
// Errors at any step show a single antd toast and reset busy. The
// chat session may have been created server-side even if /ask failed;
// that's intentional — the user can re-ask from the chat input box.

import { useCallback, useState } from "react";
import { message } from "antd";

import { useChat } from "../../../hooks/ChatContext";
import { askQuestion, getSession } from "../../../services/api";
import { searchKbHandoff } from "./journeyApi";


// Sprint 13.36 — per (journey, scope) chat-session cache. Survives
// component remounts within the same page load so a "Discuss with
// LogIQ" repeat-click on the same ticket reuses its existing chat
// session instead of forking a fresh one. Module-scoped on purpose:
// the cache is keyed by (journeySessionId, scopeIncidentId), so two
// different journeys (or the same journey at different scopes) still
// each get their own chat. A full page reload clears the cache,
// which is fine — the backend persists the chats and a fresh click
// will create a fresh one if the cache is cold.
const _handoffCache = new Map();
const _cacheKey = (journeySessionId, scope) =>
  `${journeySessionId}::${scope || "_global"}`;


// Sprint 13.36 — server-side dedupe of consecutive identical user
// turns. The initial click path persists the prefilled question
// TWICE on the backend (once by searchKbHandoff when the chat
// session is created, once by askQuestion when the answer is
// generated). On the first visit only one of those is visible
// because SET_SESSION runs between the two writes; on a reuse-rehydrate
// both are visible. This helper collapses any user turn that exactly
// duplicates the immediately preceding user turn (same role + same
// content, no assistant turn between them) before dispatching the
// session into the reducer.
function _dedupeConsecutiveUserTurns(sessionData) {
  if (!sessionData || !Array.isArray(sessionData.messages)) return sessionData;
  const out = [];
  for (const m of sessionData.messages) {
    const prev = out[out.length - 1];
    const isDuplicateUserEcho =
      prev
      && m
      && prev.role === "user"
      && m.role === "user"
      && (prev.content || "") === (m.content || "");
    if (isDuplicateUserEcho) continue;
    out.push(m);
  }
  return { ...sessionData, messages: out };
}


export default function useChatHandoff(journeySessionId) {
  const { dispatch } = useChat();
  const [busy, setBusy] = useState(false);

  const askInChat = useCallback(
    // Sprint 12.1 — optional 2nd arg `scopeIncidentId` lets per-bullet
    // callers (Stage 0 "Ask in Chat") scope the resulting chat session
    // to the bullet's source ticket. When omitted (Stage 4 / generic
    // callers), the chat opens with the existing global Search-in-KB
    // behaviour — no behavioural change for prior callers.
    async (prefilledMessage, scopeIncidentId) => {
      const text = (prefilledMessage || "").trim();
      if (!text) return;
      if (busy) return;
      if (!journeySessionId) {
        message.error("Cannot open chat — journey session is not loaded yet.");
        return;
      }
      setBusy(true);
      try {
        // Sprint 13.36 — reuse path: if we've already opened a chat
        // for this (journey, ticket) tuple in this page-load, jump
        // back to that chat instead of forking a new one. The cache
        // is invalidated lazily when the cached session no longer
        // resolves server-side (e.g. deleted between visits) — we
        // drop the stale entry and fall through to the create-fresh
        // path so the engineer always lands on a working chat.
        const cacheKey = _cacheKey(journeySessionId, scopeIncidentId);
        const cachedId = _handoffCache.get(cacheKey);
        if (cachedId) {
          try {
            const sessRes = await getSession(cachedId);
            dispatch({
              type: "SET_MODE",
              payload: { selectedMode: "troubleshooting", subMode: null },
            });
            dispatch({
              type: "SET_SESSION",
              payload: _dedupeConsecutiveUserTurns(sessRes.data),
            });
            return;
          } catch (reuseErr) {
            // eslint-disable-next-line no-console
            console.warn(
              "[useChatHandoff] cached chat session no longer reachable; creating a fresh one",
              reuseErr,
            );
            _handoffCache.delete(cacheKey);
          }
        }

        const { chat_session_id } = await searchKbHandoff(
          journeySessionId,
          text,
          scopeIncidentId || null,
        );

        dispatch({
          type: "SET_MODE",
          payload: { selectedMode: "troubleshooting", subMode: null },
        });

        try {
          const sessRes = await getSession(chat_session_id);
          dispatch({ type: "SET_SESSION", payload: sessRes.data });
        } catch (sessErr) {
          // eslint-disable-next-line no-console
          console.warn("[useChatHandoff] SET_SESSION hydration failed", sessErr);
          dispatch({ type: "ADD_USER_MESSAGE", payload: text });
        }

        dispatch({ type: "SET_LOADING", payload: true });
        try {
          const askRes = await askQuestion(text, chat_session_id);
          const askData = askRes.data;
          dispatch({
            type: "ADD_ASSISTANT_MESSAGE",
            payload: {
              answer: askData.answer,
              sources: askData.sources || [],
              confidence: askData.confidence,
              processing_time_ms: askData.processing_time_ms,
              sessionId: askData.session_id,
              context_stats: askData.context_stats || null,
              needs_clarification: askData.needs_clarification,
              clarification_id: askData.clarification_id,
              clarification_options: askData.clarification_options,
              clarification_context: askData.clarification_context,
            },
          });
        } catch (askErr) {
          // eslint-disable-next-line no-console
          console.warn("[useChatHandoff] /ask follow-up failed", askErr);
        } finally {
          dispatch({ type: "SET_LOADING", payload: false });
        }

        // Cache the freshly-created chat session so the next click
        // on the same (journey, ticket) tuple reuses it.
        _handoffCache.set(cacheKey, chat_session_id);
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("[useChatHandoff] handoff failed", err);
        message.error("Could not open chat. Please try again.");
      } finally {
        setBusy(false);
      }
    },
    // Sprint 12.1 — scopeIncidentId is captured per-call (passed as a
    // function arg, not closed over), so it's intentionally NOT listed
    // in the dependency array.
    [journeySessionId, dispatch, busy],
  );

  return { busy, askInChat };
}
