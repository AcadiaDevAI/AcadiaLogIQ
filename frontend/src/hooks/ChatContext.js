import React, { createContext, useContext, useReducer } from "react";

const ChatContext = createContext();

const initialState = {
  sessionId: null,
  messages: [],
  sessions: [],
  uploadedFiles: [],
  isLoading: false,
  isUploading: false,
  sidebarOpen: true,
  // Gate for the hover-to-peek auto-expand. Only the Resolution
  // Journey ("blocks") screen flips this to true — everywhere else
  // the sidebar should respond purely to manual button clicks, even
  // if the engineer collapsed it manually.
  sidebarHoverPeekEnabled: false,
  sidebarTab: "chat",
  userRole: "admin",  // "admin" | "user" — controls upload visibility

  // ── Guided workflow (Sprint 1) ──────────────────────
  // Null when the user has not picked a mode yet. When a
  // mode is selected on the landing page, these hold the
  // canonical values and the chat UI becomes mode-aware.
  selectedMode: null,
  subMode: null,
  conversationContextActive: false,
  customerName: null,
  technologyDomain: null,
  ticketId: null,
  issueSummary: null,
  formData: null,

  // ── Sprint 2 ────────────────────────────────────────
  // pendingContextBreak: populated when /ask returns a
  // context_break signal. Null when no modal should show.
  // Shape (mirrors context_stats keys):
  //   { source, category, matched_phrase, active_mode,
  //     active_sub_mode, triggering_query }
  pendingContextBreak: null,

  // ── Sprint 10.4 ─────────────────────────────────────
  // sessionMetadata: arbitrary per-session metadata the backend
  // attaches to the chat_sessions response. Currently used to surface
  // `journey_session_id` so ChatArea can render the "Back to
  // Resolution Journey" banner when the chat originated from a Stage 4
  // handoff. Shape: { journey_session_id?: string }.
  sessionMetadata: {},

  // ── Sprint 11 ───────────────────────────────────────
  // journeyResumeSessionId: when the chat-side "Return to Stages"
  // or "Escalate to Tier 2" button is clicked from
  // JourneyMessageActions, this carries the journey session id
  // back to LandingRouter. LandingRouter watches it, restores
  // screen=tier1 with that session, then dispatches
  // CLEAR_JOURNEY_RESUME so the field doesn't re-fire on every
  // subsequent state change. Replaces the broken Sprint 10.5
  // window.location.href = `/tier1/journey/<id>` navigation, which
  // never worked because the React app has no URL routing.
  journeyResumeSessionId: null,

  // US Pharma — true when the current chat was opened via the "KB SOP"
  // quick action from the landing / intake screen (NEW_CHAT with this flag).
  // ChatArea shows a "Back to screen" button while it's set so the engineer
  // can return to the landing screen. Journey Stage 4 chats never set it.
  kbSearchFromLanding: false,

  // US Pharma — the Store ID a landing/intake "Discuss Store Specific with
  // LogIQ" chat is scoped to (set alongside a pre-created store-scoped
  // session via NEW_CHAT payload.scopeStoreId). Drives the ChatArea banner
  // label so the engineer sees which store the chat is restricted to. Null
  // for every non-store-scoped chat.
  kbScopeStoreId: null,

  // True for any freshly-opened chat (NEW_CHAT — e.g. the sidebar "New Chat"
  // button), so ChatArea shows a "Back to screen" button that returns the
  // user to the landing screen. Cleared when a history session is loaded
  // (SET_SESSION) so old chats don't show it. All orgs.
  backToScreen: false,
};

function reducer(state, action) {
  switch (action.type) {
    // ── Load a session from backend ──────────────────────
    // Normalizes sources (dict → flat array) and preserves
    // the "feedback" field ("like"/"dislike"/undefined) so
    // thumbs up/down colors persist after sign-out and reload.
    case "SET_SESSION":
      return {
        ...state,
        sessionId: action.payload.id,
        messages: (action.payload.messages || []).map((msg, _idx, _arr) => {
          let sources = msg.sources;
          if (sources && !Array.isArray(sources)) {
            sources = [
              ...(sources.docs || []),
              ...(sources.logs || []),
              ...(sources.kb || []),
            ].filter(Boolean);
          }
          return {
            ...msg,
            sources: sources || [],
            // US Pharma (Idea A) — the store-orientation opener persists its
            // sample-question chips in session metadata (first message's
            // `_session_metadata.suggestions`). Re-attach them to the first
            // assistant message so the chips survive reload.
            suggestions:
              msg.suggestions ||
              (_idx === _arr.findIndex((m) => m.role === "assistant")
                ? action.payload.metadata?.suggestions || null
                : null),
            // Preserve feedback state from backend ("like", "dislike", or undefined)
            feedback: msg.feedback || null,
            semanticCacheId:
              msg.context_stats?.semantic_cache_id || msg.semanticCacheId || null,
          };
        }),
        // ── Pull mode fields from the session payload if backend
        //    attaches them (Sprint 2 will start populating these).
        //
        //    Sprint 12 — fallback hardened. When the backend omits
        //    selected_mode (older chat rows, or any session created
        //    before mode persistence landed) AND the session carries
        //    messages, default to "troubleshooting" so AppLayout
        //    routes to ChatArea. Previously we fell back to
        //    state.selectedMode, which left the user stuck on the
        //    LandingRouter (Tier-1 intake form) when they clicked a
        //    history entry from inside the Tier-1 Workspace
        //    (selectedMode=null at that point).
        selectedMode:
          action.payload.selected_mode ??
          ((action.payload.messages && action.payload.messages.length > 0)
            ? "troubleshooting"
            : state.selectedMode),
        subMode: action.payload.sub_mode ?? state.subMode,
        conversationContextActive:
          action.payload.conversation_context_active ?? state.conversationContextActive,
        customerName: action.payload.customer_name ?? state.customerName,
        technologyDomain: action.payload.technology_domain ?? state.technologyDomain,
        ticketId: action.payload.ticket_id ?? state.ticketId,
        issueSummary: action.payload.issue_summary ?? state.issueSummary,
        formData: action.payload.form_data ?? state.formData,
        // Sprint 10.4 — capture session-level metadata (e.g.
        // journey_session_id) so ChatArea can render the back-to-
        // journey banner. Default to {} when the backend omits it.
        sessionMetadata: action.payload.metadata ?? {},
        // Loading a specific chat from history is never the landing KB SOP
        // chat — clear the flag so its "Back to screen" button doesn't leak.
        kbSearchFromLanding: false,
        kbScopeStoreId: null,
        // A loaded history session is not a fresh New Chat → no back button.
        backToScreen: false,
      };

    case "NEW_CHAT":
      // Sprint 12 — "New Chat" now opens a blank ChatArea (empty
      // EmptyState + ChatInput) instead of bouncing the user back to
      // the LandingRouter (which, post-Sprint-11, lands on the Tier-1
      // Copilot intake form by default — unwanted for a "fresh chat"
      // gesture). selectedMode is held at "troubleshooting" so:
      //   1. AppLayout's `!state.selectedMode` gate stays false ⇒
      //      ChatArea continues to render.
      //   2. /ask uses the same ticket-related retrieval path that
      //      the Stage 4 KB-handoff chat uses (useChatHandoff.js sets
      //      the same mode), keeping search corpus consistent.
      //   3. The next /ask call creates a new backend session that
      //      will be stored with selected_mode="troubleshooting", so
      //      a later chat-history click round-trips back into ChatArea
      //      via SET_SESSION cleanly.
      return {
        ...state,
        // US Pharma — the "Discuss Store Specific with LogIQ" dialog
        // pre-creates a store-scoped session server-side and passes its id
        // here so the FIRST /ask already runs against that scoped session.
        // A plain "New Chat" sends no payload → null (fresh session on first
        // /ask, unchanged behavior).
        sessionId: (action.payload && action.payload.sessionId) || null,
        messages: [],
        selectedMode: "troubleshooting",
        subMode: null,
        conversationContextActive: true,
        customerName: null,
        technologyDomain: null,
        ticketId: null,
        issueSummary: null,
        formData: null,
        pendingContextBreak: null,
        // Sprint 10.4 — clear journey-session linkage on NEW_CHAT so
        // the "Back to Resolution Journey" banner doesn't carry over
        // from a prior journey-originated chat.
        sessionMetadata: {},
        // US Pharma — set when the KB SOP quick action opens this chat
        // (dispatch NEW_CHAT with payload.kbSearchFromLanding). A plain
        // "New Chat" click sends no payload, so the flag clears.
        kbSearchFromLanding: !!(action.payload && action.payload.kbSearchFromLanding),
        // US Pharma — Store ID this chat is scoped to (from the Store-ID
        // dialog). Null for a plain New Chat.
        kbScopeStoreId: (action.payload && action.payload.scopeStoreId) || null,
        // Any New Chat (incl. the sidebar "New Chat") shows a "Back to
        // screen" button so the user can return to the landing screen.
        backToScreen: true,
      };

    // US Pharma — leave the landing-opened KB SOP chat and return to the
    // landing screen. Clearing selectedMode makes AppLayout fall back to
    // LandingRouter (its `!selectedMode` gate). Also wipes the transient
    // chat so the landing starts clean.
    case "EXIT_KB_SEARCH":
      return {
        ...state,
        selectedMode: null,
        subMode: null,
        conversationContextActive: false,
        kbSearchFromLanding: false,
        kbScopeStoreId: null,
        backToScreen: false,
        sessionId: null,
        messages: [],
        pendingContextBreak: null,
        sessionMetadata: {},
      };

    case "ADD_USER_MESSAGE":
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            role: "user",
            content: action.payload,
            timestamp: new Date().toISOString(),
          },
        ],
      };

    case "ADD_ASSISTANT_MESSAGE":
      // Guard: never append a phantom assistant bubble. A valid assistant
      // message has either a non-empty answer OR a clarification payload.
      if (
        !action.payload ||
        (!action.payload.answer && !action.payload.needs_clarification)
      ) {
        return state;
      }
      return {
        ...state,
        sessionId: action.payload.sessionId || state.sessionId,
        messages: [
          ...state.messages,
          {
            role: "assistant",
            content: action.payload.answer || "",
            sources: action.payload.sources || [],
            // US Pharma (Idea A) — clickable sample-question chips rendered
            // under a store-orientation opener. Null for normal answers.
            suggestions: action.payload.suggestions || null,
            confidence: action.payload.confidence,
            processingTime: action.payload.processing_time_ms,
            timestamp: new Date().toISOString(),
            feedback: null, // no feedback yet on new messages
            // Brief 5 / Part 1 — carry semantic cache id forward so dislike can
            // invalidate the specific cached row that produced this answer.
            semanticCacheId:
              action.payload.context_stats?.semantic_cache_id || null,

            // Interactive clarification fields (Brief 6)
            needsClarification: action.payload.needs_clarification === true,
            clarificationId: action.payload.clarification_id || null,
            clarificationOptions: action.payload.clarification_options || null,
            clarificationContext: action.payload.clarification_context || null,
            clarificationSelectedId: null,

            // Sprint 2 — pattern response fields. Null when the
            // backend didn't attach pattern data (pre-flag shape
            // unchanged).
            patternActive: !!action.payload.context_stats?.pattern_active,
            patternTopic: action.payload.context_stats?.pattern_topic || null,
            patternData: action.payload.context_stats?.pattern_data || null,

            // Sprint 3B — low-similarity banner signal. Pulled from
            // the top-level response field (confidence_band). When
            // absent (Sprint 3B flag off or cached path), the banner
            // does not render.
            confidenceBand: action.payload.confidence_band || null,

            // Sprint 3B — KB pivot marker. Used when this message is
            // itself a pivot follow-up (kb_guidance | kb_empty). The
            // primary /ask response never sets this; only the 👎 flow
            // adds it via ADD_ASSISTANT_MESSAGE with pivot_type set.
            pivotType: action.payload.pivot_type || null,
            // Preserve the original user query on the assistant
            // message so a 👎 can send it back to /feedback/state
            // for the KB pivot lookup.
            originalQuery: action.payload.original_query || null,
          },
        ],
      };

    // Mark a clarification option as selected (prevents re-click + dims others)
    case "SET_CLARIFICATION_SELECTED":
      return {
        ...state,
        messages: state.messages.map((msg, i) =>
          i === action.payload.index
            ? { ...msg, clarificationSelectedId: action.payload.optionId }
            : msg
        ),
      };

    // Sprint 4 — set the session_id after /fingerprint/lookup creates
    // one on the user's first interaction. Does not touch messages or
    // mode state; the subsequent ADD_USER_MESSAGE / ADD_ASSISTANT_MESSAGE
    // dispatches append to the newly-attached session.
    case "SET_SESSION_ID":
      return { ...state, sessionId: action.payload || null };

    // ── Guided workflow reducer cases (Sprint 1) ────────
    case "SET_MODE":
      return {
        ...state,
        selectedMode: action.payload.selectedMode || null,
        subMode: action.payload.subMode || null,
        conversationContextActive: !!action.payload.selectedMode,
      };

    // ── Sprint 11 — chat → journey return path ──────────
    // Set the resume sid AND clear selectedMode in one shot so
    // AppLayout falls through from ChatArea back to LandingRouter,
    // which can then mount Tier1Workspace for the resumed session.
    // Payload: { journeySessionId: string }.
    case "RESUME_JOURNEY":
      return {
        ...state,
        journeyResumeSessionId: action.payload?.journeySessionId || null,
        // Clear chat mode so AppLayout shows LandingRouter on next render.
        selectedMode: null,
        subMode: null,
        conversationContextActive: false,
      };

    // Idempotent consumer-side clear. LandingRouter dispatches this
    // after it has read the resume sid and mounted Tier1Workspace,
    // so the field doesn't re-fire on subsequent renders.
    case "CLEAR_JOURNEY_RESUME":
      return {
        ...state,
        journeyResumeSessionId: null,
      };

    case "SET_SUB_MODE":
      return { ...state, subMode: action.payload || null };

    case "SET_FORM_DATA":
      return { ...state, formData: action.payload || null };

    case "RESET_MODE_STATE":
      return {
        ...state,
        selectedMode: null,
        subMode: null,
        conversationContextActive: false,
        customerName: null,
        technologyDomain: null,
        ticketId: null,
        issueSummary: null,
        formData: null,
        pendingContextBreak: null,
        kbSearchFromLanding: false,
        kbScopeStoreId: null,
        backToScreen: false,
      };

    // ── Sprint 2 ──────────────────────────────────────
    // Set to a payload dict to surface the context-break
    // modal; set to null to dismiss.
    case "SET_PENDING_CONTEXT_BREAK":
      return { ...state, pendingContextBreak: action.payload || null };

    // ── Persist like/dislike on a specific message ───────
    // Called after user clicks thumbs up or down.
    // payload: { index: number, feedback: "like" | "dislike" | null }
    case "SET_MESSAGE_FEEDBACK":
      return {
        ...state,
        messages: state.messages.map((msg, i) =>
          i === action.payload.index
            ? { ...msg, feedback: action.payload.feedback }
            : msg
        ),
      };

    case "SET_SESSIONS":
      return { ...state, sessions: action.payload };

    case "SET_FILES":
      return { ...state, uploadedFiles: action.payload };

    case "ADD_FILE":
      return {
        ...state,
        uploadedFiles: [action.payload, ...state.uploadedFiles],
      };

    case "UPDATE_FILE_STATUS":
      return {
        ...state,
        uploadedFiles: state.uploadedFiles.map((f) =>
          f.id === action.payload.id
            ? { ...f, status: action.payload.status }
            : f
        ),
      };

    case "SET_LOADING":
      return { ...state, isLoading: action.payload };

    case "SET_UPLOADING":
      return { ...state, isUploading: action.payload };

    case "TOGGLE_SIDEBAR":
      return { ...state, sidebarOpen: !state.sidebarOpen };

    // Explicit set (true = visible, false = collapsed). Used by flows
    // that want a deterministic sidebar state on entry (e.g. journey
    // mount → collapsed, Stage 4 KB handoff into chat → expanded,
    // chat "Return to Stages" → collapsed again).
    case "SET_SIDEBAR":
      return { ...state, sidebarOpen: !!action.payload };

    // Sidebar hover-peek gate. ResolutionJourney enables this on
    // mount and clears it on unmount, so the auto-expand only fires
    // while the engineer is on the blocks screen.
    case "SET_SIDEBAR_HOVER_PEEK":
      return { ...state, sidebarHoverPeekEnabled: !!action.payload };

    case "SET_SIDEBAR_TAB":
      return { ...state, sidebarTab: action.payload };

    // ── Admin / User role toggle ─────────────────────────
    case "SET_USER_ROLE":
      return { ...state, userRole: action.payload };

    case "RESET_ALL":
      return { ...initialState, sidebarOpen: state.sidebarOpen, userRole: state.userRole };

    default:
      return state;
  }
}

export function ChatProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, initialState);
  const value = React.useMemo(() => ({ state, dispatch }), [state]);
  return <ChatContext.Provider value={value}>{children}</ChatContext.Provider>;
}

export function useChat() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChat must be used within ChatProvider");
  return ctx;
}
