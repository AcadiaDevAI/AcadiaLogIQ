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
        messages: (action.payload.messages || []).map((msg) => {
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
            // Preserve feedback state from backend ("like", "dislike", or undefined)
            feedback: msg.feedback || null,
            semanticCacheId:
              msg.context_stats?.semantic_cache_id || msg.semanticCacheId || null,
          };
        }),
        // ── Pull mode fields from the session payload if backend
        //    attaches them (Sprint 2 will start populating these).
        //    Safe fall-through to current state when fields are absent.
        selectedMode: action.payload.selected_mode ?? state.selectedMode,
        subMode: action.payload.sub_mode ?? state.subMode,
        conversationContextActive:
          action.payload.conversation_context_active ?? state.conversationContextActive,
        customerName: action.payload.customer_name ?? state.customerName,
        technologyDomain: action.payload.technology_domain ?? state.technologyDomain,
        ticketId: action.payload.ticket_id ?? state.ticketId,
        issueSummary: action.payload.issue_summary ?? state.issueSummary,
        formData: action.payload.form_data ?? state.formData,
      };

    case "NEW_CHAT":
      return {
        ...state,
        sessionId: null,
        messages: [],
        // Guided workflow: NEW_CHAT always returns to the landing page.
        selectedMode: null,
        subMode: null,
        conversationContextActive: false,
        customerName: null,
        technologyDomain: null,
        ticketId: null,
        issueSummary: null,
        formData: null,
        pendingContextBreak: null,
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

    // ── Guided workflow reducer cases (Sprint 1) ────────
    case "SET_MODE":
      return {
        ...state,
        selectedMode: action.payload.selectedMode || null,
        subMode: action.payload.subMode || null,
        conversationContextActive: !!action.payload.selectedMode,
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
