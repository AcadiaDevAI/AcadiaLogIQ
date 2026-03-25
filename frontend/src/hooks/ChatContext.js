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
          };
        }),
      };

    case "NEW_CHAT":
      return { ...state, sessionId: null, messages: [] };

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
      return {
        ...state,
        sessionId: action.payload.sessionId || state.sessionId,
        messages: [
          ...state.messages,
          {
            role: "assistant",
            content: action.payload.answer,
            sources: action.payload.sources || [],
            confidence: action.payload.confidence,
            processingTime: action.payload.processing_time_ms,
            timestamp: new Date().toISOString(),
            feedback: null, // no feedback yet on new messages
          },
        ],
      };

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
