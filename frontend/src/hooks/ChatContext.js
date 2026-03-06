import React, { createContext, useContext, useReducer, useCallback } from "react";

const ChatContext = createContext();

const initialState = {
  // Current session
  sessionId: null,
  messages: [],

  // Sidebar data
  sessions: [],
  uploadedFiles: [],

  // UI states
  isLoading: false,
  isUploading: false,
  sidebarOpen: true,
  sidebarTab: "chat", // "chat" | "upload" | "files"
};

function reducer(state, action) {
  switch (action.type) {
    case "SET_SESSION":
      return {
        ...state,
        sessionId: action.payload.id,
        messages: action.payload.messages || [],
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
          },
        ],
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

    case "UPDATE_FILE_STATUS": {
      return {
        ...state,
        uploadedFiles: state.uploadedFiles.map((f) =>
          f.id === action.payload.id
            ? { ...f, status: action.payload.status }
            : f
        ),
      };
    }

    case "SET_LOADING":
      return { ...state, isLoading: action.payload };

    case "SET_UPLOADING":
      return { ...state, isUploading: action.payload };

    case "TOGGLE_SIDEBAR":
      return { ...state, sidebarOpen: !state.sidebarOpen };

    case "SET_SIDEBAR_TAB":
      return { ...state, sidebarTab: action.payload };

    case "RESET_ALL":
      return { ...initialState, sidebarOpen: state.sidebarOpen };

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
