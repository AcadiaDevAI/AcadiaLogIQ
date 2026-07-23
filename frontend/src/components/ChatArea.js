import React, { useRef, useEffect, useCallback, useState } from "react";
import { message, Button } from "antd";
import { ArrowLeftOutlined } from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { askQuestion, listSessions } from "../services/api";
import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";
import ModeBadge from "./ModeBadge";
import CustomerForm from "./CustomerForm";
import TechForm from "./TechForm";
import PatternResponseCard from "./PatternResponseCard";
import ContextBreakModal from "./ContextBreakModal";

function TypingIndicator() {
  return (
    <div className="flex gap-3 px-4 py-4 md:px-8 lg:px-16 xl:px-24 t-bg-assistant animate-fade-in">
      <div
        className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
        style={{ background: "linear-gradient(135deg, #6366f1, #7c3aed)" }}
      >
        <span className="text-white text-xs font-bold">A</span>
      </div>
      <div className="flex items-center gap-1.5 pt-2">
        <div className="typing-dot" />
        <div className="typing-dot" />
        <div className="typing-dot" />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex items-center justify-center px-4">
      <div className="text-center max-w-md">
        <h2 className="text-xl font-bold t-text mb-2">Acadia Log IQ</h2>
        <p className="t-text-muted text-sm mb-8">
          LogIQ - AI-Assisted Operational Intelligence.
        </p>
      </div>
    </div>
  );
}

export default function ChatArea() {
  const { state, dispatch } = useChat();
  const scrollRef = useRef(null);

  // Sprint 3A — prefill value pushed by post-👍 action chips. ChatInput
  // consumes it on mount-delta and clears via onPrefillConsumed so user
  // edits aren't clobbered. Flag-off: chips never render, so this state
  // stays null for the life of the session.
  const [prefillValue, setPrefillValue] = useState(null);

  // Auth interceptor is now in AuthGate.js — not needed here anymore.

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({
        top: scrollRef.current.scrollHeight,
        behavior: "smooth",
      });
    }
  }, [state.messages, state.isLoading]);

  const handleSend = useCallback(
    async (question) => {
      dispatch({ type: "ADD_USER_MESSAGE", payload: question });
      dispatch({ type: "SET_LOADING", payload: true });

      try {
        const res = await askQuestion(question, state.sessionId);
        const data = res.data;

        dispatch({
          type: "ADD_ASSISTANT_MESSAGE",
          payload: {
            answer: data.answer,
            sources: data.sources || [],
            confidence: data.confidence,
            processing_time_ms: data.processing_time_ms,
            sessionId: data.session_id,
            context_stats: data.context_stats || null,
            needs_clarification: data.needs_clarification,
            clarification_id: data.clarification_id,
            clarification_options: data.clarification_options,
            clarification_context: data.clarification_context,
          },
        });
        dispatch({ type: "SET_LOADING", payload: false });

        // Sprint 2 — surface context-break modal if the backend emitted the hint.
        if (data.context_stats?.context_break === true) {
          dispatch({
            type: "SET_PENDING_CONTEXT_BREAK",
            payload: {
              source: data.context_stats.context_break_source || "",
              category: data.context_stats.context_break_category || "",
              matched_phrase: data.context_stats.context_break_matched_phrase || "",
              active_mode: data.context_stats.context_break_active_mode || "",
              active_sub_mode: data.context_stats.context_break_active_sub_mode || "",
              triggering_query: question,
            },
          });
        }

        // Fire-and-forget sessions refresh — does not block UI or loading state
        listSessions()
          .then((sessRes) =>
            dispatch({ type: "SET_SESSIONS", payload: sessRes.data.sessions || [] })
          )
          .catch(() => {});
      } catch (err) {
        const detail = err?.response?.data?.error || err?.message || "Something went wrong";
        message.error(detail);
        dispatch({
          type: "ADD_ASSISTANT_MESSAGE",
          payload: { answer: `Error: ${detail}. Please try again.`, sources: [], confidence: 0 },
        });
        dispatch({ type: "SET_LOADING", payload: false });
      }
    },
    [state.sessionId, dispatch]
  );

  const handleClarificationSelect = useCallback(
    async (messageIndex, optionId, freeText) => {
      const clarificationMessage = state.messages[messageIndex];
      if (!clarificationMessage || !clarificationMessage.clarificationId) return;

      dispatch({
        type: "SET_CLARIFICATION_SELECTED",
        payload: { index: messageIndex, optionId },
      });

      const picked = clarificationMessage.clarificationOptions?.find(
        (o) => o.id === optionId
      );
      const userFacingText =
        optionId === "opt_other"
          ? freeText
          : picked?.label || "Selected option";

      dispatch({ type: "ADD_USER_MESSAGE", payload: userFacingText });
      dispatch({ type: "SET_LOADING", payload: true });

      try {
        const originalQuery =
          state.messages
            .slice(0, messageIndex)
            .reverse()
            .find((m) => m.role === "user")?.content || "";

        const res = await askQuestion(originalQuery, state.sessionId, {
          clarificationId: clarificationMessage.clarificationId,
          selectedOptionId: optionId,
          freeText: freeText || null,
        });
        const data = res.data;

        dispatch({
          type: "ADD_ASSISTANT_MESSAGE",
          payload: {
            answer: data.answer,
            sources: data.sources || [],
            confidence: data.confidence,
            processing_time_ms: data.processing_time_ms,
            sessionId: data.session_id,
            context_stats: data.context_stats || null,
            needs_clarification: data.needs_clarification,
            clarification_id: data.clarification_id,
            clarification_options: data.clarification_options,
            clarification_context: data.clarification_context,
          },
        });
        dispatch({ type: "SET_LOADING", payload: false });

        if (data.context_stats?.context_break === true) {
          dispatch({
            type: "SET_PENDING_CONTEXT_BREAK",
            payload: {
              source: data.context_stats.context_break_source || "",
              category: data.context_stats.context_break_category || "",
              matched_phrase: data.context_stats.context_break_matched_phrase || "",
              active_mode: data.context_stats.context_break_active_mode || "",
              active_sub_mode: data.context_stats.context_break_active_sub_mode || "",
              triggering_query: userFacingText,
            },
          });
        }

        listSessions()
          .then((sessRes) =>
            dispatch({ type: "SET_SESSIONS", payload: sessRes.data.sessions || [] })
          )
          .catch(() => {});
      } catch (err) {
        const detail = err?.response?.data?.error || err?.message || "Something went wrong";
        message.error(detail);
        dispatch({
          type: "ADD_ASSISTANT_MESSAGE",
          payload: { answer: `Error: ${detail}. Please try again.`, sources: [], confidence: 0 },
        });
        dispatch({ type: "SET_LOADING", payload: false });
      }
    },
    [state.messages, state.sessionId, dispatch]
  );

  // Sprint 2 — pre-message form gating. Only renders when the session
  // has no messages yet AND the selected sub-mode has a form component.
  // Otherwise EmptyState still renders.
  const showSprint2Form =
    state.messages.length === 0 &&
    state.selectedMode === "troubleshooting" &&
    (state.subMode === "customer_specific" ||
      state.subMode === "technology_specific");

  const renderEmptyArea = () => {
    if (!showSprint2Form) {
      return <EmptyState />;
    }
    if (state.subMode === "customer_specific") {
      return <CustomerForm onSeed={handleSend} />;
    }
    if (state.subMode === "technology_specific") {
      return <TechForm onSeed={handleSend} />;
    }
    return <EmptyState />;
  };

  return (
    <div className="flex flex-col h-screen flex-1 t-bg-primary relative">
      <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ zIndex: 0, opacity: 0.05 }}>
        <img src="/logo.png" alt="Acadia Watermark" className="w-[350px] md:w-[420px] lg:w-[500px] object-contain select-none" />
      </div>

      {state.isUploading && (
        <div
          className="px-4 py-2 text-xs flex items-center gap-2 border-b"
          style={{
            backgroundColor: "var(--bg-tertiary)",
            borderColor: "var(--border-color)",
            color: "var(--text-muted)",
            zIndex: 20,
          }}
        >
          <span className="typing-dot" style={{ animationDelay: "0s" }} />
          <span>File upload in progress — you can keep chatting</span>
        </div>
      )}

      {/* US Pharma — KB SOP chat opened from the landing / intake screen.
          A "Back to screen" button returns the engineer to the landing
          screen. Journey Stage 4 chats don't set this flag, so they never
          show it. */}
      {(state.kbSearchFromLanding || state.backToScreen) && (
        <div
          className="px-4 py-2 border-b flex items-center justify-between"
          style={{
            backgroundColor: "var(--bg-tertiary)",
            borderColor: "var(--border-color)",
            zIndex: 20,
          }}
        >
          <span className="text-xs t-text-muted">
            {state.kbScopeStoreId
              ? `Store ${state.kbScopeStoreId} — store-scoped chat`
              : state.kbSearchFromLanding
              ? "Knowledge Base & SOP search — all files"
              : "New chat"}
          </span>
          <Button
            size="small"
            icon={<ArrowLeftOutlined />}
            onClick={() => dispatch({ type: "EXIT_KB_SEARCH" })}
          >
            Back to screen
          </Button>
        </div>
      )}

      <div ref={scrollRef} className="flex-1 overflow-y-auto relative" style={{ zIndex: 10 }}>
        {state.selectedMode && (
          <div className="px-4 pt-3 md:px-8 lg:px-16 xl:px-24">
            <ModeBadge mode={state.selectedMode} subMode={state.subMode} />
          </div>
        )}
        {state.messages.length === 0 ? (
          renderEmptyArea()
        ) : (
          <div className="pb-4">
            {state.messages.map((msg, i) => (
              <React.Fragment key={i}>
                <ChatMessage
                  msg={msg}
                  index={i}
                  sessionId={state.sessionId}
                  onClarificationSelect={handleClarificationSelect}
                  clarificationDisabled={state.isLoading}
                  onPrefillInput={setPrefillValue}
                  onSuggestionClick={handleSend}
                />
                {msg.role === "assistant" &&
                  msg.patternActive &&
                  msg.patternData && (
                    <PatternResponseCard
                      topic={msg.patternTopic}
                      data={msg.patternData}
                    />
                  )}
              </React.Fragment>
            ))}
            {state.isLoading &&
              state.messages.length > 0 &&
              state.messages[state.messages.length - 1]?.role === "user" && (
                <TypingIndicator />
              )}
          </div>
        )}
      </div>

      <ChatInput
        onSend={handleSend}
        prefillValue={prefillValue}
        onPrefillConsumed={() => setPrefillValue(null)}
      />

      <ContextBreakModal />
    </div>
  );
}
