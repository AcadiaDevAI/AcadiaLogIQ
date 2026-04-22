import React, { useState } from "react";
import { Modal, Button, message } from "antd";
import { useChat } from "../hooks/ChatContext";
import { resetSessionContext } from "../services/api";

/**
 * ContextBreakModal — surfaced when the backend detects a likely
 * topic/mode switch inside a locked session (PRD §2).
 *
 * Props:
 *   onStartNew?: () => void  — invoked AFTER the session context is
 *                              reset; the parent can use this to push
 *                              the user back to the landing page.
 *
 * Behavior:
 *   - "Continue in this mode" simply dismisses and keeps state.
 *   - "Start new context" calls POST /chat/sessions/{id}/context/reset,
 *     clears ChatContext mode state, dismisses, and lets the parent
 *     return the user to the landing page.
 */
export default function ContextBreakModal({ onStartNew }) {
  const { state, dispatch } = useChat();
  const [working, setWorking] = useState(false);

  const payload = state.pendingContextBreak;
  const open = !!payload;

  const dismiss = () => {
    dispatch({ type: "SET_PENDING_CONTEXT_BREAK", payload: null });
  };

  const handleContinue = () => {
    dismiss();
  };

  const handleStartNew = async () => {
    setWorking(true);
    try {
      if (state.sessionId) {
        try {
          await resetSessionContext(state.sessionId);
        } catch (err) {
          console.warn("[ContextBreakModal] reset failed", err);
          message.warning("Could not fully reset context — trying locally.");
        }
      }
      dispatch({ type: "RESET_MODE_STATE" });
      dismiss();
      if (onStartNew) onStartNew();
    } finally {
      setWorking(false);
    }
  };

  const activeModeLabel = payload?.active_mode
    ? payload.active_mode.replace(/_/g, " ")
    : "";
  const hintPhrase = payload?.matched_phrase || payload?.llm_reason || "";

  return (
    <Modal
      open={open}
      onCancel={handleContinue}
      closable={!working}
      maskClosable={!working}
      footer={[
        <Button key="continue" onClick={handleContinue} disabled={working}>
          Continue in this mode
        </Button>,
        <Button
          key="new"
          type="primary"
          danger
          loading={working}
          onClick={handleStartNew}
          style={{ minWidth: 160 }}
        >
          Start new context
        </Button>,
      ]}
      title="Switch to a new topic?"
    >
      <div className="text-sm t-text">
        <p className="mb-2">
          This looks like a new topic or mode
          {activeModeLabel ? (
            <>
              {" "}
              — we're currently locked in <strong>{activeModeLabel}</strong>.
            </>
          ) : (
            "."
          )}
        </p>
        {hintPhrase && (
          <p className="t-text-muted text-xs mb-2">
            Signal detected:{" "}
            <span className="italic">"{hintPhrase}"</span>
          </p>
        )}
        <p className="t-text-muted text-xs">
          Choose <strong>Continue</strong> to stay in the current context, or{" "}
          <strong>Start new</strong> to clear it and pick a fresh mode.
        </p>
      </div>
    </Modal>
  );
}
