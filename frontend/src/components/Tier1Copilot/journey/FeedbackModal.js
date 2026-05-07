// Sprint 13.2 — Shared feedback modal for journey Helpful + Dislike.
//
// Mirrors the chat ChatMessage feedback modal (positive/negative copy,
// 2000-char textarea, Submit/Skip buttons) so an engineer who's used
// the chat interface gets identical behaviour on the journey panels.
// Single component reused by HelpfulButton and DislikeButton — keeps
// modal copy in one place if it ever needs changing.
//
// Submission goes through the existing /feedback/submit SES endpoint.
// The journey caller passes `sessionId` + `stage` so the email lands
// with stage context (no chat-specific message_index — that field
// stays null for journey feedback).

import React, { useState } from "react";
import { Input, Modal, message } from "antd";

import { submitFeedback } from "../../../services/api";

const { TextArea } = Input;


export default function FeedbackModal({
  open,
  variant,            // "like" | "dislike"
  sessionId,
  stage,              // journey stage label (e.g. "stage_3", "pivot_insights")
  onClose,
}) {
  const [feedbackText, setFeedbackText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const isLike = variant === "like";

  const handleSubmit = async () => {
    setSubmitting(true);
    try {
      await submitFeedback({
        // Journey feedback has no message_index — the unit is the
        // stage panel. Stage rides on the answer field as context
        // so the SES email reads naturally without backend schema
        // changes.
        session_id: sessionId || null,
        message_index: null,
        feedback_type: variant,
        feedback_text: feedbackText.trim(),
        question: stage ? `Tier-1 Journey · ${stage}` : "Tier-1 Journey",
        answer: null,
      });
      message.success("Thank you for your feedback!");
      setFeedbackText("");
      onClose?.();
    } catch {
      message.error("Failed to send feedback. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleCancel = () => {
    setFeedbackText("");
    onClose?.();
  };

  return (
    <Modal
      title={isLike ? "👍 Give positive feedback" : "👎 Give negative feedback"}
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      okText="Submit"
      cancelText="Skip"
      confirmLoading={submitting}
      destroyOnClose
    >
      <p className="t-text-muted text-sm mb-3">
        {isLike
          ? "What did you find useful about this stage? (optional)"
          : "What could be improved about this stage? (optional)"}
      </p>
      <TextArea
        rows={4}
        maxLength={2000}
        showCount
        placeholder={
          isLike
            ? "e.g., The historical match summary was spot-on…"
            : "e.g., The pivot signal was confusing…"
        }
        value={feedbackText}
        onChange={(e) => setFeedbackText(e.target.value)}
        autoFocus
      />
    </Modal>
  );
}
