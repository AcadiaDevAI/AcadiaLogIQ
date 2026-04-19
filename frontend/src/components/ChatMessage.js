import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { Tag, Tooltip, Collapse, Modal, Input, message } from "antd";
import {
  UserOutlined,
  RobotOutlined,
  FileTextOutlined,
  ClockCircleOutlined,
  ThunderboltOutlined,
  CopyOutlined,
  CheckOutlined,
  LikeOutlined,
  LikeFilled,
  DislikeOutlined,
  DislikeFilled,
} from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { saveFeedbackState, submitFeedback } from "../services/api";
import ClarificationOptions from "./ClarificationOptions";

const { TextArea } = Input;

/**
 * ChatMessage
 *
 * Feedback flow:
 *   👍 Like → icon turns green + "Give positive feedback" dialog opens
 *   👎 Dislike → icon turns red + "Give negative feedback" dialog opens
 *   Both dialogs: user types optional message (up to 1200 chars) → sent via SES email
 *   Like/dislike state is persisted in backend session → survives refresh/sign-out
 */
export default function ChatMessage({ msg, index, sessionId, onClarificationSelect, clarificationDisabled }) {
  const { dispatch } = useChat();
  const [copied, setCopied] = useState(false);

  // Feedback state: read from msg.feedback (persisted from backend) or local override
  const feedbackState = msg.feedback || null; // "like" | "dislike" | null

  // Modal state
  const [showModal, setShowModal] = useState(false);
  const [modalType, setModalType] = useState(null); // "like" | "dislike"
  const [feedbackText, setFeedbackText] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const isUser = msg.role === "user";

  const handleCopy = () => {
    navigator.clipboard.writeText(msg.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // ── Handle Like click ──────────────────────────────────
  const handleLike = () => {
    if (feedbackState === "like") return; // already liked

    // Update local state immediately
    dispatch({ type: "SET_MESSAGE_FEEDBACK", payload: { index, feedback: "like" } });

    // Persist to backend (so it survives refresh/sign-out)
    if (sessionId) {
      saveFeedbackState(
        sessionId,
        index,
        "like",
        msg.semanticCacheId || null,
      ).catch(() => {});
    }

    // Open positive feedback dialog
    setModalType("like");
    setFeedbackText("");
    setShowModal(true);
  };

  // ── Handle Dislike click ───────────────────────────────
  const handleDislike = () => {
    if (feedbackState === "dislike") return; // already disliked

    dispatch({ type: "SET_MESSAGE_FEEDBACK", payload: { index, feedback: "dislike" } });

    if (sessionId) {
      // Pass the semantic cache id (if any) so the backend can invalidate
      // that specific cached row — protecting every subsequent user.
      saveFeedbackState(
        sessionId,
        index,
        "dislike",
        msg.semanticCacheId || null,
      ).catch(() => {});
    }

    // Open negative feedback dialog
    setModalType("dislike");
    setFeedbackText("");
    setShowModal(true);
  };

  // ── Submit feedback message ────────────────────────────
  const handleSubmitFeedback = async () => {
    setSubmitting(true);
    try {
      await submitFeedback({
        session_id: sessionId || null,
        message_index: index,
        feedback_type: modalType,
        feedback_text: feedbackText.trim(),
        question: msg._question || null,
        answer: msg.content?.substring(0, 500) || null,
      });
      message.success("Thank you for your feedback!");
      setShowModal(false);
      setFeedbackText("");
    } catch {
      message.error("Failed to send feedback. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  // ── Close modal (skip feedback text, icon state already saved) ──
  const handleCancelModal = () => {
    setShowModal(false);
    setFeedbackText("");
    // Icon state is already persisted — closing just skips the text
  };

  const confidenceColor = (c) => {
    if (c >= 0.8) return "#10b981";
    if (c >= 0.6) return "#3b82f6";
    if (c >= 0.4) return "#f59e0b";
    return "#ef4444";
  };
  const confidenceLabel = (c) => {
    if (c >= 0.8) return "High";
    if (c >= 0.6) return "Good";
    if (c >= 0.4) return "Medium";
    return "Low";
  };

  const allSources = Array.isArray(msg.sources) ? msg.sources.filter(Boolean) : [];

  // Modal title and placeholder change based on like vs dislike
  const isLikeModal = modalType === "like";

  return (
    <div
      className="flex gap-3 px-4 py-4 md:px-8 lg:px-16 xl:px-24 animate-slide-up"
      style={{ backgroundColor: isUser ? "transparent" : "var(--msg-assistant-bg)" }}
    >
      {/* Avatar */}
      <div
        className="flex-shrink-0 w-8 h-8 rounded-lg flex items-center justify-center"
        style={
          isUser
            ? { backgroundColor: "var(--brand-light)", border: "1px solid var(--brand-accent)" }
            : { background: "linear-gradient(135deg, #0A3F63, #0A3F63)" }
        }
      >
        {isUser ? (
          <UserOutlined style={{ color: "var(--brand-accent)", fontSize: 12 }} />
        ) : (
          <RobotOutlined style={{ color: "#fff", fontSize: 18 }} />
        )}
      </div>

      {/* Content */}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-xs font-semibold t-text-secondary">
            {isUser ? "You" : "Acadia AI"}
          </span>
          {msg.timestamp && (
            <span className="text-[10px] t-text-faint">
              {new Date(msg.timestamp).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
            </span>
          )}
          {!isUser && msg.processingTime && (
            <Tooltip title="Processing time">
              <span className="text-[10px] t-text-faint flex items-center gap-0.5">
                <ClockCircleOutlined />
                {(msg.processingTime / 1000).toFixed(1)}s
              </span>
            </Tooltip>
          )}
          {!isUser && msg.confidence !== undefined && (
            <Tooltip title={`Confidence: ${(msg.confidence * 100).toFixed(0)}%`}>
              <span className="text-[10px] flex items-center gap-0.5" style={{ color: confidenceColor(msg.confidence) }}>
                <ThunderboltOutlined />
                {confidenceLabel(msg.confidence)}
              </span>
            </Tooltip>
          )}
        </div>

        <div className="markdown-body">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
        </div>

        {!isUser && msg.needsClarification && msg.clarificationOptions && (
          <ClarificationOptions
            options={msg.clarificationOptions}
            selectedId={msg.clarificationSelectedId}
            disabled={!!clarificationDisabled}
            onSelect={(optionId, freeText) => {
              if (typeof onClarificationSelect === "function") {
                onClarificationSelect(index, optionId, freeText);
              }
            }}
          />
        )}

        {!isUser && allSources.length > 0 && (
          <Collapse
            ghost size="small" className="mt-3"
            items={[{
              key: "sources",
              label: (
                <span className="text-xs font-medium" style={{ color: "var(--brand-accent)" }}>
                  <FileTextOutlined className="mr-1" />Sources ({allSources.length})
                </span>
              ),
              children: (
                <div className="flex flex-wrap gap-1.5">
                  {allSources.map((s, i) => (
                    <Tag key={i} color="blue" className="text-[10px] rounded-md">{s}</Tag>
                  ))}
                </div>
              ),
            }]}
          />
        )}

        {/* Action buttons: Copy | 👍 Like | 👎 Dislike */}
        {!isUser && (
          <div className="mt-2 flex items-center gap-3">
            <Tooltip title={copied ? "Copied!" : "Copy"}>
              <button onClick={handleCopy} style={{ cursor: "pointer", background: "none", border: "none", padding: 0, color: "var(--text-faint)" }} className="text-xs">
                {copied ? <CheckOutlined style={{ color: "#10b981" }} /> : <CopyOutlined />}
              </button>
            </Tooltip>

            {/* Like — green when active, opens positive feedback dialog */}
            <Tooltip title={feedbackState === "like" ? "You liked this" : "Like"}>
              <button
                onClick={handleLike}
                style={{
                  cursor: feedbackState === "like" ? "default" : "pointer",
                  background: "none", border: "none", padding: 0,
                  color: feedbackState === "like" ? "#10b981" : "var(--text-faint)",
                }}
                className="text-sm"
              >
                {feedbackState === "like" ? <LikeFilled /> : <LikeOutlined />}
              </button>
            </Tooltip>

            {/* Dislike — red when active, opens negative feedback dialog */}
            <Tooltip title={feedbackState === "dislike" ? "You disliked this" : "Dislike"}>
              <button
                onClick={handleDislike}
                style={{
                  cursor: feedbackState === "dislike" ? "default" : "pointer",
                  background: "none", border: "none", padding: 0,
                  color: feedbackState === "dislike" ? "#ef4444" : "var(--text-faint)",
                }}
                className="text-sm"
              >
                {feedbackState === "dislike" ? <DislikeFilled /> : <DislikeOutlined />}
              </button>
            </Tooltip>
          </div>
        )}
      </div>

      {/* Feedback Dialog — shown for both like and dislike */}
      <Modal
        title={isLikeModal ? "👍 Give positive feedback" : "👎 Give negative feedback"}
        open={showModal}
        onOk={handleSubmitFeedback}
        onCancel={handleCancelModal}
        okText="Submit"
        cancelText="Skip"
        confirmLoading={submitting}
      >
        <p className="t-text-muted text-sm mb-3">
          {isLikeModal
            ? "What did you like about this response? (optional)"
            : "What could be improved about this response? (optional)"}
        </p>
        <TextArea
          rows={4}
          maxLength={2000}
          showCount
          placeholder={
            isLikeModal
              ? "e.g., The answer was accurate and well-structured..."
              : "e.g., The answer missed important details about..."
          }
          value={feedbackText}
          onChange={(e) => setFeedbackText(e.target.value)}
          autoFocus
        />
      </Modal>
    </div>
  );
}