import React, { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github.css";
import { Tag, Tooltip, Collapse, Modal, Input, message } from "antd";
import { markdownComponents } from "./markdownComponents";
import { settings } from "../config/clientSettings";
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
import { useOrg } from "../hooks/OrgContext";
import { isUSPharma } from "../orgs/registry";
import { saveFeedbackState, submitFeedback } from "../services/api";
import ClarificationOptions from "./ClarificationOptions";
import JourneyMessageActions from "./journey-chat/JourneyMessageActions";

// ─────────────────────────────────────────────────────────────
// Sprint 3A — deterministic post-👍 action chip registry.
//
// Each chip is either:
//   - type: "prefill" → sets chat input to query(identifiers); user
//     reviews and sends manually. No auto-submit (safety gate).
//   - type: "copy"    → copies a reformatted version of the answer to
//     clipboard and shows a toast.
//
// Chip labels/queries use the FIRST identifier extracted from the answer
// text. When needsId=true and no identifier is extractable, the chip is
// hidden. Zero backend involvement — chip generation is pure UI.
// ─────────────────────────────────────────────────────────────

const CHIP_SETS = {
  troubleshooting: [
    { id: "postmortem",  label: "Draft post-mortem",       type: "prefill", needsId: true,
      query: (id) => `Draft a post-mortem for ${id} based on this analysis` },
    { id: "similar",     label: "Find similar incidents",  type: "prefill", needsId: true,
      query: (id) => `Show tickets with similar root cause to ${id}` },
    { id: "resolution",  label: "Copy as resolution note", type: "copy",    needsId: false },
  ],
  ticket_handling: [
    { id: "validate",    label: "Validate ticket fields",  type: "prefill", needsId: true,
      query: (id) => `Validate all fields for ${id} against our ticket standards` },
    { id: "next_steps",  label: "List next steps",         type: "prefill", needsId: true,
      query: (id) => `What are the next steps for ${id}?` },
    { id: "handoff",     label: "Copy as handoff note",    type: "copy",    needsId: false },
  ],
  escalation: [
    { id: "escalate",    label: "Draft escalation email",  type: "prefill", needsId: true,
      query: (id) => `Draft an escalation email for ${id} to senior leadership` },
    { id: "similar_p1",  label: "Find prior P1s",          type: "prefill", needsId: false,
      query: () => `Show historical P1 incidents with similar impact` },
    { id: "brief",       label: "Copy as exec brief",      type: "copy",    needsId: false },
  ],
  vendor_oem: [
    { id: "vendor_case", label: "Draft vendor ticket",     type: "prefill", needsId: true,
      query: (id) => `Reformat ${id} as a vendor support case` },
    { id: "oem_export",  label: "Export as OEM case",      type: "copy",    needsId: false },
    { id: "reproduce",   label: "List reproduction steps", type: "prefill", needsId: true,
      query: (id) => `List exact reproduction steps for ${id}` },
  ],
};

// Sprint 3A-REVISED — only troubleshooting chips are active now.
// Other mode chips wire in when Sprint 3C (escalation), 3D (ticket
// handling), or 3E (vendor_oem) activates their respective backend
// sprint. Chip strings above (CHIP_SETS.ticket_handling, .escalation,
// .vendor_oem) remain defined so each future sprint only has to add
// the mode name here.
// Sprint 3C/3D/3E — escalation, ticket_handling, vendor_oem chips are
// all active alongside troubleshooting.
const CHIP_ACTIVE_MODES = new Set([
  "troubleshooting",
  "escalation",
  "ticket_handling",
  "vendor_oem",
]);

const EXTRACT_ID_RE = /\bINC-[A-Z]+-\d+\b|\bINC-\d+\b|\bALPHA-\d+\b|\bTITAN-\d+\b|\bNEBULA-\d+\b/;

function extractFirstId(answerText) {
  if (!answerText) return null;
  const m = answerText.match(EXTRACT_ID_RE);
  return m ? m[0] : null;
}

function formatForClipboard(chipId, answerText, id) {
  const headers = {
    resolution: `# Resolution Note${id ? ` — ${id}` : ""}\n\n`,
    handoff:    `# Handoff Note${id ? ` — ${id}` : ""}\n\n`,
    brief:      `# Executive Brief${id ? ` — ${id}` : ""}\n\nSeverity: review required\n\n`,
    oem_export: `# OEM Case Submission${id ? ` — Related: ${id}` : ""}\n\n`,
  };
  return (headers[chipId] || "") + (answerText || "");
}

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
export default function ChatMessage({ msg, index, sessionId, onClarificationSelect, clarificationDisabled, onPrefillInput, onSuggestionClick }) {
  const { state, dispatch } = useChat();
  const [copied, setCopied] = useState(false);

  // Feedback state: read from msg.feedback (persisted from backend) or local override
  const feedbackState = msg.feedback || null; // "like" | "dislike" | null

  // Sprint 3A — post-👍 action chips. Shows when "like" feedback exists
  // on an assistant message AND the session mode supports chips.
  const showActionChips = (
    feedbackState === "like"
    && !!state.selectedMode
    && CHIP_ACTIVE_MODES.has(state.selectedMode)   // Sprint 3A-REVISED
    && msg.role === "assistant"
  );
  const _chipSet = showActionChips ? (CHIP_SETS[state.selectedMode] || []) : [];
  const _extractedId = _chipSet.length > 0 ? extractFirstId(msg.content) : null;
  const _visibleChips = _chipSet.filter((c) => !c.needsId || _extractedId);

  const handleChipClick = (chip) => {
    if (chip.type === "prefill") {
      const q = _extractedId ? chip.query(_extractedId) : chip.query();
      onPrefillInput?.(q);
    } else if (chip.type === "copy") {
      const formatted = formatForClipboard(chip.id, msg.content, _extractedId);
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard
          .writeText(formatted)
          .then(() => message.success("Copied to clipboard"))
          .catch(() => message.error("Copy failed — browser blocked clipboard access"));
      } else {
        message.error("Clipboard API not available in this browser");
      }
    }
  };

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
      // Sprint 3B — pull the original user question (the previous message
      // in the transcript) and thread it to the backend so a troubleshooting
      // 👎 can trigger the KB/runbook pivot. Outside troubleshooting mode
      // or with the flag off, backend simply ignores these fields.
      const priorUserMessage = state.messages
        .slice(0, index)
        .reverse()
        .find((m) => m.role === "user");
      const originalQuery = priorUserMessage?.content || null;
      const currentMode = state.selectedMode || null;

      // Pass the semantic cache id (if any) so the backend can invalidate
      // that specific cached row — protecting every subsequent user.
      saveFeedbackState(
        sessionId,
        index,
        "dislike",
        msg.semanticCacheId || null,
        currentMode,
        originalQuery,
      )
        .then((res) => {
          const pivot = res?.data?.pivot;
          if (!pivot) return;
          if (pivot.kind === "kb_guidance" && pivot.answer) {
            dispatch({
              type: "ADD_ASSISTANT_MESSAGE",
              payload: {
                answer: pivot.answer,
                sources: [],
                confidence: 0.7,
                pivot_type: "kb_guidance",
              },
            });
          } else if (pivot.kind === "no_kb_match" && pivot.message) {
            dispatch({
              type: "ADD_ASSISTANT_MESSAGE",
              payload: {
                answer: pivot.message,
                sources: [],
                confidence: 0.5,
                pivot_type: "kb_empty",
              },
            });
          }
          // pivot.kind === "dedupe" → no new message
        })
        .catch(() => {});
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

  // US Pharma — hide sources from EVERYONE (UI-only), including org admins
  // and platform super-admins; Acadia + other orgs are unaffected.
  // Hides BOTH the "Sources (N)" section (allSources → []) AND any inline
  // "[Source: …]" / "(Sources: …)" mentions — plus a trailing "Sources:"
  // line — that the model echoed into the answer.
  const { activeOrg } = useOrg();
  const hideSources = isUSPharma(activeOrg?.slug);

  const stripSourceMentions = (text) =>
    (text || "")
      // trailing "## Source documents" / "## Sources used" markdown section
      // (the KB-search prompt appends this footer to every answer).
      .replace(/\n#{1,6}\s*source\s+documents?\b[\s\S]*$/i, "")
      .replace(/\n#{1,6}\s*sources\s+used\b[\s\S]*$/i, "")
      .replace(/\[\s*sources?\s*:[^\]]*\]/gi, "")
      .replace(/\(\s*sources?\s*:[^)]*\)/gi, "")
      // trailing "Sources:" line (to end of answer)
      .replace(/\n+\s*sources?\s*:.*$/is, "")
      .replace(/[ \t]{2,}/g, " ")
      .trim();

  const allSources = hideSources
    ? []
    : Array.isArray(msg.sources)
    ? msg.sources.filter(Boolean)
    : [];
  const displayContent = hideSources ? stripSourceMentions(msg.content) : msg.content;

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
            : { background: "linear-gradient(135deg, var(--acadia-primary), var(--acadia-primary))" }
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

        {/* Sprint 3B — low-similarity banner. Renders when the backend
            tagged this answer with confidence_band="low" (top ticket-
            history chunk scored below LOW_SIMILARITY_THRESHOLD). Absent
            field → no banner, preserving pre-3B visuals. */}
        {!isUser && msg.confidenceBand === "low" && (
          <div
            role="status"
            style={{
              marginBottom: 10,
              padding: "8px 12px",
              borderRadius: 6,
              border: "1px solid #f59e0b",
              background: "rgba(245, 158, 11, 0.08)",
              color: "#92400e",
              fontSize: 12,
              lineHeight: 1.4,
            }}
          >
            ⚠️ <strong>Low similarity match</strong> — limited historical
            data found. Consider refining your query or providing more
            context.
          </div>
        )}

        <div className="markdown-body">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={settings.RICH_FORMATTING_ENABLED ? [rehypeHighlight] : []}
            components={settings.RICH_FORMATTING_ENABLED ? markdownComponents : undefined}
          >
            {displayContent}
          </ReactMarkdown>
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

        {/* US Pharma (Idea A) — sample-question chips under the store
            orientation opener. Clicking one sends it immediately via the
            chat's normal ask flow (store-scoped by the session). */}
        {!isUser && Array.isArray(msg.suggestions) && msg.suggestions.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            {msg.suggestions.map((q, i) => (
              <button
                key={i}
                type="button"
                onClick={() => !clarificationDisabled && onSuggestionClick?.(q)}
                disabled={clarificationDisabled}
                className="text-xs"
                style={{
                  cursor: clarificationDisabled ? "not-allowed" : "pointer",
                  padding: "6px 12px",
                  borderRadius: 9999,
                  border: "1px solid var(--brand-accent)",
                  background: "var(--brand-light)",
                  color: "var(--brand-accent)",
                  fontWeight: 500,
                  opacity: clarificationDisabled ? 0.6 : 1,
                }}
              >
                {q}
              </button>
            ))}
          </div>
        )}

        {/* Action buttons: Copy | 👍 Like | 👎 Dislike
            Sprint 10.5 §2.4 — JourneyMessageActions appended on the
            right via marginLeft: auto when the chat originated from a
            Stage 4 Search-KB handoff (state.sessionMetadata carries
            journey_session_id). Returns null otherwise so non-journey
            chats render this row identically to before 10.5. */}
        {!isUser && (
          <div className="mt-2 flex items-center gap-3" style={{ flexWrap: "wrap" }}>
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

            <div
              className="message-actions-journey"
              style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}
            >
              <JourneyMessageActions
                journeySessionId={state.sessionMetadata?.journey_session_id}
              />
            </div>
          </div>
        )}

        {/* Sprint 3A — post-👍 action chips. Rendered only when
            showActionChips is true AND at least one chip survives the
            needsId filter. Flag-off path: _visibleChips is [], this block
            renders nothing — pre-3A visuals preserved byte-for-byte. */}
        {!isUser && _visibleChips.length > 0 && (
          <div
            role="group"
            aria-label="Follow-up actions"
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: 6,
              marginTop: 10,
              paddingTop: 8,
              borderTop: "1px solid var(--border-color)",
            }}
          >
            {_visibleChips.map((chip) => (
              <button
                key={chip.id}
                onClick={() => handleChipClick(chip)}
                style={{
                  fontSize: 12,
                  padding: "4px 10px",
                  border: "1px solid var(--border-color)",
                  borderRadius: 16,
                  background: "transparent",
                  cursor: "pointer",
                  color: "var(--text-primary)",
                }}
              >
                {chip.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Feedback Dialog — shown for both like and dislike */}
      <Modal
        centered
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