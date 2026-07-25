// EscalationChatWidget — floating bottom-right chat for the Escalation KB.
//
// States:
//   * hidden     — nothing rendered
//   * bubble     — chat-bubble emoji launcher only (after minimize)
//   * expanded   — full chat panel
//
// The widget listens for window event `acadia:open-escalation-chat` with
// detail `{section, label}` (dispatched by EscalationProcedureModal).
// On receipt it sets the active section, opens expanded, and resets the
// transcript so the user starts a fresh thread per leaf pick.

import React, { useEffect, useRef, useState } from "react";
import { Button, Input, Spin, Tooltip, Typography, message } from "antd";
import { CloseOutlined, MinusOutlined, SendOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { askEscalation } from "./escalationApi";
import { useOrg } from "../../hooks/OrgContext";
import { isUSPharma } from "../../orgs/registry";

// Accent gradients + shadows. Acadia keeps the premium blue; US Pharma
// swaps to the org's soft-red "aurora" so the escalation chat matches the
// "Discuss with LogIQ" theme (--aurora-1 = #FF8A9B). Kept as plain hexes
// (not the private uspharma palette) since this is a shared component.
const BLUE_GRADIENT = "linear-gradient(135deg, #1E4FAF 0%, #5B8DEF 100%)";
const BLUE_SHADOW =
  "0 12px 28px -10px rgba(30, 79, 175, 0.55), 0 4px 12px rgba(0,0,0,0.15)";
const USP_RED_GRADIENT =
  "linear-gradient(135deg, #FF8A9B 0%, #E31837 55%, #B00E24 100%)";
const USP_RED_SHADOW =
  "0 12px 28px -10px rgba(227, 24, 55, 0.55), 0 4px 12px rgba(0,0,0,0.15)";


const { Text } = Typography;

// Defensive cleanup: even with strong prompt rules, the model occasionally
// leaks inline page citations or a trailing "Sources:" line. Strip them so the
// rendered bubble stays clean.
const cleanAnswer = (raw) => {
  if (!raw) return "";
  return raw
    // [p.3], [p. 3], [pp.3-4], [p.3, p.4], [P. 3]
    .replace(/\s*\[\s*pp?\.?\s*\d+(?:\s*[-,–]\s*p?\.?\s*\d+)*\s*\]/gi, "")
    // (p.3), (page 3), (pp. 3-4)
    .replace(/\s*\(\s*(?:pp?\.?|pages?)\s*\d+(?:\s*[-,–]\s*\d+)*\s*\)/gi, "")
    // bare "p.3" / "pp.3-4" left dangling
    .replace(/\s*\bpp?\.\s*\d+(?:\s*[-,–]\s*\d+)*\b/gi, "")
    // "on page 3" / "see page 3"
    .replace(/\s*\b(?:on|see|per|from|in)\s+pages?\s+\d+(?:\s*[-,–]\s*\d+)*\b/gi, "")
    // trailing Sources: line
    .replace(/\n+\s*sources?\s*:.*$/is, "")
    // collapse orphan whitespace before punctuation: "when :" -> "when:"
    .replace(/\s+([,.;:!?])/g, "$1")
    // collapse double spaces left behind
    .replace(/[ \t]{2,}/g, " ")
    .trim();
};

const MD_COMPONENTS = {
  p: ({ node, ...props }) => (
    <p style={{ margin: "0 0 8px 0" }} {...props} />
  ),
  ul: ({ node, ...props }) => (
    <ul style={{ margin: "4px 0 8px 0", paddingLeft: 20 }} {...props} />
  ),
  ol: ({ node, ...props }) => (
    <ol style={{ margin: "4px 0 8px 0", paddingLeft: 20 }} {...props} />
  ),
  li: ({ node, ...props }) => (
    <li style={{ margin: "2px 0" }} {...props} />
  ),
  strong: ({ node, ...props }) => (
    <span style={{ fontWeight: 600 }} {...props} />
  ),
  h1: ({ node, ...props }) => (
    <div style={{ fontWeight: 600, margin: "4px 0" }} {...props} />
  ),
  h2: ({ node, ...props }) => (
    <div style={{ fontWeight: 600, margin: "4px 0" }} {...props} />
  ),
  h3: ({ node, ...props }) => (
    <div style={{ fontWeight: 600, margin: "4px 0" }} {...props} />
  ),
};


const LAUNCHER_STYLE = {
  position: "fixed",
  right: 24,
  bottom: 24,
  width: 56,
  height: 56,
  borderRadius: "50%",
  background: BLUE_GRADIENT,
  color: "#fff",
  border: "none",
  fontSize: 26,
  cursor: "pointer",
  boxShadow: BLUE_SHADOW,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  zIndex: 2147483600,
};

const PANEL_STYLE = {
  position: "fixed",
  right: 24,
  bottom: 24,
  width: 380,
  maxWidth: "calc(100vw - 48px)",
  height: 540,
  maxHeight: "calc(100vh - 48px)",
  background: "var(--bg-secondary, #ffffff)",
  borderRadius: 16,
  border: "1px solid var(--border, #e5e7eb)",
  boxShadow: "0 30px 60px -20px rgba(15, 23, 42, 0.35), 0 12px 24px rgba(15, 23, 42, 0.12)",
  display: "flex",
  flexDirection: "column",
  overflow: "hidden",
  zIndex: 2147483600,
};


export default function EscalationChatWidget() {
  const [mode, setMode] = useState("hidden");
  const [section, setSection] = useState(null);
  const [label, setLabel] = useState(null);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [sending, setSending] = useState(false);

  // US Pharma repaints the escalation chat's blue accent to the org's
  // soft-red aurora (matches the "Discuss with LogIQ" theme). Every other
  // org keeps the premium blue.
  const { activeOrg } = useOrg();
  const usp = isUSPharma(activeOrg?.slug);
  const accentBg = usp ? USP_RED_GRADIENT : BLUE_GRADIENT;
  const accentShadow = usp ? USP_RED_SHADOW : BLUE_SHADOW;

  const scrollRef = useRef(null);

  // Listen for the modal -> widget hand-off event.
  useEffect(() => {
    const handler = (event) => {
      const detail = event?.detail || {};
      if (!detail.section) return;
      setSection(detail.section);
      setLabel(detail.label || detail.section);
      setMessages([
        {
          role: "assistant",
          text: `Hi — I'm LogIQ Assistance. I can answer questions about the ${
            detail.label || detail.section
          } section of the Escalation Procedures KB. Ask away.`,
        },
      ]);
      setDraft("");
      setMode("expanded");
    };
    window.addEventListener("acadia:open-escalation-chat", handler);
    return () =>
      window.removeEventListener("acadia:open-escalation-chat", handler);
  }, []);

  // Auto-scroll on new messages.
  useEffect(() => {
    if (mode !== "expanded") return;
    const el = scrollRef.current;
    if (el) {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages, mode, sending]);

  const handleSend = async () => {
    const text = draft.trim();
    if (!text || sending || !section) return;
    const userTurn = { role: "user", text };
    const nextMessages = [...messages, userTurn];
    setMessages(nextMessages);
    setDraft("");
    setSending(true);
    try {
      const history = nextMessages
        .slice(0, -1)
        .filter((m) => m.role === "user" || m.role === "assistant")
        .map((m) => ({ role: m.role, text: m.text }));
      const data = await askEscalation({
        section,
        question: text,
        history,
      });
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: cleanAnswer(data.answer) },
      ]);
    } catch (err) {
      const detail =
        err?.response?.data?.detail || err?.message || "Request failed.";
      message.error(`Chat failed: ${detail}`);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `Sorry — I couldn't answer that. (${detail})`,
        },
      ]);
    } finally {
      setSending(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  if (mode === "hidden") return null;

  if (mode === "bubble") {
    return (
      <Tooltip
        title={`Resume LogIQ Assistance (${label || "Escalation"})`}
        placement="left"
      >
        <button
          type="button"
          aria-label="Open escalation chat"
          style={usp ? { ...LAUNCHER_STYLE, background: accentBg, boxShadow: accentShadow } : LAUNCHER_STYLE}
          onClick={() => setMode("expanded")}
        >
          <span role="img" aria-hidden>
            💬
          </span>
        </button>
      </Tooltip>
    );
  }

  return (
    <div style={PANEL_STYLE} role="dialog" aria-label="Escalation chat">
      {/* Header */}
      <div
        style={{
          padding: "12px 14px",
          background: accentBg,
          color: "#fff",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span role="img" aria-hidden style={{ fontSize: 18 }}>
            💬
          </span>
          <div style={{ lineHeight: 1.2 }}>
            <div style={{ fontWeight: 600, fontSize: 14 }}>
              LogIQ Assistance
            </div>
            <div style={{ fontSize: 11, opacity: 0.85 }}>
              {label ? `${label} · ` : ""}Escalation Procedures KB
            </div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <Button
            size="small"
            type="text"
            icon={<MinusOutlined style={{ color: "#fff" }} />}
            onClick={() => setMode("bubble")}
            aria-label="Minimize"
          />
          <Button
            size="small"
            type="text"
            icon={<CloseOutlined style={{ color: "#fff" }} />}
            onClick={() => setMode("hidden")}
            aria-label="Close"
          />
        </div>
      </div>

      {/* Transcript */}
      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: 12,
          background: "var(--bg-primary, #fafafa)",
        }}
      >
        {messages.map((m, idx) => {
          const isUser = m.role === "user";
          return (
            <div
              key={idx}
              style={{
                display: "flex",
                justifyContent: isUser ? "flex-end" : "flex-start",
                marginBottom: 10,
              }}
            >
              <div
                style={{
                  maxWidth: "82%",
                  padding: "9px 12px",
                  borderRadius: 12,
                  background: isUser
                    ? accentBg
                    : "var(--bg-secondary, #fff)",
                  color: isUser ? "#fff" : "var(--text, #0f172a)",
                  border: isUser
                    ? "none"
                    : "1px solid var(--border, #e5e7eb)",
                  fontSize: 13.5,
                  lineHeight: 1.5,
                  whiteSpace: isUser ? "pre-wrap" : "normal",
                  wordBreak: "break-word",
                }}
              >
                {isUser ? (
                  m.text
                ) : (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={MD_COMPONENTS}
                  >
                    {m.text}
                  </ReactMarkdown>
                )}
              </div>
            </div>
          );
        })}
        {sending && (
          <div style={{ display: "flex", justifyContent: "flex-start" }}>
            <div
              style={{
                padding: "8px 11px",
                borderRadius: 12,
                background: "var(--bg-secondary, #fff)",
                border: "1px solid var(--border, #e5e7eb)",
              }}
            >
              <Spin size="small" />{" "}
              <Text type="secondary" style={{ marginLeft: 6, fontSize: 12 }}>
                Thinking...
              </Text>
            </div>
          </div>
        )}
      </div>

      {/* Composer */}
      <div
        style={{
          padding: 10,
          borderTop: "1px solid var(--border, #e5e7eb)",
          background: "var(--bg-secondary, #fff)",
          display: "flex",
          gap: 8,
        }}
      >
        <Input.TextArea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onKeyDown={handleKeyDown}
          autoSize={{ minRows: 1, maxRows: 4 }}
          placeholder={`Ask about ${label || "this section"}...`}
          disabled={sending}
        />
        <Button
          type="primary"
          icon={<SendOutlined />}
          onClick={handleSend}
          loading={sending}
          disabled={!draft.trim()}
        />
      </div>
    </div>
  );
}
