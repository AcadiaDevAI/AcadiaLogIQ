// HelpfulButton — premium pill upgrade (UI only, logic untouched).
//
// LOGIC PRESERVED:
//   1. Click → optimistic UI flip + parent onMarkedHelpful callback
//   2. POST telemetry helpful_clicked
//   3. Open FeedbackModal (positive-feedback variant)
//
// PRESENTATION: pill-shaped button with the aurora "good" accent
// (sage-green #46D69A) — coherent with the landing-page pill bar and
// the rest of the journey's premium chrome. Used on every journey
// stage's footer (Stage 0/2/3/4/5) so the upgrade lands everywhere.

import React, { useState } from "react";
import { message } from "antd";
import { CheckCircleFilled, LikeOutlined } from "@ant-design/icons";

import FeedbackModal from "./FeedbackModal";
import { postJourneyEvent } from "./journeyApi";


// ─── Accent palette for the Helpful pill ─────────────────────────────
const A = {
  c:         "#46D69A",                       // primary line/icon
  soft:      "rgba(70, 214, 154, 0.14)",      // idle gradient base
  hoverFill: "rgba(70, 214, 154, 0.28)",      // hover gradient intensifies
  ring:      "rgba(70, 214, 154, 0.55)",      // hover ring
};

const basePillStyle = {
  display: "inline-flex",
  alignItems: "center",
  gap: 6,
  padding: "8px 16px",
  height: 36,
  borderRadius: 9999,
  background: `linear-gradient(135deg, ${A.soft}, rgba(255,255,255,0.40))`,
  backdropFilter: "blur(10px) saturate(150%)",
  WebkitBackdropFilter: "blur(10px) saturate(150%)",
  border: `1px solid ${A.c}40`,
  color: A.c,
  fontFamily:
    "var(--font-body, 'Geist', 'Inter', system-ui, sans-serif)",
  fontSize: 13,
  fontWeight: 600,
  letterSpacing: "0.01em",
  cursor: "pointer",
  whiteSpace: "nowrap",
  boxShadow: "0 1px 3px rgba(10, 16, 24, 0.06)",
  transition:
    "transform 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
    "background 220ms cubic-bezier(0.16, 1, 0.3, 1), " +
    "border-color 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
    "box-shadow 220ms cubic-bezier(0.16, 1, 0.3, 1)",
};


export default function HelpfulButton({
  sessionId,
  stage,
  onMarkedHelpful,
  onStartNewTicket,
  disabled = false,
}) {
  const [submitted, setSubmitted] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  // ─── LOGIC — BYTE-IDENTICAL TO PRE-REVAMP ───────────────────────
  const handleClick = async () => {
    if (submitted || disabled) return;
    setSubmitted(true);
    if (typeof onMarkedHelpful === "function") {
      try { onMarkedHelpful(stage); } catch { /* ignore */ }
    }
    setModalOpen(true);
    try {
      await postJourneyEvent(sessionId, stage, "helpful_clicked");
      message.success("Thanks — captured.", 2);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.helpful] telemetry POST failed", err);
    }
  };

  // ─── SUBMITTED STATE — premium confirmation chip + pill CTA ─────
  if (submitted) {
    return (
      <>
        <div style={{ display: "inline-flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          {/* Confirmation chip in aurora-good tone */}
          <span
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              padding: "6px 12px",
              borderRadius: 9999,
              background: `linear-gradient(135deg, ${A.soft}, rgba(255,255,255,0.50))`,
              border: `1px solid ${A.c}55`,
              color: A.c,
              fontFamily: "var(--font-body, 'Geist', system-ui, sans-serif)",
              fontSize: 12.5,
              fontWeight: 600,
              boxShadow: `0 0 0 1px ${A.c}1A inset`,
            }}
          >
            <CheckCircleFilled style={{ fontSize: 14 }} />
            Marked Helpful
          </span>

          {/* "Start a new ticket" — premium pill, teal accent (matches
              the landing Ticket Filter pill so the icon feels familiar). */}
          {typeof onStartNewTicket === "function" && (
            <StartNewTicketPill onClick={onStartNewTicket} />
          )}
        </div>
        <FeedbackModal
          open={modalOpen}
          variant="like"
          sessionId={sessionId}
          stage={stage}
          onClose={() => setModalOpen(false)}
        />
      </>
    );
  }

  // ─── IDLE STATE — the Helpful pill ──────────────────────────────
  const idleBg = `linear-gradient(135deg, ${A.soft}, rgba(255,255,255,0.40))`;
  const hoverBg = `linear-gradient(135deg, ${A.hoverFill}, ${A.soft})`;

  return (
    <button
      type="button"
      onClick={handleClick}
      disabled={disabled}
      onMouseEnter={(e) => {
        if (disabled) return;
        e.currentTarget.style.transform = "translateY(-1px)";
        e.currentTarget.style.background = hoverBg;
        e.currentTarget.style.borderColor = A.ring;
        e.currentTarget.style.boxShadow =
          `0 8px 22px -8px ${A.ring}, 0 0 0 1px ${A.ring} inset`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.background = idleBg;
        e.currentTarget.style.borderColor = `${A.c}40`;
        e.currentTarget.style.boxShadow = "0 1px 3px rgba(10, 16, 24, 0.06)";
      }}
      style={{
        ...basePillStyle,
        opacity: disabled ? 0.5 : 1,
        cursor: disabled ? "not-allowed" : "pointer",
      }}
    >
      <LikeOutlined style={{ fontSize: 14 }} />
      <span>Helpful</span>
    </button>
  );
}


// ─── Premium "Start a new ticket" pill — teal accent ────────────────
function StartNewTicketPill({ onClick }) {
  const T = {
    c:         "#0BA89F",
    soft:      "rgba(124, 237, 229, 0.18)",
    hoverFill: "rgba(124, 237, 229, 0.34)",
    ring:      "rgba(11, 168, 159, 0.55)",
  };
  const idleBg = `linear-gradient(135deg, ${T.soft}, rgba(255,255,255,0.40))`;
  const hoverBg = `linear-gradient(135deg, ${T.hoverFill}, ${T.soft})`;

  return (
    <button
      type="button"
      onClick={onClick}
      onMouseEnter={(e) => {
        e.currentTarget.style.transform = "translateY(-1px)";
        e.currentTarget.style.background = hoverBg;
        e.currentTarget.style.borderColor = T.ring;
        e.currentTarget.style.boxShadow =
          `0 8px 22px -8px ${T.ring}, 0 0 0 1px ${T.ring} inset`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.background = idleBg;
        e.currentTarget.style.borderColor = `${T.c}40`;
        e.currentTarget.style.boxShadow = "0 1px 3px rgba(10, 16, 24, 0.06)";
      }}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "6px 14px",
        height: 32,
        borderRadius: 9999,
        background: idleBg,
        backdropFilter: "blur(10px) saturate(150%)",
        WebkitBackdropFilter: "blur(10px) saturate(150%)",
        border: `1px solid ${T.c}40`,
        color: T.c,
        fontFamily:
          "var(--font-body, 'Geist', 'Inter', system-ui, sans-serif)",
        fontSize: 12.5,
        fontWeight: 600,
        letterSpacing: "0.01em",
        cursor: "pointer",
        whiteSpace: "nowrap",
        boxShadow: "0 1px 3px rgba(10, 16, 24, 0.06)",
        transition:
          "transform 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "background 220ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "border-color 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "box-shadow 220ms cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      Start a new ticket
    </button>
  );
}
