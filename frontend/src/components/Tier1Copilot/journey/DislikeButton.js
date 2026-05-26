// DislikeButton — premium pill upgrade (UI only, logic untouched).
//
// LOGIC PRESERVED:
//   1. Click → optimistic UI flip to "Marked needs work"
//   2. POST telemetry disliked_clicked
//   3. Open FeedbackModal (negative variant)
//
// PRESENTATION: pill in muted coral so it reads as "feedback" not
// "danger" — coherent with HelpfulButton, EscalateButton, and the
// landing-page pill bar.

import React, { useState } from "react";
import { message } from "antd";
import { CloseCircleFilled, DislikeOutlined } from "@ant-design/icons";

import FeedbackModal from "./FeedbackModal";
import { postJourneyEvent } from "./journeyApi";


const A = {
  c:         "#E0455F",                        // coral primary
  soft:      "rgba(255, 94, 122, 0.12)",       // idle gradient base
  hoverFill: "rgba(255, 94, 122, 0.24)",       // hover intensifies
  ring:      "rgba(224, 69, 95, 0.55)",        // hover ring
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


export default function DislikeButton({
  sessionId,
  stage,
  onMarkedDisliked,
  disabled = false,
}) {
  const [submitted, setSubmitted] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  // ─── LOGIC — BYTE-IDENTICAL ───────────────────────────────────
  const handleClick = async () => {
    if (submitted || disabled) return;
    setSubmitted(true);
    if (typeof onMarkedDisliked === "function") {
      try { onMarkedDisliked(stage); } catch { /* ignore */ }
    }
    setModalOpen(true);
    try {
      await postJourneyEvent(sessionId, stage, "disliked_clicked");
      message.success("Thanks — captured.", 2);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.dislike] telemetry POST failed", err);
    }
  };

  // ─── SUBMITTED STATE — premium confirmation chip ───────────────
  if (submitted) {
    return (
      <>
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
          <CloseCircleFilled style={{ fontSize: 14 }} />
          Marked needs work
        </span>
        <FeedbackModal
          open={modalOpen}
          variant="dislike"
          sessionId={sessionId}
          stage={stage}
          onClose={() => setModalOpen(false)}
        />
      </>
    );
  }

  // ─── IDLE STATE — pill ─────────────────────────────────────────
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
      <DislikeOutlined style={{ fontSize: 14 }} />
      <span>Dislike</span>
    </button>
  );
}
