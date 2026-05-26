// EscalateButton — premium pill upgrade (UI only, logic untouched).
//
// LOGIC PRESERVED:
//   1. Click → open EscalationReasonModal
//   2. On modal submit (handleProceed) → telemetry next_stage_clicked
//      + onReveal("stage_5")
//
// PRESENTATION: amber pill — signals "warning / escalate" without
// reading as a destructive action. Lives on every stage's footer.

import React, { useState } from "react";
import { Tooltip } from "antd";
import { ExportOutlined, LoadingOutlined } from "@ant-design/icons";

import EscalationReasonModal from "./EscalationReasonModal";
import { postJourneyEvent } from "./journeyApi";


// ─── Amber escalate palette ───────────────────────────────────────
const A = {
  c:         "#D88A1A",                          // amber primary
  soft:      "rgba(255, 179, 71, 0.16)",         // idle base
  hoverFill: "rgba(255, 179, 71, 0.30)",         // hover intensifies
  ring:      "rgba(216, 138, 26, 0.55)",         // hover ring
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


export default function EscalateButton({
  sessionId,
  fromStage,
  onReveal,
  disabled = false,
  label = "Escalate to Tier 2",
}) {
  const [busy, setBusy] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  // ─── LOGIC — BYTE-IDENTICAL ───────────────────────────────────
  const handleClick = () => {
    if (busy || disabled) return;
    setModalOpen(true);
  };

  const handleProceed = async () => {
    setBusy(true);
    postJourneyEvent(
      sessionId, fromStage, "next_stage_clicked", { to: "stage_5" },
    ).catch((err) => {
      // eslint-disable-next-line no-console
      console.warn("[journey.escalate] telemetry POST failed", err);
    });
    try {
      if (typeof onReveal === "function") {
        await onReveal("stage_5");
      }
    } finally {
      setBusy(false);
    }
  };

  const idleBg = `linear-gradient(135deg, ${A.soft}, rgba(255,255,255,0.40))`;
  const hoverBg = `linear-gradient(135deg, ${A.hoverFill}, ${A.soft})`;
  const isDisabled = busy || disabled;

  return (
    <>
      <Tooltip title="Skip the rest of the journey and open the Tier-2 escalation package">
        <button
          type="button"
          onClick={handleClick}
          disabled={isDisabled}
          onMouseEnter={(e) => {
            if (isDisabled) return;
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
            opacity: isDisabled ? 0.6 : 1,
            cursor: isDisabled ? "not-allowed" : "pointer",
          }}
        >
          {busy ? (
            <LoadingOutlined style={{ fontSize: 14 }} />
          ) : (
            <ExportOutlined style={{ fontSize: 14 }} />
          )}
          <span>{label}</span>
        </button>
      </Tooltip>
      <EscalationReasonModal
        open={modalOpen}
        sessionId={sessionId}
        fromStage={fromStage}
        onClose={() => setModalOpen(false)}
        onProceed={handleProceed}
      />
    </>
  );
}
