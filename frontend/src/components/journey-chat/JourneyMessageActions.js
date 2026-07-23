// Sprint 10.5 §2.3 — inline journey actions on assistant chat messages.
//
// LOGIC PRESERVED (Sprint 11 + 13.25 contracts intact):
//   - Renders [Return to Stages] + [Escalate to Tier 2] inline beside
//     the per-message Copy / 👍 / 👎 row, gated by `journey_session_id`.
//   - "Return to Stages" → state-based RESUME_JOURNEY dispatch.
//   - "Escalate to Tier 2" → opens EscalationReasonModal; on Submit
//     records two telemetry events in order (escalation_initiated_from_chat,
//     stage_advanced), then dispatches RESUME_JOURNEY which sends the
//     user back to Stage 5.
//
// PRESENTATION UPGRADE (this revision):
//   - Replaced AntD `type="text"` small Buttons with premium pills
//     matching the journey footer aesthetic (HelpfulButton /
//     DislikeButton / EscalateButton style).
//   - Compact size (height 30px) — they sit inside chat-message
//     footer rows next to Copy / Like / Dislike actions; keeping
//     them smaller than the in-journey 36px pills preserves the
//     visual hierarchy.
//   - "Return to Stages" → iris accent (#5B8DEF) — matches the
//     landing-page RCA pill and reads as "navigation back".
//   - "Escalate to Tier 2" → amber accent (#D88A1A) — matches the
//     in-journey EscalateButton so the same action wears the same
//     colour anywhere it appears in the app.
//
// Loading + disabled states preserved.

import React from "react";
import { Tooltip, message as antMessage } from "antd";
import {
  ArrowLeftOutlined,
  ExportOutlined,
  LoadingOutlined,
} from "@ant-design/icons";

import { useChat } from "../../hooks/ChatContext";
import EscalationReasonModal from "../Tier1Copilot/journey/EscalationReasonModal";
import { postJourneyEvent } from "../Tier1Copilot/journey/journeyApi";
import { useOrg } from "../../hooks/OrgContext";
import { isUSPharma } from "../../orgs/registry";


// ─── Premium pill — compact variant for chat-message action rows ─────
// Identical visual language to the journey footer pills, scaled down
// (30px height vs 36px) so they nest cleanly alongside Copy / 👍 / 👎.
function CompactPill({ accent, icon, label, onClick, disabled, loading }) {
  const A = accent;
  const idleBg = `linear-gradient(135deg, ${A.soft}, rgba(255,255,255,0.40))`;
  const hoverBg = `linear-gradient(135deg, ${A.hoverFill}, ${A.soft})`;
  const isDisabled = disabled || loading;

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isDisabled}
      onMouseEnter={(e) => {
        if (isDisabled) return;
        e.currentTarget.style.transform = "translateY(-1px)";
        e.currentTarget.style.background = hoverBg;
        e.currentTarget.style.borderColor = A.ring;
        e.currentTarget.style.boxShadow =
          `0 6px 18px -6px ${A.ring}, 0 0 0 1px ${A.ring} inset`;
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.transform = "translateY(0)";
        e.currentTarget.style.background = idleBg;
        e.currentTarget.style.borderColor = `${A.c}40`;
        e.currentTarget.style.boxShadow = "0 1px 2px rgba(10, 16, 24, 0.05)";
      }}
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        padding: "5px 12px",
        height: 30,
        borderRadius: 9999,
        background: idleBg,
        backdropFilter: "blur(10px) saturate(150%)",
        WebkitBackdropFilter: "blur(10px) saturate(150%)",
        border: `1px solid ${A.c}40`,
        color: A.c,
        fontFamily:
          "var(--font-body, 'Geist', 'Inter', system-ui, sans-serif)",
        fontSize: 12.5,
        fontWeight: 600,
        letterSpacing: "0.01em",
        cursor: isDisabled ? "not-allowed" : "pointer",
        whiteSpace: "nowrap",
        opacity: isDisabled ? 0.6 : 1,
        boxShadow: "0 1px 2px rgba(10, 16, 24, 0.05)",
        transition:
          "transform 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "background 220ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "border-color 180ms cubic-bezier(0.16, 1, 0.3, 1), " +
          "box-shadow 220ms cubic-bezier(0.16, 1, 0.3, 1)",
      }}
    >
      {loading ? (
        <LoadingOutlined style={{ fontSize: 13 }} />
      ) : (
        <span style={{ display: "inline-flex", alignItems: "center", fontSize: 13 }}>
          {icon}
        </span>
      )}
      <span>{label}</span>
    </button>
  );
}


// ─── Accent palettes — match the corresponding in-journey buttons ───
const ACCENT_BACK = {
  c:         "#5B8DEF",                       // iris (RCA pill / navigation)
  soft:      "rgba(91, 141, 239, 0.14)",
  hoverFill: "rgba(91, 141, 239, 0.28)",
  ring:      "rgba(91, 141, 239, 0.55)",
};

const ACCENT_ESCALATE = {
  c:         "#D88A1A",                       // amber (matches EscalateButton)
  soft:      "rgba(255, 179, 71, 0.16)",
  hoverFill: "rgba(255, 179, 71, 0.30)",
  ring:      "rgba(216, 138, 26, 0.55)",
};


export default function JourneyMessageActions({ journeySessionId }) {
  const { dispatch } = useChat();
  const { activeOrg } = useOrg();
  const [escalating, setEscalating] = React.useState(false);
  // Sprint 13.25 — modal-gated escalate
  const [modalOpen, setModalOpen] = React.useState(false);

  // US Pharma renames this action to "Handoff / Escalate".
  const escalateLabel = isUSPharma(activeOrg?.slug)
    ? "Handoff / Escalate"
    : "Escalate to Tier 2";

  if (!journeySessionId) return null;

  // ─── LOGIC — UNCHANGED, plus sidebar auto-collapse ──────────────
  const handleBack = () => {
    // Returning to the journey = entering the "Preliminary Tier 1
    // Checks" surface again. Collapse the sidebar immediately so the
    // transition feels instant; ResolutionJourney's mount effect
    // will reconfirm the same on remount.
    dispatch({ type: "SET_SIDEBAR", payload: false });
    dispatch({
      type: "RESUME_JOURNEY",
      payload: { journeySessionId },
    });
  };

  const handleEscalateClick = () => {
    if (escalating) return;
    setModalOpen(true);
  };

  const handleEscalateProceed = async () => {
    setEscalating(true);
    try {
      await postJourneyEvent(
        journeySessionId,
        "stage_5",
        "escalation_initiated_from_chat",
      );
      await postJourneyEvent(
        journeySessionId,
        "stage_5",
        "stage_advanced",
      );
      dispatch({
        type: "RESUME_JOURNEY",
        payload: { journeySessionId },
      });
    } catch (err) {
      antMessage.error("Could not open escalation. Please try again.");
      setEscalating(false);
    }
    // We do NOT clear escalating in the success path — the dispatch
    // above triggers AppLayout to swap LandingRouter in and unmount.
  };

  return (
    <>
      <Tooltip title="Return to the Resolution Journey panels">
        {/* Tooltip's child must accept refs cleanly — wrap in a span
            so the unstyled <button> works with AntD's Tooltip without
            a forwardRef warning. */}
        <span style={{ display: "inline-flex" }}>
          <CompactPill
            accent={ACCENT_BACK}
            icon={<ArrowLeftOutlined />}
            label="Return to Stages"
            onClick={handleBack}
          />
        </span>
      </Tooltip>
      <Tooltip title="Open the escalation package for Tier-2 handoff">
        <span style={{ display: "inline-flex" }}>
          <CompactPill
            accent={ACCENT_ESCALATE}
            icon={<ExportOutlined />}
            label={escalateLabel}
            onClick={handleEscalateClick}
            loading={escalating}
          />
        </span>
      </Tooltip>
      <EscalationReasonModal
        open={modalOpen}
        sessionId={journeySessionId}
        fromStage="chat"
        onClose={() => setModalOpen(false)}
        onProceed={handleEscalateProceed}
      />
    </>
  );
}
