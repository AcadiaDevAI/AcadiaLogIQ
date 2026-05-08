// Sprint 11 — Escalate to Tier 2 button.
//
// Available on every stage so an engineer can short-circuit to the
// escalation package at any point in the journey. Visually demoted
// (default antd type, not "primary") so it doesn't compete with the
// stage's own "advance to next stage" CTA.
//
// Sprint 13.25 — clicks now open EscalationReasonModal first; the
// telemetry + reveal flow runs only after the engineer selects at
// least one trigger reason and clicks Submit. Cancel from the modal
// keeps the engineer where they were (no telemetry, no reveal). The
// onClick contract for the parent caller is unchanged.

import React, { useState } from "react";
import { Button, Tooltip } from "antd";
import { ExportOutlined, LoadingOutlined } from "@ant-design/icons";

import EscalationReasonModal from "./EscalationReasonModal";
import { postJourneyEvent } from "./journeyApi";


export default function EscalateButton({
  sessionId,
  fromStage,
  onReveal,
  disabled = false,
  label = "Escalate to Tier 2",
}) {
  const [busy, setBusy] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const handleClick = () => {
    if (busy || disabled) return;
    setModalOpen(true);
  };

  // Sprint 13.25 — `handleProceed` is what runs AFTER the engineer
  // submits the trigger-classification modal. Body is exactly the
  // pre-13.25 click handler — telemetry first (fire-and-forget),
  // then reveal. The modal handles email + close itself.
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

  return (
    <>
      <Tooltip title="Skip the rest of the journey and open the Tier-2 escalation package">
        <Button
          onClick={handleClick}
          disabled={busy || disabled}
          icon={busy ? <LoadingOutlined /> : <ExportOutlined />}
        >
          {label}
        </Button>
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
