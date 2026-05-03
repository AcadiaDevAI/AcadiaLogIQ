// Sprint 11 — Escalate to Tier 2 button.
//
// Available on every stage so an engineer can short-circuit to the
// escalation package at any point in the journey. Visually demoted
// (default antd type, not "primary") so it doesn't compete with the
// stage's own "advance to next stage" CTA.
//
// Click semantics (mirrors NextStageButton):
//   1. POST tier1_journey_events {fromStage, "next_stage_clicked", to:"stage_5"}
//      — backend's fetch_traversal_log uses this to record "advanced
//      at HH:MM UTC" for the originating stage.
//   2. Call parent's onReveal("stage_5") to mount Stage 5 in the
//      journey panel.
//
// The escalation package's content automatically reflects how far
// the engineer got: stages they viewed are in the traversal log,
// later stages aren't. No backend truncation knob needed — the
// log naturally records only the events that happened.

import React, { useState } from "react";
import { Button, Tooltip } from "antd";
import { ExportOutlined, LoadingOutlined } from "@ant-design/icons";

import { postJourneyEvent } from "./journeyApi";


export default function EscalateButton({
  sessionId,
  fromStage,
  onReveal,
  disabled = false,
  label = "Escalate to Tier 2",
}) {
  const [busy, setBusy] = useState(false);

  const handleClick = async () => {
    if (busy || disabled) return;
    setBusy(true);

    // Telemetry first (fire-and-forget) so the traversal log captures
    // the click even if the reveal fetch fails.
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
    <Tooltip title="Skip the rest of the journey and open the Tier-2 escalation package">
      <Button
        onClick={handleClick}
        disabled={busy || disabled}
        icon={busy ? <LoadingOutlined /> : <ExportOutlined />}
      >
        {label}
      </Button>
    </Tooltip>
  );
}
