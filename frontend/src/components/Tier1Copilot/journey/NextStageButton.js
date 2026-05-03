// Sprint 10 — Next-stage button.
//
// Per spec §2: descriptive label IS the affordance ("Historical
// Matches & Possible Causes ▶", not "Next →"). Click telemetry is
// fired before the parent's onReveal callback runs the fetch.
// Disabled while loading; small spinner replaces the chevron.

import React, { useState } from "react";
import { Button } from "antd";
import { RightOutlined, LoadingOutlined } from "@ant-design/icons";

import { postJourneyEvent } from "./journeyApi";

export default function NextStageButton({
  sessionId,
  fromStage,
  toStage,
  label,
  onReveal,
  disabled = false,
}) {
  const [busy, setBusy] = useState(false);

  const handleClick = async () => {
    if (busy || disabled) return;
    setBusy(true);

    // Fire telemetry first (fire-and-forget — we don't await it
    // before calling onReveal because the user shouldn't wait on it).
    postJourneyEvent(sessionId, fromStage, "next_stage_clicked", { to: toStage })
      .catch((err) => {
        // eslint-disable-next-line no-console
        console.warn("[journey.next_stage] telemetry POST failed", err);
      });

    try {
      if (typeof onReveal === "function") {
        await onReveal(toStage);
      }
    } finally {
      setBusy(false);
    }
  };

  return (
    <Button
      type="primary"
      onClick={handleClick}
      disabled={busy || disabled}
      icon={busy ? <LoadingOutlined /> : <RightOutlined />}
      iconPosition="end"
      style={{ minWidth: 240 }}
    >
      {label}
    </Button>
  );
}
