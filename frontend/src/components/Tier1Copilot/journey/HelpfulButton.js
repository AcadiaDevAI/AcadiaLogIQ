// Sprint 10 — Helpful button (telemetry only; never advances stages).
//
// Per spec §2 + §11:
//   - Click → POST /tier1/journey/{sid}/event with event_type=helpful_clicked
//   - Optimistic flip to a green checkmark + brief "Thanks — captured." toast
//   - Reveals an inline "Start a new ticket" affordance
//   - Errors log to console only — never block the user, never advance flow
//
// Exposes `onMarkedHelpful` so the parent stepper can update its
// `helpfulPerStage` state without re-querying the server.

import React, { useState } from "react";
import { Button, Space, message } from "antd";
import { CheckCircleFilled, LikeOutlined } from "@ant-design/icons";

import { postJourneyEvent } from "./journeyApi";

export default function HelpfulButton({
  sessionId,
  stage,
  onMarkedHelpful,
  onStartNewTicket,
  disabled = false,
}) {
  const [submitted, setSubmitted] = useState(false);

  const handleClick = async () => {
    if (submitted || disabled) return;
    // Optimistic UI flip — never block on the network round-trip.
    setSubmitted(true);
    if (typeof onMarkedHelpful === "function") {
      try {
        onMarkedHelpful(stage);
      } catch {
        /* ignore — parent state isn't critical for telemetry */
      }
    }
    try {
      await postJourneyEvent(sessionId, stage, "helpful_clicked");
      message.success("Thanks — captured.", 2);
    } catch (err) {
      // Telemetry failures are non-blocking by design.
      // eslint-disable-next-line no-console
      console.warn("[journey.helpful] telemetry POST failed", err);
    }
  };

  if (submitted) {
    return (
      <Space size="small" style={{ marginTop: 4 }}>
        <CheckCircleFilled style={{ color: "#0A7A3F", fontSize: 16 }} />
        <span style={{ color: "#0A7A3F", fontSize: 13, fontWeight: 500 }}>
          Marked Helpful
        </span>
        {typeof onStartNewTicket === "function" && (
          <Button
            type="link"
            size="small"
            onClick={onStartNewTicket}
            style={{ padding: 0 }}
          >
            Start a new ticket
          </Button>
        )}
      </Space>
    );
  }

  return (
    <Button
      onClick={handleClick}
      icon={<LikeOutlined />}
      disabled={disabled}
      size="small"
    >
      Helpful
    </Button>
  );
}
