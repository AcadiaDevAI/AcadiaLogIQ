// Sprint 10 — Helpful button.
//
// Sprint 13.2 — UPGRADED to mirror chat's Like procedure. Behaviour:
//   1. Click → optimistic UI flip to "Marked Helpful" + parent
//      `onMarkedHelpful` callback (existing contract preserved).
//   2. POST telemetry event ``helpful_clicked`` (existing contract).
//   3. ALSO open the shared FeedbackModal so the engineer can submit
//      an optional positive comment that emails via SES — the same
//      thing chat's 👍 does.
//
// Pre-13.2 behaviour was telemetry-only (no modal, no email). Every
// caller (Stage 0/2/3/4/5, Pivot Insights, Environment Context) is
// shape-identical — they pass {sessionId, stage, onMarkedHelpful,
// onStartNewTicket, disabled} — so the upgrade is invisible to
// callsites.

import React, { useState } from "react";
import { Button, Space, message } from "antd";
import { CheckCircleFilled, LikeOutlined } from "@ant-design/icons";

import FeedbackModal from "./FeedbackModal";
import { postJourneyEvent } from "./journeyApi";

export default function HelpfulButton({
  sessionId,
  stage,
  onMarkedHelpful,
  onStartNewTicket,
  disabled = false,
}) {
  const [submitted, setSubmitted] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

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
    // Sprint 13.2 — open the positive-feedback modal in parallel
    // with the telemetry POST. The modal's own /feedback/submit call
    // is independent; the engineer can Skip without losing the
    // helpful_clicked signal already in flight.
    setModalOpen(true);
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
      <>
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
