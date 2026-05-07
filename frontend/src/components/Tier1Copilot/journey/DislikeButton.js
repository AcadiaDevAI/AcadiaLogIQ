// Sprint 13.2 — Dislike button (mirrors chat's 👎 procedure).
//
// Behaviour matches HelpfulButton structurally:
//   1. Click → optimistic UI flip to "Marked needs work".
//   2. POST telemetry event ``disliked_clicked`` so prod monitoring
//      sees stage-level dissatisfaction signals alongside helpful_clicked.
//   3. Open the shared FeedbackModal in negative-feedback variant —
//      identical UX to chat's Dislike modal so an engineer who has
//      used chat doesn't have to learn anything new.
//
// Sits beside HelpfulButton on every stage panel's footer. Doesn't
// interfere with the existing helpful tracking (separate state,
// separate telemetry event). Engineer can mark Helpful AND Dislike
// on the same stage if they want to leave both kinds of feedback —
// the buttons don't lock each other.

import React, { useState } from "react";
import { Button, Space, message } from "antd";
import { CloseCircleFilled, DislikeOutlined } from "@ant-design/icons";

import FeedbackModal from "./FeedbackModal";
import { postJourneyEvent } from "./journeyApi";


export default function DislikeButton({
  sessionId,
  stage,
  onMarkedDisliked,    // optional — parent stepper may track per-stage
  disabled = false,
}) {
  const [submitted, setSubmitted] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);

  const handleClick = async () => {
    if (submitted || disabled) return;
    setSubmitted(true);
    if (typeof onMarkedDisliked === "function") {
      try {
        onMarkedDisliked(stage);
      } catch {
        /* ignore — parent state isn't critical for telemetry */
      }
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

  if (submitted) {
    return (
      <>
        <Space size="small" style={{ marginTop: 4 }}>
          <CloseCircleFilled style={{ color: "#B03A2E", fontSize: 16 }} />
          <span style={{ color: "#B03A2E", fontSize: 13, fontWeight: 500 }}>
            Marked needs work
          </span>
        </Space>
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

  return (
    <Button
      onClick={handleClick}
      icon={<DislikeOutlined />}
      disabled={disabled}
      size="small"
    >
      Dislike
    </Button>
  );
}
