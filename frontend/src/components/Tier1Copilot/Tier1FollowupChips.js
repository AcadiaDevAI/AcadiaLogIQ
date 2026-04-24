import React, { useState } from "react";
import { Button, Space, message } from "antd";
import { FOLLOWUP_ACTIONS } from "./tier1Constants";
import { sendFeedback } from "./tier1Api";

/**
 * Tier1FollowupChips — 👍 / 👎 + 5 follow-up actions.
 *
 * After 👍 we show 2 primary CTAs (generate ticket update / start over).
 * After 👎 we show the 5 action chips defined in FOLLOWUP_ACTIONS.
 */
export default function Tier1FollowupChips({
  responseId,
  sessionId,
  onNewAlert,
  // Sprint 7 — parent can intercept each action chip and open a
  // card instead of showing the Sprint-6 "stub" toast. When an
  // onAction handler returns a truthy value, the default stub
  // message is suppressed.
  onAction,
}) {
  const [thumb, setThumb] = useState(null); // "up" | "down" | null

  const postFeedback = async (helpful, followUpAction) => {
    if (!responseId) return;
    try {
      await sendFeedback({
        response_id: responseId,
        helpful,
        follow_up_action: followUpAction || null,
        session_id: sessionId || "tier1-local",
      });
    } catch (err) {
      message.warning("Could not record feedback — the session continues.");
    }
  };

  const handleThumbUp = () => {
    setThumb("up");
    postFeedback(true, null);
  };
  const handleThumbDown = () => {
    setThumb("down");
    postFeedback(false, null);
  };
  const handleAction = (actionKey) => {
    postFeedback(false, actionKey);
    // Sprint 7 — parent may open a card for this action. If the
    // parent handles it (returns truthy), skip the stub toast.
    if (onAction) {
      const handled = onAction(actionKey);
      if (handled) return;
    }
    if (actionKey === "next_best_solution") {
      message.info("Next-best solution will appear on the next submit.");
    } else if (actionKey === "escalation_note") {
      message.info("Escalation note copied to clipboard (stub).");
    } else {
      message.info("Recorded. Feature copy coming in next iteration.");
    }
  };

  if (thumb === "up") {
    return (
      <div className="mt-4 flex justify-center gap-3">
        <Button size="large" onClick={onNewAlert}>
          Start a new alert
        </Button>
      </div>
    );
  }

  if (thumb === "down") {
    return (
      <div className="mt-4">
        <div className="t-text-muted text-xs mb-2 text-center">
          What would help more?
        </div>
        <div className="flex justify-center flex-wrap gap-2">
          {FOLLOWUP_ACTIONS.map((a) => (
            <Button
              key={a.key}
              size="small"
              onClick={() => handleAction(a.key)}
            >
              {a.label}
            </Button>
          ))}
        </div>
        <div className="mt-3 flex justify-center">
          <Button size="large" onClick={onNewAlert}>
            Start a new alert
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="mt-4 flex justify-center">
      <Space>
        <Button size="large" onClick={handleThumbUp}>
          👍 Helpful
        </Button>
        <Button size="large" onClick={handleThumbDown}>
          👎 Not quite
        </Button>
      </Space>
    </div>
  );
}
