import React from "react";
import { Modal } from "antd";

/**
 * Sprint 7 — StuckDetectionModal
 * Fires when the session state's stuck_nudge flag flips to true.
 * Two choices: build escalation package, or dismiss and keep
 * troubleshooting. Either choice marks the session's
 * stuck_nudge_shown=true server-side (via the status endpoint) so
 * this modal never re-fires in the same session.
 */
export default function StuckDetectionModal({
  open,
  elapsedSeconds = 0,
  onEscalate,
  onDismiss,
}) {
  const minutes = Math.max(1, Math.round(elapsedSeconds / 60));
  return (
    <Modal
      open={open}
      onCancel={onDismiss}
      onOk={onEscalate}
      okText="Build Escalation Package"
      cancelText="Keep Troubleshooting"
      title="Need a hand?"
      closable
    >
      <p className="t-text text-sm">
        You&apos;ve been on this issue for {minutes} minute
        {minutes === 1 ? "" : "s"}. Want to escalate with what
        you&apos;ve gathered so far?
      </p>
    </Modal>
  );
}
