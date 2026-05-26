// Sprint 13.25 — Escalation trigger classification modal.
//
// Pops up before any Escalate-to-Tier-2 flow runs (in-journey
// EscalateButton + chat-side JourneyMessageActions). The engineer
// MUST select at least one trigger reason; on submit, the modal
//   1. POSTs the selected reasons to /feedback/submit (same SES
//      route as the Helpful/Dislike flow — lands in the same
//      dev@acadia inbox).
//   2. Runs the parent's `onProceed` callback, which contains the
//      existing escalation flow (telemetry + reveal Stage 5 / chat
//      RESUME_JOURNEY dispatch).
//
// Failure-open on email: a SES error must NOT block the escalation
// — the engineer's primary need is to escalate; the email is
// secondary. We log the failure and proceed.

import React, { useState } from "react";
import { Checkbox, Modal, Space, Typography } from "antd";

import { submitFeedback } from "../../../services/api";


const { Paragraph } = Typography;


// Spec-verbatim reasons. Order preserved from the user's spec so
// the engineer reads them in the intended priority sequence.
// Sprint 13.31 — "Steps attempted but failed" removed at the user's
// request; the corresponding sentence in the backend map has also
// been removed so the note can never surface that wording.
const ESCALATION_REASONS = [
  "No matching historical confidence",
  "Time threshold exceeded",
  "User skipped troubleshooting",
  "Policy-driven escalation (P1 auto-escalate)",
];


// Sprint 13.31 — persist the selected reasons by session_id so the
// Stage 5 handoff-note POST can include them. Both modal entry points
// (in-journey EscalateButton + chat-side JourneyMessageActions) write
// here; ResolutionJourney + Stage5EscalationPackage hydrate from here.
// Failure-open on any storage error — the email still went out and
// the escalation still proceeds; only the note's Reason block degrades.
const _escalationReasonsStorageKey = (sessionId) =>
  sessionId ? `tier1_escalation_reasons_${sessionId}` : null;

function _saveEscalationReasons(sessionId, reasons) {
  const key = _escalationReasonsStorageKey(sessionId);
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(reasons || []));
  } catch {
    /* quota / privacy-mode — ignore */
  }
}


export default function EscalationReasonModal({
  open,
  sessionId,
  fromStage,           // e.g. "stage_3", "stage_4", "chat" — for email subject
  onClose,
  onProceed,           // async () => { ...existing escalation flow... }
}) {
  const [selected, setSelected] = useState([]);
  const [submitting, setSubmitting] = useState(false);

  const canSubmit = selected.length > 0 && !submitting;

  const handleSubmit = async () => {
    if (!canSubmit) return;
    setSubmitting(true);

    // Sprint 13.31 — stash the selection FIRST so the Stage 5
    // handoff-note POST can read it even if the email step throws.
    _saveEscalationReasons(sessionId, selected);

    // 1. Fire-and-forget email via the existing /feedback/submit
    //    endpoint. We `await` it so the engineer briefly sees the
    //    submitting spinner, but we DON'T block the escalation on
    //    its outcome — failure-open by design.
    try {
      await submitFeedback({
        session_id: sessionId || null,
        message_index: null,
        feedback_type: "escalation_trigger",
        feedback_text: `Selected escalation triggers: ${selected.join("; ")}`,
        question: `Tier-1 Journey · ${fromStage || "unknown stage"} · Escalate to Tier 2`,
        answer: null,
      });
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn(
        "[escalation_reason_modal] feedback email failed — proceeding "
        + "with escalation anyway", err,
      );
    }

    // 2. Run the parent's existing escalation flow (telemetry +
    //    reveal Stage 5 / RESUME_JOURNEY).
    try {
      if (typeof onProceed === "function") {
        await onProceed(selected);
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error(
        "[escalation_reason_modal] onProceed failed", err,
      );
    } finally {
      setSubmitting(false);
      // Reset the modal's state so the next open starts clean.
      setSelected([]);
      onClose?.();
    }
  };

  const handleCancel = () => {
    if (submitting) return;
    setSelected([]);
    onClose?.();
  };

  return (
    <Modal
      centered
      title="Why are you escalating to Tier 2?"
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      okText="Submit & Escalate"
      cancelText="Cancel"
      okButtonProps={{ disabled: !canSubmit, loading: submitting }}
      cancelButtonProps={{ disabled: submitting }}
      destroyOnClose
    >
      <Paragraph type="secondary" style={{ marginBottom: 12, fontSize: 13 }}>
        Select one or more triggers (at least one required) so Tier-2
        has context for the handoff. Your selection is also emailed
        to the team for analytics.
      </Paragraph>
      <Checkbox.Group
        value={selected}
        onChange={(vals) => setSelected(vals)}
      >
        <Space direction="vertical" size={6}>
          {ESCALATION_REASONS.map((r) => (
            <Checkbox key={r} value={r}>
              {r}
            </Checkbox>
          ))}
        </Space>
      </Checkbox.Group>
    </Modal>
  );
}
