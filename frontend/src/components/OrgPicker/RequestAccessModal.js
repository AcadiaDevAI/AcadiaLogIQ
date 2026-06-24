/**
 * RequestAccessModal — antd Modal asking the user to optionally
 * justify a request to join an org they don't currently belong to.
 *
 * Submit → POST /organizations/{slug}/request-access. Shows a toast
 * for success / duplicate / failure and closes on success.
 *
 * The justification field is optional — backend accepts null. We
 * cap at 1000 characters to match the backend Pydantic Field
 * constraint (see backend/tenancy/routes.py AccessRequestBody).
 */

import React, { useState } from "react";
import { Modal, Input, message } from "antd";

import { requestOrganizationAccess } from "../../services/api";

const MAX_JUSTIFICATION = 1000;

export default function RequestAccessModal({ open, org, onClose }) {
  const [justification, setJustification] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const handleClose = () => {
    if (submitting) return;
    setJustification("");
    onClose && onClose();
  };

  const handleSubmit = async () => {
    if (!org?.slug) return;
    setSubmitting(true);
    try {
      const res = await requestOrganizationAccess(
        org.slug,
        justification.trim() || null
      );
      const status = res?.data?.status || "created";
      if (status === "already_member") {
        message.info(`You are already a member of ${org.name}. Refresh to enter.`);
      } else {
        message.success(`Access request sent to ${org.name}. An admin will respond.`);
      }
      setJustification("");
      onClose && onClose();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      const code = detail?.error_code;
      if (code === "duplicate_request") {
        message.warning(
          detail?.message || `You already have a pending request for ${org.name}.`
        );
      } else if (code === "org_not_found") {
        message.error("That organization no longer exists.");
      } else {
        // eslint-disable-next-line no-console
        console.error("[RequestAccessModal] submit failed", err);
        message.error("Could not send the request. Try again in a moment.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title={org ? `Request access to ${org.name}` : "Request access"}
      open={open}
      okText="Send request"
      cancelText="Cancel"
      confirmLoading={submitting}
      onOk={handleSubmit}
      onCancel={handleClose}
      maskClosable={!submitting}
      destroyOnClose
    >
      <p style={{ marginTop: 0 }}>
        An admin of this organization will review your request. You may
        optionally explain why you need access.
      </p>
      <Input.TextArea
        rows={4}
        value={justification}
        maxLength={MAX_JUSTIFICATION}
        showCount
        placeholder="Optional: why do you need access?"
        onChange={(e) => setJustification(e.target.value)}
        disabled={submitting}
      />
    </Modal>
  );
}
