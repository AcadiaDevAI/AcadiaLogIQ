// EscalationUploadDialog — US Pharma "Escalation Procedure" surface.
//
// Uses the SAME persistent backend as Acadia (GET /escalation/status +
// POST /escalation/upload → per-org kb.json), so the escalation document is
// uploaded ONCE and never needs re-uploading. Two US-Pharma differences from
// Acadia's flow:
//   1. Accepts BOTH PDF and JSON (Acadia is PDF-only). JSON matrices are
//      ingested whole under the catch-all "general" section.
//   2. After a successful upload — or immediately, when a doc already
//      exists — it drops the user straight into the escalation chatbot
//      instead of Acadia's vendor-section picker.
//
// Upload is admin-only: enforced server-side by require_org_admin, and the
// dragger is shown only to admins here. Members land directly in the chat
// (to query a doc an admin already uploaded).
//
// US-Pharma-only: mounted/opened solely by the US Pharma branch in App.js.

import React, { useEffect, useState } from "react";
import { Modal, Upload, Progress, Spin, message } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import {
  getEscalationStatus,
  uploadEscalationPdf,
} from "../../components/EscalationProcedure/escalationApi";

const { Dragger } = Upload;

// Open the shared floating escalation chat (the same widget Acadia hands off
// to). US Pharma has no vendor sections, so we use the catch-all "general"
// section, which searches the whole KB.
function openEscalationChat() {
  window.dispatchEvent(
    new CustomEvent("acadia:open-escalation-chat", {
      detail: { section: "general", label: "Escalation" },
    })
  );
}

export default function EscalationUploadDialog({ open, onClose, isAdmin }) {
  const [checking, setChecking] = useState(true);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);

  const close = () => {
    setChecking(true);
    setBusy(false);
    setProgress(0);
    onClose && onClose();
  };

  // On open, check whether this org already has an escalation doc. If so,
  // skip straight to the chat (no re-upload). Otherwise show the uploader
  // (admins) or an info note (members).
  useEffect(() => {
    if (!open) return undefined;
    let cancelled = false;
    setChecking(true);
    setBusy(false);
    setProgress(0);
    (async () => {
      try {
        const status = await getEscalationStatus();
        if (cancelled) return;
        if (status?.ready) {
          // Already uploaded — open the chat directly, no re-upload.
          openEscalationChat();
          close();
          return;
        }
        if (!isAdmin) {
          message.info(
            "No escalation document has been uploaded yet. Please ask an admin to upload one."
          );
          close();
          return;
        }
        setChecking(false); // admin + not ready → show the uploader
      } catch (err) {
        if (cancelled) return;
        message.error("Could not check the escalation document. Please try again.");
        close();
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open]);

  const handleFile = async (file) => {
    setBusy(true);
    setProgress(0);
    try {
      await uploadEscalationPdf(file, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
        },
      });
      message.success("Uploaded — opening chat");
      openEscalationChat();
      close();
    } catch (err) {
      const detail =
        err?.response?.data?.detail || err?.message || "Upload failed";
      message.error(detail, 8);
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      title="Escalation Procedure — upload document"
      footer={null}
      onCancel={busy ? undefined : close}
      maskClosable={!busy}
      destroyOnClose
    >
      {checking ? (
        <div style={{ textAlign: "center", padding: "28px 0" }}>
          <Spin />
          <p style={{ marginTop: 12, color: "var(--text-muted, #6b7280)" }}>
            Checking escalation document…
          </p>
        </div>
      ) : (
        <>
          <p style={{ marginBottom: 12, color: "var(--text-muted, #6b7280)" }}>
            Upload the escalation document (PDF or JSON). It&apos;s indexed once
            and persists — you won&apos;t need to upload it again. You&apos;ll be
            taken straight to the chat to ask escalation questions.
          </p>
          <Dragger
            accept=".json,.pdf,application/json,application/pdf"
            multiple={false}
            showUploadList={false}
            disabled={busy}
            beforeUpload={(file) => {
              handleFile(file);
              return false; // prevent AntD's built-in auto-upload
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              Click or drag a PDF / JSON file to upload
            </p>
          </Dragger>
          {busy && <Progress percent={progress} style={{ marginTop: 12 }} />}
        </>
      )}
    </Modal>
  );
}
