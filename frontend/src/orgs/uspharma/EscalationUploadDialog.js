// EscalationUploadDialog — US Pharma "Escalation Procedure" surface.
//
// Persistent per-org escalation KB (GET /escalation/status +
// POST /escalation/upload + DELETE /escalation/kb). Role-split behavior:
//
//   • MEMBER  — no dialog. If a document exists, the escalation chatbot
//     opens directly; if none exists, the chat does NOT open (a note tells
//     them to ask an admin).
//
//   • ADMIN   — ALWAYS opens this management dialog (even when a document
//     already exists). The admin can UPLOAD a new document (PDF/JSON) or
//     DELETE the existing one. The "Open Chat" button is enabled only when
//     at least one document exists; clicking it launches the escalation
//     chatbot. With no document, the chat cannot be opened.
//
// Upload/delete are admin-only server-side (require_org_admin) too.
// US-Pharma-only: mounted/opened solely by the US Pharma branch in App.js.

import React, { useEffect, useState } from "react";
import { Modal, Upload, Progress, Spin, Button, Popconfirm, message } from "antd";
import { InboxOutlined, DeleteOutlined, FileTextOutlined } from "@ant-design/icons";
import {
  getEscalationStatus,
  uploadEscalationPdf,
  deleteEscalationKb,
} from "../../components/EscalationProcedure/escalationApi";

const { Dragger } = Upload;

// Read a File as text (Promise wrapper around FileReader) so we can validate
// JSON in the browser before spending a round-trip on the upload.
function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error || new Error("Could not read file"));
    reader.readAsText(file);
  });
}

// Client-side JSON pre-check. Mirrors the backend's parse_json guard so a
// malformed matrix is caught instantly (with a line/column hint) instead of
// bouncing off the server as a 400. Returns null when valid, or a
// human-readable error string when the JSON can't be parsed. Non-JSON files
// (PDF) are skipped — they're validated server-side.
async function validateJsonFile(file) {
  const name = (file?.name || "").toLowerCase();
  if (!name.endsWith(".json")) return null;
  let text;
  try {
    text = await readFileText(file);
  } catch (e) {
    return "Could not read the file. Please try again.";
  }
  // Strip a leading UTF-8 BOM (Windows editors add one), which otherwise
  // makes JSON.parse fail on an invisible character — matches the backend's
  // utf-8-sig decode.
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  if (!text.trim()) return "The JSON file is empty.";
  try {
    JSON.parse(text);
    return null;
  } catch (e) {
    return `Invalid JSON: ${e.message}`;
  }
}

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
  const [status, setStatus] = useState(null); // { ready, filename, updated_at }
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);

  const hasDoc = !!status?.ready;

  const close = () => {
    setChecking(true);
    setStatus(null);
    setBusy(false);
    setProgress(0);
    onClose && onClose();
  };

  const refreshStatus = async () => {
    const s = await getEscalationStatus();
    setStatus(s);
    return s;
  };

  // On open: members are routed straight to the chat (or told to ask an
  // admin when nothing is uploaded) — no dialog UI. Admins always land on
  // the management panel.
  useEffect(() => {
    if (!open) return undefined;
    let cancelled = false;
    setChecking(true);
    setStatus(null);
    setBusy(false);
    setProgress(0);
    (async () => {
      try {
        const s = await getEscalationStatus();
        if (cancelled) return;
        if (!isAdmin) {
          if (s?.ready) {
            openEscalationChat();
          } else {
            message.info(
              "No escalation document is available yet. Please ask an admin to upload one."
            );
          }
          close();
          return;
        }
        // Admin — always show the management panel.
        setStatus(s);
        setChecking(false);
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

  const handleUpload = async (file) => {
    // Pre-flight JSON validation — reject a malformed matrix in the browser
    // with a precise error before uploading, instead of eating a server 400.
    const jsonError = await validateJsonFile(file);
    if (jsonError) {
      message.error(jsonError, 8);
      return;
    }
    setBusy(true);
    setProgress(0);
    try {
      await uploadEscalationPdf(file, {
        onUploadProgress: (e) => {
          if (e.total) setProgress(Math.round((e.loaded / e.total) * 100));
        },
      });
      message.success("Uploaded.");
      await refreshStatus();
    } catch (err) {
      message.error(
        err?.response?.data?.detail || err?.message || "Upload failed",
        8
      );
    } finally {
      setBusy(false);
    }
  };

  const handleDelete = async () => {
    setBusy(true);
    try {
      await deleteEscalationKb();
      message.success("Deleted.");
      await refreshStatus();
    } catch (err) {
      message.error(
        err?.response?.data?.detail || err?.message || "Delete failed",
        8
      );
    } finally {
      setBusy(false);
    }
  };

  // OK / Submit — opens the chatbot, but ONLY when a document exists.
  const handleOpenChat = () => {
    if (!hasDoc) return;
    openEscalationChat();
    close();
  };

  const footer = checking
    ? null
    : [
        <Button key="cancel" onClick={close} disabled={busy}>
          Cancel
        </Button>,
        <Button
          key="ok"
          type="primary"
          disabled={!hasDoc || busy}
          onClick={handleOpenChat}
          title={hasDoc ? "Open the escalation chat" : "Upload a document first"}
        >
          Open Chat
        </Button>,
      ];

  return (
    <Modal
      open={open}
      title="Escalation Procedure — manage document"
      footer={footer}
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
          {/* Current document + delete */}
          {hasDoc ? (
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                gap: 12,
                padding: "10px 12px",
                marginBottom: 14,
                borderRadius: 8,
                border: "1px solid var(--border, #e5e7eb)",
                background: "var(--bg-secondary, #fafafa)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
                <FileTextOutlined style={{ color: "var(--brand-accent, #E31837)" }} />
                <div style={{ minWidth: 0 }}>
                  <div
                    style={{
                      fontWeight: 600,
                      fontSize: 13,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {status.filename || "Escalation document"}
                  </div>
                  {status.updated_at && (
                    <div style={{ fontSize: 11, color: "var(--text-muted, #6b7280)" }}>
                      Uploaded {new Date(status.updated_at).toLocaleString()}
                    </div>
                  )}
                </div>
              </div>
              <Popconfirm
                title="Delete the escalation document?"
                description="The chat won't be available until a new one is uploaded."
                okText="Delete"
                okButtonProps={{ danger: true }}
                onConfirm={handleDelete}
                disabled={busy}
              >
                <Button danger size="small" icon={<DeleteOutlined />} disabled={busy}>
                  Delete
                </Button>
              </Popconfirm>
            </div>
          ) : (
            <p style={{ marginBottom: 14, color: "var(--text-muted, #6b7280)" }}>
              No escalation document uploaded yet. Upload one (PDF or JSON) to
              enable the chat.
            </p>
          )}

          {/* Upload (replace / add) */}
          <Dragger
            accept=".json,.pdf,application/json,application/pdf"
            multiple={false}
            showUploadList={false}
            disabled={busy}
            beforeUpload={(file) => {
              handleUpload(file);
              return false; // prevent AntD's built-in auto-upload
            }}
          >
            <p className="ant-upload-drag-icon">
              <InboxOutlined />
            </p>
            <p className="ant-upload-text">
              {hasDoc
                ? "Click or drag a PDF / JSON file to replace"
                : // "Click or drag a PDF / JSON file to upload"
                  "Click or drag a file to upload"}
            </p>
          </Dragger>
          {busy && <Progress percent={progress} style={{ marginTop: 12 }} />}
        </>
      )}
    </Modal>
  );
}
