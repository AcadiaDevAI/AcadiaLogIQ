// EscalationUploadDialog — US Pharma "Escalation Procedure" surface.
//
// Replaces Acadia's fixed-PDF vendor-section modal for US Pharma
// (escalation_mode = "upload_chat"). The engineer uploads an escalation
// document — JSON of ANY structure (ingested by the recursive, lossless
// chunker) or a PDF — and, once it's indexed, is dropped straight into the
// chatbot to ask escalation questions against it.
//
// US-Pharma-only: mounted/opened solely by the US Pharma branch in App.js.

import React, { useState } from "react";
import { Modal, Upload, Progress, message } from "antd";
import { InboxOutlined } from "@ant-design/icons";
import { useChat } from "../../hooks/ChatContext";
import { uploadFileV2, getUploadStatus } from "../../services/api";

const { Dragger } = Upload;

export default function EscalationUploadDialog({ open, onClose }) {
  const { dispatch } = useChat();
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [statusText, setStatusText] = useState("");

  const close = () => {
    setBusy(false);
    setProgress(0);
    setStatusText("");
    onClose && onClose();
  };

  // Poll ingestion to completion so the data is searchable before we open the
  // chat. Mirrors UploadPanel.pollJob (status "done" | "failed" | "error").
  const pollJob = async (jobId) => {
    for (let i = 0; i < 120; i++) {
      try {
        const res = await getUploadStatus(jobId);
        const { status, error } = res.data;
        if (status === "done") return { ok: true };
        if (status === "failed" || status === "error") return { ok: false, error };
      } catch {
        /* transient — keep polling */
      }
      await new Promise((r) => setTimeout(r, 1500));
    }
    return { ok: false, error: "timed out" };
  };

  const handleFile = async (file) => {
    setBusy(true);
    setProgress(0);
    setStatusText("Uploading…");
    try {
      const res = await uploadFileV2(file, "kb", (pct) => setProgress(pct));
      const jobId = res?.data?.job_id;
      setStatusText("Indexing…");
      const done = await pollJob(jobId);
      if (!done.ok) {
        message.error(done.error ? `Indexing failed — ${done.error}` : "Indexing failed", 8);
        setBusy(false);
        return;
      }
      message.success("Uploaded — opening chat");
      // Navigate to the chatbot: a general chat over all files (incl. the
      // just-uploaded escalation doc). kbSearchFromLanding surfaces the
      // "Back to screen" button.
      dispatch({ type: "NEW_CHAT", payload: { kbSearchFromLanding: true } });
      close();
    } catch (err) {
      const detail = err?.response?.data?.error || err?.message || "Upload failed";
      message.error(detail);
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
      <p style={{ marginBottom: 12, color: "var(--text-muted, #6b7280)" }}>
        Upload an escalation document — <strong>JSON</strong> (any structure) or{" "}
        <strong>PDF</strong>. It’s indexed automatically and you’ll be taken to
        the chat to ask escalation questions.
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
        <p className="ant-upload-text">Click or drag a JSON / PDF here</p>
        <p className="ant-upload-hint">Accepts .json and .pdf</p>
      </Dragger>
      {busy && (
        <div style={{ marginTop: 14 }}>
          <Progress percent={progress} status="active" />
          <div style={{ marginTop: 6, fontSize: 12, color: "var(--text-muted, #6b7280)" }}>
            {statusText}
          </div>
        </div>
      )}
    </Modal>
  );
}
