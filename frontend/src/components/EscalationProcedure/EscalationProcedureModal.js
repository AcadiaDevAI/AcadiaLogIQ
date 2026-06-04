// EscalationProcedureModal — entry picker for the Escalation Procedures KB.
//
// Flow:
//   1. On mount, fetch /escalation/status.
//   2. If the KB isn't loaded yet → show a one-time upload dragger.
//      Once any user uploads the consolidated PDF, the server persists
//      it and every later call lands straight in the picker.
//   3. With the KB ready → show three radio cards (OEM Vendor / Telco
//      / Third-party Coordinations). Picking a radio reveals its
//      sub-options inline.
//   4. Clicking a sub-option closes the modal and dispatches
//      `acadia:open-escalation-chat` with `{section, label}` so the
//      floating chat widget mounted at App level opens scoped to that
//      section.

import React, { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Modal,
  Radio,
  Space,
  Spin,
  Tooltip,
  Typography,
  Upload,
  message,
} from "antd";
import { AlertOutlined, InboxOutlined } from "@ant-design/icons";

import {
  ESCALATION_CATEGORIES,
  ESCALATION_KB_FILENAME,
} from "./escalationConstants";
import {
  deleteEscalationKb,
  getEscalationStatus,
  uploadEscalationPdf,
} from "./escalationApi";


const { Paragraph, Text } = Typography;
const { Dragger } = Upload;


// Strip the upload-pipeline job-id prefix (32-hex or UUID-with-hyphens
// followed by "_") and any path segments so the picker shows a clean
// "Escalation_Procedures_KB.pdf" instead of
// "4c28ee09e4cf4fd7bae1400bb56bce0c_Escalation_Procedures_KB.pdf"
// or "tenants/foo/4c28.../Escalation_Procedures_KB.pdf".
const prettifyKbFilename = (raw) => {
  if (!raw) return "—";
  const tail = String(raw).split("/").pop() || raw;
  return tail
    .replace(/^[0-9a-f]{32}_/i, "")
    .replace(/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}_/i, "");
};


export default function EscalationProcedureModal({ open, onClose }) {
  const [loadingStatus, setLoadingStatus] = useState(false);
  const [status, setStatus] = useState(null);
  const [error, setError] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [selectedCategory, setSelectedCategory] = useState(null);

  const refreshStatus = async () => {
    setLoadingStatus(true);
    setError(null);
    try {
      const data = await getEscalationStatus();
      setStatus(data);
      if (!data?.ready && data?.error) {
        setError(data.error);
      }
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          err?.message ||
          "Failed to load Escalation KB status.",
      );
    } finally {
      setLoadingStatus(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    setSelectedCategory(null);
    refreshStatus();
  }, [open]);

  const handleUpload = async (file) => {
    setUploading(true);
    setError(null);
    try {
      const data = await uploadEscalationPdf(file);
      message.success(
        `Escalation KB ready — ${data.total_chunks} chunks indexed across ${
          Object.keys(data.sections || {}).length
        } section(s).`,
      );
      await refreshStatus();
    } catch (err) {
      const detail =
        err?.response?.data?.detail || err?.message || "Upload failed.";
      setError(detail);
      message.error(`Upload failed: ${detail}`);
    } finally {
      setUploading(false);
    }
    // Tell antd not to fire its own POST — we handled it.
    return false;
  };

  const handleDelete = () => {
    Modal.confirm({
      title: "Delete the Escalation KB?",
      content:
        "This removes the indexed KB and any matching PDF on the server. You'll need to upload the PDF again to use the picker.",
      okText: "Delete",
      okButtonProps: { danger: true },
      cancelText: "Cancel",
      onOk: async () => {
        setDeleting(true);
        setError(null);
        try {
          await deleteEscalationKb();
          message.success("Escalation KB deleted.");
          setSelectedCategory(null);
          await refreshStatus();
        } catch (err) {
          const detail =
            err?.response?.data?.detail || err?.message || "Delete failed.";
          setError(detail);
          message.error(`Delete failed: ${detail}`);
        } finally {
          setDeleting(false);
        }
      },
    });
  };

  const handlePickLeaf = (leaf) => {
    if (!leaf) return;
    onClose?.();
    setTimeout(() => {
      window.dispatchEvent(
        new CustomEvent("acadia:open-escalation-chat", {
          detail: { section: leaf.section, label: leaf.label },
        }),
      );
    }, 60);
  };

  const renderUploadStep = () => (
    <div>
      <Alert
        type="info"
        showIcon
        message={`Upload ${ESCALATION_KB_FILENAME} to activate the four scoped chatbots.`}
        description="This is a one-time step. Once uploaded, every user lands straight in the picker on subsequent visits."
        style={{ marginBottom: 16 }}
      />
      <Dragger
        accept=".pdf"
        multiple={false}
        showUploadList={false}
        beforeUpload={handleUpload}
        disabled={uploading}
      >
        <p className="ant-upload-drag-icon">
          <InboxOutlined />
        </p>
        <p className="ant-upload-text">
          {uploading
            ? "Uploading and indexing..."
            : `Click or drag ${ESCALATION_KB_FILENAME} here`}
        </p>
        <p className="ant-upload-hint">
          PDF only. The file is parsed, embedded, and stored on the server so
          every leaf below can answer from its own section.
        </p>
      </Dragger>
      {uploading && (
        <div style={{ marginTop: 16, textAlign: "center" }}>
          <Spin />{" "}
          <Text type="secondary" style={{ marginLeft: 8 }}>
            Parsing and embedding...
          </Text>
        </div>
      )}
      {error && !uploading && (
        <Alert
          type="error"
          showIcon
          message={error}
          style={{ marginTop: 16 }}
        />
      )}
    </div>
  );

  const renderPicker = () => {
    const sectionsReady = status?.sections || {};

    return (
      <div>
        <Paragraph type="secondary" style={{ marginBottom: 16 }}>
          Pick a category, then pick the vendor / partner you want to ask about.
          A chat widget opens at the bottom-right scoped to that vendor's
          section in the Escalation Procedures KB.
        </Paragraph>

        <Radio.Group
          value={selectedCategory}
          onChange={(e) => setSelectedCategory(e.target.value)}
          style={{ display: "flex", flexDirection: "column", gap: 12 }}
        >
          {ESCALATION_CATEGORIES.map((cat) => {
            const isActive = selectedCategory === cat.key;
            return (
              <div
                key={cat.key}
                style={{
                  border: `1px solid ${
                    isActive
                      ? "var(--acadia-primary, #1E4FAF)"
                      : "var(--border, #e5e7eb)"
                  }`,
                  borderRadius: 10,
                  padding: "12px 14px",
                  background: isActive
                    ? "rgba(30, 79, 175, 0.05)"
                    : "transparent",
                  transition: "all 150ms ease",
                }}
              >
                <Radio value={cat.key} style={{ fontWeight: 600 }}>
                  {cat.label}
                </Radio>
                {isActive && (
                  <div
                    style={{
                      marginTop: 10,
                      display: "flex",
                      flexWrap: "wrap",
                      gap: 8,
                      paddingLeft: 26,
                    }}
                  >
                    {cat.leaves.map((leaf) => {
                      const hasChunks = (sectionsReady[leaf.section] || 0) > 0;
                      return (
                        <Button
                          key={leaf.section}
                          type="primary"
                          icon={<AlertOutlined />}
                          onClick={() => handlePickLeaf(leaf)}
                          disabled={!hasChunks}
                          title={
                            hasChunks
                              ? `Open ${leaf.label} chat`
                              : `No ${leaf.label} section was detected in the PDF.`
                          }
                        >
                          {leaf.label}
                        </Button>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </Radio.Group>

        <div
          style={{
            marginTop: 18,
            padding: "10px 12px",
            background: "rgba(0,0,0,0.03)",
            borderRadius: 8,
            fontSize: 12,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 12,
          }}
        >
          <Text type="secondary" style={{ flex: 1, minWidth: 0 }}>
            KB file:{" "}
            <Text code title={status?.filename || ""}>
              {prettifyKbFilename(status?.filename)}
            </Text>{" "}
            · Sections detected:{" "}
            <Text code>{Object.keys(sectionsReady).length}</Text>
          </Text>
          <Tooltip title="Delete KB file">
            <Button
              type="text"
              size="small"
              danger
              onClick={handleDelete}
              loading={deleting}
              aria-label="Delete KB file"
              style={{ fontSize: 16, lineHeight: 1, padding: "0 6px" }}
            >
              <span role="img" aria-hidden>
                🗑️
              </span>
            </Button>
          </Tooltip>
        </div>
      </div>
    );
  };

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      title={
        <Space>
          <AlertOutlined style={{ color: "var(--acadia-primary, #1E4FAF)" }} />
          <span>Escalation Procedure</span>
        </Space>
      }
      width={620}
      destroyOnClose
    >
      {loadingStatus ? (
        <div style={{ textAlign: "center", padding: 40 }}>
          <Spin />
          <div style={{ marginTop: 12 }}>
            <Text type="secondary">Loading Escalation KB...</Text>
          </div>
        </div>
      ) : status?.ready ? (
        renderPicker()
      ) : (
        renderUploadStep()
      )}
    </Modal>
  );
}
