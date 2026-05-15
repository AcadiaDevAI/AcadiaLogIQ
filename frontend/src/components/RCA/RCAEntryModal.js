// Sprint 13.32.6 — RCA entry modal.
//
// Triggered by the sidebar's RCA button (primary, sits below New
// Chat). Collects the inputs needed to drive the right-pane
// RCAFlow:
//
//   * Incident number (free text — e.g. "INC-LAN-88902"), OR
//   * An uploaded .xlsx / .csv (UI complete; backend Excel parsing
//     is a placeholder for this iteration — see RCAFlow comments).
//   * Two checkboxes selecting which RCA panels to render:
//       - Internal RCA (12-section technical)
//       - External RCA (7-section customer-facing)
//     At least one must be ticked.
//
// On submit (click "Generate") the modal calls
// `onSubmit({ incidentNumber, panels, file })` and closes itself.
// The parent (AppLayout) takes that payload, flips the right pane
// to RCAFlow with `initialPayload` set, and lets RCAFlow auto-run
// the request.
//
// Failure modes — all surfaced inline (no toasts):
//   * Neither ticket nor file provided      → "Enter a ticket
//                                              number or upload a
//                                              file."
//   * No panel selected                     → "Pick at least one
//                                              RCA type."
//   * File over size cap or wrong extension → Upload's own
//                                              before-upload check.

import React, { useState } from "react";
import {
  Alert,
  Button,
  Checkbox,
  Divider,
  Input,
  Modal,
  Space,
  Typography,
  Upload,
  message,
} from "antd";
import {
  FileSearchOutlined,
  InboxOutlined,
} from "@ant-design/icons";


const { Paragraph, Text } = Typography;
const { Dragger } = Upload;


// Accepted extensions for the upload zone. Kept narrow on purpose —
// the prompt path expects ticket-shaped data; Excel / CSV is the only
// thing the engineer will reasonably have to hand.
const ACCEPTED_EXT = ".xlsx,.csv";
const MAX_FILE_MB = 10;


export default function RCAEntryModal({ open, onClose, onSubmit }) {
  const [incidentNumber, setIncidentNumber] = useState("");
  const [internalChecked, setInternalChecked] = useState(true);
  const [externalChecked, setExternalChecked] = useState(true);
  const [file, setFile] = useState(null);
  const [error, setError] = useState(null);

  const reset = () => {
    setIncidentNumber("");
    setInternalChecked(true);
    setExternalChecked(true);
    setFile(null);
    setError(null);
  };

  const handleCancel = () => {
    reset();
    onClose?.();
  };

  const handleSubmit = () => {
    const inc = incidentNumber.trim();
    if (!inc && !file) {
      setError("Enter a ticket number or upload a file before generating.");
      return;
    }
    if (!internalChecked && !externalChecked) {
      setError("Pick at least one RCA type (Internal, External, or both).");
      return;
    }
    setError(null);
    const payload = {
      incidentNumber: inc || null,
      panels: {
        internal: !!internalChecked,
        external: !!externalChecked,
      },
      file: file || null,
    };
    onSubmit?.(payload);
    reset();
    onClose?.();
  };

  // Upload's beforeUpload is the validation gate. Returning false
  // prevents AntD from auto-POSTing the file — we want to hand it
  // to the parent on Generate click, not on file drop.
  const beforeUpload = (f) => {
    const lower = (f.name || "").toLowerCase();
    const okExt = lower.endsWith(".xlsx") || lower.endsWith(".csv");
    if (!okExt) {
      message.error("Only .xlsx and .csv files are supported.");
      return Upload.LIST_IGNORE;
    }
    const sizeMb = f.size / (1024 * 1024);
    if (sizeMb > MAX_FILE_MB) {
      message.error(`File is ${sizeMb.toFixed(1)} MB; max is ${MAX_FILE_MB} MB.`);
      return Upload.LIST_IGNORE;
    }
    setFile(f);
    setError(null);
    return false;   // stop auto-upload
  };

  return (
    <Modal
      title={
        <Space>
          <FileSearchOutlined style={{ color: "var(--acadia-primary)" }} />
          <span>Generate RCA</span>
        </Space>
      }
      open={open}
      onCancel={handleCancel}
      footer={[
        <Button key="cancel" onClick={handleCancel}>Cancel</Button>,
        <Button
          key="submit"
          type="primary"
          onClick={handleSubmit}
          style={{
            backgroundColor: "var(--acadia-primary)",
            borderColor: "var(--acadia-primary)",
          }}
        >
          Generate
        </Button>,
      ]}
      destroyOnClose
      width={520}
    >
      <Paragraph type="secondary" style={{ marginBottom: 16, fontSize: 13 }}>
        Provide a historical ticket number, or upload a spreadsheet of
        ticket records. Pick which RCA flavours you want produced.
      </Paragraph>

      <Text strong style={{ display: "block", marginBottom: 6 }}>
        Ticket number
      </Text>
      <Input
        placeholder="e.g. INC-LAN-88902"
        value={incidentNumber}
        onChange={(e) => setIncidentNumber(e.target.value)}
        prefix={<FileSearchOutlined style={{ color: "#94a3b8" }} />}
        size="middle"
        aria-label="Incident number"
        onPressEnter={handleSubmit}
      />

      <Divider style={{ margin: "16px 0", color: "#94a3b8" }} plain>
        OR
      </Divider>

      <Text strong style={{ display: "block", marginBottom: 6 }}>
        Upload spreadsheet (.xlsx, .csv)
      </Text>
      <Dragger
        accept={ACCEPTED_EXT}
        multiple={false}
        beforeUpload={beforeUpload}
        showUploadList={!!file}
        fileList={file ? [{ uid: "1", name: file.name, status: "done" }] : []}
        onRemove={() => {
          setFile(null);
          return true;
        }}
        style={{ padding: 8 }}
      >
        <p className="ant-upload-drag-icon" style={{ marginBottom: 4 }}>
          <InboxOutlined style={{ color: "var(--acadia-primary)" }} />
        </p>
        <p className="ant-upload-text" style={{ fontSize: 13 }}>
          Click or drag a file to this area
        </p>
        <p className="ant-upload-hint" style={{ fontSize: 11 }}>
          .xlsx or .csv only · max {MAX_FILE_MB} MB
        </p>
      </Dragger>

      <Divider style={{ margin: "16px 0" }} />

      <Text strong style={{ display: "block", marginBottom: 8 }}>
        RCA types
      </Text>
      <Space direction="vertical" size={4}>
        <Checkbox
          checked={internalChecked}
          onChange={(e) => setInternalChecked(e.target.checked)}
        >
          <Text strong>Internal RCA</Text>
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            — full 12-section technical, engineering audience
          </Text>
        </Checkbox>
        <Checkbox
          checked={externalChecked}
          onChange={(e) => setExternalChecked(e.target.checked)}
        >
          <Text strong>External RCA</Text>
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            — customer-facing, plain English, externally safe
          </Text>
        </Checkbox>
      </Space>

      {error ? (
        <Alert
          type="error"
          showIcon
          message={error}
          style={{ marginTop: 14 }}
        />
      ) : null}
    </Modal>
  );
}
