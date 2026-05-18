// Gap Analysis — entry modal.
//
// Mirrors the RCAEntryModal pattern intentionally so the engineer
// has a consistent mental model across the two report-generation
// flows: enter a ticket number, choose which report flavours you
// want produced, click Generate.
//
// Two checkboxes are exposed:
//   * Gap Analysis Report      (default ON)
//   * Blameless Post-Mortem    (default ON)
// At least one must remain ticked. The selection is forwarded to
// GapAnalysisFlow via the `panels` payload — the flow filters the
// Collapse to render only the requested panels, leaving the other
// untouched even when the backend returned it.
//
// Why no file-upload dragger here (RCA has one)?
// ──────────────────────────────────────────────
// The RCA modal carries an .xlsx/.csv dragger that's currently a UI
// scaffold (backend Excel-parse is pending). The Gap Analysis flow
// does not need that scaffold — keep this modal lean and ship the
// upload affordance only when the parser is actually wired.

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
} from "antd";
import { FileSearchOutlined } from "@ant-design/icons";


const { Paragraph, Text } = Typography;


export default function GapAnalysisEntryModal({ open, onClose, onSubmit }) {
  const [incidentNumber, setIncidentNumber] = useState("");
  // Both checkboxes default ON. The expected UX is "produce both
  // reports" — opting out of one is the rarer path.
  const [gapChecked, setGapChecked] = useState(true);
  const [postMortemChecked, setPostMortemChecked] = useState(true);
  const [error, setError] = useState(null);

  const reset = () => {
    setIncidentNumber("");
    setGapChecked(true);
    setPostMortemChecked(true);
    setError(null);
  };

  const handleCancel = () => {
    reset();
    onClose?.();
  };

  const handleSubmit = () => {
    const inc = incidentNumber.trim();
    if (!inc) {
      setError("Enter a ticket number to continue.");
      return;
    }
    if (!gapChecked && !postMortemChecked) {
      setError(
        "Pick at least one report type "
        + "(Gap Analysis, Post-Mortem, or both).",
      );
      return;
    }
    setError(null);
    // Payload mirrors RCA's shape — `panels` is a flat object with a
    // boolean per panel. GapAnalysisFlow consumes both fields and
    // filters its Collapse accordingly.
    const payload = {
      incidentNumber: inc,
      panels: {
        gapAnalysis: !!gapChecked,
        postMortem: !!postMortemChecked,
      },
    };
    onSubmit?.(payload);
    reset();
    onClose?.();
  };

  return (
    <Modal
      title={
        <Space>
          <FileSearchOutlined style={{ color: "var(--acadia-primary)" }} />
          <span>Generate Gap Analysis</span>
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
        Enter a historical ticket number and pick which reports you
        want produced. Both default ON because the two views are
        complementary — Gap Analysis surfaces the structural breakdown,
        the Blameless Post-Mortem explains the story and the actions.
      </Paragraph>

      <Text strong style={{ display: "block", marginBottom: 6 }}>
        Ticket number
      </Text>
      <Input
        placeholder="e.g. INC-LAN-88902"
        value={incidentNumber}
        onChange={(e) => {
          setIncidentNumber(e.target.value);
          if (error) setError(null);
        }}
        prefix={<FileSearchOutlined style={{ color: "#94a3b8" }} />}
        size="middle"
        aria-label="Incident number"
        onPressEnter={handleSubmit}
        autoFocus
      />

      <Divider style={{ margin: "16px 0" }} />

      <Text strong style={{ display: "block", marginBottom: 8 }}>
        Report types
      </Text>
      <Space direction="vertical" size={4}>
        <Checkbox
          checked={gapChecked}
          onChange={(e) => setGapChecked(e.target.checked)}
        >
          <Text strong>Gap Analysis Report</Text>
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            — Technical, Process & Communication / Silo, with severity scoring
          </Text>
        </Checkbox>
        <Checkbox
          checked={postMortemChecked}
          onChange={(e) => setPostMortemChecked(e.target.checked)}
        >
          <Text strong>Blameless Post-Mortem</Text>
          <Text type="secondary" style={{ marginLeft: 8, fontSize: 12 }}>
            — SRE 13-section narrative report, no blame language
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
