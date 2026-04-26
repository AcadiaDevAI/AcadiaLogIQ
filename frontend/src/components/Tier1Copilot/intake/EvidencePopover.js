import React from "react";
import { Popover, Tag } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";

/**
 * Sprint 9 — EvidencePopover
 *
 * Click an evidence snippet to see the LLM's exact substring quote
 * (helps the engineer trust the extraction).
 */
export default function EvidencePopover({ field, value, snippet }) {
  if (!snippet) {
    return (
      <span style={{ fontSize: 12, color: "var(--text-muted, #94A3B8)" }}>
        {value || "—"}
      </span>
    );
  }
  const content = (
    <div style={{ maxWidth: 320 }}>
      <div style={{ fontSize: 11, color: "#94A3B8", marginBottom: 4 }}>
        Why this {field}?
      </div>
      <div style={{ fontSize: 13, fontStyle: "italic" }}>
        “{snippet}”
      </div>
    </div>
  );
  return (
    <Popover content={content} trigger={["click", "hover"]}>
      <Tag
        icon={<InfoCircleOutlined />}
        style={{ cursor: "help", margin: 0 }}
      >
        {value || "—"}
      </Tag>
    </Popover>
  );
}
