import React from "react";
import { Radio, Tooltip } from "antd";
import { INTAKE_SOURCES } from "../tier1Constants";

/**
 * Sprint 9 — SourceToggle
 *
 * Top-of-form intake source picker. Default is "alert" (Sprint 6 form
 * unchanged). Switching to email/phone/portal/chat/note shows the
 * UniversalIntakePanel below.
 */
export default function SourceToggle({ value, onChange, disabled }) {
  return (
    <Tooltip title="Where is this issue coming from?">
      <Radio.Group
        value={value || "alert"}
        onChange={(e) => onChange && onChange(e.target.value)}
        optionType="button"
        buttonStyle="solid"
        disabled={disabled}
        size="middle"
        aria-label="Intake source"
      >
        {INTAKE_SOURCES.map((s) => (
          <Radio.Button key={s.value} value={s.value}>
            {s.label}
          </Radio.Button>
        ))}
      </Radio.Group>
    </Tooltip>
  );
}
