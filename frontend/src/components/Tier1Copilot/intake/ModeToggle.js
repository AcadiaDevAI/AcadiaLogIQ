import React from "react";
import { Radio, Tooltip } from "antd";
import { INTAKE_MODES } from "../tier1Constants";

/**
 * Sprint 11 — ModeToggle
 *
 * Top-level intake mode picker. Two values:
 *   proactive — alert flow (machine-generated monitoring signal)
 *   reactive  — channel paste (email / phone / portal / chat / note)
 *
 * Replaces the Sprint 9 SourceToggle (which surfaced all six channels
 * at the top level). The five reactive sub-channels now live inside
 * the Reactive panel via REACTIVE_SUB_CHANNELS so the backend
 * extraction templates still get a specific channel string.
 */
export default function ModeToggle({ value, onChange, disabled }) {
  return (
    <Tooltip title="Where is this issue coming from?">
      <Radio.Group
        value={value || "proactive"}
        onChange={(e) => onChange && onChange(e.target.value)}
        optionType="button"
        buttonStyle="solid"
        disabled={disabled}
        size="middle"
        aria-label="Intake mode"
      >
        {INTAKE_MODES.map((m) => (
          <Radio.Button key={m.value} value={m.value}>
            {m.label}
          </Radio.Button>
        ))}
      </Radio.Group>
    </Tooltip>
  );
}
