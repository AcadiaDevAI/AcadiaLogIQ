// Sprint 12.8 — Preserve Evidence & Escalate footer.
//
// Always-visible, static instruction rendered at the bottom of every
// Resolution Journey. Pairs with PreliminaryTier1ChecksHeader at the
// top. Verbatim spec text — no paraphrasing, no abbreviating. Amber
// tone (warning, not error) so it reads as a precaution the engineer
// must remember when the SLA clock is ticking, not as a failure.
//
// No data dependency, no LLM, no gating. Always rendered.

import React from "react";
import { Alert } from "antd";
import { ExclamationCircleOutlined } from "@ant-design/icons";


export default function PreserveEvidenceFooter() {
  return (
    <Alert
      type="warning"
      showIcon
      icon={<ExclamationCircleOutlined />}
      style={{ marginBottom: 16 }}
      message={
        <span style={{ fontWeight: 600 }}>
          Preserve Evidence &amp; Escalate
        </span>
      }
      description={
        "If resolution isn't reached within the SLA window, export logs and " +
        "diagnostic dumps before escalating to Tier 2 to prevent data loss " +
        "from reboots or cleared caches."
      }
    />
  );
}
