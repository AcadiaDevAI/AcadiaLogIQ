// Sprint 12.8 — Preliminary Tier 1 Checks header.
//
// Always-visible, static informational card rendered at the top of
// every Resolution Journey. The four checks below are the
// pre-flight protocol Tier 1 must complete on every ticket
// regardless of cohort data — surfaced verbatim per the spec so the
// engineer sees them on every load (no data dependency, no LLM, no
// gating). Pairs with PreserveEvidenceFooter at the bottom of the
// journey container.

import React from "react";
import { Card, Typography } from "antd";
import { SafetyCertificateOutlined } from "@ant-design/icons";

const { Title, Paragraph, Text } = Typography;


// Bullets are kept as data so the spec text lives in one place and
// the JSX render stays mechanical. Keep the strings byte-identical
// to the spec — no abbreviating, no paraphrasing.
const CHECKS = [
  {
    label: "Define Impact",
    body:
      "Identify if the issue is Isolated (single user), Localized (site-wide), " +
      "or Global (systemic) to set the correct priority level and notification " +
      "chain.",
  },
  {
    label: "Validate via Change Logs",
    body:
      "Cross-reference the incident start time with the Central Change " +
      "Management records to identify recent patches, config updates, or " +
      "hardware swaps as the likely root cause.",
  },
  {
    label: "Verify Basic Connectivity",
    body:
      "Conduct \"Sanity Checks\" (Ping, Tracert, DNS Lookup) to confirm the " +
      "network path is clear and that credentials/accounts are not simply " +
      "locked or expired.",
  },
  {
    label: "Monitor Resource & Service Health",
    body:
      "Check for Resource Exhaustion (CPU/RAM >90%) or stopped critical " +
      "services; perform a single manual restart of non-system-critical " +
      "processes if indicated by the SOP.",
  },
];


export default function PreliminaryTier1ChecksHeader() {
  return (
    <Card
      style={{
        marginBottom: 16,
        borderLeft: "4px solid #1F6FEB",
        background: "var(--surface-2, #f7faff)",
      }}
    >
      <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>
        <SafetyCertificateOutlined style={{ marginRight: 8 }} />
        Preliminary Tier 1 Checks
      </Title>
      <Paragraph type="secondary" style={{ marginBottom: 12, fontSize: 12 }}>
        Complete these pre-flight checks on every ticket before drilling
        into the cohort findings below.
      </Paragraph>
      <ol style={{ marginBottom: 0, paddingLeft: 20 }}>
        {CHECKS.map((c) => (
          <li key={c.label} style={{ marginBottom: 6 }}>
            <Text strong>{c.label}:</Text> {c.body}
          </li>
        ))}
      </ol>
    </Card>
  );
}
