// Sprint 12.8 — Preliminary Tier 1 Checks header.
//
// Static informational card rendered at the top of every Resolution
// Journey. The four checks below are the pre-flight protocol Tier 1
// must complete on every ticket regardless of cohort data — surfaced
// verbatim per the spec.
//
// Sprint 13.5 — collapsed by default with a down-chevron expand
// affordance, so the panel takes minimal vertical space on first
// paint. Engineer can expand to read the full checklist; the title +
// "expand" hint remain visible at all times so the panel never
// disappears entirely. No data dependency, no LLM, no gating.

import React, { useState } from "react";
import { Card, Typography } from "antd";
import { DownOutlined, RightOutlined, SafetyCertificateOutlined } from "@ant-design/icons";

import { useTheme } from "../../../hooks/ThemeContext";

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
  // Sprint 13.5 — collapsed by default. Engineer clicks the header
  // (or chevron) to expand. We use plain useState rather than AntD's
  // Collapse so the title row + chevron can act as a single click
  // target with the spec's exact title + intro copy.
  const [expanded, setExpanded] = useState(false);

  // Sprint 13.32.13 — dark-theme fix. The card's inline background
  // was pinned to `var(--surface-2, #f7faff)`; `--surface-2` is not
  // defined in this app's theme system, so the fallback (#f7faff —
  // a blue-tinted near-white) applied in EVERY theme, leaving the
  // panel white on dark mode with text that swaps light → invisible.
  // We now branch on isDark: light mode keeps the original #f7faff
  // exactly (zero visual change), dark mode picks --bg-tertiary
  // (#1e1e28) so the panel sits as an elevated dark surface above
  // the page bg. Nothing else about this component changed.
  //
  // Premium revamp — pale-blue background for the pre-flight card.
  //
  // IMPORTANT: under the premium theme, `src/theme/premium.css`
  // overrides `.ant-card { background: var(--glass-2) !important }`
  // — which means setting `style.background` on the Card outer
  // gets clobbered. We instead set the background on `bodyStyle`
  // (= `.ant-card-body`), which premium.css does NOT override, so
  // the colour reliably shows in every theme.
  //
  //   Light: #93C5FD (Tailwind blue-300) — clearly pale-blue,
  //          unmistakable next to the white data cards below.
  //   Dark : rgba(56, 189, 248, 0.22)     — sky-aurora wash that
  //          reads as the same accent against the deeper bg.
  const { isDark } = useTheme();
  const cardBackground = isDark
    ? "rgba(56, 189, 248, 0.22)"
    : "#BFDBFE"; // Tailwind blue-200 — clearly pale blue without
                 // dominating the page. (Blue-100 #DBEAFE was too
                 // faint to register against the page bg.)

  return (
    <Card
      style={{
        marginBottom: 16,
        // Original left-rail blue restored.
        borderLeft: "4px solid #1F6FEB",
        // Outer background — works for legacy themes (where no
        // `!important` rule overrides it). On premium the bodyStyle
        // below carries the colour.
        background: cardBackground,
      }}
      bodyStyle={{
        paddingTop: 12,
        paddingBottom: 12,
        // Premium-theme-safe carrier for the pale-blue colour.
        background: cardBackground,
        borderRadius: "inherit",
      }}
    >
      <div
        onClick={() => setExpanded((v) => !v)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setExpanded((v) => !v);
          }
        }}
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          cursor: "pointer",
          userSelect: "none",
        }}
        aria-expanded={expanded}
      >
        {/* Header text + icon back to the legacy default colour
            (AntD Title's inherited text colour). Only the card
            BACKGROUND is the vibrant pale-blue accent now. */}
        <Title level={5} style={{ marginTop: 0, marginBottom: 0 }}>
          <SafetyCertificateOutlined style={{ marginRight: 8 }} />
          Preliminary Tier 1 Checks
        </Title>
        {expanded ? (
          <DownOutlined style={{ fontSize: 12, color: "#1F6FEB" }} />
        ) : (
          <RightOutlined style={{ fontSize: 12, color: "#1F6FEB" }} />
        )}
      </div>

      {/* Sprint 13.6 — intro line moved out of the collapsed-only
          block so it stays visible on the collapsed card too.
          Engineer always sees what the panel is for, then clicks
          to read the full four-bullet checklist underneath. */}
      <Paragraph
        type="secondary"
        style={{ marginTop: 8, marginBottom: 0, fontSize: 12 }}
      >
        Complete these pre-flight checks on every ticket before drilling
        into the cohort findings below.
      </Paragraph>

      {expanded ? (
        <ol style={{ marginTop: 12, marginBottom: 0, paddingLeft: 20 }}>
          {CHECKS.map((c) => (
            <li key={c.label} style={{ marginBottom: 6 }}>
              <Text strong>{c.label}:</Text> {c.body}
            </li>
          ))}
        </ol>
      ) : null}
    </Card>
  );
}
