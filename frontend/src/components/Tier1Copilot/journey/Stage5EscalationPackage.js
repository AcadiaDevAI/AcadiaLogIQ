// Sprint 10 Stage 5 — Operational Handoff (Escalate to Tier 2 view).
//
// Sprint 12.7 — adds the Escalation Routing & Vendor/OEM Engagement
// section (deduped Resolution_Groups + Team_Paths + recommended
// Tier-2 entry candidates) and the "Generate Tier 2 Escalation
// Handoff" action.
//
// Sprint 12.9 — the Sprint 7 EscalationPackageCard render is now
// suppressed. Clicking "Escalate to Tier 2" from any stage lands
// the engineer here, and "here" is now exactly two things:
//   1. Escalation Routing & Vendor/OEM Engagement (data view)
//   2. Generate Tier 2 Escalation Handoff (LLM-backed action)
// The Sprint 7 package render + its `expanded`/`autoExpand` toggle
// + the EscalationPackageCard import are preserved as commented
// blocks; reinstate by un-commenting all three together.

import React, { useEffect, useState } from "react";
import { Alert, Button, Card, Modal, Space, Spin, Tag, Tooltip, Typography, message } from "antd";
import { CopyOutlined, LoadingOutlined, ReloadOutlined } from "@ant-design/icons";

import { useTheme } from "../../../hooks/ThemeContext";

// Sprint 12.9 — Sprint 7's full EscalationPackageCard render is
// suppressed. Clicking "Escalate to Tier 2" now lands the engineer
// on the Escalation Routing & Vendor/OEM Engagement view + the
// Generate Tier 2 Escalation Handoff action. Reinstate the import
// (and the JSX block below) if the Sprint 7 package view is ever
// brought back.
// import EscalationPackageCard from "../EscalationPackageCard";
import DislikeButton from "./DislikeButton";
import HelpfulButton from "./HelpfulButton";
import CardWatermark from "./CardWatermark";
import {
  fetchEscalationRouting,
  generateEscalationHandoffNote,
  getActivityVersion,
  subscribeActivityVersion,
} from "./journeyApi";

// Sprint 13.30 — stale-note pulse animation. Injected once on first
// import via a module-level guard so the keyframes exist for the
// entire app lifetime, not re-injected per render. The
// `prefers-reduced-motion: reduce` media query disables the
// animation for users who've opted out (WCAG 2.3.3 + browser
// vestibular-trigger settings) — the dot still appears as a static
// indicator, just without the visual pulse.
let _acadiaPulseStylesInjected = false;
function _ensurePulseStyles() {
  if (typeof document === "undefined" || _acadiaPulseStylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-pulse", "1");
  style.textContent = `
    @keyframes acadia-pulse-dot {
      0%, 100% { transform: scale(1);   opacity: 1;    box-shadow: 0 0 0 0 rgba(11, 49, 92, 0.55); }
      50%      { transform: scale(1.15); opacity: 0.85; box-shadow: 0 0 0 6px rgba(11, 49, 92, 0); }
    }
    .acadia-stale-dot {
      animation: acadia-pulse-dot 1.4s ease-in-out infinite;
    }
    @media (prefers-reduced-motion: reduce) {
      .acadia-stale-dot { animation: none !important; }
    }
  `;
  document.head.appendChild(style);
  _acadiaPulseStylesInjected = true;
}

const { Title, Paragraph, Text } = Typography;


// ─────────────────────────────────────────────────────────────
// Block 04 — Operational Handoff (heading only).
// Mirrors the Block 02 / Block 03 eyebrow + display + gradient
// italic accent treatment so the four cohort/journey panels
// (Best Historical Match, Guided Workflow, Knowledge Base &
// SOP Reference, Operational Handoff) form a consistent visual
// family. Scope: `.b04-*` so nothing bleeds into the inner
// Escalation Routing / Handoff Note sub-sections.
// ─────────────────────────────────────────────────────────────
const B04_BLUE = {
  50:  "#EFF6FF",
  200: "#BFDBFE",
  400: "#60A5FA",
  500: "#3B82F6",
  600: "#2563EB",
  700: "#1D4ED8",
};
const B04_CSS = `
.b04-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 12px; border-radius: 9999px;
  background: ${B04_BLUE[50]};
  border: 1px solid ${B04_BLUE[200]};
  color: ${B04_BLUE[700]};
  font-family: var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace);
  font-size: 10.5px; font-weight: 500; letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.b04-eyebrow__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: ${B04_BLUE[500]};
  box-shadow: 0 0 8px ${B04_BLUE[400]};
}
.b04-display {
  font-family: var(--font-display, 'Instrument Serif', Georgia, serif);
  font-size: clamp(24px, 2.8vw, 32px);
  font-weight: 400; line-height: 1.12; letter-spacing: -0.018em;
  margin: 0 0 6px 0; color: #0F172A;
}
.b04-accent {
  font-style: italic; font-weight: 400;
  background: linear-gradient(135deg, ${B04_BLUE[400]} 0%, ${B04_BLUE[600]} 60%, ${B04_BLUE[700]} 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
`;
let _b04StylesInjected = false;
function _ensureB04Styles() {
  if (typeof document === "undefined" || _b04StylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-block04", "1");
  style.textContent = B04_CSS;
  document.head.appendChild(style);
  _b04StylesInjected = true;
}


// ────────────────────────────────────────────────────────────
// Routing section — deduped groups, paths, and Tier-2 candidates.
// Hidden when the cohort carries no routing data at all so the
// engineer never sees an empty stub.
// ────────────────────────────────────────────────────────────
function EscalationRoutingSection({ routing }) {
  if (!routing || routing.empty) return null;

  return (
    <div style={{ marginTop: 16 }}>
      <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>
        Escalation Routing & Vendor/OEM Engagement
      </Title>

      {routing.recommended_tier2_teams && routing.recommended_tier2_teams.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Recommended Tier 2 entry: </Text>
          <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
            {routing.recommended_tier2_teams.map((t) => (
              <Tag
                key={t.team}
                color={t.occurrence_count > 1 ? "red" : "volcano"}
                title={
                  t.example_path
                    ? `Seen in ${t.occurrence_count} ticket${t.occurrence_count === 1 ? "" : "s"} via path: ${t.example_path}`
                    : undefined
                }
              >
                {t.team}
                {t.occurrence_count > 1 ? ` ×${t.occurrence_count}` : ""}
              </Tag>
            ))}
          </Space>
        </div>
      ) : null}

      {routing.team_paths && routing.team_paths.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Historical team paths:</Text>
          <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
            {routing.team_paths.map((p) => (
              <li key={p}>
                <Text code>{p}</Text>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {routing.resolution_groups && routing.resolution_groups.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Resolution groups across cohort:</Text>{" "}
          <Space size={[4, 4]} wrap style={{ marginTop: 4 }}>
            {routing.resolution_groups.map((g) => (
              <Tag key={g} color="geekblue">
                {g}
              </Tag>
            ))}
          </Space>
        </div>
      ) : null}

      {/* Vendor / OEM block — wired forward-compat. The corpus
          today carries 0 Vendor_OEM_Engagement records. When
          ingestion populates the field, this block lights up
          automatically. Honest empty-state copy meanwhile. */}
      {routing.forensic_data_required && routing.forensic_data_required.length > 0 ? (
        <div style={{ marginBottom: 12 }}>
          <Text strong>Forensic data required before vendor/OEM engagement:</Text>
          <ul style={{ marginTop: 4, marginBottom: 0, paddingLeft: 20 }}>
            {routing.forensic_data_required.map((f) => (
              <li key={f}>{f}</li>
            ))}
          </ul>
        </div>
      ) : routing.vendor_records && routing.vendor_records.length === 0 ? (
        // Sprint 13.23 — promoted from gray secondary text to bold
        // black so the Tier-2 reader's eye lands on it. This line
        // was previously also duplicated as a bold residual at the
        // bottom of the copyable note; the duplicate has been
        // removed in favour of this single canonical placement.
        <Paragraph style={{ marginBottom: 12 }}>
          <Text strong style={{ color: "var(--text-primary, #000)" }}>
            No vendor/OEM engagement records found in this cohort.
          </Text>
        </Paragraph>
      ) : null}

      {routing.cohort_size ? (
        <Paragraph style={{ marginBottom: 0, marginTop: 4 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Aggregated from {routing.tickets_with_data} of{" "}
            {routing.cohort_size} cohort ticket
            {routing.cohort_size === 1 ? "" : "s"}.
          </Text>
        </Paragraph>
      ) : null}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────
// Handoff Report — 10-step blue scale + 9 type roles + TimeCard grid
//
// Replaces the previous monospace <pre> render with a structured
// card grid. Parses the LLM/template-generated handoff `note` into
// {eyebrow, title, subhead, activities[], diagnostics[], reason}
// and renders TimeCards (default / major / full / diag) inside a
// CardsGrid, with a ReasonStrip below.
//
// Scope: `.hr-*` so this stylesheet cannot bleed into any other
// panel. Buttons (Copy / Regenerate) and the underlying `note`
// data source remain UNCHANGED.
// ─────────────────────────────────────────────────────────────
const HR_BLUE = {
  50:  "#EFF6FF",
  100: "#DBEAFE",
  200: "#BFDBFE",
  300: "#93C5FD",
  400: "#60A5FA",
  500: "#3B82F6",
  600: "#2563EB",
  700: "#1D4ED8",
  800: "#1E40AF",
  900: "#1E3A8A",
};
const HR_CSS = `
.hr-root {
  font-family: var(--font-body, 'Geist', 'Inter', system-ui, sans-serif);
  color: #0F172A;
}
.hr-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 12px; border-radius: 9999px;
  background: ${HR_BLUE[50]};
  border: 1px solid ${HR_BLUE[200]};
  color: ${HR_BLUE[800]};
  font-family: var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace);
  font-size: 10.5px; font-weight: 500; letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 8px;
}
.hr-eyebrow__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: ${HR_BLUE[500]};
  box-shadow: 0 0 8px ${HR_BLUE[400]};
}
.hr-title {
  font-family: var(--font-display, 'Instrument Serif', Georgia, serif);
  font-size: clamp(22px, 2.4vw, 28px);
  font-weight: 400; line-height: 1.15; letter-spacing: -0.018em;
  margin: 0 0 4px 0; color: #0F172A;
}
.hr-titleAccent {
  font-style: italic; font-weight: 400;
  background: linear-gradient(135deg, ${HR_BLUE[400]} 0%, ${HR_BLUE[600]} 60%, ${HR_BLUE[800]} 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
.hr-subhead {
  font-size: 13px; line-height: 1.55; color: #475569;
  margin: 0 0 16px 0;
}
.hr-sectionHead {
  display: inline-flex; align-items: center; gap: 8px;
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-size: 11px; font-weight: 600; letter-spacing: 0.10em;
  text-transform: uppercase; color: ${HR_BLUE[700]};
  margin: 18px 0 10px 0;
}
.hr-sectionHead::before {
  content: ""; display: inline-block; width: 18px; height: 1px;
  background: ${HR_BLUE[300]};
}
.hr-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 12px;
}
/* TimeCard — default variant. */
.hr-card {
  position: relative;
  background: #FFFFFF;
  border: 1px solid ${HR_BLUE[100]};
  border-radius: 14px;
  padding: 14px 16px;
  display: flex; flex-direction: column; gap: 6px;
  min-height: 96px;
  transition: border-color 180ms ease, box-shadow 180ms ease;
}
.hr-card:hover { border-color: ${HR_BLUE[300]}; box-shadow: 0 4px 16px -8px rgba(37, 99, 235, 0.18); }
/* Major variant — index-0 (longest-duration) card. Spans 2 grid
   columns on wide layouts so it reads as the dominant activity. */
.hr-card--major {
  background: linear-gradient(135deg, ${HR_BLUE[50]} 0%, #FFFFFF 70%);
  border-color: ${HR_BLUE[200]};
}
/* Wide variant — spans both columns of the 2-col grid. Used when
   Search KB / SOP Reference should fill row 2 because Discuss with
   LogIQ was never opened. */
.hr-card--wide {
  grid-column: 1 / -1;
}
/* Full variant — total card. Always last and full-width. */
.hr-card--full {
  grid-column: 1 / -1;
  background: linear-gradient(135deg, ${HR_BLUE[600]} 0%, ${HR_BLUE[800]} 100%);
  border-color: ${HR_BLUE[700]};
  color: #FFFFFF;
}
.hr-card--full .hr-cardLabel,
.hr-card--full .hr-cardNum,
.hr-card--full .hr-cardTag { color: #FFFFFF; }
.hr-card--full .hr-cardTag {
  background: rgba(255, 255, 255, 0.14);
  border-color: rgba(255, 255, 255, 0.28);
}
/* Diag variant — diagnostic activity card (carries a codeChip). */
.hr-card--diag {
  background: #FFFFFF;
  border-color: ${HR_BLUE[100]};
}
/* Type roles inside a card */
.hr-cardK {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 10px; font-weight: 600; letter-spacing: 0.12em;
  text-transform: uppercase; color: ${HR_BLUE[500]};
}
.hr-cardLabel {
  font-size: 13px; font-weight: 600; line-height: 1.4;
  color: #0F172A;
}
.hr-cardNum {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 22px; font-weight: 600; line-height: 1.1;
  color: ${HR_BLUE[800]};
  margin-top: 2px;
}
.hr-card--full .hr-cardNum { font-size: 28px; }
.hr-cardNum small {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-size: 12px; font-weight: 500; color: inherit;
  opacity: 0.75; margin-left: 3px;
}
.hr-cardNum__active {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
  margin-left: 10px; opacity: 0.85;
}
.hr-cardTag {
  display: inline-flex; align-items: center; gap: 4px;
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 11px; font-weight: 500;
  background: ${HR_BLUE[50]};
  border: 1px solid ${HR_BLUE[200]};
  color: ${HR_BLUE[700]};
  padding: 2px 8px; border-radius: 9999px;
}
.hr-ticket {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 11.5px; font-weight: 600;
  background: ${HR_BLUE[50]};
  border: 1px solid ${HR_BLUE[200]};
  color: ${HR_BLUE[800]};
  padding: 1px 7px; border-radius: 6px;
  white-space: nowrap;
}
.hr-tickets { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 4px; }
.hr-codeChip {
  display: inline-block;
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  white-space: nowrap;
  background: ${HR_BLUE[50]};
  border: 1px solid ${HR_BLUE[200]};
  color: ${HR_BLUE[900]};
  font-size: 11.5px;
  padding: 2px 8px;
  border-radius: 6px;
  max-width: 100%; overflow-x: auto;
}
.hr-codeChips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
/* ReasonStrip */
.hr-reason {
  display: flex; align-items: flex-start; gap: 10px;
  margin-top: 18px;
  padding: 12px 14px;
  background: ${HR_BLUE[50]};
  border-left: 3px solid ${HR_BLUE[600]};
  border-radius: 8px;
  font-size: 13px; line-height: 1.55; color: ${HR_BLUE[900]};
}
.hr-reason__label {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-size: 10.5px; font-weight: 600; letter-spacing: 0.14em;
  text-transform: uppercase; color: ${HR_BLUE[700]};
  white-space: nowrap; padding-top: 1px;
}
/* Diag-row layout */
.hr-diagList {
  display: flex; flex-direction: column; gap: 10px;
}
.hr-diagRow {
  display: grid; grid-template-columns: 24px 1fr;
  gap: 10px;
  padding: 10px 12px;
  background: #FFFFFF;
  border: 1px solid ${HR_BLUE[100]};
  border-radius: 10px;
}
.hr-diagIdx {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 12px; font-weight: 600; color: ${HR_BLUE[600]};
  padding-top: 1px;
}
.hr-diagText {
  font-size: 13px; line-height: 1.55; color: #0F172A;
}
`;
let _hrStylesInjected = false;
function _ensureHrStyles() {
  if (typeof document === "undefined" || _hrStylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-handoff-report", "1");
  style.textContent = HR_CSS;
  document.head.appendChild(style);
  _hrStylesInjected = true;
}


// Convert "1018s" / "(34m 32s)" / "16m 48s" / loose forms into a
// canonical {seconds, label} pair so cards always render the same
// shape. "16m 58s" preferred over "1018s" per spec.
function _normaliseDuration(raw) {
  if (!raw) return { seconds: 0, label: "—" };
  const txt = String(raw).trim();
  // Already canonical (e.g. "16m 48s", "9s", "34m 32s")
  const mPlusS = txt.match(/(\d+)\s*m\s*(\d+)\s*s/i);
  if (mPlusS) {
    const secs = parseInt(mPlusS[1], 10) * 60 + parseInt(mPlusS[2], 10);
    return { seconds: secs, label: `${mPlusS[1]}m ${mPlusS[2]}s` };
  }
  const onlyM = txt.match(/^(\d+)\s*m\s*$/i);
  if (onlyM) {
    return { seconds: parseInt(onlyM[1], 10) * 60, label: `${onlyM[1]}m` };
  }
  const onlyS = txt.match(/^(\d+)\s*s\s*$/i);
  if (onlyS) {
    const s = parseInt(onlyS[1], 10);
    if (s >= 60) {
      const m = Math.floor(s / 60);
      const r = s % 60;
      return { seconds: s, label: r === 0 ? `${m}m` : `${m}m ${r}s` };
    }
    return { seconds: s, label: `${s}s` };
  }
  const onlyDigits = txt.match(/^(\d+)$/);
  if (onlyDigits) {
    const s = parseInt(onlyDigits[1], 10);
    if (s >= 60) {
      const m = Math.floor(s / 60);
      const r = s % 60;
      return { seconds: s, label: r === 0 ? `${m}m` : `${m}m ${r}s` };
    }
    return { seconds: s, label: `${s}s` };
  }
  return { seconds: 0, label: txt };
}


// Render the canonical "16m 58s" string with the unit halves wrapped
// in <small> so the digits stay big and prominent. Tabular numerals
// are applied via the .hr-cardNum class.
function _renderTimeRichText(label) {
  if (!label || typeof label !== "string") return label;
  // Split between digits and the unit letter so "16m 58s" renders as
  // 16<small>m</small> 58<small>s</small>.
  const parts = label.split(/(\d+)/).filter(Boolean);
  return parts.map((p, i) => {
    if (/^\d+$/.test(p)) {
      return <span key={i}>{p}</span>;
    }
    return <small key={i}>{p}</small>;
  });
}


// Parse the LLM/template handoff note text into structured sections.
// The parser is deliberately tolerant — when a bullet doesn't match a
// recognised shape, it lands in `extraBullets` (rendered as plain
// items) instead of crashing. If parsing yields nothing meaningful,
// the caller falls back to the raw <pre> render.
function _parseHandoffNote(note) {
  if (!note || typeof note !== "string") return null;
  // Section split — sections are delimited by *Header:* markers.
  // Section names we care about: "Diagnostic Summary", "Reason for Escalation".
  const sections = {};
  const sectionRe = /\*([^*]+?):\*\s*\n([\s\S]*?)(?=\n\*[^*]+?:\*|$)/g;
  let m;
  while ((m = sectionRe.exec(note)) !== null) {
    sections[m[1].trim()] = m[2].trim();
  }
  // Title line — first *...* line at the top.
  const titleMatch = note.match(/\*([^*]+)\*/);
  const titleLine = titleMatch ? titleMatch[1].trim() : "Operational Handoff";
  // Eyebrow / title / titleAccent split — e.g. "Escalation to Tier 2: Triage Complete"
  let eyebrow = "Escalation to Tier 2";
  let title = "Triage Complete";
  let titleAccent = "Complete";
  if (titleLine.includes(":")) {
    const [left, right] = titleLine.split(":").map((s) => s.trim());
    eyebrow = left;
    title = right;
    // Accent the last word of the title line.
    const words = right.split(/\s+/);
    if (words.length > 1) {
      titleAccent = words[words.length - 1];
    } else {
      titleAccent = right;
    }
  }
  // Activity bullets sit BEFORE the first *Section:* marker. Capture
  // the slice between the title and the first section header.
  let activityBody = "";
  const firstSectionIdx = note.search(/\n\*[^*]+?:\*/);
  const titleEndIdx = titleMatch
    ? note.indexOf(titleMatch[0]) + titleMatch[0].length
    : 0;
  if (firstSectionIdx > -1) {
    activityBody = note.slice(titleEndIdx, firstSectionIdx);
  } else {
    activityBody = note.slice(titleEndIdx);
  }
  // Subhead — the first non-bullet sentence after the title.
  const subheadMatch = activityBody.match(/^\s*([^\-\n][^\n]*)/);
  const subhead = subheadMatch ? subheadMatch[1].trim() : "";
  // Activity bullets — lines starting with "- " (with optional
  // continuation lines on the next indented row). Bullets that
  // carry no duration AND no ticket id are pure status lines
  // ("Tier 1 has completed initial triage.") — they don't deserve
  // a TimeCard, so we drop them here and let the subhead carry the
  // narrative context instead.
  const activities = [];
  const statusBullets = [];
  const bulletRe = /(?:^|\n)-\s+([^\n]+(?:\n {2,}[^\n]+)*)/g;
  let b;
  while ((b = bulletRe.exec(activityBody)) !== null) {
    const raw = b[1].replace(/\n {2,}/g, " ").trim();
    const parsed = _parseActivityBullet(raw);
    const hasDuration = parsed.durationSeconds > 0 || parsed.isTotal;
    const hasTickets = parsed.tickets && parsed.tickets.length > 0;
    // Sprint 13.33 — canonical 2x2 grid cards (Historical / Guided /
    // Search KB / Discuss with LogIQ) always render, even when the
    // engineer didn't engage with them (no duration, no tickets).
    // _layoutActivities handles the "Discuss not opened → Search KB
    // spans the row" rule downstream.
    const isCanonical = _classifyActivity(parsed) !== null;
    if (hasDuration || hasTickets || isCanonical) {
      activities.push(parsed);
    } else {
      statusBullets.push(raw);
    }
  }
  // Diagnostics — bullets inside the Diagnostic Summary section.
  const diagnostics = [];
  if (sections["Diagnostic Summary"]) {
    const diagBody = sections["Diagnostic Summary"];
    const dRe = /(?:^|\n)-\s+([^\n]+(?:\n[^\-][^\n]*)*)/g;
    let d;
    while ((d = dRe.exec(diagBody)) !== null) {
      const raw = d[1].replace(/\s+/g, " ").trim();
      diagnostics.push(_parseDiagBullet(raw));
    }
  }
  // Reason — bullets inside the Reason for Escalation section.
  let reason = null;
  if (sections["Reason for Escalation"]) {
    const reasonBody = sections["Reason for Escalation"];
    const reasonMatch = reasonBody.match(/-\s+([^\n]+)/);
    reason = reasonMatch ? reasonMatch[1].trim() : reasonBody.replace(/^-\s*/, "").trim();
  }
  return { eyebrow, title, titleAccent, subhead, activities, statusBullets, diagnostics, reason };
}


// Activity bullet → {label, durationLabel, durationSeconds, tickets, detail, isTotal, isKb}
function _parseActivityBullet(raw) {
  // Total line — "Total time on this triage: 17m 17s."
  const totalMatch = raw.match(/^Total time on this triage:\s*([0-9msh\s]+)\.?/i);
  if (totalMatch) {
    const d = _normaliseDuration(totalMatch[1]);
    return {
      label: "Total time on this triage",
      durationSeconds: d.seconds,
      durationLabel: d.label,
      tickets: [],
      detail: "active",
      isTotal: true,
    };
  }
  // Pull ticket IDs (INC-XXX / TKT-XXX / similar).
  const ticketMatches = raw.match(/\b[A-Z]{2,}[A-Z0-9-]*\d[A-Z0-9-]*\b/g) || [];
  const tickets = [...new Set(ticketMatches.filter((t) => /-/.test(t)))];
  // Pull duration — Sprint 13.33 — find the LAST paren whose content
  // contains a time token (digits + m/s/h). Earlier versions used
  // the FIRST paren, which broke on bullets like Historical's
  // "(engaged via per-ticket chat for INC-X) ... (3m 12s across
  // context panels)" — the engagement paren has no time units, so
  // we'd return 0 duration. Scanning all parens and picking the
  // last time-bearing one fixes that case while keeping single-paren
  // bullets working as before.
  const allParens = raw.match(/\([^)]*\d+\s*[smh]\b[^)]*\)/gi) || [];
  const durParen = allParens.length > 0 ? allParens[allParens.length - 1] : "";
  const innerDur = durParen ? durParen.slice(1, -1) : "";
  const durRaw = (innerDur.match(/\d+\s*m\s*\d+\s*s/i)
    || innerDur.match(/\d+\s*m/i)
    || innerDur.match(/\d+\s*s/i)
    || [""])[0];
  const d = _normaliseDuration(durRaw);
  // Label — strip the trailing paren + period.
  let label = raw.replace(/\s*\([^)]*\)\.?\s*$/, "").trim();
  // For lines like "Search KB / SOP reference: consulted", split on ":"
  let detail = "";
  let isKb = false;
  if (label.includes(":")) {
    const [lhs, rhs] = label.split(":").map((s) => s.trim());
    label = lhs;
    detail = rhs;
  }
  if (/Search\s+KB|SOP\s+reference/i.test(label)) {
    isKb = true;
    // KB "opened but not engaged" rule — when the detail mentions
    // 0 chat sessions, prefer "opened" over "consulted".
    if (/0\s*chat session/i.test(innerDur) || /0\s*chat/i.test(detail)) {
      detail = "opened";
    }
  }
  // Tickets line — when the bullet is the "Historical tickets" header
  // followed by a comma-separated ticket list on the next continuation
  // line, the ticketMatches array already captured them.
  return {
    label,
    durationSeconds: d.seconds,
    durationLabel: d.label,
    tickets,
    detail,
    isTotal: false,
    isKb,
  };
}


// Diagnostic bullet → {text, commands[]}
function _parseDiagBullet(raw) {
  const commands = [];
  const codeRe = /`([^`]+)`/g;
  let c;
  while ((c = codeRe.exec(raw)) !== null) {
    // Split multi-line code blocks into individual commands.
    const parts = c[1].split(/\n+/).map((s) => s.trim()).filter(Boolean);
    parts.forEach((p) => commands.push(p));
  }
  const text = raw.replace(/`[^`]+`/g, "").replace(/\s+/g, " ").replace(/\s+\.$/, ".").trim();
  return { text, commands };
}


// Sprint 13.33 — canonical 2x2 grid for the Activity Summary:
//   Row 1: Historical tickets surfaced | Guided Troubleshooting
//   Row 2: Search KB / SOP Reference   | Discuss with LogIQ
// When Discuss with LogIQ was never opened (no chat sessions), it
// is hidden and Search KB / SOP Reference spans the full row 2.
// The Total card is unchanged: rendered last, full-width.
function _classifyActivity(activity) {
  if (!activity || activity.isTotal) return null;
  const label = (activity.label || "").trim();
  if (/^(?:similar\s+)?historical\s+tickets/i.test(label)) return "historical";
  if (/^guided\s+troubleshooting/i.test(label)) return "guided";
  if (/^search\s+kb|^sop\s+reference/i.test(label)) return "search_kb";
  if (/^discuss\s+with\s+logiq/i.test(label)) return "discuss";
  return null;
}

function _isDiscussNotOpened(activity) {
  if (!activity) return true;
  const detail = (activity.detail || "").toLowerCase();
  if (/no per-ticket chat/.test(detail)) return true;
  if (/^no\b/.test(detail) || /not\s+opened/.test(detail)) return true;
  const noDuration = (activity.durationSeconds || 0) === 0;
  const noTickets = !activity.tickets || activity.tickets.length === 0;
  return noDuration && noTickets;
}

// Sprint 13.33 — per-card eyebrow labels. Replaces the generic
// "Activity" tag with the specific journey stage so each card in
// the 2x2 grid self-identifies (Context Research / Guided
// Troubleshooting / Search KB / SOP Reference / Discuss with LogIQ).
const _KIND_LABELS = {
  historical: "Context Research",
  guided: "Guided Troubleshooting",
  search_kb: "Search KB / SOP Reference",
  discuss: "Discuss with LogIQ",
};

function _layoutActivities(activities) {
  const total = activities.find((a) => a.isTotal) || null;

  const byKind = {};
  activities.forEach((a) => {
    if (a.isTotal) return;
    const kind = _classifyActivity(a);
    if (kind && !byKind[kind]) byKind[kind] = a;
  });

  const discussOpened = byKind.discuss && !_isDiscussNotOpened(byKind.discuss);

  // Sprint 13.33 — hideLabel suppresses the bold-black `.hr-cardLabel`
  // below the eyebrow on cards where the parsed label just duplicates
  // the kindLabel (Guided / SearchKB / Discuss). Historical's parsed
  // label carries dynamic context that's NOT in the eyebrow (e.g.
  // "Historical tickets surfaced for context (engaged via per-ticket
  // chat for INC-X)"), so its label stays visible.
  const styled = [];
  if (byKind.historical) {
    styled.push({
      ...byKind.historical,
      variant: "default",
      kindLabel: _KIND_LABELS.historical,
    });
  }
  if (byKind.guided) {
    styled.push({
      ...byKind.guided,
      variant: "default",
      kindLabel: _KIND_LABELS.guided,
      hideLabel: true,
    });
  }
  if (byKind.search_kb) {
    styled.push({
      ...byKind.search_kb,
      variant: discussOpened ? "default" : "wide",
      kindLabel: _KIND_LABELS.search_kb,
      hideLabel: true,
    });
  }
  if (discussOpened) {
    styled.push({
      ...byKind.discuss,
      variant: "default",
      kindLabel: _KIND_LABELS.discuss,
      hideLabel: true,
    });
  }

  // Defensive: surface any non-canonical, non-total bullets that
  // arrived as activities (e.g., future bullet kinds the backend
  // adds) so they don't silently vanish.
  activities.forEach((a) => {
    if (a.isTotal) return;
    if (_classifyActivity(a)) return;
    styled.push({ ...a, variant: "default" });
  });

  return { styled, total };
}


function TimeCard({ activity }) {
  const variantClass = activity.variant === "major"
    ? "hr-card hr-card--major"
    : activity.variant === "full"
      ? "hr-card hr-card--full"
      : activity.variant === "wide"
        ? "hr-card hr-card--wide"
        : "hr-card";
  const eyebrow = activity.kindLabel || "Activity";
  // Sprint 13.33 — `hideLabel` cards drop the bold-black label below
  // the eyebrow because it just repeats the kindLabel text (Guided
  // Troubleshooting / Search KB / Discuss with LogIQ). Cards without
  // hideLabel (Context Research, Total) still render the parsed label
  // so its dynamic context (engagement / ticket counts / etc.) stays
  // visible.
  return (
    <div className={variantClass}>
      <span className="hr-cardK">{eyebrow}</span>
      {!activity.hideLabel ? (
        <div className="hr-cardLabel">{activity.label}</div>
      ) : null}
      <div className="hr-cardNum">
        {_renderTimeRichText(activity.durationLabel)}
        {activity.isTotal ? (
          <span className="hr-cardNum__active">active</span>
        ) : null}
      </div>
      {activity.detail ? (
        <span className="hr-cardTag">{activity.detail}</span>
      ) : null}
      {activity.tickets && activity.tickets.length > 0 ? (
        <div className="hr-tickets">
          {activity.tickets.map((t) => (
            <span key={t} className="hr-ticket">{t}</span>
          ))}
        </div>
      ) : null}
    </div>
  );
}


function CardsGrid({ activities }) {
  const { styled, total } = _layoutActivities(activities);
  if (styled.length === 0 && !total) return null;
  return (
    <div className="hr-grid">
      {styled.map((a, i) => (
        <TimeCard key={i} activity={a} />
      ))}
      {total ? (
        <TimeCard activity={{ ...total, variant: "full" }} />
      ) : null}
    </div>
  );
}


function DiagList({ diagnostics }) {
  if (!diagnostics || diagnostics.length === 0) return null;
  return (
    <div className="hr-diagList">
      {diagnostics.map((d, i) => (
        <div key={i} className="hr-diagRow">
          <span className="hr-diagIdx">{String(i + 1).padStart(2, "0")}</span>
          <div>
            <div className="hr-diagText">{d.text}</div>
            {d.commands && d.commands.length > 0 ? (
              <div className="hr-codeChips">
                {d.commands.map((cmd, j) => (
                  <span key={j} className="hr-codeChip">{cmd}</span>
                ))}
              </div>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}


function ReasonStrip({ reason }) {
  if (!reason) return null;
  return (
    <div className="hr-reason" role="note">
      <span className="hr-reason__label">Reason</span>
      <span>{reason}</span>
    </div>
  );
}


// Top-level renderer. Receives the raw `note` text + falls back to
// `fallbackRenderer` (the original <pre>) when parsing yields no
// meaningful structure.
function HandoffReportBody({ note, fallbackRenderer }) {
  React.useEffect(() => { _ensureHrStyles(); }, []);
  const parsed = React.useMemo(() => _parseHandoffNote(note), [note]);
  const usable = !!(
    parsed
    && (parsed.activities.length > 0 || parsed.diagnostics.length > 0 || parsed.reason)
  );
  if (!usable) return fallbackRenderer ? fallbackRenderer() : null;
  return (
    <div className="hr-root">
      {/* eyebrow */}
      <div className="hr-eyebrow">
        <span aria-hidden className="hr-eyebrow__dot" />
        {parsed.eyebrow}
      </div>
      {/* title with accent on the last word */}
      <h2 className="hr-title">
        {parsed.title.replace(new RegExp(`\\s*${parsed.titleAccent}$`), "")}{" "}
        <em className="hr-titleAccent">{parsed.titleAccent}</em>
      </h2>
      {parsed.subhead ? <p className="hr-subhead">{parsed.subhead}</p> : null}

      {/* Status bullets that carry neither a duration nor a ticket
          (e.g. "Tier 1 has completed initial triage.") render here
          as a muted inline strip rather than empty TimeCards. */}
      {parsed.statusBullets && parsed.statusBullets.length > 0 ? (
        <ul style={{
          listStyle: "none", padding: 0, margin: "0 0 14px 0",
          display: "flex", flexWrap: "wrap", gap: "6px 14px",
          fontSize: 12.5, color: "#475569",
        }}>
          {parsed.statusBullets.map((s, i) => (
            <li key={i} style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span aria-hidden style={{
                width: 4, height: 4, borderRadius: "50%",
                background: HR_BLUE[400], display: "inline-block",
              }} />
              {s}
            </li>
          ))}
        </ul>
      ) : null}

      {parsed.activities.length > 0 ? (
        <CardsGrid activities={parsed.activities} />
      ) : null}

      {parsed.diagnostics.length > 0 ? (
        <>
          <div className="hr-sectionHead">Diagnostic Summary</div>
          <DiagList diagnostics={parsed.diagnostics} />
        </>
      ) : null}

      {parsed.reason ? <ReasonStrip reason={parsed.reason} /> : null}
    </div>
  );
}


// ────────────────────────────────────────────────────────────
// Sprint 13.17 — Auto-fetched Tier-2 handoff note. Engineer arrives
// at the Operational Handoff panel (Stage 5) and the note loads
// automatically — no Generate button. UX rationale: when an
// engineer is escalating in a crisis, the report should be ready
// the moment they land here, not behind another click.
//
// The backend assembles the note from:
//   - Cohort tickets (incident-number list + executive summaries)
//   - Stage 3 consolidated read-only steps Tier-1 saw
//   - Traversal log (which stages Tier-1 actually visited)
//   - Stage 5 routing aggregation (Resolution_Groups + Team_Path +
//     Vendor_OEM forensic data)
//
// Failure-open: backend always returns a usable note. When
// `used_fallback=true`, an Alert tells the engineer the diagnostic
// sentence is heuristic and offers Regenerate.
// ────────────────────────────────────────────────────────────
function HandoffNoteAction({ sessionId, attemptedStage3Steps, onRefreshRouting }) {
  // Sprint 13.32.13 — dark-theme fix. The handoff-note <pre> below
  // had `background: var(--surface-2, #fafafa)`; `--surface-2` is
  // undefined in this app's theme system so the fallback (#fafafa)
  // applied in every theme, leaving the Operational Handoff content
  // block solid white in dark mode while text colour swapped to
  // light. We now derive bg + text from the live theme; light theme
  // keeps the original #fafafa exactly.
  const { isDark } = useTheme();
  const notePreBg = isDark ? "#16161d" : "#fafafa";
  const notePreText = isDark ? "#e2e8f0" : "inherit";
  const notePreBorder = isDark ? "#2a2a3d" : "#f0f0f0";

  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);
  const [usedFallback, setUsedFallback] = useState(false);

  // Sprint 13.30 — stale-note detection. `noteVersion` snapshots the
  // journey activity counter at fetch time; `currentActivityVersion`
  // tracks the live counter via subscription. When they diverge,
  // the note no longer reflects the engineer's latest journey state.
  // Initial null on noteVersion → first fetch hasn't completed yet →
  // never stale (don't false-flag a still-loading note).
  const [noteVersion, setNoteVersion] = useState(null);
  const [currentActivityVersion, setCurrentActivityVersion] = useState(
    () => getActivityVersion(),
  );
  React.useEffect(() => {
    _ensurePulseStyles();
    const unsubscribe = subscribeActivityVersion((v) => {
      setCurrentActivityVersion(v);
    });
    return unsubscribe;
  }, []);
  const isStale =
    noteVersion !== null
    && currentActivityVersion > noteVersion
    && !!note
    && !busy;

  // Sprint 13.19 — extract the ticked step numbers from the lifted
  // map. JSON.stringify on the dep array keeps useCallback stable
  // across re-renders that don't actually change the tick state.
  const attemptedStepNumbers = React.useMemo(() => {
    if (!attemptedStage3Steps) return [];
    return Object.entries(attemptedStage3Steps)
      .filter(([, v]) => !!v)
      .map(([k]) => Number(k))
      .filter((n) => Number.isFinite(n) && n > 0);
  }, [attemptedStage3Steps]);

  const fetchNote = React.useCallback(async (opts = {}) => {
    if (!sessionId) return "";
    setBusy(true);
    try {
      // Sprint 13.24 PERF — auto-fetches use the cache (fast);
      // Regenerate button passes force=true to bust it server-side.
      const data = await generateEscalationHandoffNote(
        sessionId,
        attemptedStepNumbers,
        { force: !!opts.force },
      );
      const fresh = data?.note || "";
      setNote(fresh);
      setUsedFallback(!!data?.used_fallback);
      // Sprint 13.30 — stamp the version AT successful fetch
      // completion. Any bump that lands after this stamps a higher
      // version → next render flips isStale=true. Stamping at
      // completion (vs. start) intentionally treats events that
      // arrived *during* the fetch as "after" the note — those
      // weren't reflected in this run, so they should still trigger
      // the stale signal.
      setNoteVersion(getActivityVersion());
      return fresh;
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[journey.handoff_note] generate failed", err);
      message.error("Could not generate the escalation note. Please try again.");
      return "";
    } finally {
      setBusy(false);
    }
  }, [sessionId, attemptedStepNumbers]);

  // Auto-fetch on mount + whenever sessionId or the ticked-step set
  // changes. Re-fetching on tick changes means engineers can toggle
  // boxes after landing on Stage 5 and Regenerate to refresh.
  React.useEffect(() => {
    fetchNote();
  }, [fetchNote]);

  // Sprint 13.30 — Regenerate is the hard-reload path: bust caches
  // server-side, refetch routing, and refetch the note. Used both by
  // the Regenerate button and by the Copy-while-stale modal's
  // "Regenerate first" CTA. Returns the fresh note string so the
  // caller (e.g. Copy-while-stale) can chain a clipboard write
  // without racing setState.
  const regenerate = React.useCallback(async () => {
    if (typeof onRefreshRouting === "function") {
      onRefreshRouting();
    }
    return fetchNote({ force: true });
  }, [fetchNote, onRefreshRouting]);

  const doCopy = async (text) => {
    const target = (typeof text === "string" ? text : note) || "";
    if (!target) return;
    try {
      await navigator.clipboard.writeText(target);
      message.success("Escalation note copied.");
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.handoff_note] clipboard write failed", err);
      message.warning("Copy failed — select the text manually.");
    }
  };

  // Sprint 13.30 — Copy guardrail. When stale, show a confirm modal
  // with two CTAs:
  //   - "Regenerate & copy" (default, primary, Acadia color) →
  //     regen + auto-copy the freshly returned note.
  //   - "Copy current" (secondary) → copy the existing stale note
  //     as-is. Engineer's choice; we don't block them outright.
  // Catches the failure mode where the engineer's eyes miss the
  // pulsing dot and they paste a stale note into the Tier-2 ticket.
  const handleCopy = () => {
    if (!note) return;
    if (!isStale) {
      doCopy();
      return;
    }
    Modal.confirm({
      title: "Updates available",
      content:
        "Activity has occurred in this journey since this note was generated. "
        + "Regenerate to refresh, or copy the current note as-is.",
      okText: "Regenerate & copy",
      cancelText: "Copy current",
      okButtonProps: {
        type: "primary",
        style: {
          background: "var(--acadia-primary)",
          borderColor: "var(--acadia-primary)",
        },
      },
      onOk: async () => {
        const fresh = await regenerate();
        await doCopy(fresh);
      },
      onCancel: () => doCopy(),
    });
  };

  return (
    <div style={{ marginTop: 16 }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: 8,
          gap: 8,
          flexWrap: "wrap",
        }}
      >
        {/* Sprint 13.21 — header relabeled at user's request. */}
        <Text strong>Operational Handoff</Text>
        <Space>
          <Button
            size="small"
            icon={<CopyOutlined />}
            onClick={handleCopy}
            disabled={busy || !note}
          >
            Copy
          </Button>
          {/* Sprint 13.24 — Regenerate restored. Click bypasses the
              session-keyed consolidated-ledger cache (force=true)
              and triggers a fresh LLM call.
              Sprint 13.30 — Regenerate is a hard reload of the
              entire Stage 5 view (note + routing + cohort cache)
              and gains a stale-state affordance: when journey
              activity has occurred since the last successful note
              fetch (`isStale === true`), the button switches from
              the default outlined style to an Acadia-primary fill,
              shows a pulsing dot indicator (CSS keyframes; honors
              prefers-reduced-motion), and surfaces a tooltip
              explaining what to do. The non-stale (steady) state
              is the original Sprint 13.24 button — visually
              identical to before, so engineers who never trigger
              activity past Stage 5 see no behavioural change. */}
          <Tooltip
            title={
              isStale
                ? "Activity has occurred in this journey since this note was generated. Click to refresh."
                : "Regenerate the handoff note (bypasses cache)."
            }
          >
            <Button
              size="small"
              onClick={regenerate}
              disabled={busy}
              icon={busy ? <LoadingOutlined /> : <ReloadOutlined />}
              type={isStale ? "primary" : "default"}
              style={
                isStale
                  ? {
                      background: "var(--acadia-primary)",
                      borderColor: "var(--acadia-primary)",
                      color: "#fff",
                      position: "relative",
                      paddingRight: 18,
                    }
                  : { position: "relative" }
              }
            >
              {isStale ? "Refresh" : "Regenerate"}
              {isStale ? (
                <span
                  className="acadia-stale-dot"
                  aria-hidden="true"
                  style={{
                    position: "absolute",
                    top: -3,
                    right: -3,
                    width: 9,
                    height: 9,
                    borderRadius: "50%",
                    background: "var(--acadia-primary, #0b315c)",
                    border: "2px solid var(--bg-secondary, #fff)",
                    pointerEvents: "none",
                  }}
                />
              ) : null}
            </Button>
          </Tooltip>
        </Space>
      </div>

      {busy && !note ? (
        <div style={{ padding: 24, textAlign: "center" }}>
          <Spin />
          <div
            style={{
              marginTop: 8,
              fontSize: 12,
              color: "var(--text-muted, #6b7280)",
            }}
          >
            Composing the handoff note from Tier-1's journey…
          </div>
        </div>
      ) : null}

      {usedFallback ? (
        <Alert
          type="warning"
          showIcon
          message="LLM unavailable — used template fallback. The diagnostic-summary sentence is heuristic; click Regenerate to retry."
          style={{ marginBottom: 8 }}
        />
      ) : null}

      {/* Structured Handoff Report — 10-step blue scale, 9 type
          roles, TimeCard grid (default / major / full / diag) +
          ReasonStrip. Parses the same `note` payload the original
          <pre> showed, so the underlying data source is UNCHANGED.
          When the parser can't extract anything meaningful (very
          short / malformed notes), it falls back to the original
          monospace <pre> render — that fallback is the JSX block
          inside `fallbackRenderer` below.

          OLD render preserved as the fallback path so the engineer
          never sees a blank panel even if parsing yields nothing:

          ── OLD (also the fallback) ──
          <pre style={{
            background: notePreBg, color: notePreText,
            border: `1px solid ${notePreBorder}`, borderRadius: 4,
            padding: 12, whiteSpace: "pre-wrap", wordBreak: "break-word",
            fontFamily: "var(--font-monospace, ui-monospace, SFMono-Regular, Menlo, monospace)",
            fontSize: 13, lineHeight: 1.45, maxHeight: 560, overflow: "auto",
          }}>
            {note}
          </pre>
          ── /OLD ── */}
      {/* Sprint 13.34 — reverted to the original monospace <pre>
          render at the user's request. The structured Handoff Report
          (HandoffReportBody + TimeCards grid + diagnostics + reason
          strip) is no longer the default; the engineer now sees the
          raw note text exactly as the backend assembled it.
          HandoffReportBody and the _parseHandoffNote/_layoutActivities
          helpers above are intentionally kept in the file so the
          structured render can be reinstated with a single-line swap
          if needed. */}
      {note ? (
        <pre
          style={{
            background: notePreBg,
            color: notePreText,
            border: `1px solid ${notePreBorder}`,
            borderRadius: 4,
            padding: 12,
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
            fontFamily:
              "var(--font-monospace, ui-monospace, SFMono-Regular, Menlo, monospace)",
            fontSize: 13,
            lineHeight: 1.45,
            maxHeight: 560,
            overflow: "auto",
          }}
        >
          {note}
        </pre>
      ) : null}
    </div>
  );
}


export default function Stage5EscalationPackage({
  // Sprint 12.9 — `data` (Sprint 7 Tier1EscalationPackage) and
  // `autoExpand` are kept in the destructure for callsite
  // back-compat, but no longer drive any rendering. The panel now
  // shows only Escalation Routing & Vendor/OEM Engagement + the
  // Generate Tier 2 Escalation Handoff action.
  data,            // eslint-disable-line no-unused-vars
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  helpfulMarked,
  autoExpand = true,  // eslint-disable-line no-unused-vars
  // Sprint 13.19 — Stage 3 checkbox state lifted from
  // ResolutionJourney. HandoffNoteAction reads ticked step
  // numbers from this map and posts them with the note request.
  attemptedStage3Steps,
}) {
  // Block 04 — inject scoped CSS once on first mount. Idempotent.
  React.useEffect(() => { _ensureB04Styles(); }, []);

  // Sprint 12.9 — `expanded` / `autoExpand` toggle suppressed along
  // with the Sprint 7 EscalationPackageCard render. Re-enable if
  // that view is ever reinstated.
  // const [expanded, setExpanded] = useState(autoExpand);

  // Sprint 12.7 — Escalation Routing fetched on mount. Failure here
  // only hides the routing block; the Generate-Handoff button still
  // works. Fire-and-forget; no spinner blocking the panel.
  // Sprint 13.30 — fetcher lifted to a useCallback so the child
  // HandoffNoteAction's Regenerate button can invoke it (alongside
  // the note refetch) for a hard-reload of the whole Stage 5 view.
  // The auto-fetch on mount is preserved verbatim — only Regenerate
  // gains an extra trigger path.
  const [routing, setRouting] = useState(null);
  const fetchRouting = React.useCallback(async () => {
    if (!sessionId) return;
    try {
      const r = await fetchEscalationRouting(sessionId);
      setRouting(r);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.warn("[journey.escalation_routing] fetch failed", err);
    }
  }, [sessionId]);
  useEffect(() => {
    fetchRouting();
  }, [fetchRouting]);

  // Sprint 12.9 — `if (!data)` early-out removed. The new view does
  // not depend on the Sprint 7 package payload, so a /stage-5 fetch
  // failure no longer blanks the card. Reinstate alongside the
  // EscalationPackageCard if Sprint 7 view returns.

  return (
    <Card
      style={{
        marginBottom: 16,
        borderLeft: "4px solid #B03A2E",
        // Premium revamp — per-card Acadia watermark.
        position: "relative",
        overflow: "hidden",
      }}
      // US Pharma — pale-red panel fill (matches Preliminary Tier 1
      // Checks). --usp-panel-bg is defined only under .org-uspharma, so
      // every other org falls back to transparent (unchanged glass card).
      bodyStyle={{ background: "var(--usp-panel-bg, transparent)" }}
    >
      {/* Sprint 13.35 — reverted to the original centered Acadia
          watermark at the user's request. The smaller top-right
          version is preserved below as a comment for easy revert.

          ── ALT (top-right, size 120, opacity 0.10) ──
          <CardWatermark position="top-right" size={120} opacity={0.10} />
          ── /ALT ── */}
      <CardWatermark />
      {/* ─── Block 04 — premium heading replacement (heading only) ───
          OLD heading preserved below for reference. Card wrapper +
          <CardWatermark/> + footer (Helpful/Dislike) + inner
          "Operational Handoff" sub-section banner (line ~415) all
          UNCHANGED — only the top-level panel title is re-typeset
          to match the Best Historical Match / Guided Workflow /
          Knowledge Base & SOP Reference header treatment.

          ── OLD ──
          <Title level={5} style={{ marginTop: 0, position: "relative", zIndex: 1 }}>
            Operational Handoff
          </Title>
          ── /OLD ── */}
      <div style={{ position: "relative", zIndex: 1, marginTop: 0, marginBottom: 8 }}>
        <div className="b04-eyebrow">
          <span aria-hidden className="b04-eyebrow__dot" />
          Tier-2 Handoff
        </div>
        <h2 className="b04-display">
          Operational <em className="b04-accent">handoff</em>
        </h2>
      </div>

      {/* Sprint 12.9 — Sprint 7 EscalationPackageCard render
          suppressed. Clicking "Escalate to Tier 2" now lands the
          engineer on the Escalation Routing & Vendor/OEM Engagement
          view (below) + the Generate Tier 2 Escalation Handoff
          action. The full Sprint 7 package view is preserved on
          disk; uncomment the block below + the EscalationPackageCard
          import + the `expanded`/`autoExpand` state above to
          reinstate. */}
      {/*
      {expanded ? (
        <EscalationPackageCard pkg={data} onClose={() => setExpanded(false)} />
      ) : (
        <Paragraph type="secondary" style={{ marginBottom: 12 }}>
          The escalation package is ready. Open it to copy the
          Tier-2 handoff text.
        </Paragraph>
      )}
      */}

      {/* Sprint 12.7 — Routing section. Hidden when empty.
          Sprint 12.9 — primary content of the Escalate-to-Tier-2 view. */}
      <EscalationRoutingSection routing={routing} />

      {/* Sprint 13.17 — Tier-2 handoff note. Auto-fetches on mount;
          no manual Generate button. The note text is what Tier-2
          will paste into ServiceNow / their ticket system.
          Sprint 13.19 — receives the lifted Stage 3 checkbox state
          so the note's Diagnostic Summary reflects ONLY the steps
          the engineer actually ticked. Re-fetches when ticks change. */}
      <HandoffNoteAction
        sessionId={sessionId}
        attemptedStage3Steps={attemptedStage3Steps}
        onRefreshRouting={fetchRouting}
      />

      <div
        style={{
          marginTop: 16,
          paddingTop: 12,
          borderTop: "1px solid var(--border-color, #f0f0f0)",
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <HelpfulButton
            sessionId={sessionId}
            stage="stage_5"
            onMarkedHelpful={onMarkedHelpful}
            onStartNewTicket={onStartNewTicket}
            disabled={helpfulMarked}
          />
          <DislikeButton
            sessionId={sessionId}
            stage="stage_5"
          />
        </div>
        <div style={{ display: "flex", gap: 8 }}>
          {/* Sprint 12.7.1 — "Show escalation package" CTA suppressed:
              with autoExpand=true (default) the package is already
              visible on first paint; the button only appeared after the
              engineer manually collapsed via EscalationPackageCard's
              Close affordance, which is rare on this card. Reinstate
              by un-commenting if the journey is ever switched to a
              collapsed-by-default Stage 5 (autoExpand={false}). */}
          {/*
          {!expanded ? (
            <Button type="primary" onClick={() => setExpanded(true)}>
              Show escalation package
            </Button>
          ) : null}
          */}
          {typeof onStartNewTicket === "function" ? (
            <Button onClick={onStartNewTicket}>New Ticket</Button>
          ) : null}
        </div>
      </div>
    </Card>
  );
}
