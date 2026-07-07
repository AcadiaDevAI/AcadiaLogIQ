// Sprint 10.2 Stage 0 — Best-Ticket Distillation banner.
// Sprint 10.3 — added evidence_strength branching; CorpusStatsTail
// extracted to its own file with a default export.
//
// Renders the cohort's highest-quality past resolution. The headline
// copy varies by evidence_strength so a score-3 ticket isn't labelled
// "best-rated" (which implies high confidence).
//
// Sparse fallback (data.sparse=true) renders the spec's spare copy.
// No buttons — Stage 0 is informational lead-in only.

import React from "react";
import { Button, Card, List, Tag, Tooltip, Typography } from "antd";
import { MessageOutlined, ExportOutlined } from "@ant-design/icons";

import CardWatermark from "../../../components/Tier1Copilot/journey/CardWatermark";
import CorpusStatsTail from "../../../components/Tier1Copilot/journey/CorpusStatsTail";
import DislikeButton from "../../../components/Tier1Copilot/journey/DislikeButton";
import EscalateButton from "../../../components/Tier1Copilot/journey/EscalateButton";
import HelpfulButton from "../../../components/Tier1Copilot/journey/HelpfulButton";
// Sprint 13.10 — Stage 0 now offers a direct shortcut to Stage 3
// (Guided Troubleshooting Workflow) alongside Escalate, so the
// engineer can skip the intermediate Pivot Insights / Stage 2
// panels when they want to go straight to the playbook.
import NextStageButton from "../../../components/Tier1Copilot/journey/NextStageButton";
import { STAGE_LABELS } from "../../../components/Tier1Copilot/tier1Constants";
import { stripLeadingNumber } from "../../../components/Tier1Copilot/journey/stepText";
import useChatHandoff from "../../../components/Tier1Copilot/journey/useChatHandoff";

const { Title, Text, Paragraph } = Typography;


// ─────────────────────────────────────────────────────────────
// Block 01 — Best Historical Match & Recommended Resolution
// Scoped 7-step blue scale + 8 type roles.
// Class names are `.b01-*` so this stylesheet cannot bleed into
// any other panel (Stage 2 / Stage 3 / RCA / Gap Analysis all
// remain visually untouched). Injected once per session via the
// module guard below.
// Roles:
//   eyebrow  — small uppercase mono label above the headline
//   display  — large headline ("We found N similar matches.")
//   accent   — gradient-painted word inside the headline ("match")
//   subhead  — secondary line beneath the headline (carries cohort)
//   rowIndex — 01 / 02 / 03 … left-side row counter (tabular nums)
//   rowBody  — main descriptive text inside the row
//   rowId    — inline ticket id (e.g. INC-PHOENIX-402)
//   rowLink  — "Discuss →" right-aligned per-row action
// ─────────────────────────────────────────────────────────────
const B01_BLUE = {
  50:  "#EFF6FF",
  100: "#DBEAFE",
  200: "#BFDBFE",
  400: "#60A5FA",
  500: "#3B82F6",
  600: "#2563EB",
  700: "#1D4ED8",
};
const B01_CSS = `
.b01-root { color: #0F172A; }
.b01-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 12px; border-radius: 9999px;
  background: ${B01_BLUE[50]};
  border: 1px solid ${B01_BLUE[200]};
  color: ${B01_BLUE[700]};
  font-family: var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace);
  font-size: 10.5px; font-weight: 500; letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.b01-eyebrow__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: ${B01_BLUE[500]};
  box-shadow: 0 0 8px ${B01_BLUE[400]};
}
.b01-display {
  font-family: var(--font-display, 'Instrument Serif', Georgia, serif);
  font-size: clamp(26px, 3.2vw, 36px);
  font-weight: 400; line-height: 1.12; letter-spacing: -0.018em;
  margin: 0 0 6px 0; color: #0F172A;
}
.b01-accent {
  font-style: italic; font-weight: 400;
  background: linear-gradient(135deg, ${B01_BLUE[400]} 0%, ${B01_BLUE[600]} 60%, ${B01_BLUE[700]} 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
.b01-subhead {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 13px; line-height: 1.5; color: #475569;
  margin: 0 0 18px 0;
}
.b01-rows { list-style: none; padding: 0; margin: 0 0 4px 0; }
.b01-row {
  display: grid;
  grid-template-columns: 40px 1fr auto;
  gap: 14px;
  align-items: start;
  padding: 10px 0;
  border-top: 1px solid ${B01_BLUE[100]};
}
.b01-row:first-child { border-top: none; padding-top: 4px; }
.b01-rowIndex {
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 13px; font-weight: 600; color: ${B01_BLUE[600]};
  line-height: 1.55; padding-top: 1px;
}
.b01-rowBody {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 14px; line-height: 1.55; color: #0F172A;
}
.b01-rowId {
  display: inline-block;
  font-family: var(--font-mono, 'Geist Mono', monospace);
  font-feature-settings: 'tnum' 1;
  font-size: 12.5px; font-weight: 600;
  color: ${B01_BLUE[700]};
  background: ${B01_BLUE[50]};
  border: 1px solid ${B01_BLUE[200]};
  padding: 1px 8px; border-radius: 6px;
  margin-right: 8px;
}
.b01-rowLink {
  font-family: var(--font-body, 'Geist', system-ui, sans-serif);
  font-size: 12.5px; font-weight: 600;
  color: ${B01_BLUE[600]};
  background: transparent; border: none; padding: 4px 0; cursor: pointer;
  white-space: nowrap;
  transition: color 160ms ease, transform 160ms ease;
}
.b01-rowLink:hover:not(:disabled) { color: ${B01_BLUE[700]}; transform: translateX(2px); }
.b01-rowLink:disabled { color: ${B01_BLUE[400]}; cursor: not-allowed; }
`;
let _b01StylesInjected = false;
function _ensureB01Styles() {
  if (typeof document === "undefined" || _b01StylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-block01", "1");
  style.textContent = B01_CSS;
  document.head.appendChild(style);
  _b01StylesInjected = true;
}


function formatMinutes(mins) {
  if (mins == null) return null;
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}


function ringColor(cleanPct) {
  if (cleanPct >= 80) return "#0A7A3F"; // green
  if (cleanPct >= 50) return "#C9870B"; // amber
  return "#B03A2E";                     // red
}


// Sprint 12.1 — Each "How they did it" bullet ends with " - INC-XXX"
// (suffix appended by backend `_resolution_steps`). Extract that
// trailing incident id so the per-bullet "Ask in chat" handoff can
// scope the resulting chat session to that source ticket only.
//
// Returns the incident id string (e.g. "INC-PHOENIX-402") or null
// when no recognisable suffix is found — a null scope just means the
// chat opens unscoped (global Search-in-KB behaviour), which is a
// safe fall-back. Tolerant matcher: accepts a hyphenated alphanumeric
// id with at least one dash, matching the project's INC-* / TKT-*
// patterns without hard-coding a specific prefix.
function extractTrailingIncidentId(stepText) {
  if (typeof stepText !== "string") return null;
  // Pattern: " - <UPPERCASE-TOKEN-WITH-DASH>" anchored at end of string,
  // optional trailing whitespace tolerated.
  const m = stepText.match(/\s-\s([A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+)\s*$/);
  return m ? m[1] : null;
}


// Sprint 13.7 — headline simplified. The previous evidence-strength
// branching appended "The best fix came from INC-XXX (rated N/5)" /
// "Closest match: INC-XXX" tails — those have been removed at the
// request of the user. The headline now reads only the cohort-count
// preamble; the per-ticket detail rendered below the headline carries
// the source incident IDs directly. `formatMinutes` and the unused
// branches are left in place via this single-line builder so the
// imports / signatures don't need a follow-up cleanup.
//
// Sprint 13.32.3 — count made dynamic against the RENDERED list, not
// `data.cohort_size`. Backend's cohort_size counts every cohort dict
// regardless of whether it has a usable Incident_Summary; the list
// below ("Possible details are:") only renders entries that *do*
// have one. When one of the cohort tickets has no summary, the user
// saw e.g. "We found 5 similar instances" but only 4 bullets. Now
// we count what the engineer can actually see. cohort_size remains
// the fallback only when the summaries field is absent entirely.
// Pluralisation also fixed (was appending "s" to "issue" instead of
// "instance"; the new copy switches the word that varies).
function buildHeadline(data) {
  const visible = Array.isArray(data.top5_incident_summaries)
    ? data.top5_incident_summaries.length
    : 0;
  const n = visible || data.cohort_size || 0;
  const noun = n === 1 ? "instance" : "instances";
  return `We found ${n} similar ${noun} for this issue.`;
}


export default function Stage0BestTicketDistillation({
  data,
  sessionId,
  onReveal,
  // Sprint 13.2 — Stage 0 now carries Helpful + Dislike alongside
  // the existing Escalate. Parent passes the same onMarkedHelpful /
  // onStartNewTicket / helpfulMarked plumbing it already supplies
  // to every other stage panel.
  onMarkedHelpful,
  onStartNewTicket,
  helpfulMarked,
}) {
  // Sprint 11 — per-step "Ask in chat" links. Hook is a no-op when
  // sessionId is missing (defensive — Stage 0 should always have one).
  const { busy: handoffBusy, askInChat } = useChatHandoff(sessionId);

  // Block 01 — inject scoped CSS once on first mount. Idempotent.
  React.useEffect(() => { _ensureB01Styles(); }, []);

  if (!data) return null;

  // ── Sparse case (cohort empty) ──
  if (data.sparse) {
    return (
      <Card
        style={{
          marginBottom: 16,
          borderLeft: "4px solid #6B6B6B",
          position: "relative",
          overflow: "hidden",
        }}
      >
        <CardWatermark />
        <Title level={5} style={{ marginTop: 0, position: "relative", zIndex: 1 }}>Best Historical Match & Recommended Resolution</Title>
        {/* Sprint 13.8 — profile_match line suppressed in the sparse
            case too, for parity with the main panel above. */}
        {/*
        {data.profile_match ? (
          <Paragraph style={{ marginBottom: 8 }}>
            <Text type="secondary">{data.profile_match}</Text>
          </Paragraph>
        ) : null}
        */}
        <Paragraph style={{ marginBottom: 0 }}>
          {/* Sprint 13.7 — softer no-data copy. */}
          No relevant historical tickets were found for this issue.
          Please review the recommended actions below.
        </Paragraph>
      </Card>
    );
  }

  const color = ringColor(data.clean_resolution_percent || 0);

  return (
    <Card
      style={{
        marginBottom: 16,
        borderLeft: `4px solid ${color}`,
        // Premium revamp — per-card Acadia watermark, centered behind
        // content. Card needs `position: relative` + `overflow: hidden`
        // for the absolute-positioned <CardWatermark> to sit inside
        // the card's rounded border.
        position: "relative",
        overflow: "hidden",
      }}
    >
      <CardWatermark />
      {/* ─── Block 01 — Best Historical Match (premium typography) ─────
          The Card wrapper, <CardWatermark/>, footer (Helpful/Dislike/
          Escalate/NextStage) all remain UNCHANGED. Only the title +
          headline + bullet-list region below has been re-typeset
          against the 8-role scale defined in B01_CSS above.

          The OLD render is preserved in the comment block immediately
          below so it can be reinstated by uncommenting and removing
          the new block:

          ── OLD (Sprint 13.7 / 13.8) ──
          <Title level={5} style={{ marginTop: 0, marginBottom: 16, position: "relative", zIndex: 1 }}>
            Best Historical Match & Recommended Resolution
          </Title>
          <Paragraph style={{ marginBottom: 12 }}>
            <Text strong>{buildHeadline(data)}</Text>
          </Paragraph>
          {data.top5_incident_summaries && data.top5_incident_summaries.length > 0 ? (
            <div style={{ marginBottom: 8 }}>
              <Text strong>Possible details are:</Text>
              <List
                size="small"
                dataSource={data.top5_incident_summaries}
                renderItem={(s, i) => {
                  const cleaned = stripLeadingNumber(s);
                  const bulletIncident =
                    extractTrailingIncidentId(cleaned) || data.best_incident || null;
                  return (
                    <List.Item key={i} style={{ paddingLeft: 8, display: "flex", alignItems: "flex-start", gap: 8 }}>
                      <span style={{ flex: 1 }}>{i + 1}. {cleaned}</span>
                      {sessionId && cleaned ? (
                        <Tooltip title={bulletIncident ? `Discuss this with Logic — the chat will be scoped to ${bulletIncident}` : "Discuss this with Logic"}>
                          <Button type="link" size="small" icon={<MessageOutlined />}
                            loading={handoffBusy} onClick={() => askInChat(cleaned, bulletIncident)}
                            style={{ paddingLeft: 0, paddingRight: 0 }}>
                            Discuss with LogIQ
                          </Button>
                        </Tooltip>
                      ) : null}
                    </List.Item>
                  );
                }}
              />
            </div>
          ) : null}
          ── /OLD ── */}
      <div className="b01-root" style={{ position: "relative", zIndex: 1 }}>
        {/* Role: eyebrow — small uppercase mono chip */}
        <div className="b01-eyebrow">
          <span aria-hidden className="b01-eyebrow__dot" />
          Best Historical Match
        </div>

        {/* Role: display + accent word "match" (must literally be the
            accent per spec). Cohort count is computed off the live
            visible-rows count exactly as `buildHeadline` did, so no
            data-shape change. */}
        {(() => {
          const visibleCount = Array.isArray(data.top5_incident_summaries)
            ? data.top5_incident_summaries.length
            : 0;
          const n = visibleCount || data.cohort_size || 0;
          const noun = n === 1 ? "match" : "matches";
          return (
            <h2 className="b01-display">
              We found {n} similar{" "}
              <em className="b01-accent">{noun}</em> for this issue.
            </h2>
          );
        })()}

        {/* Role: subhead — carries cohort size (formatted with
            thousands separators, e.g. "14,847" when the corpus is
            that large). `data.corpus_size` is the upstream library
            count; falls back to cohort_size when the larger figure
            isn't shipped, and is omitted entirely when neither is
            available so we never show a dishonest "0 resolved" line. */}
        {(() => {
          const corpus = (
            typeof data.corpus_size === "number" ? data.corpus_size
              : (typeof data.total_resolved === "number" ? data.total_resolved
                : (typeof data.cohort_size === "number" ? data.cohort_size : null))
          );
          if (corpus == null) return null;
          return (
            <p className="b01-subhead">
              Drawn from {corpus.toLocaleString("en-US")} resolved tickets in
              the Acadia knowledge base.
            </p>
          );
        })()}

        {/* Numbered rows — rowIndex (01/02…), rowBody w/ inline rowId
            chip, rowLink ("Discuss →"). Rendered when top5 exists. */}
        {data.top5_incident_summaries && data.top5_incident_summaries.length > 0 ? (
          <ul className="b01-rows">
            {data.top5_incident_summaries.map((s, i) => {
              const cleaned = stripLeadingNumber(s);
              const bulletIncident =
                extractTrailingIncidentId(cleaned) || data.best_incident || null;
              // Strip the trailing " - INC-XXX" suffix from the body
              // so the id renders ONCE (as the inline chip) instead of
              // being duplicated at the end of the sentence.
              const bodyText = bulletIncident
                ? cleaned.replace(/\s*-\s*[A-Z][A-Z0-9-]+\s*$/, "")
                : cleaned;
              const idx = String(i + 1).padStart(2, "0");
              return (
                <li className="b01-row" key={i}>
                  <span className="b01-rowIndex">{idx}</span>
                  <span className="b01-rowBody">
                    {bulletIncident ? (
                      <span className="b01-rowId">{bulletIncident}</span>
                    ) : null}
                    {bodyText}
                  </span>
                  {sessionId && cleaned ? (
                    <Tooltip
                      title={
                        bulletIncident
                          ? `Discuss this with LogIQ — chat scoped to ${bulletIncident}`
                          : "Discuss this with LogIQ"
                      }
                    >
                      <button
                        type="button"
                        className="b01-rowLink"
                        disabled={handoffBusy}
                        onClick={() => askInChat(cleaned, bulletIncident)}
                        aria-label={`Discuss ${bulletIncident || "this match"} with LogIQ`}
                        style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                      >
                        Discuss with LogIQ
                        <ExportOutlined aria-hidden="true" style={{ fontSize: 12 }} />
                      </button>
                    </Tooltip>
                  ) : null}
                </li>
              );
            })}
          </ul>
        ) : null}
      </div>

      {data.critical_intervention ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Critical intervention: </Text>
          {data.critical_intervention}
        </Paragraph>
      ) : null}

      {/* Sprint 10.3 — CorpusStatsTail is its own component now. */}
      <CorpusStatsTail data={data} />

      {/* Sprint 11 — Escalate button on every stage. Stage 0 has no
          "next stage" CTA of its own (informational lead-in), so this
          sits alone at the bottom right. Click → Stage 5 with a
          traversal log that records "viewed Stage 0, advanced to
          stage_5 at HH:MM UTC". */}
      {sessionId && typeof onReveal === "function" ? (
        <div
          style={{
            marginTop: 12,
            paddingTop: 12,
            borderTop: "1px solid var(--border-color, #f0f0f0)",
            display: "flex",
            flexWrap: "wrap",
            gap: 12,
            justifyContent: "space-between",
            alignItems: "center",
          }}
        >
          {/* Sprint 13.2 — Helpful + Dislike pair, mirrors every
              other stage's footer. Helpful opens the positive-feedback
              modal (chat parity); Dislike opens the negative one. */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <HelpfulButton
              sessionId={sessionId}
              stage="stage_0"
              onMarkedHelpful={onMarkedHelpful}
              onStartNewTicket={onStartNewTicket}
              disabled={helpfulMarked}
            />
            <DislikeButton
              sessionId={sessionId}
              stage="stage_0"
            />
          </div>
          {/* US Pharma edit — the "Guided Troubleshooting Workflow"
              shortcut (→ Stage 3) is replaced by a direct "KB SOP"
              shortcut that advances to Stage 4 (Search KB / SOP
              handoff). Escalate-to-Tier-2 remains alongside it. */}
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <EscalateButton
              sessionId={sessionId}
              fromStage="stage_0"
              onReveal={onReveal}
            />
            <NextStageButton sessionId={sessionId} fromStage="stage_0" toStage="stage_4" label="KB SOP" onReveal={onReveal} />
          </div>
        </div>
      ) : null}
    </Card>
  );
}
