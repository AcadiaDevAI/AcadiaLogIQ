// Sprint 10.2 — Pivot Insights panel.
//
// Sprint 10.4 §2 + §3 — plain-English copy + redundant-card hide:
//   1. Heading "What NOT to chase" → "Known dead ends"
//   2. Smoking Gun half hidden when derived_from !== "mental_pivot_aggregate"
//      (the primary_fix_fallback case duplicated Stage 0's "What worked")
//   3. "Bypass:" label → "Recommended fix:"
//   4. DNC empty-state copy rewritten in plain English
//   5. below_threshold branch removed (taxonomy collapsed in §4)
//   6. DNC section always renders — engineer always sees either the
//      list or honest "we looked, here's what we found (or didn't)"

import React from "react";
import { Card, Divider, List, Tag, Typography } from "antd";
import { ThunderboltOutlined, CloseCircleOutlined } from "@ant-design/icons";

import DislikeButton from "./DislikeButton";
import HelpfulButton from "./HelpfulButton";
import EscalateButton from "./EscalateButton";
import NextStageButton from "./NextStageButton";
import { STAGE_LABELS } from "../tier1Constants";

const { Title, Paragraph, Text } = Typography;


// Sprint 10.4 §4 — empty-state copy by reason. Plain English only.
// `below_threshold` was removed from the schema in 10.4 since
// MIN_OCCURRENCE_COUNT=1 makes it unreachable.
const DNC_EMPTY_COPY_BY_REASON = {
  no_data:
    "No 'dead end' data available for these past tickets. Apply standard differential diagnosis.",
  no_recurring:
    "Each past ticket investigated different leads — no shared dead ends to flag.",
};

const DNC_FALLBACK_EMPTY_COPY = DNC_EMPTY_COPY_BY_REASON.no_data;


// ─────────────────────────────────────────────────────────────
// Smoking Gun half — Sprint 10.4 §3: only renders when the cohort's
// Knowledge_Base actually carries pivot data. The primary_fix_fallback
// case is hidden because it just echoes Stage 0's "What worked" line.
// ─────────────────────────────────────────────────────────────
function SmokingGunSection({ data }) {
  if (!data) return null;

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 8 }}>
        <div>
          <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>
            <ThunderboltOutlined style={{ marginRight: 8 }} />
            What to look for
          </Title>
        </div>
        {data.frequency_in_cohort_percent ? (
          <Tag color="orange">{data.frequency_in_cohort_percent}% of similar tickets</Tag>
        ) : null}
      </div>

      {data.pivot_signal ? (
        <Paragraph style={{ marginBottom: 8, marginTop: 8 }}>
          <Text strong>Pivot signal: </Text>
          {data.pivot_signal}
        </Paragraph>
      ) : null}
      {data.bypass_instruction ? (
        <Paragraph style={{ marginBottom: 8 }}>
          {/* Sprint 10.4 §2.1 — "Bypass:" → "Recommended fix:" */}
          <Text strong>Recommended fix: </Text>
          {data.bypass_instruction}
        </Paragraph>
      ) : null}
      {data.recommended_action ? (
        <Paragraph style={{ marginBottom: 8 }}>
          <Text strong>Recommended action: </Text>
          <Text code>{data.recommended_action}</Text>
        </Paragraph>
      ) : null}
      {data.seen_in_incidents && data.seen_in_incidents.length > 0 ? (
        <Paragraph style={{ marginBottom: 0 }}>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Seen in: {data.seen_in_incidents.join(", ")}
          </Text>
        </Paragraph>
      ) : null}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────
// Known dead ends section — Sprint 10.4 §4.4: always renders.
// ─────────────────────────────────────────────────────────────
function DoNotChaseSection({ data }) {
  if (!data) return null;
  const isEmpty = data.empty || !Array.isArray(data.entries) || data.entries.length === 0;
  const emptyCopy = isEmpty
    ? (DNC_EMPTY_COPY_BY_REASON[data.reason] || DNC_FALLBACK_EMPTY_COPY)
    : null;

  return (
    <div>
      <Title level={5} style={{ marginTop: 0 }}>
        <CloseCircleOutlined style={{ marginRight: 8 }} />
        {/* Sprint 13.3 — reverted "Known dead ends" → "What NOT to chase"
            so the heading matches the spec wording the LLM synthesis
            prompt is also based on. */}
        What NOT to chase
      </Title>

      {isEmpty ? (
        <Paragraph type="secondary">{emptyCopy}</Paragraph>
      ) : (
        <>
          {/* Sprint 13.4 — render strictly to spec: misleading_signal +
              rule-out logic. The previous "seen N×" tag, "Why:" prefix,
              and "Seen in: <incident>" footer were UI decoration not
              in the procedure spec — they made the panel feel cluttered
              and competed with the LLM-synthesised prose. */}
          <List
            itemLayout="vertical"
            size="small"
            dataSource={data.entries}
            renderItem={(entry, idx) => (
              <List.Item key={`${entry.misleading_signal}-${idx}`} style={{ paddingBottom: 8 }}>
                <div>
                  <Text strong>{entry.misleading_signal}</Text>
                </div>
                {entry.rule_out_logic ? (
                  <Paragraph style={{ marginTop: 4, marginBottom: 0 }}>
                    {entry.rule_out_logic}
                  </Paragraph>
                ) : null}
              </List.Item>
            )}
          />
          {/* Sprint 13.3 — small footnote when the LLM polish pass
              didn't run; engineer sees verbatim source text. Mirrors
              Stage 3's "(verbatim source — synthesis unavailable)". */}
          {data.synthesis_skipped ? (
            <Paragraph
              type="secondary"
              style={{ marginTop: 4, marginBottom: 0, fontSize: 12 }}
            >
              (Showing verbatim source text — synthesis unavailable.)
            </Paragraph>
          ) : null}
        </>
      )}
    </div>
  );
}


// ─────────────────────────────────────────────────────────────
// Combined panel
// ─────────────────────────────────────────────────────────────
export default function PivotInsightsPanel({
  data,                    // {smoking_gun, do_not_chase}
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,                // (toStage) → fetches + flips reveal[stage_2]
  helpfulMarked,
}) {
  if (!data) return null;
  const smokingGun = data.smoking_gun;
  const doNotChase = data.do_not_chase;

  // Sprint 10.4 §3.2 — only render the "What to look for" half when
  // it carries genuinely new info (KB-derived pivot signal). The
  // primary_fix_fallback variant duplicated Stage 0; "empty" had
  // nothing to show. Sprint 12.3 — also render when a single ticket
  // carries pivot data ("mental_pivot_single"); the data is real
  // even if the cohort threshold wasn't met.
  const showSmokingGun =
    smokingGun?.derived_from === "mental_pivot_aggregate" ||
    smokingGun?.derived_from === "mental_pivot_single";

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #C9870B" }}>
      {/* Sprint 13.10 — "What to look for" (Smoking Gun) section
          suppressed at the user's request. Backend still computes
          `smoking_gun` and ships it on /initial; only the JSX render
          is commented. Reinstate by un-commenting the line below
          if the Smoking Gun half is ever wanted again. */}
      {/* {showSmokingGun && <SmokingGunSection data={smokingGun} />} */}
      {/* Sprint 13.9 — "What NOT to chase" section suppressed at the
          user's request. Backend still computes `do_not_chase` and
          ships it on /initial; only the JSX render is commented.
          The Divider above it is also suppressed since with
          DoNotChaseSection gone there's nothing for it to separate.
          Reinstate by un-commenting the two lines below; no other
          plumbing change required. */}
      {/*
      {showSmokingGun && <Divider style={{ margin: "16px 0" }} />}
      <DoNotChaseSection data={doNotChase} />
      */}

      {/* Single Helpful + single next-stage button. */}
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
            stage="pivot_insights"
            onMarkedHelpful={onMarkedHelpful}
            onStartNewTicket={onStartNewTicket}
            disabled={helpfulMarked}
          />
          <DislikeButton
            sessionId={sessionId}
            stage="pivot_insights"
          />
        </div>
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {/* Sprint 11 — Escalate option alongside the primary
              "advance to next stage" CTA. fromStage matches the
              merged-Pivot-Insights telemetry surface so the
              traversal log labels the click correctly. */}
          <EscalateButton
            sessionId={sessionId}
            fromStage="pivot_insights"
            onReveal={onReveal}
          />
          <NextStageButton
            sessionId={sessionId}
            fromStage="pivot_insights"
            toStage="stage_2"
            label={STAGE_LABELS.stage_2}
            onReveal={onReveal}
          />
        </div>
      </div>
    </Card>
  );
}
