import React from "react";
import { Alert, Button, Card, Tag, Tooltip } from "antd";
import {
  DownloadOutlined,
  LeftOutlined,
  RightOutlined,
} from "@ant-design/icons";
import {
  CONFIDENCE_COLOR,
  CONFIDENCE_LABEL,
  CONFIDENCE_LABEL_V2,
  TIER1_DOWNLOAD_DEMO_ON, // Sprint 8.1 demo
  TIER1_PROGRESSIVE_ON,
  TIER1_UX_FIXES_ON,
} from "./tier1Constants";
import SkeletonCard from "./SkeletonCard";
import { useTier1Theme } from "../../theme/ThemeProvider";
import { downloadAllMatches } from "./demoDownloads"; // Sprint 8.1 demo

/**
 * Tier1AnswerCard — 8-section Tier-1 answer + optional Sprint 7
 * arrow pagination + confidence banner.
 *
 * Sprint 6 contract preserved: when Sprint 7 props (matchIndex,
 * totalMatches, onPrev, onNext) are not supplied, the card renders
 * exactly the Sprint 6 layout. The `progressive` flag (default False
 * for Sprint 6 callers) also controls the new Low/None banners.
 */
export default function Tier1AnswerCard({
  result,
  progressive = false,
  matchIndex = 0,
  totalMatches = 0,
  onPrev,
  onNext,
  // Sprint 8 additions
  paginating = false,
  // Sprint 8.1 demo — props wired through from Tier1Workspace so the
  // "Download all 5" button can fetch + serialise each rank.
  sessionId = null,
  fetchMatchByIndex = null,
  alertPayload = null,
}) {
  const { tokens, isModern } = useTier1Theme();
  if (!result) return null;
  const {
    matched_incident: matchedIncident,
    confidence,
    similar_count: similarCount,
    cache_hit: cacheHit,
    answer,
  } = result;
  const a = answer || {};
  const checks = Array.isArray(a.recommended_first_checks)
    ? a.recommended_first_checks
    : [];

  const confidenceColor = CONFIDENCE_COLOR[confidence] || CONFIDENCE_COLOR.None;
  const sprint7 = progressive && TIER1_PROGRESSIVE_ON;
  // Sprint 8 — prefer the neutralised label dict when the UX-fixes
  // flag is on. Fall back to Sprint 7's legacy strings otherwise.
  const labelV2 =
    TIER1_UX_FIXES_ON && sprint7
      ? CONFIDENCE_LABEL_V2[confidence]
      : null;
  const label = labelV2
    ? labelV2.primary
    : sprint7
    ? CONFIDENCE_LABEL[confidence] || `${confidence} confidence`
    : `${confidence} confidence`;
  const subline = labelV2 ? labelV2.subline : null;
  const showArrows = sprint7 && totalMatches > 1;

  const cardStyle = isModern
    ? {
        backgroundColor: tokens.surfaceBase,
        borderColor: tokens.surfaceElevated,
        borderRadius: tokens.radiusLg,
        boxShadow: tokens.shadowMd,
      }
    : {
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
      };

  // Sprint 8.1 demo — shared download button (rendered in both paginating
  // and normal states so testers can grab the bundle either way).
  const downloadButton = TIER1_DOWNLOAD_DEMO_ON ? (
    <Tooltip title="Download all 5 matches as .txt (demo only)">
      <Button
        size="small"
        icon={<DownloadOutlined />}
        onClick={() =>
          downloadAllMatches(sessionId, fetchMatchByIndex, alertPayload)
        }
        disabled={!sessionId || !fetchMatchByIndex}
      >
        Download all 5
      </Button>
    </Tooltip>
  ) : null;

  // Sprint 8 — while a rank-N fetch is in flight, replace the body with
  // a SkeletonCard but KEEP the header + arrows interactive so the
  // engineer can cancel/redirect without waiting.
  if (paginating) {
    return (
      <Card bodyStyle={{ padding: 24 }} style={cardStyle}>
        {showArrows && (
          <div className="flex justify-between items-center mb-3">
            <Button
              size="small"
              icon={<LeftOutlined />}
              disabled={matchIndex <= 0}
              onClick={onPrev}
            >
              Prev
            </Button>
            <span className="t-text-muted text-xs">
              {matchIndex + 1} / {totalMatches}
            </span>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              {downloadButton /* Sprint 8.1 demo */}
              <Button
                size="small"
                icon={<RightOutlined />}
                disabled={matchIndex >= totalMatches - 1}
                onClick={onNext}
              >
                Next
              </Button>
            </div>
          </div>
        )}
        <SkeletonCard variant="answer" />
      </Card>
    );
  }

  return (
    <Card bodyStyle={{ padding: 24 }} style={cardStyle}>
      {showArrows && (
        <div className="flex justify-between items-center mb-3">
          <Button
            size="small"
            icon={<LeftOutlined />}
            disabled={matchIndex <= 0}
            onClick={onPrev}
          >
            Prev
          </Button>
          <span className="t-text-muted text-xs">
            {matchIndex + 1} / {totalMatches}
          </span>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            {downloadButton /* Sprint 8.1 demo */}
            <Button
              size="small"
              icon={<RightOutlined />}
              disabled={matchIndex >= totalMatches - 1}
              onClick={onNext}
            >
              Next
            </Button>
          </div>
        </div>
      )}
      {/* Sprint 8.1 demo — when arrows aren't shown (single match),
          still expose the download button on its own row. */}
      {!showArrows && downloadButton && (
        <div className="flex justify-end mb-2">{downloadButton}</div>
      )}

      <div className="flex justify-between items-center mb-4 flex-wrap gap-2">
        <div>
          <div className="t-text-muted text-xs uppercase tracking-wide">
            Matched incident
          </div>
          <div className="t-text font-semibold">
            {matchedIncident || "— no historical match —"}
          </div>
        </div>
        <div className="flex gap-2 items-center" style={{ textAlign: "right" }}>
          <div>
            <Tag color={confidenceColor} style={{ fontWeight: 600 }}>
              {label}
            </Tag>
            {subline && (
              <div
                className="t-text-muted"
                style={{ fontSize: 11, marginTop: 2 }}
              >
                {subline}
              </div>
            )}
          </div>
          <Tag>{similarCount || 0} similar</Tag>
          {cacheHit ? <Tag color="purple">cache hit</Tag> : null}
        </div>
      </div>

      {sprint7 && confidence === "Low" && !TIER1_UX_FIXES_ON && (
        <Alert
          type="warning"
          showIcon
          className="mb-4"
          message="Weak match — use as reference only"
          description="The retrieval found a ticket with partial overlap. Verify before acting on its recommendation."
        />
      )}
      {sprint7 && confidence === "None" && !TIER1_UX_FIXES_ON && (
        <Alert
          type="error"
          showIcon
          className="mb-4"
          message="No historical evidence"
          description="Consider opening a new incident and capturing your observation for future matches."
        />
      )}
      {/* Sprint 8 — the low/none copy has moved into the header
          confidence subline, so the harsh inline Alert blocks are
          suppressed when the UX-fixes flag is on. */}

      <Section title="Issue Understanding" body={a.issue_understanding} />
      <Section title="Historical Match" body={a.historical_match} />
      <Section title="Most Likely Cause" body={a.most_likely_cause} />
      <SectionList title="Recommended First Checks" items={checks} />
      <Section title="Most Likely Fix" body={a.most_likely_fix} />
      <Section title="Validation" body={a.validation} />
      <Section title="Escalate If" body={a.escalate_if} />
      <Section title="Follow-up Question" body={a.follow_up_question} />
    </Card>
  );
}

function Section({ title, body }) {
  if (!body) return null;
  return (
    <div className="mb-4">
      <div className="t-text font-semibold mb-1">{title}</div>
      <div className="t-text text-sm whitespace-pre-wrap">{body}</div>
    </div>
  );
}

function SectionList({ title, items }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="mb-4">
      <div className="t-text font-semibold mb-1">{title}</div>
      <ol className="list-decimal list-inside t-text text-sm space-y-1">
        {items.map((it, i) => (
          <li key={i}>{it}</li>
        ))}
      </ol>
    </div>
  );
}
