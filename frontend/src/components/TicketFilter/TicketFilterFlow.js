// Ticket Filter — right-pane flow.
//
// Sidebar's "Ticket Filter" button flips AppLayout's
// ``ticketFilterOpen`` flag and this component takes over the right
// pane. The engineer picks two dropdown values (SLA_Target_Met +
// Resolution_Quality_Score), clicks "Filter Tickets", and the
// matching ticket list appears as paginated cards.
//
// Design constraints
// ------------------
// * Independent of RCA / Gap Analysis / Chat — no shared imports
//   into those folders, no shared state, no shared CSS.
// * Submit button is DISABLED until both dropdowns have values
//   (matches the backend's required-both contract).
// * Empty result set renders the Ant Design ``<Empty>`` "No tickets
//   match these filters" — NOT an error state.
// * Pagination is bounded (50 max per page, hard-capped server-side
//   too) so a runaway click can't drag in 10K rows.

import React, { useState } from "react";
import {
  Alert,
  Button,
  Card,
  Empty,
  Pagination,
  Select,
  Space,
  Spin,
  Tag,
  Typography,
  message,
} from "antd";
import {
  FilterOutlined,
  LoadingOutlined,
  CheckCircleTwoTone,
  CloseCircleTwoTone,
} from "@ant-design/icons";

import { filterTickets } from "./ticketFilterApi";
import BackArrowButton from "../common/BackArrowButton";
import useSidebarPeek from "../../hooks/useSidebarPeek";


const { Title, Paragraph, Text } = Typography;


const SLA_OPTIONS = [
  { value: "True",  label: "True" },
  { value: "False", label: "False" },
];

const SCORE_OPTIONS = [
  { value: "1", label: "1" },
  { value: "2", label: "2" },
  { value: "3", label: "3" },
  { value: "4", label: "4" },
  { value: "5", label: "5" },
];

const DEFAULT_PAGE_SIZE = 20;


function PriorityTag({ value }) {
  // Lightweight colour mapping — P1 red, P2 orange, P3 blue,
  // unknown grey. Helps the engineer scan the list visually.
  if (!value) return null;
  const colour =
    value === "P1" ? "red"
    : value === "P2" ? "orange"
    : value === "P3" ? "blue"
    : "default";
  return <Tag color={colour}>{value}</Tag>;
}


function SlaPill({ value }) {
  if (value === "True") {
    return (
      <Tag icon={<CheckCircleTwoTone twoToneColor="#52c41a" />} color="success">
        SLA met
      </Tag>
    );
  }
  if (value === "False") {
    return (
      <Tag icon={<CloseCircleTwoTone twoToneColor="#ff4d4f" />} color="error">
        SLA missed
      </Tag>
    );
  }
  return null;
}


function StatusTag({ value }) {
  if (!value) return null;
  return <Tag>{value}</Tag>;
}


function ScoreTag({ value }) {
  if (!value) return null;
  // Visual cue: 4-5 green, 3 amber, 1-2 red.
  const n = parseInt(value, 10);
  const colour =
    n >= 4 ? "green"
    : n === 3 ? "gold"
    : n >= 1 ? "red"
    : "default";
  return <Tag color={colour}>Score {value}/5</Tag>;
}


// One ticket row.
//
// Top row: incident number + priority / status / SLA / score tags.
// Below: the ticket's INCIDENT narrative from Incident_Summary —
// this is the primary thing the engineer reads on the card.
//
// We deliberately do NOT render customer_name or timestamp here.
// The backend still ships those fields on the wire for future
// admin views; the current UI shows what the engineer actually
// needs to triage the result list.
function TicketCard({ ticket }) {
  return (
    <Card
      key={ticket.incident_number}
      size="small"
      style={{ marginBottom: 8 }}
      bodyStyle={{ padding: "12px 14px" }}
    >
      <Space size="middle" wrap style={{ marginBottom: ticket.incident ? 8 : 0 }}>
        <Text strong style={{ fontSize: 14 }}>
          {ticket.incident_number}
        </Text>
        <PriorityTag value={ticket.priority} />
        <StatusTag value={ticket.ticket_status} />
        <SlaPill value={ticket.sla_target_met} />
        <ScoreTag value={ticket.resolution_quality_score} />
      </Space>
      {ticket.incident ? (
        <Paragraph
          style={{
            marginBottom: 0,
            fontSize: 13,
            lineHeight: 1.55,
            color: "var(--text-primary, #1f2937)",
          }}
        >
          {ticket.incident}
        </Paragraph>
      ) : null}
    </Card>
  );
}


export default function TicketFilterFlow({ onReturnToStages }) {
  // Collapse the sidebar + enable hover-peek while this flow is
  // open, identical to the blocks screen. Hook restores the sidebar
  // to its expanded default on unmount, so leaving Ticket Filter
  // returns the engineer to landing/chat with an expanded sidebar.
  useSidebarPeek();

  // Dropdown selections. Both default to undefined so the submit
  // button stays disabled until the user makes both choices.
  const [sla, setSla] = useState(undefined);
  const [score, setScore] = useState(undefined);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);  // {tickets, total, page, page_size, has_more}
  const [page, setPage] = useState(1);

  const canSubmit = !!sla && !!score && !busy;

  // Run a fresh filter on the current selections — used for the
  // initial submit AND for any page change while the same filter
  // is active.
  const runFilter = async (nextPage) => {
    const p = Math.max(1, parseInt(nextPage || 1, 10));
    setBusy(true);
    setError(null);
    try {
      const data = await filterTickets({
        slaTargetMet: sla,
        resolutionQualityScore: score,
        page: p,
        pageSize: DEFAULT_PAGE_SIZE,
      });
      setResult(data);
      setPage(p);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[ticket_filter] query failed", err);
      const status = err?.response?.status;
      if (status === 429) {
        setError("Too many filter requests — wait a moment and try again.");
      } else if (status === 422) {
        setError("Invalid filter values. Please pick from the dropdowns.");
      } else {
        setError("Something went wrong filtering tickets. Please try again.");
      }
      setResult(null);
      message.error("Filter failed");
    } finally {
      setBusy(false);
    }
  };

  const handleSubmit = () => {
    if (!canSubmit) return;
    runFilter(1);
  };

  // Changing either dropdown invalidates the current result set so
  // the user doesn't see stale data with a "stale filter" mismatch.
  const handleSlaChange = (v) => {
    setSla(v);
    setResult(null);
    setError(null);
  };
  const handleScoreChange = (v) => {
    setScore(v);
    setResult(null);
    setError(null);
  };

  return (
    <div className="flex flex-col h-full w-full t-bg-primary">
      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="w-full max-w-5xl mx-auto">
          {/* Top-left back affordance — quick bail-out path. The
              bottom-right "Return to Stages" button below stays as
              the deliberate-completion exit. Both wire into the
              same onReturnToStages prop so they share behaviour. */}
          <div style={{ marginBottom: 12 }}>
            <BackArrowButton
              onClick={() => {
                if (typeof onReturnToStages === "function") {
                  onReturnToStages();
                }
              }}
            />
          </div>

          {/* Header */}
          <div style={{ marginBottom: 16 }}>
            <Title level={3} style={{ marginBottom: 4 }}>
              Ticket Filter
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Pull historical tickets that match an SLA outcome and a
              resolution-quality score. Both fields are required.
            </Paragraph>
          </div>

          {/* Filter form */}
          <Card style={{ marginBottom: 16 }} bodyStyle={{ padding: 16 }}>
            <Space size="middle" wrap>
              <div style={{ minWidth: 200 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  SLA target met
                </Text>
                <Select
                  placeholder="Select SLA outcome"
                  value={sla}
                  onChange={handleSlaChange}
                  options={SLA_OPTIONS}
                  style={{ width: "100%", marginTop: 4 }}
                  disabled={busy}
                  aria-label="SLA target met"
                />
              </div>
              <div style={{ minWidth: 200 }}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  Resolution quality score
                </Text>
                <Select
                  placeholder="Select score"
                  value={score}
                  onChange={handleScoreChange}
                  options={SCORE_OPTIONS}
                  style={{ width: "100%", marginTop: 4 }}
                  disabled={busy}
                  aria-label="Resolution quality score"
                />
              </div>
              <div style={{ alignSelf: "flex-end" }}>
                <Button
                  type="primary"
                  size="large"
                  icon={busy ? <LoadingOutlined /> : <FilterOutlined />}
                  onClick={handleSubmit}
                  disabled={!canSubmit}
                >
                  {busy ? "Filtering…" : "Filter Tickets"}
                </Button>
              </div>
            </Space>

            {error ? (
              <Alert
                type="error"
                showIcon
                message={error}
                style={{ marginTop: 12 }}
              />
            ) : null}
          </Card>

          {/* Loading state */}
          {busy && !result ? (
            <Card>
              <div style={{ textAlign: "center", padding: 32 }}>
                <Spin size="large" />
                <Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
                  Searching matching tickets…
                </Paragraph>
              </div>
            </Card>
          ) : null}

          {/* Results */}
          {result && !busy ? (
            result.total === 0 ? (
              <Card>
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description="No tickets match these filters"
                />
              </Card>
            ) : (
              <>
                <div style={{ marginBottom: 8 }}>
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    Showing {result.tickets.length} of {result.total} matching tickets
                  </Text>
                </div>
                {result.tickets.map((t) => (
                  <TicketCard key={t.incident_number} ticket={t} />
                ))}
                {result.total > result.page_size ? (
                  <div style={{ marginTop: 16, textAlign: "right" }}>
                    <Pagination
                      current={result.page}
                      pageSize={result.page_size}
                      total={result.total}
                      onChange={(p) => runFilter(p)}
                      showSizeChanger={false}
                      disabled={busy}
                    />
                  </div>
                ) : null}
              </>
            )
          ) : null}
        </div>
      </div>
    </div>
  );
}
