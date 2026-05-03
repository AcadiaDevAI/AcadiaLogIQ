// Sprint 10 Stage 0 — Confidence Lead.
//
// Per spec §3.1 + §9: banner with profile match, headline sentence,
// frequency, success rate, avg time vs platform median. Color ring
// derived from success_rate_percent: ≥80% high, 50-79% medium, <50% low.
// NO BUTTONS — Stage 0 is informational lead-in only.
//
// Sparse case (data.sparse=true): suppress percentages and render the
// spare copy "Limited historical data — proceed with the playbook below."

import React from "react";
import { Card, Row, Col, Statistic, Tag, Typography } from "antd";

const { Title, Text, Paragraph } = Typography;


function ringColor(successRate) {
  if (successRate >= 80) return "#0A7A3F";   // high — green
  if (successRate >= 50) return "#C9870B";   // medium — amber
  return "#B03A2E";                          // low — red
}


function ringTone(successRate) {
  if (successRate >= 80) return "High";
  if (successRate >= 50) return "Medium";
  return "Low";
}


function formatMinutes(mins) {
  if (mins == null) return null;
  if (mins < 60) return `${mins}m`;
  const h = Math.floor(mins / 60);
  const m = mins % 60;
  return m > 0 ? `${h}h ${m}m` : `${h}h`;
}


function buildHeadline(d) {
  const seen = d.seen_count || 0;
  const resolved = d.resolved_count || 0;
  const pct = d.success_rate_percent || 0;
  const months = d.lookback_months || 18;
  if (seen <= 0) return "";
  return (
    `We have seen this ${seen} time${seen === 1 ? "" : "s"} in the last ` +
    `${months} months. The recommended approach worked ` +
    `${resolved} of those ${seen} time${seen === 1 ? "" : "s"} (${pct}%).`
  );
}


export default function Stage0ConfidenceLead({ data }) {
  if (!data) return null;

  // Sparse case — render the placeholder banner per spec
  if (data.sparse) {
    return (
      <Card style={{ marginBottom: 16, borderLeft: "4px solid #6B6B6B" }}>
        <Title level={5} style={{ marginTop: 0 }}>Stage 0 — Confidence Lead</Title>
        <Paragraph style={{ marginBottom: 0 }}>
          {data.profile_match ? (
            <Text type="secondary">{data.profile_match}</Text>
          ) : null}
        </Paragraph>
        <Paragraph style={{ marginTop: 8, marginBottom: 0 }}>
          Limited historical data — proceed with the playbook below.
        </Paragraph>
      </Card>
    );
  }

  const color = ringColor(data.success_rate_percent || 0);
  const tone = ringTone(data.success_rate_percent || 0);
  const avgFmt = formatMinutes(data.avg_minutes_to_resolve);
  const medianFmt = formatMinutes(data.platform_median_minutes);

  return (
    <Card
      style={{
        marginBottom: 16,
        borderLeft: `4px solid ${color}`,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", flexWrap: "wrap", gap: 12 }}>
        <div style={{ minWidth: 240, flex: 1 }}>
          <Title level={5} style={{ marginTop: 0, marginBottom: 4 }}>
            Stage 0 — Confidence Lead
          </Title>
          {data.profile_match ? (
            <Tag style={{ marginBottom: 8 }}>{data.profile_match}</Tag>
          ) : null}
          <Paragraph style={{ marginBottom: 0 }}>
            <Text strong>{buildHeadline(data)}</Text>
          </Paragraph>
        </div>
        <Tag color={color} style={{ fontSize: 14, fontWeight: 600, padding: "4px 12px" }}>
          {tone} confidence · {data.success_rate_percent}%
        </Tag>
      </div>

      <Row gutter={16} style={{ marginTop: 16 }}>
        <Col flex="auto">
          <Statistic
            title="Times seen"
            value={data.seen_count}
            suffix={
              data.corpus_size
                ? <Text type="secondary" style={{ fontSize: 12 }}>
                    of {data.corpus_size} ({data.seen_percent_of_corpus}%)
                  </Text>
                : null
            }
          />
        </Col>
        <Col flex="auto">
          <Statistic
            title="Resolved cleanly"
            value={data.resolved_count}
            suffix={<Text type="secondary" style={{ fontSize: 12 }}>of {data.seen_count}</Text>}
          />
        </Col>
        {avgFmt ? (
          <Col flex="auto">
            <Statistic
              title="Avg time to resolve"
              value={avgFmt}
              valueStyle={{ fontSize: 22 }}
              suffix={
                medianFmt
                  ? <Text type="secondary" style={{ fontSize: 12 }}>vs platform median {medianFmt}</Text>
                  : null
              }
            />
          </Col>
        ) : null}
      </Row>
    </Card>
  );
}
