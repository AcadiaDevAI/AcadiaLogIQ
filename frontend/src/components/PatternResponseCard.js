import React from "react";
import { Card, Tag } from "antd";
import {
  BarChartOutlined,
  CheckCircleOutlined,
  ThunderboltOutlined,
} from "@ant-design/icons";

/**
 * PatternResponseCard — Troubleshooting Pattern Response (PRD §4).
 *
 * Renders the historical pattern summary the backend attaches to an
 * assistant turn in context_stats.pattern_data:
 *   { occurrence_count, timeframe_months, top_actions, most_successful,
 *     success_rate, success_count, total_count, recent_count_30d,
 *     confidence_score, ...,
 *     insufficient_confidence (transparency path) }
 *
 * Renders nothing for the pre-flag shape or when data is absent.
 */
export default function PatternResponseCard({ topic, data }) {
  if (!data || typeof data !== "object") return null;

  if (data.insufficient_confidence) {
    return (
      <div className="px-4 md:px-8 lg:px-16 xl:px-24 pb-2">
        <Card
          size="small"
          style={{
            backgroundColor: "var(--bg-secondary)",
            borderColor: "var(--border-color)",
          }}
          bodyStyle={{ padding: "10px 14px" }}
        >
          <div className="flex items-center gap-2 text-xs t-text-muted">
            <BarChartOutlined />
            <span>{data.message || "Limited historical data for reliable patterns."}</span>
          </div>
        </Card>
      </div>
    );
  }

  const count = Number(data.occurrence_count || 0);
  const months = Number(data.timeframe_months || 0);
  const topActions = Array.isArray(data.top_actions) ? data.top_actions.slice(0, 3) : [];
  const mostSuccessful = data.most_successful || null;
  const successRate = Number(data.success_rate || 0);
  const successCount = Number(data.success_count || 0);
  const total = Number(data.total_count || count || 0);
  const recent = Number(data.recent_count_30d || 0);

  return (
    <div className="px-4 md:px-8 lg:px-16 xl:px-24 pb-2">
      <Card
        size="small"
        title={
          <div className="flex items-center gap-2 text-sm">
            <BarChartOutlined style={{ color: "var(--brand-accent)" }} />
            <span className="t-text">
              Historical pattern{topic ? ` · ${topic}` : ""}
            </span>
          </div>
        }
        style={{
          backgroundColor: "var(--bg-secondary)",
          borderColor: "var(--border-color)",
        }}
        bodyStyle={{ padding: 14 }}
        headStyle={{ borderColor: "var(--border-color)", padding: "8px 14px" }}
      >
        <div className="flex flex-wrap gap-2 mb-3">
          <Tag color="blue">
            Occurred {count} time{count === 1 ? "" : "s"}
            {months > 0 ? ` in ${months} mo` : ""}
          </Tag>
          {recent > 0 && <Tag color="geekblue">{recent} in last 30 days</Tag>}
          {total > 0 && (
            <Tag color={successRate >= 0.5 ? "green" : "orange"}>
              Resolved {Math.round(successRate * 100)}% ({successCount}/{total})
            </Tag>
          )}
        </div>

        {topActions.length > 0 && (
          <div className="mb-3">
            <p className="text-xs font-medium t-text-muted mb-1">
              <ThunderboltOutlined /> Top {topActions.length} action
              {topActions.length === 1 ? "" : "s"}
            </p>
            <ol className="list-decimal ml-5 text-sm t-text space-y-0.5">
              {topActions.map((a, idx) => (
                <li key={idx}>
                  <span>{a.action}</span>
                  {typeof a.count === "number" && a.count > 0 && (
                    <span className="t-text-faint text-xs ml-1">
                      ({a.count}×)
                    </span>
                  )}
                </li>
              ))}
            </ol>
          </div>
        )}

        {mostSuccessful && (
          <div>
            <p className="text-xs font-medium t-text-muted mb-1">
              <CheckCircleOutlined /> Most successful resolution
            </p>
            <p className="text-sm t-text">{mostSuccessful}</p>
          </div>
        )}
      </Card>
    </div>
  );
}
