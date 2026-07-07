import React, { useCallback, useEffect, useState } from "react";
import { Segmented, Table, Statistic, Spin, Empty, Button, Tooltip, message } from "antd";
import { ReloadOutlined, ThunderboltOutlined } from "@ant-design/icons";
import { getTokenConsumption } from "../../services/api";
import { useOrg } from "../../hooks/OrgContext";

// Admin-only view of THIS org's Bedrock token consumption. The numbers are
// RLS-scoped server-side to the caller's active org, so switching orgs in the
// picker re-scopes everything here automatically. Backend independently gates
// the endpoint with require_org_admin — this component is only mounted for
// admins as a convenience.

const PERIOD_OPTIONS = [
  { label: "Today", value: "today" },
  { label: "This Month", value: "month" },
  { label: "All Time", value: "all" },
];

const fmtInt = (n) => (n || 0).toLocaleString();
const fmtCost = (n) => `$${(n || 0).toFixed(4)}`;

export default function TokenUsagePanel() {
  const { activeOrg } = useOrg();
  const [period, setPeriod] = useState("month");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (p) => {
    setLoading(true);
    try {
      const res = await getTokenConsumption(p);
      setData(res.data);
    } catch (err) {
      const detail = err?.response?.data?.detail || err?.message || "Failed to load usage";
      message.error(detail);
      setData(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load(period);
  }, [period, load]);

  const totals = data?.totals || {};

  const featureColumns = [
    { title: "Feature", dataIndex: "feature", key: "feature" },
    { title: "Calls", dataIndex: "call_count", key: "call_count", align: "right", render: fmtInt },
    { title: "In", dataIndex: "input_tokens", key: "input_tokens", align: "right", render: fmtInt },
    { title: "Out", dataIndex: "output_tokens", key: "output_tokens", align: "right", render: fmtInt },
    { title: "Cost", dataIndex: "cost_usd", key: "cost_usd", align: "right", render: fmtCost },
  ];

  const modelColumns = [
    { title: "Model", dataIndex: "model_id", key: "model_id", ellipsis: true },
    { title: "In", dataIndex: "input_tokens", key: "input_tokens", align: "right", render: fmtInt },
    { title: "Out", dataIndex: "output_tokens", key: "output_tokens", align: "right", render: fmtInt },
    { title: "Cost", dataIndex: "cost_usd", key: "cost_usd", align: "right", render: fmtCost },
  ];

  return (
    <div className="flex flex-col gap-3 overflow-y-auto max-h-[calc(100vh-380px)] pb-4">
      <div className="flex items-center gap-2">
        <ThunderboltOutlined style={{ color: "#6366f1" }} />
        <span className="text-xs font-semibold t-text-secondary">
          Token Usage{activeOrg?.name ? ` — ${activeOrg.name}` : ""}
        </span>
        <Tooltip title="Refresh">
          <Button
            type="text"
            size="small"
            icon={<ReloadOutlined />}
            onClick={() => load(period)}
            className="ml-auto"
          />
        </Tooltip>
      </div>

      <Segmented
        size="small"
        block
        value={period}
        onChange={setPeriod}
        options={PERIOD_OPTIONS}
      />

      {loading ? (
        <div className="flex justify-center py-8">
          <Spin />
        </div>
      ) : !data || (totals.call_count || 0) === 0 ? (
        <Empty
          image={Empty.PRESENTED_IMAGE_SIMPLE}
          description={<span className="t-text-muted text-xs">No usage recorded for this period</span>}
        />
      ) : (
        <>
          <div className="grid grid-cols-3 gap-2">
            <div className="t-bg-tertiary rounded-lg px-2 py-2 text-center">
              <Statistic
                title={<span className="text-[10px] t-text-muted">Cost</span>}
                value={totals.cost_usd || 0}
                precision={4}
                prefix="$"
                valueStyle={{ fontSize: 16, color: "var(--brand-accent)" }}
              />
            </div>
            <div className="t-bg-tertiary rounded-lg px-2 py-2 text-center">
              <Statistic
                title={<span className="text-[10px] t-text-muted">Total tokens</span>}
                value={totals.total_tokens || 0}
                valueStyle={{ fontSize: 16 }}
              />
            </div>
            <div className="t-bg-tertiary rounded-lg px-2 py-2 text-center">
              <Statistic
                title={<span className="text-[10px] t-text-muted">Calls</span>}
                value={totals.call_count || 0}
                valueStyle={{ fontSize: 16 }}
              />
            </div>
          </div>

          <div>
            <div className="text-[11px] t-text-muted mb-1 mt-1">By feature</div>
            <Table
              size="small"
              rowKey="feature"
              columns={featureColumns}
              dataSource={data.by_feature || []}
              pagination={false}
              scroll={{ x: true }}
            />
          </div>

          <div>
            <div className="text-[11px] t-text-muted mb-1 mt-1">By model</div>
            <Table
              size="small"
              rowKey="model_id"
              columns={modelColumns}
              dataSource={data.by_model || []}
              pagination={false}
              scroll={{ x: true }}
            />
          </div>
        </>
      )}
    </div>
  );
}
