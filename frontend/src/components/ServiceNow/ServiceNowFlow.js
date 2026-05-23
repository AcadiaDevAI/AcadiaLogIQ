// ServiceNow — right-pane flow.
//
// Sidebar's "Connect to ServiceNow" button flips AppLayout's
// ``serviceNowOpen`` flag and this component takes over the right
// pane. It triggers one backend call (GET /ticket-filter/servicenow)
// on mount, then renders the returned JSON in a scrollable card.
//
// Design constraints
// ------------------
// * Independent of Ticket Filter / RCA / Gap Analysis / Chat — no
//   shared imports into those folders, no shared state, no shared CSS.
// * Auto-fetches on mount so the user lands on a populated view
//   without an extra click; a "Refresh" button re-runs the call.
// * Errors (503 = creds missing, 502 = ServiceNow unreachable,
//   429 = rate limit) surface as Ant Design Alerts — never silent.
// * "Return to Stages" button at bottom-right mirrors the Ticket
//   Filter UX so muscle memory carries across.

import React, { useEffect, useState } from "react";
import {
  Alert,
  Button,
  Card,
  Space,
  Spin,
  Tag,
  Typography,
  message,
} from "antd";
import {
  ApiOutlined,
  ArrowLeftOutlined,
  LoadingOutlined,
  ReloadOutlined,
} from "@ant-design/icons";

import { fetchServiceNowIncidents } from "./serviceNowApi";


const { Title, Paragraph, Text } = Typography;


export default function ServiceNowFlow({ onReturnToStages }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);

  const runFetch = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const data = await fetchServiceNowIncidents();
      setResult(data);
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[servicenow] call failed", err);
      const status = err?.response?.status;
      const detail = err?.response?.data?.detail;
      if (status === 503) {
        setError(
          detail ||
            "ServiceNow credentials are not configured on the server. " +
              "Set SERVICE_NOW_* env vars in backend/env.bvk or AWS Secrets Manager.",
        );
      } else if (status === 502) {
        setError(
          detail ||
            "Could not reach ServiceNow. Check the instance URL, credentials, and that the instance is awake.",
        );
      } else if (status === 429) {
        setError("Too many ServiceNow requests — wait a moment and try again.");
      } else if (status === 401) {
        setError("Not signed in. Please sign in again.");
      } else {
        setError("ServiceNow call failed. Please try again.");
      }
      setResult(null);
      message.error("ServiceNow call failed");
    } finally {
      setBusy(false);
    }
  };

  // Auto-fetch on mount. Empty dep array — only the first render
  // triggers; subsequent reloads happen via the Refresh button.
  useEffect(() => { runFetch(); }, []);

  return (
    <div className="flex flex-col h-full w-full t-bg-primary">
      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-4 py-6">
        <div className="w-full max-w-5xl mx-auto">
          {/* Header */}
          <div style={{ marginBottom: 16 }}>
            <Title level={3} style={{ marginBottom: 4 }}>
              <Space size="small">
                <ApiOutlined />
                Connect to ServiceNow
              </Space>
            </Title>
            <Paragraph type="secondary" style={{ marginBottom: 0 }}>
              Fetches priority-1 incidents from the configured ServiceNow
              instance, parses the XML response, and returns the JSON below.
            </Paragraph>
          </div>

          {/* Action bar */}
          <Card style={{ marginBottom: 16 }} bodyStyle={{ padding: 16 }}>
            <Space size="middle" align="center" wrap>
              <Button
                type="primary"
                size="large"
                icon={busy ? <LoadingOutlined /> : <ReloadOutlined />}
                onClick={runFetch}
                disabled={busy}
              >
                {busy ? "Connecting…" : "Refresh"}
              </Button>
              {result ? (
                <Text type="secondary" style={{ fontSize: 13 }}>
                  Fetched <Tag color="blue">{result.count}</Tag> priority-1
                  incident{result.count === 1 ? "" : "s"} from{" "}
                  <Text code style={{ fontSize: 12 }}>
                    {result.instance_url}
                  </Text>
                </Text>
              ) : null}
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
                  Calling ServiceNow…
                </Paragraph>
              </div>
            </Card>
          ) : null}

          {/* Result */}
          {result && !busy ? (
            <Card
              size="small"
              title={
                <Space size="small">
                  <ApiOutlined />
                  <Text strong>ServiceNow — priority=1 incidents</Text>
                  <Tag color="blue">{result.count}</Tag>
                </Space>
              }
              extra={
                <Text type="secondary" style={{ fontSize: 11 }}>
                  {result.instance_url}
                </Text>
              }
            >
              <pre
                style={{
                  margin: 0,
                  maxHeight: 560,
                  overflow: "auto",
                  fontSize: 12,
                  lineHeight: 1.5,
                  background: "var(--color-background-secondary, #f6f8fa)",
                  padding: 12,
                  borderRadius: 6,
                  whiteSpace: "pre-wrap",
                  wordBreak: "break-word",
                }}
              >
                {JSON.stringify(result, null, 2)}
              </pre>
            </Card>
          ) : null}
        </div>
      </div>

      {/* Fixed bottom-right Return button — same UX as Ticket Filter / RCA / Gap. */}
      <div
        style={{
          padding: "12px 16px",
          borderTop: "1px solid var(--border-color, #e5e7eb)",
          display: "flex",
          justifyContent: "flex-end",
          flexShrink: 0,
        }}
      >
        <Button
          type="default"
          icon={<ArrowLeftOutlined />}
          onClick={() => {
            if (typeof onReturnToStages === "function") {
              onReturnToStages();
            }
          }}
        >
          Return to Stages
        </Button>
      </div>
    </div>
  );
}
