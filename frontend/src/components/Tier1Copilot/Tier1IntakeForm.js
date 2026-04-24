import React, { useState } from "react";
import { Button, Card, Form, Input, Select } from "antd";
import { DownOutlined, UpOutlined } from "@ant-design/icons";
import {
  SEVERITY_OPTIONS,
  TECHNOLOGY_OPTIONS,
  TIER1_UX_FIXES_ON,
} from "./tier1Constants";
import SeverityChipSelector from "./SeverityChipSelector";
import AssetAutocomplete from "./AssetAutocomplete";
import { useTier1Theme } from "../../theme/ThemeProvider";

/**
 * Tier1IntakeForm — Sprint 6 default + Sprint 8 progressive variant.
 *
 * When REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND !== "true" the component
 * renders the Sprint 6 flat 9-field form (byte-identical to the
 * pre-Sprint-8 experience). When the flag is on, it renders the
 * progressive design spec'd in §6 — severity chips, 3 required fields
 * prominent, 6 optional collapsed behind "Add more context".
 */
export default function Tier1IntakeForm(props) {
  if (TIER1_UX_FIXES_ON) {
    return <ProgressiveIntakeForm {...props} />;
  }
  return <ClassicIntakeForm {...props} />;
}


function ClassicIntakeForm({
  sessionId,
  busy,
  onSubmit,
  onBack,
}) {
  const [form] = Form.useForm();
  const [canSubmit, setCanSubmit] = useState(false);

  const recomputeCanSubmit = () => {
    const values = form.getFieldsValue(["severity", "asset_name", "alert_type"]);
    const allFilled =
      !!values.severity &&
      !!(values.asset_name && values.asset_name.trim()) &&
      !!(values.alert_type && values.alert_type.trim());
    const hasErrors = form
      .getFieldsError()
      .some((f) => (f.errors && f.errors.length > 0));
    setCanSubmit(allFilled && !hasErrors);
  };

  const handleFinish = (values) => {
    const payload = {
      severity: values.severity,
      asset_name: (values.asset_name || "").trim(),
      alert_type: (values.alert_type || "").trim(),
      customer: (values.customer || "").trim() || null,
      location: (values.location || "").trim() || null,
      technology: values.technology || null,
      ip_or_device_id: (values.ip_or_device_id || "").trim() || null,
      error_code: (values.error_code || "").trim() || null,
      notes: (values.notes || "").trim() || null,
      session_id: sessionId || "tier1-local",
    };
    onSubmit && onSubmit(payload);
  };

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-8 t-bg-primary">
      <div className="w-full max-w-3xl">
        <div className="text-center mb-6">
          <h1 className="text-xl font-bold t-text">Tier-1 Alert Copilot</h1>
          <p className="t-text-muted text-sm mt-1">
            Enter what you see. The copilot retrieves the closest historical
            incident and returns an 8-section troubleshooting answer.
          </p>
        </div>

        <Card
          bodyStyle={{ padding: 24 }}
          style={{
            backgroundColor: "var(--bg-secondary)",
            borderColor: "var(--border-color)",
          }}
        >
          <Form
            form={form}
            layout="vertical"
            onFinish={handleFinish}
            onValuesChange={recomputeCanSubmit}
            onFieldsChange={recomputeCanSubmit}
            requiredMark="optional"
          >
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <Form.Item
                label="Severity"
                name="severity"
                rules={[{ required: true, message: "Pick a severity" }]}
              >
                <Select
                  placeholder="Select severity"
                  options={SEVERITY_OPTIONS}
                  size="large"
                />
              </Form.Item>

              <Form.Item
                label="Technology"
                name="technology"
                help="Optional — helps alias expansion"
              >
                <Select
                  placeholder="Select technology (optional)"
                  options={TECHNOLOGY_OPTIONS}
                  size="large"
                  allowClear
                />
              </Form.Item>

              <Form.Item
                label="Asset name"
                name="asset_name"
                rules={[{ required: true, message: "Asset name required" }]}
              >
                <Input
                  size="large"
                  placeholder="e.g. V-Desktop Environment"
                  maxLength={200}
                />
              </Form.Item>

              <Form.Item
                label="Alert type"
                name="alert_type"
                rules={[{ required: true, message: "Alert type required" }]}
              >
                <Input
                  size="large"
                  placeholder="e.g. Desktop Slowness"
                  maxLength={200}
                />
              </Form.Item>

              <Form.Item label="Customer" name="customer">
                <Input
                  size="large"
                  placeholder="Customer name (optional)"
                  maxLength={200}
                />
              </Form.Item>

              <Form.Item label="Location" name="location">
                <Input
                  size="large"
                  placeholder="Site / region (optional)"
                  maxLength={200}
                />
              </Form.Item>

              <Form.Item label="IP or device ID" name="ip_or_device_id">
                <Input
                  size="large"
                  placeholder="e.g. 10.0.12.4 or edge-rtr-01"
                  maxLength={200}
                />
              </Form.Item>

              <Form.Item label="Error code" name="error_code">
                <Input
                  size="large"
                  placeholder="e.g. BGP-5-ADJCHANGE"
                  maxLength={100}
                />
              </Form.Item>

              <Form.Item
                label="Notes"
                name="notes"
                className="md:col-span-2"
              >
                <Input.TextArea
                  rows={3}
                  placeholder="Anything else the copilot should see (optional)"
                  maxLength={2000}
                  showCount
                />
              </Form.Item>
            </div>

            <div className="flex justify-between items-center mt-2">
              <Button size="large" onClick={onBack} disabled={busy}>
                Back
              </Button>
              <Button
                type="primary"
                size="large"
                htmlType="submit"
                disabled={!canSubmit || busy}
                loading={busy}
                style={{
                  backgroundColor: "#0A3F63",
                  borderColor: "#0A3F63",
                  minWidth: 180,
                }}
              >
                {busy ? "Analyzing alert…" : "Analyze alert"}
              </Button>
            </div>
          </Form>
        </Card>
      </div>
    </div>
  );
}


// ─────────────────────────────────────────────────────────────
// Sprint 8 — ProgressiveIntakeForm
// ─────────────────────────────────────────────────────────────
function ProgressiveIntakeForm({
  sessionId,
  busy,
  onSubmit,
  onBack,
  recentAssets = [],
  recentAlertTypes = [],
}) {
  const { tokens, isModern } = useTier1Theme();
  const [form] = Form.useForm();
  const [severity, setSeverity] = useState(null);
  const [assetName, setAssetName] = useState("");
  const [alertType, setAlertType] = useState("");
  const [expanded, setExpanded] = useState(false);

  const canSubmit =
    !!severity
    && !!assetName.trim()
    && !!alertType.trim();

  const handleFinish = (values) => {
    const payload = {
      severity,
      asset_name: assetName.trim(),
      alert_type: alertType.trim(),
      customer: (values.customer || "").trim() || null,
      location: (values.location || "").trim() || null,
      technology: values.technology || null,
      ip_or_device_id: (values.ip_or_device_id || "").trim() || null,
      error_code: (values.error_code || "").trim() || null,
      notes: (values.notes || "").trim() || null,
      session_id: sessionId || "tier1-local",
    };
    if (onSubmit) onSubmit(payload);
  };

  const cardStyle = {
    backgroundColor: tokens.surfaceBase,
    borderColor: isModern ? tokens.surfaceElevated : "var(--border-color)",
    borderRadius: tokens.radiusLg || 16,
    boxShadow: isModern ? tokens.shadowMd : undefined,
  };

  const submitStyle = {
    minWidth: 180,
    borderRadius: tokens.radiusMd || 12,
    background: isModern
      ? tokens.gradientAccent || "#0A3F63"
      : "#0A3F63",
    borderColor: isModern ? "transparent" : "#0A3F63",
  };

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-8 t-bg-primary">
      <div className="w-full" style={{ maxWidth: 640 }}>
        <div className="text-center mb-6">
          <h1
            className="t-text"
            style={{
              fontSize: 22,
              fontWeight: 600,
              letterSpacing: "-0.01em",
              margin: 0,
            }}
          >
            What&apos;s happening?
          </h1>
          <p className="t-text-muted text-sm mt-2" style={{ margin: "6px 0 0" }}>
            Enter what you see. The copilot retrieves the closest
            historical incident and returns an 8-section troubleshooting
            answer.
          </p>
        </div>

        <Card bodyStyle={{ padding: 24 }} style={cardStyle}>
          <Form
            form={form}
            layout="vertical"
            onFinish={handleFinish}
          >
            <Form.Item
              label={<span style={{ fontWeight: 600 }}>Severity</span>}
              required
            >
              <SeverityChipSelector
                value={severity}
                onChange={setSeverity}
                disabled={busy}
              />
            </Form.Item>

            <Form.Item
              label={<span style={{ fontWeight: 600 }}>Asset or system</span>}
              required
            >
              <AssetAutocomplete
                value={assetName}
                onChange={setAssetName}
                placeholder="Start typing — e.g., V-Desktop Environment"
                suggestions={recentAssets}
                size="large"
              />
            </Form.Item>

            <Form.Item
              label={
                <span style={{ fontWeight: 600 }}>What&apos;s the alert about?</span>
              }
              required
            >
              <AssetAutocomplete
                value={alertType}
                onChange={setAlertType}
                placeholder="e.g., Desktop Slowness, BGP flap, Circuit down"
                suggestions={recentAlertTypes}
                size="large"
              />
            </Form.Item>

            <div style={{ marginTop: 4, marginBottom: 16 }}>
              <Button
                type="link"
                size="small"
                icon={expanded ? <UpOutlined /> : <DownOutlined />}
                onClick={() => setExpanded((v) => !v)}
                style={{ paddingLeft: 0 }}
              >
                {expanded ? "Hide optional fields" : "Add more context (optional)"}
              </Button>
            </div>

            {expanded && (
              <div
                className="grid grid-cols-1 md:grid-cols-2 gap-x-4"
                style={{ transition: tokens.transitionMed }}
              >
                <Form.Item label="Customer" name="customer">
                  <Input size="large" maxLength={200} allowClear />
                </Form.Item>
                <Form.Item label="Location" name="location">
                  <Input size="large" maxLength={200} allowClear />
                </Form.Item>
                <Form.Item label="Technology / domain" name="technology">
                  <Select
                    size="large"
                    allowClear
                    options={TECHNOLOGY_OPTIONS}
                    placeholder="(optional)"
                  />
                </Form.Item>
                <Form.Item label="IP / circuit / device ID" name="ip_or_device_id">
                  <Input size="large" maxLength={200} allowClear />
                </Form.Item>
                <Form.Item label="Error / alarm code" name="error_code">
                  <Input size="large" maxLength={100} allowClear />
                </Form.Item>
                <Form.Item
                  label="Notes"
                  name="notes"
                  className="md:col-span-2"
                >
                  <Input.TextArea rows={3} maxLength={2000} showCount />
                </Form.Item>
              </div>
            )}

            <div className="flex justify-between items-center mt-3">
              <Button size="large" onClick={onBack} disabled={busy}>
                Back
              </Button>
              <Button
                type="primary"
                size="large"
                htmlType="submit"
                disabled={!canSubmit || busy}
                loading={busy}
                style={submitStyle}
              >
                {busy ? "Analyzing alert…" : "Analyze alert →"}
              </Button>
            </div>
          </Form>
        </Card>
      </div>
    </div>
  );
}
