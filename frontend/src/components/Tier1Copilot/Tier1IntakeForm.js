import React, { useState } from "react";
import { Button, Card, Form, Input, Select } from "antd";
import { DownOutlined, UpOutlined } from "@ant-design/icons";
import {
  SEVERITY_OPTIONS,
  TECHNOLOGY_OPTIONS,
  TIER1_UX_FIXES_ON,
  UNIVERSAL_INTAKE_ON, // Sprint 9
} from "./tier1Constants";
import SeverityChipSelector from "./SeverityChipSelector";
import AssetAutocomplete from "./AssetAutocomplete";
import { useTier1Theme } from "../../theme/ThemeProvider";
// Sprint 9 — universal intake (paste-from-anywhere mode).
// Sprint 11 — Proactive | Reactive split-view; ModeToggle removed
// because both modes are visible side-by-side now.
import UniversalIntakePanel from "./intake/UniversalIntakePanel";

/**
 * Tier1IntakeForm — Sprint 6 default + Sprint 8 progressive variant.
 *
 * When REACT_APP_LOGIQ_TIER1_UX_FIXES_FRONTEND !== "true" the component
 * renders the Sprint 6 flat 9-field form (byte-identical to the
 * pre-Sprint-8 experience). When the flag is on, it renders the
 * progressive design spec'd in §6 — severity chips, 3 required fields
 * prominent, 6 optional collapsed behind "Add more context".
 */
// Sprint 9 — when the universal-intake frontend flag is on, the form
// is wrapped with source-selection state + an optional paste panel so
// engineers can pull a triage interpretation from email/phone/portal/
// chat/note. Flag-off path: byte-identical Sprint 6/8 form (the inner
// Classic / Progressive components are unchanged).
export default function Tier1IntakeForm(props) {
  if (UNIVERSAL_INTAKE_ON) {
    return <SourceAwareIntake {...props} />;
  }
  if (TIER1_UX_FIXES_ON) {
    return <ProgressiveIntakeForm {...props} />;
  }
  return <ClassicIntakeForm {...props} />;
}


function SourceAwareIntake(props) {
  // Sprint 11 — Side-by-side layout. Replaces the earlier toggle UX
  // (ModeToggle Proactive/Reactive) with a 2-column grid:
  //   left  = Proactive — the structured Alert form
  //   right = Reactive — paste-message box; extraction pre-fills the
  //           Alert form on the left
  // On viewports < md the two columns stack. Picking a Reactive card
  // pre-fills the Proactive form so the engineer sees their edited
  // intake without needing to switch panes.
  const [prefill, setPrefill] = useState(null);

  const handleCardPicked = (filled) => {
    setPrefill({ ...filled, _stamp: Date.now() });
  };

  const InnerForm = TIER1_UX_FIXES_ON ? ProgressiveIntakeForm : ClassicIntakeForm;

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 t-bg-primary">
      <div className="w-full" style={{ maxWidth: 1280, margin: "0 auto" }}>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Proactive — structured alert form */}
          <div>
            <div className="text-center mb-3">
              <h2
                className="t-text"
                style={{ fontSize: 18, fontWeight: 600, margin: 0 }}
              >
                Proactive
              </h2>
              <p
                className="t-text-muted"
                style={{ fontSize: 12, margin: "2px 0 0" }}
              >
                Monitoring or alert-triggered intake
              </p>
            </div>
            <InnerForm {...props} prefill={prefill} embedded />
          </div>

          {/* Reactive — paste box; output pre-fills Proactive on the left */}
          <div>
            <div className="text-center mb-3">
              <h2
                className="t-text"
                style={{ fontSize: 18, fontWeight: 600, margin: 0 }}
              >
                Reactive
              </h2>
              <p
                className="t-text-muted"
                style={{ fontSize: 12, margin: "2px 0 0" }}
              >
                Customer-reported via email, phone, portal, chat or note
              </p>
            </div>
            <UniversalIntakePanel
              source="note"
              sessionId={props.sessionId}
              onCardPicked={handleCardPicked}
              header="Tell us what's happening"
              helperText={
                "Provide device type, alert type, and a brief summary so we"
                + " can auto-fill the Proactive form on the left. Pick an"
                + " interpretation card and the form populates instantly."
              }
              placeholder={
                "e.g., V-Desktop Environment is reporting Desktop Slowness"
                + " for customer Acme since 9:30 AM. Users see lag opening"
                + " applications; ping to gateway is normal."
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}


function ClassicIntakeForm({
  sessionId,
  busy,
  onSubmit,
  onBack,
  prefill, // Sprint 9 — populated by SourceAwareIntake on card pick
}) {
  const [form] = Form.useForm();
  const [canSubmit, setCanSubmit] = useState(false);

  // Sprint 9 — apply universal-intake prefill when it arrives. Each
  // pick stamps a new _stamp so React re-runs the effect even if the
  // engineer picks the same card twice.
  React.useEffect(() => {
    if (!prefill) return;
    const fields = {};
    if (prefill.severity) fields.severity = prefill.severity;
    if (prefill.asset_name) fields.asset_name = prefill.asset_name;
    if (prefill.alert_type) fields.alert_type = prefill.alert_type;
    if (prefill.customer) fields.customer = prefill.customer;
    if (prefill.location) fields.location = prefill.location;
    if (Object.keys(fields).length > 0) {
      form.setFieldsValue(fields);
      // Trigger button-enabled recompute since we just filled fields.
      setTimeout(() => {
        const values = form.getFieldsValue(["severity", "asset_name", "alert_type"]);
        setCanSubmit(
          !!values.severity
            && !!(values.asset_name && values.asset_name.trim())
            && !!(values.alert_type && values.alert_type.trim()),
        );
      }, 0);
    }
  }, [prefill && prefill._stamp]);

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
          <h1 className="text-xl font-bold t-text">Tier-1 Alert Triage</h1>
          <p className="t-text-muted text-sm mt-1">
            Describe the incident below. We&apos;ll match it to the closest
            historical ticket and return an 8-section troubleshooting answer.
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
                  placeholder="Anything else worth noting (optional)"
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
                  backgroundColor: "var(--acadia-primary)",
                  borderColor: "var(--acadia-primary)",
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
  prefill, // Sprint 9
  // Sprint 11 fix — when rendered inside the Proactive | Reactive split
  // (SourceAwareIntake), the column already shows a "Proactive" header
  // above this form, so suppress the inner "What's happening?" title
  // block and drop the outer flex/padding wrapper. Both columns then
  // have identical vertical anchors and the Cards line up perfectly.
  embedded = false,
}) {
  const { tokens, isModern } = useTier1Theme();
  const [form] = Form.useForm();
  const [severity, setSeverity] = useState(null);
  const [assetName, setAssetName] = useState("");
  const [alertType, setAlertType] = useState("");

  // Sprint 9 — apply universal-intake prefill when it arrives.
  React.useEffect(() => {
    if (!prefill) return;
    if (prefill.severity) setSeverity(prefill.severity);
    if (prefill.asset_name) setAssetName(prefill.asset_name);
    if (prefill.alert_type) setAlertType(prefill.alert_type);
    const inner = {};
    if (prefill.customer) inner.customer = prefill.customer;
    if (prefill.location) inner.location = prefill.location;
    if (Object.keys(inner).length > 0) {
      form.setFieldsValue(inner);
    }
  }, [prefill && prefill._stamp]);
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

  // Sprint 11 — match the Reactive UniversalIntakePanel Card exactly so
  // both columns of the split intake screen read as sibling cards.
  const cardStyle = {
    backgroundColor: "var(--bg-secondary)",
    borderColor: "var(--border-color)",
    borderRadius: 12,
  };

  const submitStyle = {
    minWidth: 180,
    borderRadius: tokens.radiusMd || 12,
    background: isModern
      ? tokens.gradientAccent || "var(--acadia-primary)"
      : "var(--acadia-primary)",
    borderColor: isModern ? "transparent" : "var(--acadia-primary)",
  };

  // Sprint 11 fix — embedded mode (inside SourceAwareIntake split layout)
  // collapses the page-level wrapper so the Card aligns with the
  // Reactive column's UniversalIntakePanel Card.
  const wrapperClass = embedded
    ? "w-full"
    : "flex-1 flex items-center justify-center px-4 py-8 t-bg-primary";
  const innerWrapperStyle = embedded ? {} : { maxWidth: 640 };
  const innerWrapperClass = embedded ? "w-full" : "w-full";

  return (
    <div className={wrapperClass}>
      <div className={innerWrapperClass} style={innerWrapperStyle}>
        {!embedded && (
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
              Describe the incident below. We&apos;ll match it to the closest
              historical ticket and return an 8-section troubleshooting answer.
            </p>
          </div>
        )}

        <Card bodyStyle={{ padding: 18 }} style={cardStyle}>
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
