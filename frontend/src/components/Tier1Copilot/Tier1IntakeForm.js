import React, { useState } from "react";
import { Button, Card, Form, Input, Segmented, Select } from "antd";
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
// Premium revamp — horizontal pill bar (RCA / Gap / Filter / SN)
// pinned above the LogIQ title. Handlers come in via props from
// LandingRouter (which received them from AppLayout).
import QuickActionsBar from "../QuickActionsBar";

/**
 * Tier1IntakeForm — Sprint 6 default + Sprint 8 progressive variant.
 *
 * Renders the progressive design spec'd in §6 — severity chips, 3
 * required fields prominent, 6 optional collapsed behind
 * "Add more context".
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
  // New layout (replaces the Sprint 11 side-by-side split):
  //   * Title "LogIQ – Operational Intelligence Platform" sits a little
  //     above the vertical center of the right pane.
  //   * Beneath the title, a Proactive / Reactive segmented toggle.
  //   * Below the toggle, ONE active panel at a time:
  //       Proactive (default) — the structured ProgressiveIntakeForm
  //                              wrapped in a spherical-bordered box.
  //       Reactive            — the UniversalIntakePanel paste area.
  //   * On the Reactive side, clicking "Extract & Suggest" auto-applies
  //     the top extracted candidate to the Proactive form's prefill,
  //     and flips the active tab back to "proactive" so the engineer
  //     sees the pre-filled form immediately (no manual tab switch).
  //
  // Previous Sprint 11 two-column grid implementation is preserved in
  // git history; not inlined here because it would dwarf the new layout.
  const [prefill, setPrefill] = useState(null);
  const [activeTab, setActiveTab] = useState("proactive");

  // Card-pick handler — invoked from UniversalIntakePanel after the
  // user hits Extract & Suggest. Stamps the payload (so React re-runs
  // the prefill effect even if the same card is picked twice) and
  // flips the visible tab back to Proactive so the engineer lands on
  // the pre-filled form.
  const handleCardPicked = (filled) => {
    setPrefill({ ...filled, _stamp: Date.now() });
    setActiveTab("proactive");
  };

  const InnerForm = TIER1_UX_FIXES_ON ? ProgressiveIntakeForm : ClassicIntakeForm;

  return (
    // px-2 (was px-4) trims the side gutters so the rounded box uses
    // more of the right-pane width.
    <div className="flex-1 overflow-y-auto px-2 py-4 t-bg-primary">
      {/* Outer wrapper widened: 960 → 1400 so the Proactive / Reactive
          box stretches across the right pane instead of leaving large
          empty gutters on either side. Cap at 1400 to avoid the form
          becoming uncomfortably wide on very large monitors. */}
      <div className="w-full" style={{ maxWidth: 1400, margin: "0 auto" }}>
        {/* Premium revamp — horizontal pill bar pinned to the very top
            of the intake landing. Same as before — handlers come in
            from AppLayout. Pills render only when the matching handler
            prop is supplied. */}
        <QuickActionsBar
          onOpenRca={props.onOpenRca}
          onOpenGapAnalysis={props.onOpenGapAnalysis}
          onOpenTicketFilter={props.onOpenTicketFilter}
          onOpenServiceNow={props.onOpenServiceNow}
          marginBottom={16}
        />

        {/* Vertical spacer — pushes the title down so it sits a touch
            ABOVE the screen center. Trimmed 14vh → 6vh so the form box
            doesn't push below the fold once the wider layout reduces
            the form's overall height. */}
        <div style={{ height: "6vh" }} />

        {/* ─── Page title — premium wordmark treatment ──────────────────
            Replaces the bold black sans-serif H1 with the same visual
            language the LandingPage already uses (Instrument Serif
            display font + aurora-gradient italic + eyebrow chip), so
            this intake screen reads as part of the same brand family
            as the Acadia landing instead of a generic admin form.

            Composition:
              1. Eyebrow chip   — small uppercase mono label
                                  "OPERATIONAL INTELLIGENCE PLATFORM"
                                  in iris-aurora tint, pill border,
                                  green status dot. Matches the
                                  LandingPage eyebrowStyle.
              2. Wordmark       — "Welcome to LogIQ" in Instrument
                                  Serif. "LogIQ" is set italic and
                                  painted with the aurora gradient via
                                  background-clip:text so it pops as
                                  the focal element. Inline gradient
                                  styles (not the `.aurora-text` class)
                                  so it works regardless of which
                                  theme wrapper is active.

            Previous (heavy black bold) heading preserved for reference:

            <h1
              className="t-text"
              style={{
                fontSize: 28,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                margin: 0,
                lineHeight: 1.25,
              }}
            >
              LogIQ – Operational Intelligence Platform
            </h1>
        */}
        <div style={{ textAlign: "center", marginBottom: 16 }}>
          {/* Eyebrow chip */}
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "5px 13px",
              borderRadius: 9999,
              background: "rgba(91, 141, 239, 0.08)",
              border: "1px solid rgba(91, 141, 239, 0.22)",
              color: "var(--aurora-2, #5B8DEF)",
              fontFamily:
                "var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace)",
              fontSize: 10.5,
              fontWeight: 500,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              marginBottom: 10,
            }}
          >
            <span
              aria-hidden
              style={{
                width: 6,
                height: 6,
                borderRadius: "50%",
                background: "#10b981",
                boxShadow: "0 0 8px rgba(16, 185, 129, 0.7)",
              }}
            />
            Operational Intelligence Platform
          </div>

          {/* Wordmark — "Welcome to Log" stays bold + black so it
              reads as a single phrase; only the trailing "IQ" is
              italic + Acadia blue, painting it as the focal accent. */}
          <h1
            style={{
              fontFamily:
                "var(--font-display, 'Instrument Serif', Georgia, 'Times New Roman', serif)",
              fontSize: "clamp(34px, 4.6vw, 52px)",
              fontWeight: 700,
              lineHeight: 1.05,
              letterSpacing: "-0.018em",
              margin: 0,
              color: "var(--text, #0f172a)",
            }}
          >
            <span>Welcome to Log</span>
            <em
              style={{
                fontStyle: "italic",
                fontWeight: 400,
                color: "var(--acadia-primary, #1E4FAF)",
              }}
            >
              IQ
            </em>
          </h1>
        </div>

        {/* Proactive / Reactive toggle — segmented control reads as a
            single horizontal pill with two options. Centered under the
            title so it visually anchors the form box below. */}
        <div
          style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}
        >
          <Segmented
            value={activeTab}
            onChange={(v) => setActiveTab(v)}
            options={[
              { label: "Proactive", value: "proactive" },
              { label: "Reactive",  value: "reactive"  },
            ]}
            size="large"
          />
        </div>

        {/* Active panel — only ONE renders at a time. Both panels are
            wrapped by the spherical-bordered box styling in their own
            components (ProgressiveIntakeForm Card + UniversalIntakePanel
            Card both lift the radius for this view). */}
        {activeTab === "proactive" ? (
          <InnerForm {...props} prefill={prefill} embedded />
        ) : (
          <UniversalIntakePanel
            source="note"
            sessionId={props.sessionId}
            onCardPicked={handleCardPicked}
            header="Tell us what's happening"
            helperText={
              "Describe the issue in your own words. The system extracts"
              + " the most likely interpretation and pre-fills the"
              + " Proactive form for you."
            }
            placeholder={
              "e.g., V-Desktop Environment is reporting Desktop Slowness"
              + " for customer Acme since 9:30 AM. Users see lag opening"
              + " applications; ping to gateway is normal."
            }
          />
        )}
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

  // Severity is no longer a mandatory gate — the engineer can submit
  // with just Asset + Alert. Severity moved into the "Add more context"
  // optional block; if picked there it still rides on the payload below
  // (the backend contract is unchanged — severity is allowed to be null).
  //
  // Previous (three-field) gate kept here for reference:
  // const canSubmit =
  //   !!severity
  //   && !!assetName.trim()
  //   && !!alertType.trim();
  const canSubmit =
    !!assetName.trim()
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

  // Premium single-panel layout — the form is now the ONLY thing the
  // engineer sees in Proactive mode (no more side-by-side split), so
  // it gets a more deliberate "card" look: spherical corners, soft
  // ambient shadow, generous padding. The radius value (28px) is the
  // upper end of "rounded rectangle" before it starts to look like a
  // pill — chosen to match the Segmented toggle above and the chips
  // inside the form so the whole panel reads as one rounded family.
  //
  // Previous Sprint-11 Card style preserved for reference:
  // const cardStyle = {
  //   backgroundColor: "var(--bg-secondary)",
  //   borderColor: "var(--border-color)",
  //   borderRadius: 12,
  // };
  const cardStyle = {
    backgroundColor: "var(--bg-secondary)",
    borderColor: "var(--border-color)",
    borderRadius: 28,
    boxShadow: "0 6px 24px -8px rgba(15, 23, 42, 0.10), 0 2px 6px -2px rgba(15, 23, 42, 0.06)",
  };
  // Input style — pill-rounded so each row reads as its own rounded
  // capsule inside the outer rectangular box.
  const inputStyle = { borderRadius: 9999 };

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

        {/* Tightened body padding (18 → 14, 28 wide) and inner header
            gap so the rounded box reads shorter without losing the
            spherical-border feel. The form sits in a wider rectangle
            now (parent maxWidth 1400), so vertical padding can shrink
            without the content feeling cramped. */}
        <Card bodyStyle={{ padding: "14px 28px" }} style={cardStyle}>
          {embedded && (
            <div style={{ marginBottom: 10, textAlign: "center" }}>
              <h2
                className="t-text"
                style={{
                  fontSize: 18,
                  fontWeight: 600,
                  letterSpacing: "-0.01em",
                  margin: 0,
                  marginBottom: 2,
                }}
              >
                What&apos;s happening?
              </h2>
              <p
                className="t-text-muted"
                style={{ fontSize: 12, margin: 0, lineHeight: 1.45 }}
              >
                Describe the incident below — we&apos;ll match it to the
                closest historical ticket and return a troubleshooting
                answer.
              </p>
            </div>
          )}

          <Form
            form={form}
            layout="vertical"
            onFinish={handleFinish}
          >
            {/* Severity removed from the mandatory top section. It now
                lives inside the "Add more context" expander below and is
                fully optional. The engineer can search with just the two
                remaining required fields (Asset + Alert). The submit
                payload still carries `severity` (state value below) — it
                just defaults to null when the engineer skips it.

                Previous (mandatory-at-top) markup preserved for reference:

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
            */}

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
                inputStyle={inputStyle}
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
                inputStyle={inputStyle}
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
                {/* Severity — relocated here as an optional field. The
                    chip selector + state binding (`severity` /
                    `setSeverity`) are unchanged; only its position in
                    the form moved (and the field is no longer required
                    on the canSubmit gate above). The md:col-span-2 keeps
                    the chip row full-width so the chips have room to
                    breathe inside the grid. */}
                <Form.Item
                  label="Severity"
                  className="md:col-span-2"
                >
                  <SeverityChipSelector
                    value={severity}
                    onChange={setSeverity}
                    disabled={busy}
                  />
                </Form.Item>
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
