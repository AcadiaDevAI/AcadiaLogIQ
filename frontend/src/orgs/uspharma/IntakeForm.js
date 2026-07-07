import React, { useState } from "react";
import { Button, Card, Form, Input, Segmented, Select } from "antd";
import { DownOutlined, UpOutlined } from "@ant-design/icons";
import { TECHNOLOGY_OPTIONS } from "../../components/Tier1Copilot/tier1Constants";
import SeverityChipSelector from "../../components/Tier1Copilot/SeverityChipSelector";
import UniversalIntakePanel from "../../components/Tier1Copilot/intake/UniversalIntakePanel";
import QuickActionsBar from "../../components/QuickActionsBar";

/**
 * US Pharma intake — Store ID + symptom.
 *
 * US Pharma's Tier-1 flow searches historic incidents for a SPECIFIC store
 * (hard-scoped by store_id) matched on the built-in Fingerprints. So the
 * shared "Asset or system" field is replaced by a required Store ID; the
 * "What's the alert about?" symptom field is kept (it drives the fingerprint
 * match). Proactive/Reactive toggle is kept. Optional fields stay collapsed.
 *
 * This is a self-contained US-Pharma component composed from shared building
 * blocks (Segmented, QuickActionsBar, UniversalIntakePanel, SeverityChipSelector)
 * — Acadia's Tier1IntakeForm is untouched.
 *
 * Submit payload → { store_id, alert_type: <symptom>, session_id, ...optional }
 * (no asset_name). Backend /tier1/analyze hard-filters matches to store_id.
 */
export default function USPharmaIntakeForm(props) {
  const [prefill, setPrefill] = useState(null);
  const [activeTab, setActiveTab] = useState("proactive");

  const handleCardPicked = (filled) => {
    // Reactive → Proactive handoff. We can only reliably pre-fill the symptom
    // from free-text extraction; the Store ID stays for the engineer to enter.
    setPrefill({ ...filled, _stamp: Date.now() });
    setActiveTab("proactive");
  };

  return (
    <div className="flex-1 overflow-y-auto px-2 py-4 t-bg-primary">
      <div className="w-full" style={{ maxWidth: 1400, margin: "0 auto" }}>
        <QuickActionsBar
          onOpenRca={props.onOpenRca}
          onOpenGapAnalysis={props.onOpenGapAnalysis}
          onOpenEscalationProcedure={props.onOpenEscalationProcedure}
          onOpenTicketFilter={props.onOpenTicketFilter}
          onOpenServiceNow={props.onOpenServiceNow}
          marginBottom={16}
        />

        <div style={{ height: "6vh" }} />

        <div style={{ textAlign: "center", marginBottom: 16 }}>
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 8,
              padding: "5px 13px",
              borderRadius: 9999,
              background: "rgba(11, 114, 133, 0.08)",
              border: "1px solid rgba(11, 114, 133, 0.25)",
              color: "#0b7285",
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
            US Pharma · Store Support
          </div>
          <h1
            style={{
              fontFamily:
                "var(--font-display, 'Instrument Serif', Georgia, 'Times New Roman', serif)",
              fontSize: "clamp(30px, 4.2vw, 46px)",
              fontWeight: 700,
              lineHeight: 1.05,
              letterSpacing: "-0.018em",
              margin: 0,
              color: "var(--text, #0f172a)",
            }}
          >
            <span>Find a store&apos;s </span>
            <em style={{ fontStyle: "italic", fontWeight: 400, color: "#0b7285" }}>
              incident history
            </em>
          </h1>
        </div>

        <div style={{ display: "flex", justifyContent: "center", marginBottom: 12 }}>
          <Segmented
            value={activeTab}
            onChange={(v) => setActiveTab(v)}
            options={[
              { label: "Proactive", value: "proactive" },
              { label: "Reactive", value: "reactive" },
            ]}
            size="large"
          />
        </div>

        {activeTab === "proactive" ? (
          <USPharmaProactiveForm {...props} prefill={prefill} />
        ) : (
          <UniversalIntakePanel
            source="note"
            sessionId={props.sessionId}
            onCardPicked={handleCardPicked}
            header="Tell us what's happening at the store"
            helperText={
              "Describe the issue in your own words. We'll pre-fill the symptom;"
              + " enter the Store ID to search that store's incident history."
            }
            placeholder={
              "e.g., Store 3001 primary circuit down; Fortinet WAN1 showing"
              + " LOS and BGP flapping; POS failed over to 5G."
            }
          />
        )}
      </div>
    </div>
  );
}


function USPharmaProactiveForm({ sessionId, busy, onSubmit, onBack, prefill }) {
  const [form] = Form.useForm();
  const [storeId, setStoreId] = useState("");
  const [symptom, setSymptom] = useState("");
  const [severity, setSeverity] = useState(null);
  const [expanded, setExpanded] = useState(false);

  React.useEffect(() => {
    if (!prefill) return;
    // Reactive extraction fills the symptom (alert_type); Store ID stays manual.
    if (prefill.alert_type) setSymptom(prefill.alert_type);
    if (prefill.severity) setSeverity(prefill.severity);
    const inner = {};
    if (prefill.customer) inner.customer = prefill.customer;
    if (prefill.location) inner.location = prefill.location;
    if (Object.keys(inner).length > 0) form.setFieldsValue(inner);
  }, [prefill && prefill._stamp]);

  // Both Store ID and symptom are mandatory.
  const canSubmit = !!storeId.trim() && !!symptom.trim();

  const handleFinish = (values) => {
    const payload = {
      store_id: storeId.trim(),
      // Symptom drives the fingerprint match; asset_name is intentionally null
      // for US Pharma (backend treats it as optional).
      alert_type: symptom.trim(),
      asset_name: null,
      severity,
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
    backgroundColor: "var(--bg-secondary)",
    borderColor: "var(--border-color)",
    borderRadius: 28,
    boxShadow:
      "0 6px 24px -8px rgba(15, 23, 42, 0.10), 0 2px 6px -2px rgba(15, 23, 42, 0.06)",
  };
  const inputStyle = { borderRadius: 9999 };

  return (
    <div className="w-full">
      <Card bodyStyle={{ padding: "14px 28px" }} style={cardStyle}>
        <div style={{ marginBottom: 10, textAlign: "center" }}>
          <h2
            className="t-text"
            style={{ fontSize: 18, fontWeight: 600, letterSpacing: "-0.01em", margin: 0, marginBottom: 2 }}
          >
            Which store, and what&apos;s happening?
          </h2>
          <p className="t-text-muted" style={{ fontSize: 12, margin: 0, lineHeight: 1.45 }}>
            Enter the Store ID and describe the symptom — we&apos;ll match it to
            that store&apos;s past incidents and return a troubleshooting answer.
          </p>
        </div>

        <Form form={form} layout="vertical" onFinish={handleFinish}>
          <Form.Item label={<span style={{ fontWeight: 600 }}>Store ID</span>} required>
            <Input
              size="large"
              value={storeId}
              // Store IDs are numeric — strip non-digits as the engineer types.
              onChange={(e) => setStoreId(e.target.value.replace(/[^0-9]/g, ""))}
              placeholder="e.g., 3001"
              inputMode="numeric"
              maxLength={12}
              style={inputStyle}
            />
          </Form.Item>

          <Form.Item
            label={<span style={{ fontWeight: 600 }}>What&apos;s the alert about?</span>}
            required
          >
            <Input
              size="large"
              value={symptom}
              onChange={(e) => setSymptom(e.target.value)}
              placeholder="e.g., LOS on WAN1, BGP flap, circuit down, store offline"
              maxLength={200}
              style={inputStyle}
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
            <div className="grid grid-cols-1 md:grid-cols-2 gap-x-4">
              <Form.Item label="Severity" className="md:col-span-2">
                <SeverityChipSelector value={severity} onChange={setSeverity} disabled={busy} />
              </Form.Item>
              <Form.Item label="Location" name="location">
                <Input size="large" maxLength={200} allowClear />
              </Form.Item>
              <Form.Item label="Technology / domain" name="technology">
                <Select size="large" allowClear options={TECHNOLOGY_OPTIONS} placeholder="(optional)" />
              </Form.Item>
              <Form.Item label="IP / circuit / device ID" name="ip_or_device_id">
                <Input size="large" maxLength={200} allowClear />
              </Form.Item>
              <Form.Item label="Error / alarm code" name="error_code">
                <Input size="large" maxLength={100} allowClear />
              </Form.Item>
              <Form.Item label="Notes" name="notes" className="md:col-span-2">
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
              style={{ minWidth: 200, borderRadius: 12, background: "#0b7285", borderColor: "#0b7285" }}
            >
              {busy ? "Searching…" : "Find store incidents →"}
            </Button>
          </div>
        </Form>
      </Card>
    </div>
  );
}
