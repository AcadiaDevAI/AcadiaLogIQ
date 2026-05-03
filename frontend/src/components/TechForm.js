import React, { useState } from "react";
import { Card, Form, Select, Input, Button, message } from "antd";
import { useChat } from "../hooks/ChatContext";
import { patchSessionForm } from "../services/api";

/**
 * TechForm — Troubleshooting > Technology-specific sub-mode (PRD §3.B).
 *
 * Collects a canonical technology domain + optional issue details,
 * persists to the session, and seeds a starter query.
 *
 * Rendered in place of EmptyState when the flag matrix matches and
 *   selectedMode === "troubleshooting" && subMode === "technology_specific"
 *   && messages.length === 0
 */
const TECH_DOMAINS = [
  { value: "routing", label: "Routing" },
  { value: "switching", label: "Switching" },
  { value: "wireless", label: "Wireless" },
  { value: "security_firewall", label: "Security / Firewall" },
  { value: "sdwan", label: "SD-WAN" },
  { value: "voice_uc", label: "Voice / UC" },
  { value: "datacenter", label: "Data Center" },
  { value: "cloud_networking", label: "Cloud Networking" },
  { value: "observability", label: "Observability / Monitoring" },
  { value: "other", label: "Other" },
];

export default function TechForm({ onSeed }) {
  const { state, dispatch } = useChat();
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

  const handleFinish = async (values) => {
    const technologyDomain = (values.technologyDomain || "").trim();
    const vendor = (values.vendor || "").trim();
    const issueSummary = (values.issueSummary || "").trim();
    const additionalContext = (values.additionalContext || "").trim();

    if (!technologyDomain) {
      message.warning("Technology domain is required.");
      return;
    }

    setSubmitting(true);
    try {
      if (state.sessionId) {
        try {
          await patchSessionForm(state.sessionId, {
            technology_domain: technologyDomain || null,
            issue_summary: issueSummary || null,
            form_data: {
              sub_mode: "technology_specific",
              technology_domain: technologyDomain,
              vendor: vendor,
              issue_summary: issueSummary,
              additional_context: additionalContext,
            },
          });
        } catch (err) {
          if (err?.response?.status === 403) {
            console.info("[TechForm] patch 403 — Sprint 2 backend flag off");
          } else {
            console.warn("[TechForm] patch failed", err);
          }
        }
      }

      dispatch({
        type: "SET_FORM_DATA",
        payload: {
          sub_mode: "technology_specific",
          technology_domain: technologyDomain,
          vendor: vendor,
          issue_summary: issueSummary,
          additional_context: additionalContext,
        },
      });

      const domainLabel =
        TECH_DOMAINS.find((d) => d.value === technologyDomain)?.label ||
        technologyDomain;

      const seedQuery = issueSummary
        ? `${issueSummary} — Technology: ${domainLabel}${
            vendor ? `, Vendor: ${vendor}` : ""
          }${additionalContext ? `, Context: ${additionalContext}` : ""}`
        : `Common issues and fixes for ${domainLabel}${
            vendor ? ` (${vendor})` : ""
          }?`;

      if (onSeed) {
        onSeed(seedQuery);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-6">
      <div className="w-full max-w-xl">
        <Card
          bodyStyle={{ padding: 20 }}
          style={{
            backgroundColor: "var(--bg-secondary)",
            borderColor: "var(--border-color)",
          }}
        >
          <div className="mb-4">
            <h2 className="text-base font-semibold t-text mb-1">
              Technology-specific troubleshooting
            </h2>
            <p className="t-text-muted text-xs">
              Pick a domain and we'll surface historical patterns relevant to
              that technology.
            </p>
          </div>
          <Form
            form={form}
            layout="vertical"
            onFinish={handleFinish}
            requiredMark={false}
          >
            <Form.Item
              name="technologyDomain"
              label={<span className="t-text text-sm">Technology domain</span>}
              rules={[{ required: true, message: "Please pick a domain" }]}
            >
              <Select
                placeholder="Select a technology"
                options={TECH_DOMAINS}
                showSearch
                optionFilterProp="label"
              />
            </Form.Item>
            <Form.Item
              name="vendor"
              label={<span className="t-text text-sm">Vendor (optional)</span>}
            >
              <Input placeholder="e.g. Cisco, Palo Alto" maxLength={80} />
            </Form.Item>
            <Form.Item
              name="issueSummary"
              label={<span className="t-text text-sm">Issue summary</span>}
            >
              <Input.TextArea
                rows={2}
                placeholder="One-line description (optional)"
                maxLength={280}
              />
            </Form.Item>
            <Form.Item
              name="additionalContext"
              label={<span className="t-text text-sm">Additional context</span>}
            >
              <Input.TextArea
                rows={3}
                placeholder="Relevant environment details (optional)"
                maxLength={600}
              />
            </Form.Item>
            <div className="flex justify-end">
              <Button
                type="primary"
                htmlType="submit"
                loading={submitting}
                style={{
                  backgroundColor: "var(--acadia-primary)",
                  borderColor: "var(--acadia-primary)",
                  minWidth: 140,
                }}
              >
                Start session
              </Button>
            </div>
          </Form>
        </Card>
      </div>
    </div>
  );
}
