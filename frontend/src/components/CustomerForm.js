import React, { useState } from "react";
import { Card, Form, Input, Button, message } from "antd";
import { useChat } from "../hooks/ChatContext";
import { patchSessionForm } from "../services/api";

/**
 * CustomerForm — Troubleshooting > Customer-specific sub-mode (PRD §3.A).
 *
 * Collects the 5 canonical customer-specific fields, persists them to
 * the session via POST /chat/sessions/{id}/mode/form, updates
 * ChatContext so downstream renders are mode-aware, and optionally
 * seeds a starter query on submit so the user gets an immediate answer.
 *
 * Rendered in place of EmptyState by ChatArea when
 *   LOGIQ_SPRINT2_FRONTEND && GUIDED_WORKFLOW_ENABLED
 *   && selectedMode === "troubleshooting"
 *   && subMode === "customer_specific"
 *   && messages.length === 0
 */
export default function CustomerForm({ onSeed }) {
  const { state, dispatch } = useChat();
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

  const handleFinish = async (values) => {
    const customerName = (values.customerName || "").trim();
    const technologyDomain = (values.technologyDomain || "").trim();
    const ticketId = (values.ticketId || "").trim();
    const issueSummary = (values.issueSummary || "").trim();
    const additionalContext = (values.additionalContext || "").trim();

    if (!customerName) {
      message.warning("Customer name is required.");
      return;
    }

    setSubmitting(true);
    try {
      if (state.sessionId) {
        try {
          await patchSessionForm(state.sessionId, {
            customer_name: customerName || null,
            technology_domain: technologyDomain || null,
            ticket_id: ticketId || null,
            issue_summary: issueSummary || null,
            form_data: {
              sub_mode: "customer_specific",
              customer_name: customerName,
              technology_domain: technologyDomain,
              ticket_id: ticketId,
              issue_summary: issueSummary,
              additional_context: additionalContext,
            },
          });
        } catch (err) {
          if (err?.response?.status === 403) {
            console.info("[CustomerForm] patch 403 — Sprint 2 backend flag off");
          } else {
            console.warn("[CustomerForm] patch failed", err);
          }
        }
      }

      dispatch({
        type: "SET_FORM_DATA",
        payload: {
          sub_mode: "customer_specific",
          customer_name: customerName,
          technology_domain: technologyDomain,
          ticket_id: ticketId,
          issue_summary: issueSummary,
          additional_context: additionalContext,
        },
      });

      const seedParts = [];
      if (issueSummary) {
        seedParts.push(`Issue: ${issueSummary}`);
      }
      seedParts.push(`Customer: ${customerName}`);
      if (technologyDomain) seedParts.push(`Technology: ${technologyDomain}`);
      if (ticketId) seedParts.push(`Ticket: ${ticketId}`);
      if (additionalContext) seedParts.push(`Context: ${additionalContext}`);

      const seedQuery = issueSummary
        ? `${issueSummary} — ${seedParts.slice(1).join(", ")}`
        : `What historical issues have we seen for ${customerName}${
            technologyDomain ? ` on ${technologyDomain}` : ""
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
              Customer-specific troubleshooting
            </h2>
            <p className="t-text-muted text-xs">
              Fill in what you know. We'll pull historical context for this
              customer before answering.
            </p>
          </div>
          <Form
            form={form}
            layout="vertical"
            onFinish={handleFinish}
            requiredMark={false}
          >
            <Form.Item
              name="customerName"
              label={<span className="t-text text-sm">Customer name</span>}
              rules={[{ required: true, message: "Customer name is required" }]}
            >
              <Input placeholder="e.g. Acme Networks" maxLength={120} />
            </Form.Item>
            <Form.Item
              name="technologyDomain"
              label={<span className="t-text text-sm">Technology domain</span>}
            >
              <Input placeholder="e.g. Routing / Cisco Catalyst" maxLength={120} />
            </Form.Item>
            <Form.Item
              name="ticketId"
              label={<span className="t-text text-sm">Ticket ID</span>}
            >
              <Input placeholder="e.g. INC-10037" maxLength={60} />
            </Form.Item>
            <Form.Item
              name="issueSummary"
              label={<span className="t-text text-sm">Issue summary</span>}
            >
              <Input.TextArea
                rows={2}
                placeholder="One-line description of what's happening"
                maxLength={280}
              />
            </Form.Item>
            <Form.Item
              name="additionalContext"
              label={<span className="t-text text-sm">Additional context</span>}
            >
              <Input.TextArea
                rows={3}
                placeholder="Anything else the assistant should know (optional)"
                maxLength={600}
              />
            </Form.Item>
            <div className="flex justify-end">
              <Button
                type="primary"
                htmlType="submit"
                loading={submitting}
                style={{
                  backgroundColor: "#0A3F63",
                  borderColor: "#0A3F63",
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
