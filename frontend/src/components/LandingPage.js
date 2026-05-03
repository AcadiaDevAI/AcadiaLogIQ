import React, { useState } from "react";
import { Radio, Button, Card, message } from "antd";
import {
  ToolOutlined,
  FileTextOutlined,
  RiseOutlined,
  ApiOutlined,
} from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { setSessionMode } from "../services/api";

/**
 * LandingPage — PRD Sections 1 and 3.
 *
 * First screen the user sees when GUIDED_WORKFLOW_ENABLED is true and no
 * mode is locked on the current session. Dispatches SET_MODE on Continue
 * so the rest of the app (ChatArea, ModeBadge) becomes mode-aware.
 *
 * Sprint 1 scope:
 *   - 4 main mode radios
 *   - Troubleshooting sub-mode radio (Customer-specific / Technology-specific)
 *   - Continue button writes mode to backend + dispatches to ChatContext
 *
 * Out of Sprint 1 scope (handled in later sprints):
 *   - Structured input forms for each sub-mode (Sprint 2)
 *   - Ticket-handling / Escalation / Vendor sub-flows (Sprint 4)
 *   - Mode-aware LLM prompts (Sprint 3)
 */
const MODE_OPTIONS = [
  {
    value: "troubleshooting",
    label: "Assistance with troubleshooting based on historical data",
    icon: <ToolOutlined />,
  },
  {
    value: "ticket_handling",
    label: "Assistance with ticket handling process",
    icon: <FileTextOutlined />,
  },
  {
    value: "escalation",
    label: "Assistance with escalation",
    icon: <RiseOutlined />,
  },
  {
    value: "vendor_oem",
    label: "Assistance with vendor / OEM engagement",
    icon: <ApiOutlined />,
  },
];

const TROUBLESHOOTING_SUB_OPTIONS = [
  { value: "customer_specific", label: "Customer specific" },
  { value: "technology_specific", label: "Technology specific" },
];

export default function LandingPage() {
  const { state, dispatch } = useChat();

  const [mode, setMode] = useState(null);
  const [subMode, setSubMode] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  const needsSubMode = mode === "troubleshooting";
  const canContinue = !!mode && (!needsSubMode || !!subMode);

  const handleContinue = async () => {
    if (!canContinue) return;
    setSubmitting(true);
    try {
      // If the session already exists (user came back to landing via
      // Change Context), persist the mode to the backend. If there's
      // no session yet, first /ask call will create it; we defer the
      // backend write until the session exists.
      if (state.sessionId) {
        await setSessionMode(state.sessionId, {
          selectedMode: mode,
          subMode: subMode || null,
        });
      }
      dispatch({
        type: "SET_MODE",
        payload: { selectedMode: mode, subMode: subMode || null },
      });
    } catch (err) {
      message.error("Could not save your selection. Please try again.");
      setSubmitting(false);
      return;
    }
    setSubmitting(false);
  };

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-8 t-bg-primary">
      <div className="w-full max-w-2xl">
        {/* Header */}
        <div className="text-center mb-6">
          <img
            src="/logo.png"
            alt="LogIQ"
            className="h-12 mx-auto mb-3 object-contain"
          />
          <h1 className="text-xl font-bold t-text">Operations Guidance Assistant</h1>
          <p className="t-text-muted text-sm mt-1">Select what you need help with</p>
        </div>

        {/* Main mode card */}
        <Card
          className="mb-4"
          bodyStyle={{ padding: 20 }}
          style={{ backgroundColor: "var(--bg-secondary)", borderColor: "var(--border-color)" }}
        >
          <p className="text-sm font-medium t-text mb-3">
            What can I help you with today?
          </p>
          <Radio.Group
            value={mode}
            onChange={(e) => {
              setMode(e.target.value);
              setSubMode(null);
            }}
            className="flex flex-col gap-2"
          >
            {MODE_OPTIONS.map((opt) => (
              <Radio
                key={opt.value}
                value={opt.value}
                className="t-text py-1.5 pl-1"
              >
                <span className="inline-flex items-center gap-2">
                  <span style={{ color: "var(--brand-accent)" }}>{opt.icon}</span>
                  {opt.label}
                </span>
              </Radio>
            ))}
          </Radio.Group>
        </Card>

        {/* Secondary sub-mode card — troubleshooting only in Sprint 1 */}
        {needsSubMode && (
          <Card
            className="mb-4"
            bodyStyle={{ padding: 20 }}
            style={{ backgroundColor: "var(--bg-secondary)", borderColor: "var(--border-color)" }}
          >
            <p className="text-sm font-medium t-text mb-3">
              Please choose troubleshooting context:
            </p>
            <Radio.Group
              value={subMode}
              onChange={(e) => setSubMode(e.target.value)}
              className="flex flex-col gap-2"
            >
              {TROUBLESHOOTING_SUB_OPTIONS.map((opt) => (
                <Radio
                  key={opt.value}
                  value={opt.value}
                  className="t-text py-1.5 pl-1"
                >
                  {opt.label}
                </Radio>
              ))}
            </Radio.Group>
          </Card>
        )}

        {/* Continue */}
        <div className="flex justify-center">
          <Button
            type="primary"
            size="large"
            onClick={handleContinue}
            disabled={!canContinue || submitting}
            loading={submitting}
            style={{
              backgroundColor: canContinue ? "var(--acadia-primary)" : undefined,
              borderColor: canContinue ? "var(--acadia-primary)" : undefined,
              minWidth: 160,
            }}
          >
            Continue
          </Button>
        </div>

        <p className="text-center t-text-faint text-[10px] mt-6">
          Your selection sets the working context for this session.
        </p>
      </div>
    </div>
  );
}
