import React, { useMemo, useState } from "react";
import { Button, Card, Progress, Space, Tag, Tooltip, message } from "antd";
import {
  CheckCircleFilled,
  DownloadOutlined,
  MinusCircleOutlined,
} from "@ant-design/icons";
import useAutoScrollIntoView from "../../hooks/useAutoScrollIntoView";
import { useTier1Theme } from "../../theme/ThemeProvider";
import {
  TIER1_DOWNLOAD_DEMO_ON, // Sprint 8.1 demo
  TIER1_UX_FIXES_ON,
} from "./tier1Constants";
import { downloadDiagnostics } from "./demoDownloads"; // Sprint 8.1 demo

/**
 * Sprint 7 + Sprint 8 — DeeperDiagnosticsCard
 *
 * Sprint 7 rendered Continue/Skip/Escalate buttons but left several
 * transitions incomplete. Sprint 8 (Track A) fixes:
 *   - Continue on final step closes the card (finalize branch).
 *   - Skip marks the step as visually struck (progress indicator +
 *     local step-status log).
 *   - Skip to escalation closes the card AND raises the callback so
 *     the parent opens the Escalation Package with pre-filled
 *     what_tried.
 *   - next_question.options render as clickable buttons that record
 *     the answer in `what_tried` and advance the step.
 *   - Auto-scroll the body into view on step change.
 *
 * Sprint 7 callback contract is preserved: `onContinueStep`,
 * `onSkipStep`, `onSkipToEscalation`, `onClose` all fire with the same
 * shape. A new optional prop `onAnswerQuestion` receives the
 * multiple-choice answer (skipped gracefully when not supplied).
 */
export default function DeeperDiagnosticsCard({
  diagnostics,
  onContinueStep,
  onSkipStep,
  onSkipToEscalation,
  onAnswerQuestion,
  onClose,
  // Sprint 8.1 demo — used by the Download button. Workspace threads
  // whatTried + matchedIncident through so the .txt export is complete.
  whatTried = [],
  matchedIncident = null,
}) {
  const [currentIdx, setCurrentIdx] = useState(0);
  // Per-step status — "done" | "skipped" | undefined.
  const [stepStatus, setStepStatus] = useState({});
  const { tokens, isModern } = useTier1Theme();

  const scrollRef = useAutoScrollIntoView(true, currentIdx);

  const steps = Array.isArray(diagnostics && diagnostics.steps)
    ? diagnostics.steps
    : [];
  const total = steps.length;
  const step = steps[currentIdx];

  const cardStyle = isModern
    ? {
        backgroundColor: tokens.surfaceBase,
        borderColor: tokens.surfaceElevated,
        borderRadius: tokens.radiusLg,
        boxShadow: tokens.shadowMd,
      }
    : {
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
      };

  const closeCard = () => {
    if (onClose) onClose();
  };

  const advanceOrClose = () => {
    if (currentIdx + 1 >= total) {
      message.success("Diagnostic path completed");
      closeCard();
      return;
    }
    setCurrentIdx((i) => i + 1);
  };

  const markStatus = (idx, status) => {
    setStepStatus((prev) => ({ ...prev, [idx]: status }));
  };

  if (!diagnostics) return null;
  if (!step) {
    return (
      <Card
        ref={scrollRef}
        title="Deeper Diagnostics"
        extra={
        <Space>
          {/* Sprint 8.1 demo */}
          {TIER1_DOWNLOAD_DEMO_ON && diagnostics && (
            <Tooltip title="Download diagnostics as .txt (demo only)">
              <Button
                size="small"
                icon={<DownloadOutlined />}
                onClick={() =>
                  downloadDiagnostics(diagnostics, whatTried, matchedIncident)
                }
              >
                Download
              </Button>
            </Tooltip>
          )}
          <Button onClick={closeCard}>Close</Button>
        </Space>
      }
        style={cardStyle}
      >
        <p className="t-text text-sm">
          No diagnostic steps available for this ticket.
        </p>
      </Card>
    );
  }

  const handleContinue = () => {
    markStatus(currentIdx, "done");
    if (onContinueStep) {
      onContinueStep({
        step: step.title,
        result: "normal",
        note: "continued",
      });
    }
    advanceOrClose();
  };

  const handleSkip = () => {
    markStatus(currentIdx, "skipped");
    if (onSkipStep) {
      onSkipStep({
        step: step.title,
        result: "skipped",
      });
    }
    advanceOrClose();
  };

  const handleSkipToEscalation = () => {
    // Mark current + remaining as skipped so the escalation bundle's
    // what_tried reflects the truth.
    const snapshot = { ...stepStatus };
    for (let i = currentIdx; i < total; i += 1) {
      snapshot[i] = i === currentIdx ? "escalated" : "skipped";
    }
    setStepStatus(snapshot);
    if (onSkipToEscalation) {
      onSkipToEscalation({
        step: step.title,
        result: "escalated",
        note: "skip_to_escalation",
      });
    }
    closeCard();
  };

  const handleAnswer = (optionLabel) => {
    markStatus(currentIdx, "done");
    if (onAnswerQuestion) {
      onAnswerQuestion({
        step: step.title,
        result: "answered",
        note: optionLabel,
      });
    } else if (onContinueStep) {
      onContinueStep({
        step: step.title,
        result: "answered",
        note: optionLabel,
      });
    }
    advanceOrClose();
  };

  const progressPct = Math.round(((currentIdx + 1) / Math.max(1, total)) * 100);

  return (
    <Card
      ref={scrollRef}
      title={
        <div className="flex justify-between items-center">
          <span>Deeper Diagnostics</span>
          <Tag>{diagnostics.severity || "P?"}</Tag>
        </div>
      }
      extra={
        <Space>
          {/* Sprint 8.1 demo */}
          {TIER1_DOWNLOAD_DEMO_ON && diagnostics && (
            <Tooltip title="Download diagnostics as .txt (demo only)">
              <Button
                size="small"
                icon={<DownloadOutlined />}
                onClick={() =>
                  downloadDiagnostics(diagnostics, whatTried, matchedIncident)
                }
              >
                Download
              </Button>
            </Tooltip>
          )}
          <Button onClick={closeCard}>Close</Button>
        </Space>
      }
      bodyStyle={{ padding: 24 }}
      style={cardStyle}
    >
      <div className="mb-3">
        {TIER1_UX_FIXES_ON ? (
          <StepDots
            total={total}
            currentIdx={currentIdx}
            stepStatus={stepStatus}
          />
        ) : (
          <Progress percent={progressPct} size="small" showInfo={false} />
        )}
        <div className="t-text-muted text-xs mt-1">
          Step {currentIdx + 1} of {total}
        </div>
      </div>

      <div className="mb-2 t-text font-semibold">{step.title}</div>
      {step.what_to_check && (
        <div className="mb-2 t-text text-sm">
          <span className="t-text-muted">What to check: </span>
          {step.what_to_check}
        </div>
      )}
      {step.why && (
        <div className="mb-2 t-text text-sm">
          <span className="t-text-muted">Why: </span>
          {step.why}
        </div>
      )}
      {step.command && (
        <div className="mb-2">
          <code className="t-text text-sm bg-black/10 px-2 py-1 rounded">
            {step.command}
          </code>
        </div>
      )}
      {step.expected_result && (
        <div className="mb-2 t-text text-sm">
          <span className="t-text-muted">Expected: </span>
          {step.expected_result}
        </div>
      )}
      {step.next_action_if_abnormal && (
        <div className="mb-2 t-text text-sm">
          <span className="t-text-muted">If abnormal: </span>
          {step.next_action_if_abnormal}
        </div>
      )}

      {diagnostics.validation && currentIdx === total - 1 && (
        <div className="mt-4 t-text text-sm">
          <span className="t-text-muted">Validation: </span>
          {diagnostics.validation}
        </div>
      )}

      {/* Sprint 8 — render next_question options as clickable buttons
          on the last step so the engineer has a single-click path to
          record an answer and close the card. */}
      {TIER1_UX_FIXES_ON
        && currentIdx === total - 1
        && diagnostics.next_question
        && Array.isArray(diagnostics.next_question.options)
        && diagnostics.next_question.options.length > 0 && (
        <div className="mt-4">
          <div className="t-text-muted text-xs mb-2">
            {diagnostics.next_question.prompt || "What was the outcome?"}
          </div>
          <Space wrap>
            {diagnostics.next_question.options.map((opt) => (
              <Button key={opt} size="small" onClick={() => handleAnswer(opt)}>
                {opt}
              </Button>
            ))}
          </Space>
        </div>
      )}

      <div className="mt-4 flex justify-between">
        <Space>
          <Button
            onClick={handleContinue}
            type="primary"
            style={{
              backgroundColor: isModern ? undefined : "#0A3F63",
              borderColor: isModern ? "transparent" : "#0A3F63",
              background: isModern ? tokens.gradientAccent : undefined,
            }}
          >
            {currentIdx + 1 >= total ? "Finish →" : "Continue →"}
          </Button>
          <Tooltip title="Record as skipped and advance">
            <Button onClick={handleSkip}>Skip this step</Button>
          </Tooltip>
        </Space>
        <Button danger onClick={handleSkipToEscalation}>
          Skip to escalation
        </Button>
      </div>
    </Card>
  );
}


function StepDots({ total, currentIdx, stepStatus }) {
  const dots = useMemo(() => Array.from({ length: total }, (_, i) => i), [total]);
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
      {dots.map((i) => {
        const status = stepStatus[i];
        const isCurrent = i === currentIdx;
        let Icon = null;
        let color = "rgba(15,23,42,0.20)"; // unvisited
        if (status === "done") {
          Icon = CheckCircleFilled;
          color = "#10B981";
        } else if (status === "skipped") {
          Icon = MinusCircleOutlined;
          color = "#94A3B8";
        }
        if (isCurrent && !status) {
          color = "#1E3A8A";
        }
        return (
          <span
            key={i}
            aria-label={`Step ${i + 1}${status ? ` (${status})` : ""}`}
            style={{
              width: 14,
              height: 14,
              borderRadius: "50%",
              border: `2px solid ${color}`,
              background: isCurrent || status === "done" ? color : "transparent",
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              transition: "all 150ms ease",
            }}
          >
            {Icon && (
              <Icon style={{ color: "#fff", fontSize: 10 }} />
            )}
          </span>
        );
      })}
    </div>
  );
}
