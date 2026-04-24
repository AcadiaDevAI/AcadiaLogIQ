import React, { useState } from "react";
import { Button, Card, message } from "antd";
import { useChat } from "../hooks/ChatContext";
import FingerprintInputScreen from "./FingerprintInputScreen";
import LandingPage from "./LandingPage";
import Tier1IntakeForm from "./Tier1Copilot/Tier1IntakeForm";
import Tier1AnswerCard from "./Tier1Copilot/Tier1AnswerCard";
import Tier1FollowupChips from "./Tier1Copilot/Tier1FollowupChips";
import Tier1Workspace from "./Tier1Copilot/Tier1Workspace";
import { TIER1_PROGRESSIVE_ON } from "./Tier1Copilot/tier1Constants";
import { analyzeAlert } from "./Tier1Copilot/tier1Api";

/**
 * LandingRouter — Sprint 4 entry flow.
 *
 * Screens:
 *   - "fingerprint" (default)   user enters a fingerprint or skips
 *   - "nomatch"                 miss feedback + "Pick a mode" CTA
 *   - "modes"                   forwards to the classic LandingPage
 *
 * On match, the FingerprintInputScreen calls /fingerprint/lookup, appends
 * the resulting assistant answer into ChatContext, and dispatches
 * SET_MODE(troubleshooting) so AppLayout's `!state.selectedMode` check
 * flips and the user lands directly in ChatArea reading their answer.
 */
// Sprint 6 — Tier-1 Copilot entry button. Hidden entirely when the
// build-time flag REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND !== "true".
// Keeping the flag read at module scope so it's evaluated once and
// inlined by the bundler when off.
const TIER1_FRONTEND_ON =
  process.env.REACT_APP_LOGIQ_TIER1_COPILOT_FRONTEND === "true";

export default function LandingRouter() {
  const { dispatch } = useChat();
  const [screen, setScreen] = useState("fingerprint");
  const [lastFingerprint, setLastFingerprint] = useState(null);

  // Sprint 6 — Tier-1 local flow state (form → analyze → answer → feedback).
  // Lives on LandingRouter so "Start a new alert" can clear it in one setter.
  const [tier1Busy, setTier1Busy] = useState(false);
  const [tier1Result, setTier1Result] = useState(null);
  const [tier1SessionId] = useState(
    () => `tier1-${Math.random().toString(36).slice(2, 10)}`
  );

  const goToModes = () => setScreen("modes");
  const goToTier1 = () => {
    setTier1Result(null);
    setScreen("tier1");
  };

  const handleMatch = ({ fingerprint, answer, sessionId }) => {
    // Append the user probe + assistant answer to the transcript so
    // ChatArea renders them immediately (no second round-trip).
    dispatch({
      type: "ADD_USER_MESSAGE",
      payload: `[Fingerprint lookup] ${fingerprint}`,
    });
    dispatch({
      type: "ADD_ASSISTANT_MESSAGE",
      payload: {
        answer: answer || "",
        sources: [],
        sessionId: sessionId || null,
      },
    });
    // Flip the mode gate so AppLayout shows ChatArea next.
    dispatch({
      type: "SET_MODE",
      payload: { selectedMode: "troubleshooting", subMode: null },
    });
  };

  const handleNoMatch = ({ fingerprint }) => {
    setLastFingerprint(fingerprint);
    setScreen("nomatch");
  };

  if (screen === "fingerprint") {
    return (
      <>
        <FingerprintInputScreen
          onMatch={handleMatch}
          onNoMatch={handleNoMatch}
          onSkip={goToModes}
        />
        {TIER1_FRONTEND_ON && (
          <div className="flex justify-center pb-6">
            <Button
              size="large"
              onClick={goToTier1}
              style={{ minWidth: 200 }}
            >
              Tier-1 Copilot
            </Button>
          </div>
        )}
      </>
    );
  }

  if (screen === "tier1" && TIER1_FRONTEND_ON) {
    const handleTier1Submit = async (payload) => {
      setTier1Busy(true);
      try {
        const data = await analyzeAlert(payload);
        setTier1Result(data);
      } catch (err) {
        const detail =
          err?.response?.data?.detail || err?.message || "Analyze failed.";
        message.error(`Tier-1 analyze failed: ${detail}`);
      } finally {
        setTier1Busy(false);
      }
    };

    if (!tier1Result) {
      return (
        <Tier1IntakeForm
          sessionId={tier1SessionId}
          busy={tier1Busy}
          onSubmit={handleTier1Submit}
          onBack={() => {
            setTier1Result(null);
            setScreen("fingerprint");
          }}
        />
      );
    }

    // Sprint 7 — mount the progressive Workspace when the build-time
    // flag is baked in AND the backend returned a Sprint-7 session_id
    // (which only happens when LOGIQ_TIER1_PROGRESSIVE_BACKEND is true).
    // Either flag off → fall back to the Sprint 6 card + chips layout.
    if (TIER1_PROGRESSIVE_ON && tier1Result.session_id) {
      return (
        <Tier1Workspace
          result={tier1Result}
          onNewAlert={() => setTier1Result(null)}
        />
      );
    }

    return (
      <div className="flex-1 overflow-y-auto px-4 py-8 t-bg-primary">
        <div className="w-full max-w-3xl mx-auto">
          <Tier1AnswerCard result={tier1Result} />
          <Tier1FollowupChips
            responseId={tier1Result.response_id}
            sessionId={tier1SessionId}
            onNewAlert={() => setTier1Result(null)}
          />
        </div>
      </div>
    );
  }

  if (screen === "nomatch") {
    return (
      <div className="flex-1 flex items-center justify-center px-4 py-8 t-bg-primary">
        <div className="w-full max-w-2xl">
          <div className="text-center mb-6">
            <img
              src="/logo.png"
              alt="LogIQ"
              className="h-12 mx-auto mb-3 object-contain"
            />
            <h1 className="text-xl font-bold t-text">No exact match</h1>
            <p className="t-text-muted text-sm mt-1">
              We don&apos;t have a gold-standard write-up for{" "}
              <code>{lastFingerprint}</code> yet.
            </p>
          </div>
          <Card
            className="mb-4"
            bodyStyle={{ padding: 20 }}
            style={{
              backgroundColor: "var(--bg-secondary)",
              borderColor: "var(--border-color)",
            }}
          >
            <p className="text-sm t-text mb-2">
              Pick a workflow below to continue — the assistant will still
              search the full ticket history, SOPs, and KB for related
              guidance.
            </p>
          </Card>
          <div className="flex justify-center gap-3">
            <Button
              size="large"
              onClick={() => setScreen("fingerprint")}
            >
              Try a different fingerprint
            </Button>
            <Button
              type="primary"
              size="large"
              onClick={goToModes}
              style={{
                backgroundColor: "#0A3F63",
                borderColor: "#0A3F63",
                minWidth: 160,
              }}
            >
              Pick a mode
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return <LandingPage />;
}
