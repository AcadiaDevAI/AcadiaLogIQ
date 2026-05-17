import React, { useEffect, useState } from "react";
import { Button, Card, message } from "antd";
import { useChat } from "../hooks/ChatContext";
// Fingerprint landing disabled — Tier-1 Copilot is now the default entry.
// Keep import commented so it can be re-enabled in a single line if needed.
// import FingerprintInputScreen from "./FingerprintInputScreen";
import LandingPage from "./LandingPage";
import Tier1IntakeForm from "./Tier1Copilot/Tier1IntakeForm";
import Tier1AnswerCard from "./Tier1Copilot/Tier1AnswerCard";
import Tier1FollowupChips from "./Tier1Copilot/Tier1FollowupChips";
import Tier1Workspace from "./Tier1Copilot/Tier1Workspace";
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
export default function LandingRouter() {
  const { state, dispatch } = useChat();
  // Default landing is now Tier-1 Copilot (was "fingerprint").
  const [screen, setScreen] = useState("tier1");
  const [lastFingerprint, setLastFingerprint] = useState(null);

  // Sprint 6 — Tier-1 local flow state (form → analyze → answer → feedback).
  // Lives on LandingRouter so "Start a new alert" can clear it in one setter.
  const [tier1Busy, setTier1Busy] = useState(false);
  const [tier1Result, setTier1Result] = useState(null);
  const [tier1SessionId] = useState(
    () => `tier1-${Math.random().toString(36).slice(2, 10)}`
  );

  // ── Sprint 11 — chat → journey return path ──────────
  // When JourneyMessageActions dispatches RESUME_JOURNEY (from
  // "Return to Stages" or "Escalate to Tier 2"), ChatContext sets
  // journeyResumeSessionId AND clears selectedMode. AppLayout
  // re-mounts us; this effect picks up the resume signal, jumps
  // straight to screen="tier1" with a synthesized result blob,
  // and clears the resume field so it doesn't re-fire.
  //
  // The synthesized blob has only session_id + started_at; that is
  // enough for useTier1Session to hydrate and for the journey
  // panels to fetch their own data via /tier1/journey/<sid>/initial,
  // /resume-state, /stage-N. The match-card may show degraded data
  // until the next /match call, which is acceptable for a return
  // path (the user came in via journey, not via fresh /analyze).
  useEffect(() => {
    if (!state.journeyResumeSessionId) return;
    const sid = state.journeyResumeSessionId;
    setScreen("tier1");
    setTier1Result({
      session_id: sid,
      started_at: new Date().toISOString(),
      top_5_match_ids: [],
    });
    dispatch({ type: "CLEAR_JOURNEY_RESUME" });
  }, [state.journeyResumeSessionId, dispatch]);

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

  // ─────────────────────────────────────────────────────────────
  // Fingerprint landing screen disabled.
  // Tier-1 Copilot is now the default entry (see useState("tier1") above).
  // To re-enable: uncomment the FingerprintInputScreen import at the top
  // and uncomment the block below, and revert default screen to "fingerprint".
  // ─────────────────────────────────────────────────────────────
  // if (screen === "fingerprint") {
  //   return (
  //     <>
  //       <FingerprintInputScreen
  //         onMatch={handleMatch}
  //         onNoMatch={handleNoMatch}
  //         onSkip={goToModes}
  //       />
  //       {TIER1_FRONTEND_ON && (
  //         <div className="flex justify-center pb-6">
  //           <Button
  //             size="large"
  //             onClick={goToTier1}
  //             style={{ minWidth: 200 }}
  //           >
  //             Tier-1 Copilot
  //           </Button>
  //         </div>
  //       )}
  //     </>
  //   );
  // }

  if (screen === "tier1") {
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
            // Fingerprint screen is disabled — fall back to mode picker.
            setTier1Result(null);
            setScreen("modes");
          }}
        />
      );
    }

    // Sprint 7 — mount the progressive Workspace when the backend
    // returned a Sprint-7 session_id. Falls back to the Sprint 6 card
    // + chips layout when session_id is missing.
    if (tier1Result.session_id) {
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
            {/* Fingerprint flow disabled — "Try a different fingerprint" hidden.
            <Button
              size="large"
              onClick={() => setScreen("fingerprint")}
            >
              Try a different fingerprint
            </Button>
            */}
            <Button
              type="primary"
              size="large"
              onClick={goToModes}
              style={{
                backgroundColor: "var(--acadia-primary)",
                borderColor: "var(--acadia-primary)",
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
