import React, { useState } from "react";
import { Button, Card } from "antd";
import { useChat } from "../hooks/ChatContext";
import FingerprintInputScreen from "./FingerprintInputScreen";
import LandingPage from "./LandingPage";

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
  const { dispatch } = useChat();
  const [screen, setScreen] = useState("fingerprint");
  const [lastFingerprint, setLastFingerprint] = useState(null);

  const goToModes = () => setScreen("modes");

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
      <FingerprintInputScreen
        onMatch={handleMatch}
        onNoMatch={handleNoMatch}
        onSkip={goToModes}
      />
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
