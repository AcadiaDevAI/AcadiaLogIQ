// Sprint 10.5 §2.3 — inline journey actions on assistant chat messages.
//
// Renders [← Return to Stages] and [📋 Escalate to Tier 2] inline beside
// the existing per-message Copy / 👍 / 👎 row. ONLY rendered when the
// chat session has `journey_session_id` in its metadata — i.e. when
// the chat was created via a Stage 4 Search-KB handoff. Non-journey
// chats render the message footer unchanged.
//
// Sprint 11 — navigation rewritten.
//   The original Sprint 10.5 implementation used
//   `window.location.href = '/tier1/journey/<sid>'` for navigation.
//   That triggered a full page reload but the React app has no URL
//   routing — App.js never reads window.location.pathname. Reload
//   landed the user on the LandingPage, NOT on the journey panel.
//
//   Replaced with state-based dispatch: RESUME_JOURNEY tells
//   ChatContext to (a) carry the journey session id forward and
//   (b) clear selectedMode so AppLayout falls back to LandingRouter,
//   which picks up the resume sid via useEffect and re-mounts
//   Tier1Workspace with that session. Mirrors the Stage 4 → chat
//   handoff pattern in reverse (state-based, no URL rewrite).
//
// Order matters in handleEscalate: we record telemetry FIRST, then
// dispatch the resume. If we dispatched first, the unmount could
// cancel the in-flight POST and lose the event.

import React from "react";
import { Button, Tooltip, message as antMessage } from "antd";
import { ArrowLeftOutlined, ExportOutlined } from "@ant-design/icons";

import { useChat } from "../../hooks/ChatContext";
import { postJourneyEvent } from "../Tier1Copilot/journey/journeyApi";


export default function JourneyMessageActions({ journeySessionId }) {
  const { dispatch } = useChat();
  const [escalating, setEscalating] = React.useState(false);

  if (!journeySessionId) return null;

  const handleBack = () => {
    // Sprint 11 — state-based return. ChatContext will clear
    // selectedMode + carry the sid to LandingRouter which will
    // re-mount Tier1Workspace; resume-state then lands the user
    // at whichever stage they were last on.
    dispatch({
      type: "RESUME_JOURNEY",
      payload: { journeySessionId },
    });
  };

  const handleEscalate = async () => {
    if (escalating) return;
    setEscalating(true);
    try {
      // Sprint 10.7 §4.3 — fire TWO events in order:
      //   1. escalation_initiated_from_chat — Sprint 10.5 analytics
      //      surface, kept verbatim so existing dashboards stay valid.
      //   2. stage_advanced (stage_5) — the Sprint 10.7 resume signal,
      //      so /resume-state opens the journey at Stage 5 with the
      //      escalation package visible.
      // Order matters: write 1 first so analytics record even if the
      // resume write fails; write 2 second so the engineer's resume
      // pointer is updated. We `await` both before dispatching so the
      // unmount can't cancel either request.
      await postJourneyEvent(
        journeySessionId,
        "stage_5",
        "escalation_initiated_from_chat",
      );
      await postJourneyEvent(
        journeySessionId,
        "stage_5",
        "stage_advanced",
      );
      // Sprint 11 — state-based return. resume-state will read the
      // stage_advanced row above and land the user at Stage 5,
      // matching what Stage 4's "Escalate to Tier-2" button does
      // (which just calls onReveal('stage_5') in-place).
      dispatch({
        type: "RESUME_JOURNEY",
        payload: { journeySessionId },
      });
    } catch (err) {
      antMessage.error("Could not open escalation. Please try again.");
      setEscalating(false);
    }
    // Note: we do not clear escalating in the success path because
    // the dispatch above triggers AppLayout to swap LandingRouter in
    // and unmount this component.
  };

  return (
    <>
      <Tooltip title="Return to the Resolution Journey panels">
        <Button
          type="text"
          size="small"
          icon={<ArrowLeftOutlined />}
          onClick={handleBack}
        >
          Return to Stages
        </Button>
      </Tooltip>
      <Tooltip title="Open the escalation package for Tier-2 handoff">
        <Button
          type="text"
          size="small"
          icon={<ExportOutlined />}
          loading={escalating}
          onClick={handleEscalate}
        >
          Escalate to Tier 2
        </Button>
      </Tooltip>
    </>
  );
}
