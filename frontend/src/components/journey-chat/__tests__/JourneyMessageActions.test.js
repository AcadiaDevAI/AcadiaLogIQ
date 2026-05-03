// Sprint 10.5 §4.2 + Sprint 11 — JourneyMessageActions tests.
//
// Setup note: this codebase has no other frontend tests yet. To run
// these you need React Testing Library installed alongside the Jest
// harness that already ships with react-scripts:
//   cd frontend && npm i -D @testing-library/react @testing-library/jest-dom
// Then `npm test -- JourneyMessageActions` from frontend/.
//
// Sprint 11 rewrite — the chat-back navigation no longer uses
// window.location.href (which never worked because the React app has
// no URL routing). It now dispatches RESUME_JOURNEY into ChatContext.
// Tests verify the dispatch payload and ordering instead.

import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import JourneyMessageActions from "../JourneyMessageActions";

jest.mock(
  "../../Tier1Copilot/journey/journeyApi",
  () => ({
    postJourneyEvent: jest.fn(),
  }),
);

// Sprint 11 — useChat() returns { dispatch }. We mock the hook so the
// component can be rendered without wrapping in a real ChatProvider.
const mockDispatch = jest.fn();
jest.mock("../../../hooks/ChatContext", () => ({
  useChat: () => ({ dispatch: mockDispatch }),
}));

import { postJourneyEvent } from "../../Tier1Copilot/journey/journeyApi";


beforeEach(() => {
  postJourneyEvent.mockReset();
  mockDispatch.mockReset();
});


describe("JourneyMessageActions", () => {
  test("renders_nothing_when_journey_session_id_is_missing", () => {
    const { container } = render(
      <JourneyMessageActions journeySessionId={null} />,
    );
    // Component returns null → no DOM nodes produced.
    expect(container.firstChild).toBeNull();
    expect(screen.queryByText(/Return to Stages/i)).toBeNull();
    expect(screen.queryByText(/Escalate to Tier 2/i)).toBeNull();
  });

  test("renders_both_buttons_when_journey_session_id_present", () => {
    render(<JourneyMessageActions journeySessionId="sess_xyz" />);
    expect(screen.getByText(/Return to Stages/i)).toBeInTheDocument();
    expect(screen.getByText(/Escalate to Tier 2/i)).toBeInTheDocument();
  });

  test("back_button_dispatches_resume_journey_with_session_id", () => {
    // Sprint 11 — replaces the prior window.location.href assertion.
    // The handler must dispatch RESUME_JOURNEY so LandingRouter can
    // re-mount Tier1Workspace at the user's last stage.
    render(<JourneyMessageActions journeySessionId="sess_xyz" />);
    fireEvent.click(screen.getByText(/Return to Stages/i));
    expect(mockDispatch).toHaveBeenCalledWith({
      type: "RESUME_JOURNEY",
      payload: { journeySessionId: "sess_xyz" },
    });
  });

  test("escalate_button_records_two_events_in_order_then_dispatches_resume", async () => {
    // Sprint 10.7 §4.5 + Sprint 11 — the escalate handler records TWO
    // events before dispatching, in this order:
    //   1. escalation_initiated_from_chat  (Sprint 10.5 analytics)
    //   2. stage_advanced for stage_5      (Sprint 10.7 resume signal)
    // Both must POST before the dispatch so the events survive the
    // unmount, and the order matters because resume analytics queries
    // assume escalation_initiated_from_chat fires FIRST. The dispatch
    // (replacing the old window.location.href) triggers LandingRouter
    // to re-mount Tier1Workspace; resume-state lands the user at
    // Stage 5 from the stage_advanced row above.
    postJourneyEvent.mockResolvedValue({ ok: true });

    render(<JourneyMessageActions journeySessionId="sess_xyz" />);
    fireEvent.click(screen.getByText(/Escalate to Tier 2/i));

    await waitFor(() => {
      expect(postJourneyEvent).toHaveBeenCalledTimes(2);
    });

    const calls = postJourneyEvent.mock.calls;
    // First call: analytics
    expect(calls[0]).toEqual([
      "sess_xyz", "stage_5", "escalation_initiated_from_chat",
    ]);
    // Second call: resume signal
    expect(calls[1]).toEqual([
      "sess_xyz", "stage_5", "stage_advanced",
    ]);

    // Sprint 11 — dispatch must fire after both POSTs.
    await waitFor(() => {
      expect(mockDispatch).toHaveBeenCalledWith({
        type: "RESUME_JOURNEY",
        payload: { journeySessionId: "sess_xyz" },
      });
    });
  });

  test("escalate_button_does_not_dispatch_when_event_post_fails", async () => {
    // Sprint 11 — if the analytics POST throws, the dispatch must NOT
    // fire (otherwise we'd land the user at Stage 5 without the event
    // recorded, breaking the resume-state contract). The component
    // shows an error toast instead.
    postJourneyEvent.mockRejectedValueOnce(new Error("network"));

    render(<JourneyMessageActions journeySessionId="sess_xyz" />);
    fireEvent.click(screen.getByText(/Escalate to Tier 2/i));

    await waitFor(() => {
      expect(postJourneyEvent).toHaveBeenCalled();
    });
    // Dispatch must not have fired RESUME_JOURNEY.
    const resumeCalls = mockDispatch.mock.calls.filter(
      ([action]) => action && action.type === "RESUME_JOURNEY",
    );
    expect(resumeCalls).toHaveLength(0);
  });
});
