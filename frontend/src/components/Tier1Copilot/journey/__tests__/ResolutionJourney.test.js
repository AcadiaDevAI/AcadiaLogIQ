// Sprint 10.7 §4.5 — frontend tests for the journey resume flow.
//
// Setup note: this codebase has no other frontend tests yet. To run
// these you need React Testing Library installed alongside the Jest
// harness that already ships with react-scripts:
//   cd frontend && npm i -D @testing-library/react @testing-library/jest-dom
// The Sprint 10.5 spec flagged the missing harness as a future-sprint
// gap; this file is structured so it works the moment that gap closes.

import React from "react";
import { render, screen, waitFor } from "@testing-library/react";

// Mock the entire journeyApi module so the resume / fetch calls are
// driven from the test fixtures rather than real network requests.
jest.mock(
  "../journeyApi",
  () => ({
    fetchInitial: jest.fn(),
    fetchStage2: jest.fn(),
    fetchStage3: jest.fn(),
    fetchStage4: jest.fn(),
    fetchStage5: jest.fn(),
    getResumeState: jest.fn(),
    postJourneyEvent: jest.fn(() => Promise.resolve({ ok: true })),
  }),
);

import * as journeyApi from "../journeyApi";
import ResolutionJourney from "../ResolutionJourney";
import Stage5EscalationPackage from "../Stage5EscalationPackage";


function _initialPayload() {
  return {
    session_id: "sess_test",
    stage_0: { sparse: true, profile_match: null },
    pivot_insights: {
      smoking_gun: { empty: true, derived_from: "empty" },
      do_not_chase: { empty: true, entries: [], reason: "no_data" },
    },
  };
}


beforeEach(() => {
  Object.values(journeyApi).forEach((fn) => {
    if (typeof fn === "function" && fn.mockReset) fn.mockReset();
  });
  journeyApi.postJourneyEvent.mockResolvedValue({ ok: true });
});


describe("ResolutionJourney resume", () => {
  test("test_journey_resumes_at_stage_returned_by_api", async () => {
    // Engineer was at stage_3 → resume mounts Stages 0..3, hides 4+5.
    journeyApi.fetchInitial.mockResolvedValue(_initialPayload());
    journeyApi.getResumeState.mockResolvedValue({
      session_id: "sess_test",
      current_stage: "stage_3",
      last_event_at: "2026-04-30T12:00:00Z",
    });
    journeyApi.fetchStage2.mockResolvedValue({ matches: [] });
    journeyApi.fetchStage3.mockResolvedValue({ steps: [] });

    render(<ResolutionJourney sessionId="sess_test" />);

    // Wait for both /initial and /resume-state to settle.
    await waitFor(() => {
      expect(journeyApi.fetchInitial).toHaveBeenCalledWith("sess_test");
      expect(journeyApi.getResumeState).toHaveBeenCalledWith("sess_test");
    });

    // Stages up to and including stage_3 are mounted: their fetchers
    // have been called.
    await waitFor(() => {
      expect(journeyApi.fetchStage2).toHaveBeenCalledTimes(1);
      expect(journeyApi.fetchStage3).toHaveBeenCalledTimes(1);
    });
    // Stages beyond resumedStage are NOT mounted on initial load.
    expect(journeyApi.fetchStage4).not.toHaveBeenCalled();
    expect(journeyApi.fetchStage5).not.toHaveBeenCalled();
  });

  test("test_journey_falls_back_to_stage_0_on_resume_error", async () => {
    // resume-state throws → journey paints only Stage 0 +
    // pivot_insights, no later stage fetchers fire.
    journeyApi.fetchInitial.mockResolvedValue(_initialPayload());
    journeyApi.getResumeState.mockRejectedValue(new Error("Network down"));

    render(<ResolutionJourney sessionId="sess_test" />);

    await waitFor(() => {
      expect(journeyApi.fetchInitial).toHaveBeenCalled();
    });

    // No advance fetchers called — graceful degrade to pre-10.7
    // stage_0-only first paint.
    expect(journeyApi.fetchStage2).not.toHaveBeenCalled();
    expect(journeyApi.fetchStage3).not.toHaveBeenCalled();
    expect(journeyApi.fetchStage4).not.toHaveBeenCalled();
    expect(journeyApi.fetchStage5).not.toHaveBeenCalled();
  });
});


describe("Stage5EscalationPackage auto-expand", () => {
  test("test_stage_5_auto_expands_when_resumed_directly", () => {
    // autoExpand={true} → EscalationPackageCard rendered, no
    // "Show escalation package" toggle button.
    const fakePackage = {
      what_was_tried: ["step a", "step b"],
      summary: "Test escalation summary text.",
    };

    render(
      <Stage5EscalationPackage
        data={fakePackage}
        sessionId="sess_test"
        onMarkedHelpful={() => {}}
        helpfulMarked={false}
        autoExpand={true}
      />,
    );

    // The "Show escalation package" toggle should NOT be present —
    // the package is already expanded.
    expect(screen.queryByText(/Show escalation package/i)).toBeNull();
    // Title is always there as a smoke check.
    expect(screen.getByText(/Operational Handoff/i)).toBeInTheDocument();
  });

  test("test_stage_5_collapsed_when_autoExpand_false", () => {
    // autoExpand={false} (explicit opt-out) → toggle button visible,
    // package text hidden until clicked.
    const fakePackage = {
      what_was_tried: ["step a"],
      summary: "Hidden summary",
    };

    render(
      <Stage5EscalationPackage
        data={fakePackage}
        sessionId="sess_test"
        onMarkedHelpful={() => {}}
        helpfulMarked={false}
        autoExpand={false}
      />,
    );

    expect(screen.getByText(/Show escalation package/i)).toBeInTheDocument();
  });
});
