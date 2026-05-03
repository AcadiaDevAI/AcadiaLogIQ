// Sprint 10 — Resolution Journey orchestrator (parent state machine).
//
// Per spec §10:
//   - On mount: GET /initial. Render Stage 0 + 1A + 1B as soon as it
//     lands. SkeletonCard during load.
//   - Next-stage button click → GET the corresponding endpoint, store
//     in `data`, flip `revealed[stage_n] = true`. Server-side telemetry
//     fires from inside NextStageButton (no double-firing here).
//   - Once a later stage is revealed, earlier stages remain visible
//     above (scrollable).
//   - Helpful clicks NEVER advance the flow (telemetry only).
//   - "New Ticket" resets state and calls the parent's onNewAlert.

import React, { useCallback, useEffect, useState } from "react";
import { Alert, Spin } from "antd";

import SkeletonCard from "../SkeletonCard";

// Sprint 10.2 — Stage 0 redesigned (best-ticket distillation) and
// Stage 1A + 1B merged into PivotInsightsPanel. Old standalone
// Stage1a/1b components are no longer mounted; their files stay on
// disk for one cycle as deprecated references.
import Stage0BestTicketDistillation from "./Stage0BestTicketDistillation";
import PivotInsightsPanel from "./PivotInsightsPanel";
import Stage2HistoricalMatches from "./Stage2HistoricalMatches";
import Stage3TroubleshootingApproach from "./Stage3TroubleshootingApproach";
import Stage4SearchKBHandoff from "./Stage4SearchKBHandoff";
import Stage5EscalationPackage from "./Stage5EscalationPackage";

import {
  fetchInitial,
  fetchStage2,
  fetchStage3,
  fetchStage4,
  fetchStage5,
  getResumeState,
  postJourneyEvent,
} from "./journeyApi";


// Sprint 10.7 §4.1 — stage progression order. Index in this list maps
// directly to "how far the engineer has walked." resume-state returns
// one of these values, and we mount every stage up to and including
// it. Keep in sync with backend ResumeStateResponse.current_stage.
const STAGE_ORDER = [
  "stage_0",
  "pivot_insights",
  "stage_2",
  "stage_3",
  "stage_4",
  "stage_5",
];


export default function ResolutionJourney({ sessionId, onNewAlert }) {
  // Sprint 10.2 — pivot_insights replaces stage_1a + stage_1b in the
  // revealed map. The merged panel always renders alongside Stage 0.
  // Sprint 10.7 — initial values stay false beyond pivot_insights;
  // the resume-state effect below flips later stages to true when the
  // engineer is mid-journey on remount.
  const [revealed, setRevealed] = useState({
    stage_0: true,
    pivot_insights: true,
    stage_2: false,
    stage_3: false,
    stage_4: false,
    stage_5: false,
  });
  // Sprint 10.7 §4.2 — true when the engineer arrived directly at
  // Stage 5 (e.g. via chat-Escalate). Stage5EscalationPackage uses
  // this to render the package open immediately instead of behind a
  // toggle. Stays false during normal walked-through journeys.
  const [resumedAtStage5, setResumedAtStage5] = useState(false);

  const [helpfulPerStage, setHelpfulPerStage] = useState({});
  const [data, setData] = useState({
    initial: null,
    stage_2: null,
    stage_3: null,
    stage_4: null,
    stage_5: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // ── Initial fetch + Sprint 10.7 resume state ──
  // We run BOTH calls in parallel: /initial paints Stage 0 +
  // pivot_insights, /resume-state tells us which later stages to
  // mount immediately. Both must complete before we lift the loading
  // skeleton — partial paint would briefly show Stage 0 alone before
  // the resume index expanded the tree, causing a visible flicker on
  // every remount of an in-progress journey.
  useEffect(() => {
    if (!sessionId) {
      setError("missing-session");
      setLoading(false);
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        // Sprint 10.7 §4.1 — resume-state runs alongside /initial.
        // Catch-and-default so a resume-state failure never blocks
        // the journey from rendering; the engineer just starts at
        // Stage 0 (the pre-10.7 default).
        const [initial, resume] = await Promise.all([
          fetchInitial(sessionId),
          getResumeState(sessionId).catch(() => ({ current_stage: "stage_0" })),
        ]);
        if (cancelled) return;
        setData((prev) => ({ ...prev, initial }));

        // Stage 2-5 reveals based on the resumed stage. We only flip
        // booleans here — the per-stage payload fetches happen as
        // each panel is asked for via `reveal()` in the existing
        // forward-flow path. For stages already passed, we trigger
        // the same fetcher inline below.
        const resumedStage = resume?.current_stage || "stage_0";
        const resumedIdx = STAGE_ORDER.indexOf(resumedStage);
        if (resumedIdx > 1) {
          // Stages 2..N need their data prefetched so the panels
          // render with content (not empty cards). Run in parallel.
          const stagesToFetch = STAGE_ORDER.slice(2, resumedIdx + 1);
          await Promise.all(
            stagesToFetch.map((stage) => {
              const fetcher = {
                stage_2: fetchStage2,
                stage_3: fetchStage3,
                stage_4: fetchStage4,
                stage_5: fetchStage5,
              }[stage];
              if (!fetcher) return Promise.resolve();
              return fetcher(sessionId)
                .then((payload) => {
                  if (cancelled) return;
                  setData((prev) => ({ ...prev, [stage]: payload }));
                })
                .catch((err) => {
                  // eslint-disable-next-line no-console
                  console.warn("[journey.resume] prefetch failed", stage, err);
                });
            }),
          );
          if (cancelled) return;
          // Flip every revealed[stage] for stages 2..resumedIdx.
          setRevealed((prev) => {
            const next = { ...prev };
            for (const s of stagesToFetch) {
              next[s] = true;
            }
            return next;
          });
          if (resumedStage === "stage_5") {
            setResumedAtStage5(true);
          }
        }

        // Sprint 10.2 — fire stage_rendered telemetry for the merged
        // pivot_insights panel (replacing the separate stage_1a /
        // stage_1b events).
        for (const stage of ["stage_0", "pivot_insights"]) {
          postJourneyEvent(sessionId, stage, "stage_rendered").catch(() => {});
        }
      } catch (err) {
        if (!cancelled) setError(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => { cancelled = true; };
  }, [sessionId]);

  // ── Helpful click handler — purely advisory ──
  const markHelpful = useCallback((stage) => {
    setHelpfulPerStage((prev) => ({ ...prev, [stage]: true }));
  }, []);

  // ── Reveal handler — fetch the target stage's payload, store, flip ──
  const reveal = useCallback(
    async (toStage) => {
      const fetcher = {
        stage_2: fetchStage2,
        stage_3: fetchStage3,
        stage_4: fetchStage4,
        stage_5: fetchStage5,
      }[toStage];
      if (!fetcher) return;

      try {
        const payload = await fetcher(sessionId);
        setData((prev) => ({ ...prev, [toStage]: payload }));
        setRevealed((prev) => ({ ...prev, [toStage]: true }));
        postJourneyEvent(sessionId, toStage, "stage_rendered").catch(() => {});
        // Sprint 10.7 §3 — stage_advanced is the resume signal. Fire
        // it on every forward reveal so /resume-state has a row to
        // read on the next remount (chat-handoff round-trip, browser
        // refresh, etc.). stage_rendered is too noisy — it fires for
        // pivot_insights on initial load, which would falsely advance
        // a fresh-Stage-0 engineer past Stage 0 on remount.
        postJourneyEvent(sessionId, toStage, "stage_advanced").catch(() => {});
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error("[journey.reveal] fetch failed", toStage, err);
        // Still flip revealed so the user sees an empty state — better
        // than a phantom "loading" forever.
        setRevealed((prev) => ({ ...prev, [toStage]: true }));
      }
    },
    [sessionId],
  );

  const startNewTicket = useCallback(() => {
    if (typeof onNewAlert === "function") {
      onNewAlert();
    }
  }, [onNewAlert]);

  // ── Loading skeleton on first paint ──
  if (loading) {
    return (
      <div className="flex-1 overflow-y-auto px-4 py-6 t-bg-primary">
        <div className="w-full max-w-6xl mx-auto">
          <div style={{ textAlign: "center", padding: 24 }}>
            <Spin />
            <span style={{ marginLeft: 12 }}>Assembling your resolution journey…</span>
          </div>
          <SkeletonCard variant="answer" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 overflow-y-auto px-4 py-6 t-bg-primary">
        <div className="w-full max-w-6xl mx-auto">
          <Alert
            type="error"
            showIcon
            message="Could not load the Resolution Journey"
            description="The /initial fetch failed. Check that LOGIQ_TIER1_JOURNEY_BACKEND is true and the session_id is valid."
          />
        </div>
      </div>
    );
  }

  const initial = data.initial;
  if (!initial) return null;

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 t-bg-primary">
      <div className="w-full max-w-6xl mx-auto">
        {/* Sprint 10.2 — Stage 0 best-ticket distillation (replaces
            ConfidenceLead). Sprint 11 — sessionId now passed so the
            "How they did it" steps can offer per-step "Ask in chat"
            links via useChatHandoff. */}
        <Stage0BestTicketDistillation
          data={initial.stage_0}
          sessionId={sessionId}
          onReveal={reveal}
        />

        {/* Sprint 10.2 — merged Pivot Insights panel (Smoking Gun +
            Do Not Chase combined). Single Helpful + single
            "Historical Matches" next-stage button. */}
        <PivotInsightsPanel
          data={initial.pivot_insights}
          sessionId={sessionId}
          onMarkedHelpful={markHelpful}
          onStartNewTicket={startNewTicket}
          onReveal={reveal}
          helpfulMarked={!!helpfulPerStage.pivot_insights}
        />

        {/* Stage 2 — revealed on click */}
        {revealed.stage_2 ? (
          <Stage2HistoricalMatches
            data={data.stage_2}
            sessionId={sessionId}
            onMarkedHelpful={markHelpful}
            onStartNewTicket={startNewTicket}
            onReveal={reveal}
            helpfulMarked={!!helpfulPerStage.stage_2}
          />
        ) : null}

        {/* Stage 3 — revealed on click */}
        {revealed.stage_3 ? (
          <Stage3TroubleshootingApproach
            data={data.stage_3}
            sessionId={sessionId}
            onMarkedHelpful={markHelpful}
            onStartNewTicket={startNewTicket}
            onReveal={reveal}
            helpfulMarked={!!helpfulPerStage.stage_3}
          />
        ) : null}

        {/* Stage 4 — revealed on click */}
        {revealed.stage_4 ? (
          <Stage4SearchKBHandoff
            data={data.stage_4}
            sessionId={sessionId}
            onMarkedHelpful={markHelpful}
            onStartNewTicket={startNewTicket}
            onReveal={reveal}
            helpfulMarked={!!helpfulPerStage.stage_4}
          />
        ) : null}

        {/* Stage 5 — revealed on click. Sprint 10.7 §4.2 — autoExpand
            is true when the engineer arrived via chat-Escalate so the
            package shows expanded immediately; false on a normal
            walked-through journey so the engineer can scroll past
            other stages without the package elbowing in. */}
        {revealed.stage_5 ? (
          <Stage5EscalationPackage
            data={data.stage_5}
            sessionId={sessionId}
            onMarkedHelpful={markHelpful}
            onStartNewTicket={startNewTicket}
            helpfulMarked={!!helpfulPerStage.stage_5}
            autoExpand={resumedAtStage5}
          />
        ) : null}
      </div>
    </div>
  );
}
