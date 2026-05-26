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
// Top quick-action pill bar — same component the landing page uses,
// surfaced again on the journey page per the spec.
import QuickActionsBar from "../../QuickActionsBar";
// Premium revamp — sidebar state. The journey wants the sidebar
// collapsed by default so the engineer's eye lands on the stages,
// not on chat history. Chat handoff re-expands it; Return-to-Stages
// re-collapses (handled in their respective components).
import { useChat } from "../../../hooks/ChatContext";

// Sprint 10.2 — Stage 0 redesigned (best-ticket distillation) and
// Stage 1A + 1B merged into PivotInsightsPanel. Old standalone
// Stage1a/1b components are no longer mounted; their files stay on
// disk for one cycle as deprecated references.
// Sprint 12.4 — Environment Context lead-in panel added above Stage 0.
// Sprint 12.8 — Preliminary checks header + Preserve-evidence footer
// wrap every journey load (static, data-free, always visible).
import EnvironmentContextPanel from "./EnvironmentContextPanel";
import PreliminaryTier1ChecksHeader from "./PreliminaryTier1ChecksHeader";
import PreserveEvidenceFooter from "./PreserveEvidenceFooter";
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


// Sprint 13.22 — localStorage hydration helpers. ResolutionJourney
// remounts when the engineer opens a Stage 4 KB chat (the chat handoff
// flips the AppLayout mode → ChatArea mounts, journey unmounts) and
// remounts again on return. Without persistence, every remount loses
// `revealed` and the Stage 3 checkbox state — which surfaces as:
//   * Stage 2 panel reappearing even when the engineer never visited
//     it (the slice-based resume hydration did this).
//   * Ticked diagnostic checkboxes resetting to unchecked.
//
// localStorage is keyed by session_id so different cohorts don't
// collide. Failure-open: any read/write error returns/saves nothing
// — the journey degrades to today's pre-13.22 behaviour.
const _journeyStorageKey = (sessionId) =>
  sessionId ? `tier1_journey_state_${sessionId}` : null;

// Sprint 13.32.2 — single-key breadcrumb so AppLayout's "Return to
// Stages" handler (RCAFlow exit) can resume the engineer's most-
// recently-active journey. Written every time ResolutionJourney
// mounts with a valid sessionId. Failure-open.
const _LAST_ACTIVE_JOURNEY_KEY = "tier1_last_active_journey_session";

// Sprint 13.32.5 — RCA button in the Sidebar must only be visible
// while ResolutionJourney is actually rendered (i.e. the engineer
// is on Preliminary Tier 1 Checks / Stages 0-5). Earlier revisions
// gated the button on a localStorage breadcrumb, but that flag
// persisted across navigations and surfaced the button on the
// landing page too. The fix is a RUNTIME signal: we fire one event
// on mount, another on unmount, and the Sidebar listens. No
// persistence — the button reflects the live tree, not history.
const _JOURNEY_MOUNTED_EVENT = "acadia:journey-mounted";
const _JOURNEY_UNMOUNTED_EVENT = "acadia:journey-unmounted";

function _writeLastActiveJourneySid(sessionId) {
  if (!sessionId) return;
  try {
    localStorage.setItem(_LAST_ACTIVE_JOURNEY_KEY, sessionId);
  } catch {
    /* quota / privacy-mode — ignore */
  }
}

function _readJourneyState(sessionId) {
  const key = _journeyStorageKey(sessionId);
  if (!key) return null;
  try {
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : null;
  } catch {
    return null;
  }
}

function _writeJourneyState(sessionId, payload) {
  const key = _journeyStorageKey(sessionId);
  if (!key) return;
  try {
    localStorage.setItem(key, JSON.stringify(payload));
  } catch {
    /* quota exceeded / privacy-mode — degrade silently */
  }
}

function _clearJourneyState(sessionId) {
  const key = _journeyStorageKey(sessionId);
  if (!key) return;
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}


export default function ResolutionJourney({
  sessionId,
  onNewAlert,
  // Premium revamp — quick-action handlers threaded down from
  // AppLayout via LandingRouter → Tier1Workspace. Render the same
  // pill bar that lives on the landing page so RCA / Gap Analysis /
  // Ticket Filter / ServiceNow stay one click away while the engineer
  // is in the journey.
  onOpenRca,
  onOpenGapAnalysis,
  onOpenTicketFilter,
  onOpenServiceNow,
}) {
  // Read chat dispatch up-front; we use it below to drive the
  // sidebar visibility based on the engineer's active stage.
  const { dispatch: chatDispatch } = useChat();
  // Sprint 13.22 — initial state pulls from localStorage so it
  // survives mid-flow remounts (Stage 4 KB chat round-trip is the
  // common case). When localStorage is empty, falls back to the
  // pre-13.22 defaults.
  const _hydrated = _readJourneyState(sessionId);

  // Sprint 10.2 — pivot_insights replaces stage_1a + stage_1b in the
  // revealed map. The merged panel always renders alongside Stage 0.
  // Sprint 10.7 — initial values stay false beyond pivot_insights;
  // the resume-state effect below flips later stages to true when the
  // engineer is mid-journey on remount.
  const [revealed, setRevealed] = useState(() => {
    const defaults = {
      stage_0: true,
      pivot_insights: true,
      stage_2: false,
      stage_3: false,
      stage_4: false,
      stage_5: false,
    };
    if (_hydrated && _hydrated.revealed) {
      // Merge so any new stage keys added in future sprints inherit
      // their default (today's add → tomorrow's hydrate of an old
      // user's state).
      return { ...defaults, ...(_hydrated.revealed || {}) };
    }
    return defaults;
  });
  // Sprint 10.7 §4.2 — true when the engineer arrived directly at
  // Stage 5 (e.g. via chat-Escalate). Stage5EscalationPackage uses
  // this to render the package open immediately instead of behind a
  // toggle. Stays false during normal walked-through journeys.
  const [resumedAtStage5, setResumedAtStage5] = useState(false);

  // ── Sidebar auto-collapse on the journey surface ────────────────
  // The journey is the "Preliminary Tier 1 Checks" screen — a focus
  // surface. The engineer should look at stages, not chat history,
  // so the sidebar tucks itself away when this view mounts.
  // Everywhere else in the app (chat, landing, modals) the sidebar
  // stays at its natural expanded default. The Stage 4 KB handoff
  // and JourneyMessageActions explicitly expand it again when the
  // engineer leaves the journey for chat.
  useEffect(() => {
    // Collapse the sidebar on entry to the blocks screen and enable
    // the hover-peek gate so the engineer can briefly hover the
    // 56 px bar to expand it. The gate is mount-scoped — unmounting
    // (Stage 4 chat handoff, navigate away, return to landing) clears
    // it so the hover-expand never fires on chat or landing screens.
    //
    // The cleanup also restores SET_SIDEBAR(true). The journey is the
    // only screen where the sidebar should be collapsed by default;
    // every other screen (landing, chat, RCA, Gap Analysis, Ticket
    // Filter, ServiceNow) expects the sidebar expanded. Without this
    // restore, navigating away from the journey leaves the sidebar
    // stuck in its collapsed-on-mount state.
    //
    // Stage 4's KB handoff and JourneyMessageActions both explicitly
    // dispatch SET_SIDEBAR(true) themselves before unmounting; the
    // cleanup's restore is idempotent in those paths and serves as a
    // safety net for every other exit (Start new ticket, navigate to
    // landing, escalate, etc.).
    chatDispatch({ type: "SET_SIDEBAR", payload: false });
    chatDispatch({ type: "SET_SIDEBAR_HOVER_PEEK", payload: true });
    return () => {
      chatDispatch({ type: "SET_SIDEBAR_HOVER_PEEK", payload: false });
      chatDispatch({ type: "SET_SIDEBAR", payload: true });
    };
  }, [chatDispatch]);

  const [helpfulPerStage, setHelpfulPerStage] = useState({});
  // Sprint 13.19 — Stage 3 checkbox state lifted here so Stage 5's
  // handoff-note POST can read it. Map of {step_number: bool}.
  // Defaults empty; toggled via the callback passed down to
  // Stage3TroubleshootingApproach.
  // Sprint 13.22 — also hydrated from localStorage so checkboxes
  // survive the chat round-trip remount.
  const [attemptedStage3Steps, setAttemptedStage3Steps] = useState(() => {
    if (_hydrated && _hydrated.attemptedStage3Steps && typeof _hydrated.attemptedStage3Steps === "object") {
      return _hydrated.attemptedStage3Steps;
    }
    return {};
  });
  const toggleAttemptedStep = useCallback((stepNumber) => {
    setAttemptedStage3Steps((prev) => ({
      ...prev,
      [stepNumber]: !prev[stepNumber],
    }));
  }, []);
  const [data, setData] = useState({
    initial: null,
    stage_2: null,
    stage_3: null,
    stage_4: null,
    stage_5: null,
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  // Sprint 13.22 — persist `revealed` + `attemptedStage3Steps`
  // back to localStorage on every change. Cheap (one stringify of
  // a small object on each toggle); guarantees the state survives
  // every unmount/remount cycle without any backend round-trip.
  useEffect(() => {
    if (!sessionId) return;
    _writeJourneyState(sessionId, {
      revealed,
      attemptedStage3Steps,
    });
  }, [sessionId, revealed, attemptedStage3Steps]);

  // Sprint 13.32.2 — breadcrumb the active sessionId so AppLayout's
  // "Return to Stages" handler can RESUME_JOURNEY into this exact
  // journey on RCA exit. Runs once per mount per session.
  useEffect(() => {
    _writeLastActiveJourneySid(sessionId);
  }, [sessionId]);

  // Sprint 13.32.5 — fire mount/unmount events so the Sidebar's
  // RCA button can toggle live (hidden on the landing page,
  // visible while the journey is mounted). Gated on a valid
  // sessionId so we don't false-positive during the initial render
  // before the journey has anything to work with. Cleanup fires
  // the unmount event on every dismount path (Stage 4 chat handoff,
  // RCA flow takeover, New Ticket reset, route change).
  useEffect(() => {
    if (!sessionId) return undefined;
    try {
      if (typeof window !== "undefined") {
        window.dispatchEvent(new Event(_JOURNEY_MOUNTED_EVENT));
      }
    } catch { /* unsupported env */ }
    return () => {
      try {
        if (typeof window !== "undefined") {
          window.dispatchEvent(new Event(_JOURNEY_UNMOUNTED_EVENT));
        }
      } catch { /* unsupported env */ }
    };
  }, [sessionId]);

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

        // Sprint 13.22 — resume-state hydration tightened. The prior
        // implementation called STAGE_ORDER.slice(2, resumedIdx+1) and
        // revealed every stage in that range. That made false claims
        // about the engineer's path: a journey of "Stage 0 → Stage 3
        // → Stage 4" would re-mount with Stage 2 revealed too, even
        // though the engineer never opened the Related Incidents
        // panel. With localStorage-backed `revealed` from the hydration
        // above, we already have the engineer's exact reveal map. The
        // resume-state call now serves as a fallback for the very
        // first mount (no localStorage yet) AND only flips the
        // resumed stage itself — not the chain in between.
        const resumedStage = resume?.current_stage || "stage_0";
        const resumedIdx = STAGE_ORDER.indexOf(resumedStage);
        const _hadHydrated = !!(
          _hydrated && _hydrated.revealed && Object.keys(_hydrated.revealed).length
        );

        // Sprint 13.22 — when localStorage hydrated `revealed`, the
        // panels for those stages will mount immediately but their
        // data is in-memory only (not persisted). Prefetch payloads
        // for every revealed gated stage so panels don't render with
        // null data after a remount.
        if (_hadHydrated) {
          const hydratedStages = ["stage_2", "stage_3", "stage_4", "stage_5"]
            .filter((s) => _hydrated.revealed[s]);
          if (hydratedStages.length > 0) {
            await Promise.all(
              hydratedStages.map((stage) => {
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
                    console.warn(
                      "[journey.hydrate] prefetch failed", stage, err,
                    );
                  });
              }),
            );
          }
          if (cancelled) return;
          if (_hydrated.revealed.stage_5) {
            setResumedAtStage5(true);
          }
        }

        // Sprint 13.23 — Resume-state's `current_stage` ALWAYS wins
        // on top of localStorage. Reason: when the engineer
        // escalates from the SOP/KB chat (JourneyMessageActions →
        // postJourneyEvent stage_advanced stage_5 → RESUME_JOURNEY
        // dispatch), the journey remounts and we need to land on
        // Stage 5 even though localStorage's last save (taken before
        // the chat unmount) had revealed.stage_5=false.
        //
        // localStorage still preserves PAST visits (Stage 2 stays
        // hidden if never visited) — server-side stage_advanced is
        // authoritative only for FORWARD progress. The merge logic
        // is: hydrate from localStorage, then ALSO apply the server's
        // current_stage on top. This prevents the bug where chat-side
        // escalation came back without rendering Stage 5.
        if (resumedIdx > 1) {
          const alreadyRevealed = !!revealed[resumedStage];
          if (!alreadyRevealed) {
            // Prefetch payload for the resumed stage (the
            // localStorage-driven prefetch loop above only handles
            // hydrated stages; this case covers chat-side advance
            // that localStorage didn't yet know about).
            const fetcher = {
              stage_2: fetchStage2,
              stage_3: fetchStage3,
              stage_4: fetchStage4,
              stage_5: fetchStage5,
            }[resumedStage];
            if (fetcher && !data[resumedStage]) {
              try {
                const payload = await fetcher(sessionId);
                if (!cancelled) {
                  setData((prev) => ({ ...prev, [resumedStage]: payload }));
                }
              } catch (err) {
                // eslint-disable-next-line no-console
                console.warn(
                  "[journey.resume] prefetch failed", resumedStage, err,
                );
              }
            }
            if (cancelled) return;
            setRevealed((prev) => ({ ...prev, [resumedStage]: true }));
          }
          if (resumedStage === "stage_5") {
            setResumedAtStage5(true);
          }
        }

        // Sprint 10.2 — fire stage_rendered telemetry for the merged
        // pivot_insights panel (replacing the separate stage_1a /
        // stage_1b events).
        // Sprint 12.4 — environment_context joins the eager-paint set.
        for (const stage of ["environment_context", "stage_0", "pivot_insights"]) {
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
      if (!fetcher) {
        // Sprint 12.4 — Environment-Context's NextStageButton points
        // at "stage_0", which is auto-revealed and ships in /initial
        // (no separate fetcher). The click is still meaningful: it
        // records the engineer's "I read the env profile, advancing"
        // intent so /resume-state can place them at Stage 0 on the
        // next remount. Fire stage_advanced telemetry and exit.
        if (toStage === "stage_0") {
          postJourneyEvent(sessionId, "stage_0", "stage_advanced").catch(() => {});
        }
        return;
      }

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
    // Sprint 13.22 — wipe the persisted journey state for this
    // session so the next ticket starts with a clean reveal map +
    // empty checkbox state. Stale entries for old sessions that
    // never got cleared remain on disk; they're tiny (a few KB
    // tops) and harmless until the session id is reused.
    _clearJourneyState(sessionId);
    if (typeof onNewAlert === "function") {
      onNewAlert();
    }
  }, [onNewAlert]);

  // ── Loading skeleton on first paint ──
  if (loading) {
    return (
      <div className="flex-1 overflow-y-auto py-6 t-bg-primary" style={{ paddingLeft: 8, paddingRight: 8 }}>
        <div className="w-full mx-auto" style={{ maxWidth: "100%" }}>
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
      <div className="flex-1 overflow-y-auto py-6 t-bg-primary" style={{ paddingLeft: 8, paddingRight: 8 }}>
        <div className="w-full mx-auto" style={{ maxWidth: "100%" }}>
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
    <div className="flex-1 overflow-y-auto py-6 t-bg-primary" style={{ paddingLeft: 8, paddingRight: 8 }}>
      {/* Premium revamp — page-level Acadia watermark removed.
          Each stage card now carries its OWN watermark
          (CardWatermark inside Stage 2 / 3 / 4 / 5) so the brand
          presence is per-block rather than behind everything at
          once. The page background stays clean. */}

      {/* Premium revamp — journey grid stretches to nearly the full
          viewport width. The user explicitly asked for minimal side
          gutters; we drop the max-width cap entirely (full width)
          and leave only an 8 px page gutter on each side. */}
      <div className="w-full mx-auto" style={{ maxWidth: "100%" }}>
        {/* Premium revamp — top quick-action pill bar. Mirrors the
            landing page bar so RCA / Gap Analysis / Ticket Filter /
            Connect to ServiceNow stay one click away while the
            engineer is in the journey (including the Best Historical
            Match & Recommended Resolution panel). Each pill is
            wired through AppLayout → LandingRouter → Tier1Workspace;
            handlers are optional so legacy callers degrade silently. */}
        <QuickActionsBar
          onOpenRca={onOpenRca}
          onOpenGapAnalysis={onOpenGapAnalysis}
          onOpenTicketFilter={onOpenTicketFilter}
          onOpenServiceNow={onOpenServiceNow}
          marginBottom={20}
        />

        {/* Sprint 12.8 — Preliminary Tier 1 Checks header. Static
            informational card; always rendered at the very top,
            ahead of every cohort-derived panel. Engineer's pre-flight
            checklist (Define Impact / Validate via Change Logs /
            Verify Basic Connectivity / Monitor Resource & Service
            Health). No data dependency, no API call. */}
        <PreliminaryTier1ChecksHeader />

        {/* Sprint 13.5 — Environment Context & Tech Component Profile
            panel suppressed at the request of the user. The backend
            still computes `initial.environment_profile`; only the
            frontend render is commented. Reinstate by un-commenting
            the JSX block below; no other change required. */}
        {/*
        <EnvironmentContextPanel
          data={initial.environment_profile}
          sessionId={sessionId}
          onMarkedHelpful={markHelpful}
          onStartNewTicket={startNewTicket}
          onReveal={reveal}
          helpfulMarked={!!helpfulPerStage.environment_context}
        />
        */}

        {/* Sprint 10.2 — Stage 0 best-ticket distillation (replaces
            ConfidenceLead). Sprint 11 — sessionId now passed so the
            "How they did it" steps can offer per-step "Ask in chat"
            links via useChatHandoff. */}
        <Stage0BestTicketDistillation
          data={initial.stage_0}
          sessionId={sessionId}
          onReveal={reveal}
          onMarkedHelpful={markHelpful}
          onStartNewTicket={startNewTicket}
          helpfulMarked={!!helpfulPerStage.stage_0}
        />

        {/* Sprint 13.11 — entire PivotInsightsPanel mount suppressed
            at the user's request. The panel's body (Smoking Gun +
            What NOT to chase) was already commented inside the
            component; with the engineer also waiving the panel's
            footer (Helpful / Dislike / Escalate / Related-Incidents
            next-stage button), there's nothing left to render — so
            the whole mount is taken out. Backend `/initial` still
            ships `pivot_insights` payload (smoking_gun + do_not_chase);
            only the frontend mount is commented. Reinstate by
            un-commenting the JSX block below if the panel is ever
            wanted again. Forward navigation to Stage 2 is now
            unreachable from this surface — Stage 0's footer
            shortcut to Stage 3 ("Guided Troubleshooting Workflow")
            and the Escalate-to-Tier-2 path remain. */}
        {/*
        <PivotInsightsPanel
          data={initial.pivot_insights}
          sessionId={sessionId}
          onMarkedHelpful={markHelpful}
          onStartNewTicket={startNewTicket}
          onReveal={reveal}
          helpfulMarked={!!helpfulPerStage.pivot_insights}
        />
        */}

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
            attemptedStepsByStep3={attemptedStage3Steps}
            onToggleAttemptedStep={toggleAttemptedStep}
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
            attemptedStage3Steps={attemptedStage3Steps}
          />
        ) : null}

        {/* Sprint 12.8 — Preserve Evidence & Escalate footer. Static
            instruction; always rendered at the very bottom of the
            journey, regardless of which stages have been revealed.
            Pairs with PreliminaryTier1ChecksHeader at the top. */}
        <PreserveEvidenceFooter />
      </div>
    </div>
  );
}
