import React, { useCallback, useEffect, useRef, useState } from "react";
import { ConfigProvider, theme } from "antd";
import { ThemeProvider, useTheme } from "./hooks/ThemeContext";
// Premium theme stylesheet — selectors are scoped to `.theme-premium`,
// so it has zero effect under the legacy theme. Imported once at app
// root so the cascade is available when ACTIVE_THEME = "premium".
import "./theme/premium.css";
// Premium DARK companion — selectors are scoped to
// `.theme-premium.theme-premium-dark`, so it only paints when the
// premium theme is active AND the user has toggled to dark. Light
// premium is untouched.
import "./theme/premium-dark.css";
import { resolveAntdTheme } from "./theme/theme-config";
import { ChatProvider, useChat } from "./hooks/ChatContext";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import MobileHeader from "./components/MobileHeader";
import AuthGate from "./components/AuthGate";
import LandingRouter from "./components/LandingRouter";
import RCAFlow from "./components/RCA/RCAFlow";
import RCAEntryModal from "./components/RCA/RCAEntryModal";
import GapAnalysisFlow from "./components/GapAnalysis/GapAnalysisFlow";
import GapAnalysisEntryModal from "./components/GapAnalysis/GapAnalysisEntryModal";
import TicketFilterFlow from "./components/TicketFilter/TicketFilterFlow";
import ServiceNowFlow from "./components/ServiceNow/ServiceNowFlow";

function BuildStamp() {
  return (
    <div
      style={{
        position: "fixed",
        bottom: "8px",
        right: "12px",
        fontSize: "10px",
        color: "#888",
        opacity: 0.5,
        pointerEvents: "none",
        zIndex: 9999,
        fontFamily: "monospace",
      }}
    >
      build {process.env.REACT_APP_BUILD_TIMESTAMP || "dev"}
    </div>
  );
}

function AppLayout() {
  const { state, dispatch } = useChat();

  // Sprint 13.32 — RCA mode flag. Local to AppLayout so we don't
  // pollute ChatContext for a feature that doesn't touch chat.
  //
  // Sprint 13.32.6 — flow split into two stages:
  //   1. Sidebar RCA click → `rcaModalOpen=true` → RCAEntryModal
  //      collects {incidentNumber, panels, file}.
  //   2. Modal submit → `rcaPayload` set + `rcaOpen=true` →
  //      RCAFlow takes over the right pane and auto-runs the
  //      request using `initialPayload`.
  //   3. RCAFlow "Return to Stages" → `rcaOpen=false` and
  //      RESUME_JOURNEY dispatch (existing logic untouched).
  const [rcaModalOpen, setRcaModalOpen] = useState(false);
  const [rcaOpen, setRcaOpen] = useState(false);
  const [rcaPayload, setRcaPayload] = useState(null);

  const handleRcaModalSubmit = useCallback((payload) => {
    // Close every other right-pane flow so the render chain
    // shows the freshly-opened RCA pane (and so a stale gapOpen /
    // ticketFilterOpen / serviceNowOpen doesn't reappear when RCA returns).
    setGapOpen(false);
    setGapPayload(null);
    setTicketFilterOpen(false);
    setServiceNowOpen(false);
    setRcaPayload(payload);
    setRcaOpen(true);
    setRcaModalOpen(false);
  }, []);

  // ── Gap Analysis — independent state mirror of RCA's. ──────────
  // Lives next to the RCA flags so the two features share AppLayout's
  // "right pane takeover" semantics, but the flow components, payloads,
  // and submission handlers are fully separate.
  //
  //   1. Sidebar "Gap Analysis" click → `gapModalOpen=true`.
  //   2. Modal submit → `gapPayload` set + `gapOpen=true`. The right
  //      pane flips to GapAnalysisFlow which auto-runs the request.
  //   3. "Return" → `gapOpen=false` and (same RESUME_JOURNEY signal
  //      RCA uses) so the engineer lands back on their journey.
  const [gapModalOpen, setGapModalOpen] = useState(false);
  const [gapOpen, setGapOpen] = useState(false);
  const [gapPayload, setGapPayload] = useState(null);

  // Ticket Filter — independent right-pane feature, same mount
  // mechanics as RCA / Gap but with no modal (the dropdowns live
  // inside the flow component itself).
  const [ticketFilterOpen, setTicketFilterOpen] = useState(false);

  // ServiceNow — independent optional integration. Same mount
  // mechanics as Ticket Filter. The flow auto-fetches on mount,
  // so no modal / payload state is needed at this layer.
  const [serviceNowOpen, setServiceNowOpen] = useState(false);

  const handleGapModalSubmit = useCallback((payload) => {
    // Close every other right-pane flow so the render chain shows
    // the freshly-opened Gap Analysis pane.
    setRcaOpen(false);
    setRcaPayload(null);
    setTicketFilterOpen(false);
    setServiceNowOpen(false);
    setGapPayload(payload);
    setGapOpen(true);
    setGapModalOpen(false);
  }, []);

  // ── Back-arrow origin tracking ─────────────────────────────────
  // Ticket Filter and ServiceNow flows are right-pane takeovers that
  // can be entered from three different screens — the blocks/journey,
  // chat, or landing. The icon-only back-arrow at the top-left of
  // those flows is supposed to return the engineer to *where they
  // came from*, not unconditionally to landing.
  //
  // We snapshot the "origin" at the moment the takeover opens, then
  // restore it on close:
  //
  //   journey  → re-dispatch RESUME_JOURNEY with the captured sid so
  //              LandingRouter's resume effect re-mounts Tier1Workspace
  //              (LandingRouter's local tier1Result is destroyed on
  //              unmount, so we cannot rely on natural restoration —
  //              the explicit resume signal is required).
  //   chat     → leave selectedMode untouched; AppLayout re-renders
  //              ChatArea naturally.
  //   landing  → fall back to the old RESET_MODE_STATE +
  //              CLEAR_JOURNEY_RESUME behaviour so any stale journey
  //              breadcrumb doesn't auto-resume a session the engineer
  //              never visited.
  //
  // Journey-active detection uses the existing window events
  // (`acadia:journey-mounted` / `acadia:journey-unmounted`) that
  // ResolutionJourney already broadcasts on mount/unmount. The
  // localStorage key written by the same component supplies the
  // session id needed for RESUME_JOURNEY.
  const journeyMountedRef = useRef(false);
  useEffect(() => {
    const onMount = () => { journeyMountedRef.current = true; };
    const onUnmount = () => { journeyMountedRef.current = false; };
    window.addEventListener("acadia:journey-mounted", onMount);
    window.addEventListener("acadia:journey-unmounted", onUnmount);
    return () => {
      window.removeEventListener("acadia:journey-mounted", onMount);
      window.removeEventListener("acadia:journey-unmounted", onUnmount);
    };
  }, []);

  // Mirror state.selectedMode into a ref so captureOrigin can read the
  // freshest value without forcing the open handlers to re-create on
  // every render.
  const selectedModeRef = useRef(state.selectedMode);
  useEffect(() => {
    selectedModeRef.current = state.selectedMode;
  }, [state.selectedMode]);

  const prevOriginRef = useRef(null);

  const captureOrigin = useCallback(() => {
    if (journeyMountedRef.current) {
      let journeySid = null;
      try {
        journeySid = localStorage.getItem("tier1_last_active_journey_session");
      } catch { /* privacy-mode — degrade silently */ }
      return { type: "journey", journeySid: journeySid || null };
    }
    if (selectedModeRef.current) {
      return { type: "chat" };
    }
    return { type: "landing" };
  }, []);

  const restoreOrigin = useCallback(() => {
    const origin = prevOriginRef.current;
    prevOriginRef.current = null;
    if (origin?.type === "journey" && origin.journeySid) {
      // Re-fire the resume signal so LandingRouter's effect picks it
      // up and re-mounts Tier1Workspace at the same session id.
      dispatch({
        type: "RESUME_JOURNEY",
        payload: { journeySessionId: origin.journeySid },
      });
      return;
    }
    if (origin?.type === "chat") {
      // No dispatch needed — selectedMode was never cleared, so
      // AppLayout will re-render ChatArea once the takeover flag
      // flips off.
      return;
    }
    // origin?.type === "landing" or unknown — old behaviour.
    dispatch({ type: "RESET_MODE_STATE" });
    dispatch({ type: "CLEAR_JOURNEY_RESUME" });
  }, [dispatch]);

  // Sidebar's "Ticket Filter" button — no modal, so this is the
  // only handler the feature needs. Closes RCA + Gap on open
  // (symmetric with their handlers above).
  const handleOpenTicketFilter = useCallback(() => {
    prevOriginRef.current = captureOrigin();
    setRcaOpen(false);
    setRcaPayload(null);
    setGapOpen(false);
    setGapPayload(null);
    setServiceNowOpen(false);
    setTicketFilterOpen(true);
  }, [captureOrigin]);

  // Back-arrow / "Return to Stages" from Ticket Filter — restores
  // whichever screen the engineer came from instead of unconditionally
  // dropping them on landing.
  const handleReturnFromTicketFilter = useCallback(() => {
    restoreOrigin();
    setTicketFilterOpen(false);
  }, [restoreOrigin]);

  // Sidebar's "Connect to ServiceNow" button — no modal needed since
  // the flow auto-fetches on mount. Closes every other right-pane
  // takeover, identical pattern to handleOpenTicketFilter above.
  const handleOpenServiceNow = useCallback(() => {
    prevOriginRef.current = captureOrigin();
    setRcaOpen(false);
    setRcaPayload(null);
    setGapOpen(false);
    setGapPayload(null);
    setTicketFilterOpen(false);
    setServiceNowOpen(true);
  }, [captureOrigin]);

  // Back-arrow / "Return to Stages" from ServiceNow — restores the
  // origin screen via the shared restoreOrigin helper.
  const handleReturnFromServiceNow = useCallback(() => {
    restoreOrigin();
    setServiceNowOpen(false);
  }, [restoreOrigin]);

  const handleReturnFromGapAnalysis = useCallback(() => {
    // Mirror RCA's "Return to Stages" — drop the user on the
    // landing intake form (Proactive / Reactive picker). Reset all
    // mode state and clear any stale journey-resume signal so
    // LandingRouter renders Tier1IntakeForm fresh rather than
    // auto-mounting a previous Tier1Workspace.
    dispatch({ type: "RESET_MODE_STATE" });
    dispatch({ type: "CLEAR_JOURNEY_RESUME" });
    setGapOpen(false);
  }, [dispatch]);

  // "Return to Stages" UX — land the engineer on the Tier-1 intake
  // form (the Proactive / Reactive picker), regardless of whether
  // they were in an active journey before opening RCA. Clearing
  // selectedMode causes AppLayout's `showLanding` gate to flip true;
  // clearing journeyResumeSessionId prevents LandingRouter's resume
  // effect from synthesizing a Tier1Workspace mount.
  const handleReturnFromRca = useCallback(() => {
    dispatch({ type: "RESET_MODE_STATE" });
    dispatch({ type: "CLEAR_JOURNEY_RESUME" });
    setRcaOpen(false);
  }, [dispatch]);

  // When the user hasn't picked a mode yet, show the LandingPage instead
  // of the ChatArea. Sidebar stays visible so past sessions remain reachable.
  const showLanding = !state.selectedMode;

  return (
    <div className="flex h-screen overflow-hidden t-bg-primary">
      {/* Sidebar - desktop */}
      <div className="hidden md:flex">
        <Sidebar
          onOpenRca={() => setRcaModalOpen(true)}
          onOpenGapAnalysis={() => setGapModalOpen(true)}
          onOpenTicketFilter={handleOpenTicketFilter}
          onOpenServiceNow={handleOpenServiceNow}
        />
      </div>

      {/* Sidebar - mobile overlay */}
      {state.sidebarOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div className="flex-shrink-0">
            <Sidebar
              onOpenRca={() => setRcaModalOpen(true)}
              onOpenGapAnalysis={() => setGapModalOpen(true)}
              onOpenTicketFilter={handleOpenTicketFilter}
              onOpenServiceNow={handleOpenServiceNow}
            />
          </div>
          <div
            className="flex-1 bg-black/40"
            onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })}
          />
        </div>
      )}

      {/* Sprint 13.32.6 — RCA entry modal. Triggered by the sidebar
          RCA button (now a peer of New Chat). On submit, the payload
          flows into the right-pane RCAFlow via `initialPayload`. */}
      <RCAEntryModal
        open={rcaModalOpen}
        onClose={() => setRcaModalOpen(false)}
        onSubmit={handleRcaModalSubmit}
      />

      {/* Gap Analysis entry modal — independent of the RCA modal.
          Same flow shape: collect the incident number, hand the
          payload to AppLayout, AppLayout flips the right pane to
          GapAnalysisFlow. */}
      <GapAnalysisEntryModal
        open={gapModalOpen}
        onClose={() => setGapModalOpen(false)}
        onSubmit={handleGapModalSubmit}
      />

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <MobileHeader />
        {rcaOpen ? (
          <RCAFlow
            initialPayload={rcaPayload}
            onReturnToStages={handleReturnFromRca}
          />
        ) : gapOpen ? (
          /* Gap Analysis right-pane takeover — independent of RCA.
             Only one of the two flows is ever active at a time
             because their setters are wired to mutually exclusive
             buttons; the strict if/else preserves that invariant. */
          <GapAnalysisFlow
            initialPayload={gapPayload}
            onReturnToStages={handleReturnFromGapAnalysis}
          />
        ) : ticketFilterOpen ? (
          /* Ticket Filter right-pane takeover — independent of RCA
             and Gap. The setters above ensure rcaOpen / gapOpen are
             cleared whenever this opens, so the if/else priority
             ordering above this branch can't shadow it. */
          <TicketFilterFlow
            onReturnToStages={handleReturnFromTicketFilter}
          />
        ) : serviceNowOpen ? (
          /* ServiceNow right-pane takeover — independent optional
             integration. Mutual-exclusion with the other flows is
             enforced by handleOpenServiceNow + the symmetric closures
             added to handleRcaModalSubmit / handleGapModalSubmit /
             handleOpenTicketFilter above. */
          <ServiceNowFlow
            onReturnToStages={handleReturnFromServiceNow}
          />
        ) : showLanding ? (
          /* Landing surface — receives the four quick-action handlers
             so the top horizontal pill bar (RCA / Gap / Filter /
             ServiceNow) on the LandingPage modes-picker can open the
             corresponding right-pane flows without going through the
             sidebar. */
          <LandingRouter
            onOpenRca={() => setRcaModalOpen(true)}
            onOpenGapAnalysis={() => setGapModalOpen(true)}
            onOpenTicketFilter={handleOpenTicketFilter}
            onOpenServiceNow={handleOpenServiceNow}
          />
        ) : (
          <ChatArea />
        )}
      </div>

      {/* BuildStamp watermark removed at user request — the component
          definition above is left in place for diagnostic reuse but
          is no longer rendered on the app shell. */}
    </div>
  );
}

function ThemedApp() {
  const { isDark } = useTheme();

  // All AntD design tokens (colors, radii, font, component-specific
  // overrides) are resolved from a single switch in
  // `src/theme/theme-config.js`. Under premium → Aurora Operations
  // tokens; under legacy → original light/dark tokens (preserved
  // byte-for-byte from before the revamp).
  const antTheme = resolveAntdTheme(isDark, {
    darkAlgorithm: theme.darkAlgorithm,
    defaultAlgorithm: theme.defaultAlgorithm,
  });

  return (
    <ConfigProvider theme={antTheme}>
      <AuthGate>
        <ChatProvider>
          <AppLayout />
        </ChatProvider>
      </AuthGate>
    </ConfigProvider>
  );
}

export default function App() {
  return (
    <ThemeProvider>
      <ThemedApp />
    </ThemeProvider>
  );
}
