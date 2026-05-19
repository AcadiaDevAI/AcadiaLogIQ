import React, { useCallback, useState } from "react";
import { ConfigProvider, theme } from "antd";
import { ThemeProvider, useTheme } from "./hooks/ThemeContext";
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
    // Close the Gap Analysis flow if it's open — the render chain
    // checks rcaOpen FIRST, so without this the new RCA submit would
    // still work but a stale gapOpen=true would render Gap Analysis
    // again the moment RCA returns. Symmetric with handleGapModalSubmit
    // below.
    setGapOpen(false);
    setGapPayload(null);
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

  const handleGapModalSubmit = useCallback((payload) => {
    // Close the RCA flow if it's open — the render chain checks
    // rcaOpen BEFORE gapOpen, so without this Gap Analysis would
    // never appear when the user navigates from RCA → Gap. This was
    // the exact bug: "click Gap Analysis from RCA does nothing".
    setRcaOpen(false);
    setRcaPayload(null);
    setGapPayload(payload);
    setGapOpen(true);
    setGapModalOpen(false);
  }, []);

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
        />
      </div>

      {/* Sidebar - mobile overlay */}
      {state.sidebarOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div className="flex-shrink-0">
            <Sidebar
              onOpenRca={() => setRcaModalOpen(true)}
              onOpenGapAnalysis={() => setGapModalOpen(true)}
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
        ) : showLanding ? (
          <LandingRouter />
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

  const antTheme = isDark
    ? {
        algorithm: theme.darkAlgorithm,
        token: {
          colorPrimary: "#6366f1",
          colorBgContainer: "#16161d",
          colorBgElevated: "#1e1e28",
          colorBorder: "#2a2a3d",
          colorText: "#e2e8f0",
          colorTextSecondary: "#94a3b8",
          borderRadius: 8,
          fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
        },
        components: {
          Button: { primaryShadow: "none" },
          Collapse: { headerBg: "transparent", contentBg: "transparent" },
          Tabs: { colorBgContainer: "transparent", itemColor: "#94a3b8", itemSelectedColor: "#a5b4fc", inkBarColor: "#6366f1" },
        },
      }
    : {
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#4f46e5",
          colorBgContainer: "#ffffff",
          colorBgElevated: "#f8f9fb",
          colorBorder: "#dee2e6",
          colorText: "#1a1a2e",
          colorTextSecondary: "#495057",
          borderRadius: 8,
          fontFamily: "'Poppins', 'Inter', system-ui, sans-serif",
        },
        components: {
          Button: { primaryShadow: "none" },
          Collapse: { headerBg: "transparent", contentBg: "transparent" },
          Tabs: { colorBgContainer: "transparent", itemColor: "#495057", itemSelectedColor: "#4f46e5", inkBarColor: "#4f46e5" },
        },
      };

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
