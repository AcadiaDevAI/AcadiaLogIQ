import React, { useCallback, useState } from "react";
import { ConfigProvider, theme } from "antd";
import { ThemeProvider, useTheme } from "./hooks/ThemeContext";
import { ChatProvider, useChat } from "./hooks/ChatContext";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import MobileHeader from "./components/MobileHeader";
import AuthGate from "./components/AuthGate";
import LandingPage from "./components/LandingPage";
import LandingRouter from "./components/LandingRouter";
import RCAFlow from "./components/RCA/RCAFlow";
import RCAEntryModal from "./components/RCA/RCAEntryModal";
import { settings as clientSettings } from "./config/clientSettings";

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
    setRcaPayload(payload);
    setRcaOpen(true);
    setRcaModalOpen(false);
  }, []);

  // Sprint 13.32.2 — "Return to Stages" UX.
  //
  // Always lands the engineer in the Tier-1 GUIDED TROUBLESHOOTING
  // path (never the chat interface, never the Landing page when an
  // alternative exists).
  //
  // Flow:
  //   1. Read the last-active journey sid that ResolutionJourney
  //      breadcrumbs into localStorage every time it mounts. When
  //      present, that's the engineer's most recent live journey —
  //      RESUME_JOURNEY restores it with Preliminary Tier 1 Checks
  //      at the top.
  //   2. When no breadcrumb exists (engineer never opened a journey
  //      this session), still dispatch RESUME_JOURNEY with null so
  //      `selectedMode` is cleared. AppLayout falls through to
  //      LandingRouter, which defaults to screen="tier1" and shows
  //      Tier1IntakeForm — the entry point of the guided path,
  //      NOT the chat interface and NOT the marketing landing.
  //   3. Finally flip rcaOpen=false so the RCA pane unmounts.
  const handleReturnFromRca = useCallback(() => {
    let lastJourneySid = null;
    try {
      lastJourneySid = localStorage.getItem("tier1_last_active_journey_session");
    } catch {
      /* privacy-mode / quota — fall through */
    }

    dispatch({
      type: "RESUME_JOURNEY",
      payload: { journeySessionId: lastJourneySid || null },
    });

    setRcaOpen(false);
  }, [dispatch]);

  // When guided workflow is enabled and the user hasn't picked a mode yet,
  // show the LandingPage instead of the ChatArea. Sidebar stays visible so
  // past sessions remain reachable.
  const showLanding =
    clientSettings.GUIDED_WORKFLOW_ENABLED && !state.selectedMode;

  return (
    <div className="flex h-screen overflow-hidden t-bg-primary">
      {/* Sidebar - desktop */}
      <div className="hidden md:flex">
        <Sidebar onOpenRca={() => setRcaModalOpen(true)} />
      </div>

      {/* Sidebar - mobile overlay */}
      {state.sidebarOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div className="flex-shrink-0">
            <Sidebar onOpenRca={() => setRcaModalOpen(true)} />
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

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <MobileHeader />
        {rcaOpen ? (
          <RCAFlow
            initialPayload={rcaPayload}
            onReturnToStages={handleReturnFromRca}
          />
        ) : showLanding ? (
          clientSettings.LOGIQ_SPRINT4_FRONTEND ? (
            <LandingRouter />
          ) : (
            <LandingPage />
          )
        ) : (
          <ChatArea />
        )}
      </div>

      <BuildStamp />
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
