import React from "react";
import { ConfigProvider, theme } from "antd";
import { ThemeProvider, useTheme } from "./hooks/ThemeContext";
import { ChatProvider, useChat } from "./hooks/ChatContext";
import Sidebar from "./components/Sidebar";
import ChatArea from "./components/ChatArea";
import MobileHeader from "./components/MobileHeader";
import AuthGate from "./components/AuthGate";

const BUILD_TIMESTAMP = process.env.REACT_APP_BUILD_TIMESTAMP || "dev";

function BuildStamp() {
  return (
    <div
      className="fixed bottom-1 right-2 text-[10px] opacity-40 pointer-events-none select-none"
      style={{ zIndex: 9999, color: "var(--text-muted, #888)" }}
      title="Build timestamp — confirms the running bundle"
    >
      build {BUILD_TIMESTAMP}
    </div>
  );
}

function AppLayout() {
  const { state, dispatch } = useChat();

  return (
    <div className="flex h-screen overflow-hidden t-bg-primary">
      {/* Sidebar - desktop */}
      <div className="hidden md:flex">
        <Sidebar />
      </div>

      {/* Sidebar - mobile overlay */}
      {state.sidebarOpen && (
        <div className="md:hidden fixed inset-0 z-50 flex">
          <div className="flex-shrink-0">
            <Sidebar />
          </div>
          <div
            className="flex-1 bg-black/40"
            onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })}
          />
        </div>
      )}

      {/* Main area */}
      <div className="flex-1 flex flex-col min-w-0">
        <MobileHeader />
        <ChatArea />
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
