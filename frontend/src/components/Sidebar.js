import React, { useEffect, useRef, useState } from "react";
import { Button, Tooltip, Badge, Tabs, Empty, Popconfirm, message, Switch } from "antd";
import {
  PlusOutlined,
  MessageOutlined,
  FileOutlined,
  DeleteOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  CloudUploadOutlined,
  HistoryOutlined,
  ClearOutlined,
  BulbOutlined,
  SettingOutlined,
  UserOutlined,
  FileSearchOutlined,
} from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { useTheme } from "../hooks/ThemeContext";
import {
  listSessions,
  getSession,
  deleteSession,
  deleteAllSessions,
  listFiles,
  deleteFile,
  resetSessionContext,
} from "../services/api";
import UploadPanel from "./UploadPanel";
import UserProfile from "./UserProfile";
import VersionGroup from "./VersionGroup";

// Sprint 3-PREP-B — doc_kind badge rendered next to each filename.
// Flag-gated via REACT_APP_LOGIQ_BULK_INGEST_FRONTEND so flag-off = no
// badge (byte-identical post-PREP-A UI).
const KIND_LABELS = {
  ticket: "tickets",
  sop: "sop",
  kb: "kb",
  contact_customer: "cust",
  contact_vendor: "vnd",
  vendor_case: "case",
};
const KIND_COLORS = {
  ticket: "#3b82f6",
  sop: "#10b981",
  kb: "#8b5cf6",
  contact_customer: "#f59e0b",
  contact_vendor: "#ef4444",
  vendor_case: "#ec4899",
};
const BULK_INGEST_ENABLED =
  (process.env.REACT_APP_LOGIQ_BULK_INGEST_FRONTEND || "false").toLowerCase() === "true";

function DocKindBadge({ kind }) {
  if (!BULK_INGEST_ENABLED) return null;
  const label = KIND_LABELS[kind];
  if (!label) return null;
  return (
    <span
      className="text-[9px] px-1.5 py-0.5 rounded flex-shrink-0"
      style={{ backgroundColor: KIND_COLORS[kind] || "#64748b", color: "#fff" }}
      title={`doc_kind: ${kind}`}
    >
      {label}
    </span>
  );
}

// Group files by (normalized_name || version_family_key || name), sort
// each group descending by version_rank so the newest version is first.
function groupByVersionFamily(files) {
  const groups = new Map();
  for (const f of files || []) {
    const key = f.normalized_name || f.version_family_key || f.name;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(f);
  }
  const result = [];
  for (const [key, versions] of groups.entries()) {
    versions.sort((a, b) => {
      const ra = a.version_rank != null ? a.version_rank : 0;
      const rb = b.version_rank != null ? b.version_rank : 0;
      if (rb !== ra) return rb - ra;
      const ta = a.created_at ? new Date(a.created_at).getTime() : 0;
      const tb = b.created_at ? new Date(b.created_at).getTime() : 0;
      return tb - ta;
    });
    result.push({ key, versions });
  }
  return result;
}

export default function Sidebar({ onOpenRca }) {
  // Sprint 13.32 — `onOpenRca` is passed in from AppLayout. When the
  // RCA button below History is clicked, we call it to flip the
  // right pane over to the RCAFlow surface. Sidebar itself stays
  // mounted and visible throughout, so the user can also click
  // History items / New Chat to leave RCA at any point.
  //
  // Sprint 13.32.6 — visibility gate dropped. RCA is now a peer
  // primary action below New Chat, always visible while signed in.
  // The earlier `journeyActive` runtime gate + the localStorage
  // breadcrumb gate are no longer consulted here. The localStorage
  // breadcrumb is still WRITTEN by ResolutionJourney for the
  // separate "Return to Stages" handler — only the visibility
  // listener is gone.
  const { state, dispatch } = useChat();
  const { isDark, toggleTheme } = useTheme();
  const [loading, setLoading] = useState(false);
  // Sprint 11 — Track session ids whose DELETE is in flight so a fast
  // double-click on the same row doesn't fire two requests (the second
  // would 404 because the first already removed the row, leaving the
  // user looking at a misleading "Failed to delete" toast).
  const inFlightDeletes = useRef(new Set());

  const isAdmin = state.userRole === "admin";

  useEffect(() => {
    fetchSessions();
    fetchFiles();
  }, []);

  const fetchSessions = async () => {
    try {
      const res = await listSessions();
      dispatch({ type: "SET_SESSIONS", payload: res.data.sessions || [] });
    } catch { /* ignore */ }
  };

  const fetchFiles = async () => {
    try {
      const res = await listFiles();
      dispatch({ type: "SET_FILES", payload: res.data.files || [] });
    } catch { /* ignore */ }
  };

  const handleSelectSession = async (id) => {
    try {
      setLoading(true);
      const res = await getSession(id);
      dispatch({ type: "SET_SESSION", payload: res.data });
    } catch {
      message.error("Failed to load chat");
    } finally {
      setLoading(false);
    }
  };

  const handleDeleteSession = async (id, e) => {
    e?.stopPropagation();

    // Sprint 11 — Idempotent + optimistic delete.
    //
    // Backend symptom this guards against (per logs): the same session
    // id was DELETEd up to 5 times across 2 minutes — first call 200,
    // every subsequent call 404, because the row stayed visible in the
    // sidebar (sessions are fetched once on mount, never re-polled).
    // Each 404 fired "Failed to delete" and the user kept clicking.
    //
    // Three guards:
    //   1. In-flight Set blocks fast double-click on the same row.
    //   2. Optimistic remove from local state BEFORE the request — UI
    //      feels instant, and a 404 race never re-shows the row.
    //   3. 404 = success (idempotent DELETE semantics — the session is
    //      gone, exactly what the user wanted). Only true server / network
    //      failures (5xx, no response) restore the row + show an error.
    if (inFlightDeletes.current.has(id)) return;
    inFlightDeletes.current.add(id);

    const prevSessions = state.sessions;
    const wasActive = state.sessionId === id;

    // Optimistic remove.
    dispatch({
      type: "SET_SESSIONS",
      payload: state.sessions.filter((s) => s.id !== id),
    });
    if (wasActive) dispatch({ type: "NEW_CHAT" });

    try {
      await deleteSession(id);
      message.success("Chat deleted");
    } catch (err) {
      const status = err?.response?.status;
      if (status === 404) {
        // Session already gone (other tab, prior delete that timed out
        // client-side, owner_id mismatch). Treat as success — the row
        // was already optimistically removed and that's the right end
        // state. Use a quiet info toast so the user isn't confused.
        message.info("Chat already removed");
      } else {
        // Real failure — restore the optimistic removal so the user
        // can retry. NEW_CHAT was the right call if wasActive (we
        // can't restore the active-session pointer because the chat
        // pane already cleared); leaving them on NEW_CHAT is the
        // safer default than re-opening a chat we may not own.
        dispatch({ type: "SET_SESSIONS", payload: prevSessions });
        message.error("Failed to delete");
      }
    } finally {
      inFlightDeletes.current.delete(id);
    }
  };

  const handleClearAll = async () => {
    try {
      await deleteAllSessions();
      dispatch({ type: "SET_SESSIONS", payload: [] });
      dispatch({ type: "NEW_CHAT" });
      message.success("All chats cleared");
    } catch { message.error("Failed to clear"); }
  };

  const handleChangeContext = async () => {
    try {
      if (state.sessionId) {
        await resetSessionContext(state.sessionId);
      }
    } catch {
      // Swallow network error — still reset client-side so the user is
      // never stuck in a mode. Backend write can retry on next mode-set.
    }
    dispatch({ type: "RESET_MODE_STATE" });
    message.success("Context cleared — pick a new mode.");
  };

  const handleDeleteFile = async (fileId, fileName) => {
    try {
      await deleteFile(fileId);
      dispatch({
        type: "SET_FILES",
        payload: state.uploadedFiles.filter((f) => f.id !== fileId),
      });
      message.success(`"${fileName}" deleted`);
    } catch {
      message.error("Failed to delete file");
    }
  };

  const handleRoleToggle = (checked) => {
    dispatch({ type: "SET_USER_ROLE", payload: checked ? "admin" : "user" });
  };

  // ── Build tabs based on role ──
  const tabItems = [
    // History tab — always visible
    {
      key: "chat",
      label: <span className="flex items-center gap-1.5 text-xs"><HistoryOutlined /> History</span>,
      children: (
        <div className="flex flex-col gap-1 overflow-y-auto max-h-[calc(100vh-380px)]">
          {/* Sprint 13.32.6 — RCA button moved out of this tab.
              It now lives below New Chat in the permanent action
              area (see the New Chat row higher up in this file).
              Block intentionally left empty so the History tab
              shape stays identical to its pre-RCA layout. */}
          {state.sessions.length === 0 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span className="t-text-muted text-xs">No chats yet</span>} />
          ) : (
            state.sessions.map((s) => (
              <div
                key={s.id}
                onClick={() => handleSelectSession(s.id)}
                className={`group flex items-center justify-between px-3 py-2.5 rounded-lg cursor-pointer transition-all duration-200 ${
                  state.sessionId === s.id
                    ? "bg-brand-100 dark:bg-brand-600/20 border border-brand-200 dark:border-brand-500/30"
                    : "t-bg-hover border border-transparent"
                }`}
                style={state.sessionId === s.id ? { backgroundColor: "var(--brand-light)", borderColor: "var(--brand-accent)" } : {}}
              >
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <MessageOutlined style={{ color: "var(--brand-accent)" }} className="text-xs flex-shrink-0" />
                  <span className="text-sm truncate t-text-secondary">{s.title}</span>
                </div>
                <Tooltip title="Delete">
                  <Button
                    type="text"
                    size="small"
                    icon={<DeleteOutlined />}
                    onClick={(e) => handleDeleteSession(s.id, e)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    style={{ color: "var(--text-muted)" }}
                  />
                </Tooltip>
              </div>
            ))
          )}
        </div>
      ),
    },

    // Upload tab — Admin only
    ...(isAdmin
      ? [
          {
            key: "upload",
            label: <span className="flex items-center gap-1.5 text-xs"><CloudUploadOutlined /> Upload</span>,
            children: <UploadPanel onUploadComplete={fetchFiles} />,
          },
        ]
      : []),

    // Files tab — always visible, but delete button is Admin only
    {
      key: "files",
      label: (
        <span className="flex items-center gap-1.5 text-xs">
          <FileOutlined />{" "}
          <Badge count={state.uploadedFiles.length} size="small" style={{ backgroundColor: "#100d4b" }}>Files</Badge>
        </span>
      ),
      children: (
        <div className="flex flex-col gap-1 overflow-y-auto max-h-[calc(100vh-380px)]">
          {state.uploadedFiles.length === 0 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span className="t-text-muted text-xs">No files uploaded</span>} />
          ) : (
            groupByVersionFamily(state.uploadedFiles).map(({ key, versions }) => {
              if (versions.length === 1) {
                const f = versions[0];
                // Sprint 2.9 — red-dot indicator for files rejected at ingestion
                // (malformed JSON). Backend sets ingestion_status="invalid_json"
                // and populates ingestion_error with a line-number + reason.
                const isInvalid = f.ingestion_status === "invalid_json";
                const invalidTooltip = f.ingestion_error || "Invalid JSON structure";
                return (
                  <div key={f.id} className="group flex items-center justify-between px-3 py-2 rounded-lg t-bg-tertiary border" style={{ borderColor: isInvalid ? "#ef4444" : "var(--border-color)" }}>
                    <div className="flex items-center gap-2 min-w-0 flex-1">
                      {isInvalid ? (
                        <Tooltip title={invalidTooltip}>
                          <span
                            aria-label="Invalid file"
                            style={{
                              display: "inline-block",
                              width: 10,
                              height: 10,
                              borderRadius: "50%",
                              background: "#ef4444",
                              flexShrink: 0,
                            }}
                          />
                        </Tooltip>
                      ) : (
                        <FileOutlined style={{ color: "#6366f1" }} />
                      )}
                      <div className="min-w-0">
                        <div className="flex items-center gap-1.5">
                          <p className="text-xs t-text truncate max-w-[140px]" style={isInvalid ? { color: "#ef4444" } : undefined}>{f.name}</p>
                          <DocKindBadge kind={f.doc_kind} />
                        </div>
                        <p className="text-[10px] t-text-muted">
                          {f.size_mb != null ? `${f.size_mb.toFixed(1)}MB · ` : ""}
                          <span style={{ color: isInvalid ? "#ef4444" : f.status === "indexed" ? "#10b981" : f.status === "failed" ? "#ef4444" : "#f59e0b" }}>
                            {isInvalid ? "invalid_json" : f.status}
                          </span>
                        </p>
                      </div>
                    </div>
                    {isAdmin && (
                      <Popconfirm
                        title={`Delete "${f.name}"?`}
                        description="This will remove the file and all its indexed data."
                        onConfirm={() => handleDeleteFile(f.id, f.name)}
                        okText="Delete"
                        cancelText="Cancel"
                        okButtonProps={{ danger: true }}
                      >
                        <Tooltip title="Delete file">
                          <Button
                            type="text"
                            size="small"
                            icon={<DeleteOutlined />}
                            className="opacity-0 group-hover:opacity-100 transition-opacity flex-shrink-0"
                            style={{ color: "var(--text-muted)" }}
                            danger
                          />
                        </Tooltip>
                      </Popconfirm>
                    )}
                  </div>
                );
              }
              return (
                <VersionGroup
                  key={key}
                  versions={versions}
                  isAdmin={isAdmin}
                  onDelete={handleDeleteFile}
                />
              );
            })
          )}
        </div>
      ),
    },
  ];

  // ── Collapsed sidebar ──
  if (!state.sidebarOpen) {
    return (
      <div className="flex flex-col items-center py-4 px-1 t-bg-secondary border-r w-14 h-screen" style={{ borderColor: "var(--border-color)" }}>
        <Tooltip title="Expand sidebar" placement="right">
          <Button type="text" icon={<MenuUnfoldOutlined style={{ color: "var(--text-muted)" }} />} onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })} />
        </Tooltip>
        <div className="mt-4">
          <Tooltip title="New Chat" placement="right">
            <Button type="text" icon={<PlusOutlined style={{ color: "var(--text-muted)" }} />} onClick={() => dispatch({ type: "NEW_CHAT" })} />
          </Tooltip>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col t-bg-secondary border-r w-[280px] h-screen lg:w-[300px]" style={{ borderColor: "var(--border-color)" }}>
      {/* Header */}
      <div
        className="flex items-center px-4 py-3 border-b"
        style={{ borderColor: "var(--border-color)" }}
      >
        <div className="flex items-center">
          <div className="h-12 w-[160px] flex items-center overflow-hidden">
            <img
              src="/logo.png"
              alt="Acadia Logo"
              className="h-full w-auto object-contain"
            />
          </div>
        </div>

        <Button
          type="text"
          icon={<MenuFoldOutlined style={{ color: "var(--text-muted)" }} />}
          onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })}
          size="small"
          className="ml-auto"
        />
      </div>

      {/* New Chat + Theme Toggle */}
      <div className="px-3 pt-3 flex items-center gap-2">
        <Button
          icon={<PlusOutlined />}
          onClick={() => dispatch({ type: "NEW_CHAT" })}
          block
          className="rounded-lg h-9 font-medium text-sm flex-1"
          style={{
            backgroundColor: "var(--acadia-primary)",
            borderColor: "var(--acadia-primary)",
            color: "#fff"
          }}
        >
          New Chat
        </Button>
        <Tooltip title={isDark ? "Switch to Light" : "Switch to Dark"}>
          <Button
            type="text"
            icon={<BulbOutlined style={{ color: isDark ? "#fbbf24" : "#08324F" }} />}
            onClick={toggleTheme}
            className="flex-shrink-0"
          />
        </Tooltip>
      </div>

      {/* Sprint 13.32.6 — RCA primary action. Sits directly below
          New Chat as a permanent sidebar action. Same Acadia-primary
          colour so it reads as a peer call-to-action. Click opens the
          RCAEntryModal (ticket-number + Internal/External + file
          upload); the modal hands the captured payload to the
          right-pane RCAFlow on submit. Hidden when AppLayout didn't
          wire onOpenRca so the file stays backwards-compatible. */}
      {typeof onOpenRca === "function" ? (
        <div className="px-3 pt-2">
          <Tooltip title="Generate Internal / External RCA from a ticket number or upload">
            <Button
              icon={<FileSearchOutlined />}
              onClick={onOpenRca}
              block
              className="rounded-lg h-9 font-medium text-sm"
              style={{
                backgroundColor: "var(--acadia-primary)",
                borderColor: "var(--acadia-primary)",
                color: "#fff"
              }}
            >
              RCA
            </Button>
          </Tooltip>
        </div>
      ) : null}

      {/* Tabs */}
      <div className="flex-1 overflow-hidden px-3 pt-2">
        <Tabs defaultActiveKey="chat" items={tabItems} size="small" className="sidebar-tabs" />
      </div>

      {/* ── Role Toggle (Admin / User) ── */}
      <div
        className="flex items-center justify-between px-4 py-2.5 border-t"
        style={{ borderColor: "var(--border-color)" }}
      >
        <div className="flex items-center gap-2">
          {isAdmin ? (
            <SettingOutlined style={{ color: "var(--acadia-primary)", fontSize: 14 }} />
          ) : (
            <UserOutlined style={{ color: "var(--text-muted)", fontSize: 14 }} />
          )}
          <span className="text-xs font-medium" style={{ color: isAdmin ? "var(--acadia-primary)" : "var(--text-muted)" }}>
            {isAdmin ? "Admin" : "User"}
          </span>
        </div>
        <Switch
          checked={isAdmin}
          onChange={handleRoleToggle}
          size="small"
          style={{
            backgroundColor: isAdmin ? "var(--acadia-primary)" : undefined,
          }}
        />
      </div>

      {/* User Profile (Clerk) */}
      <UserProfile />

      {/* Footer */}
      <div className="px-3 py-3 border-t" style={{ borderColor: "var(--border-color)" }}>
        {/* {state.selectedMode && (
          <Button
            type="text"
            block
            size="small"
            onClick={handleChangeContext}
            className="t-text-muted text-xs mb-1"
          >
            Change Context
          </Button>
        )} */}
        <Popconfirm title="Clear all chat history?" onConfirm={handleClearAll} okText="Clear" cancelText="Cancel" okButtonProps={{ danger: true }}>
          <Button type="text" icon={<ClearOutlined />} block size="small" className="t-text-muted text-xs">
            Clear All History
          </Button>
        </Popconfirm>
      </div>
    </div>
  );
}
