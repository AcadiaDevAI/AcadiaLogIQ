import React, { useEffect, useState } from "react";
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
} from "../services/api";
import UploadPanel from "./UploadPanel";
import UserProfile from "./UserProfile";

export default function Sidebar() {
  const { state, dispatch } = useChat();
  const { isDark, toggleTheme } = useTheme();
  const [loading, setLoading] = useState(false);

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
    try {
      await deleteSession(id);
      dispatch({ type: "SET_SESSIONS", payload: state.sessions.filter((s) => s.id !== id) });
      if (state.sessionId === id) dispatch({ type: "NEW_CHAT" });
      message.success("Chat deleted");
    } catch { message.error("Failed to delete"); }
  };

  const handleClearAll = async () => {
    try {
      await deleteAllSessions();
      dispatch({ type: "SET_SESSIONS", payload: [] });
      dispatch({ type: "NEW_CHAT" });
      message.success("All chats cleared");
    } catch { message.error("Failed to clear"); }
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

  const tabItems = [
    {
      key: "chat",
      label: <span className="flex items-center gap-1.5 text-xs"><HistoryOutlined /> History</span>,
      children: (
        <div className="flex flex-col gap-1 overflow-y-auto max-h-[calc(100vh-380px)]">
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
    {
      key: "upload",
      label: <span className="flex items-center gap-1.5 text-xs"><CloudUploadOutlined /> Upload</span>,
      children: <UploadPanel onUploadComplete={fetchFiles} />,
    },
    {
      key: "files",
      label: (
        <span className="flex items-center gap-1.5 text-xs">
          <FileOutlined />{" "}
          <Badge count={state.uploadedFiles.length} size="small" style={{ backgroundColor: "#4f46e5" }}>Files</Badge>
        </span>
      ),
      children: (
        <div className="flex flex-col gap-1 overflow-y-auto max-h-[calc(100vh-380px)]">
          {state.uploadedFiles.length === 0 ? (
            <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description={<span className="t-text-muted text-xs">No files uploaded</span>} />
          ) : (
            state.uploadedFiles.map((f) => (
              <div key={f.id} className="group flex items-center justify-between px-3 py-2 rounded-lg t-bg-tertiary border" style={{ borderColor: "var(--border-color)" }}>
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <FileOutlined style={{ color: "#6366f1" }} />
                  <div className="min-w-0">
                    <p className="text-xs t-text truncate max-w-[140px]">{f.name}</p>
                    <p className="text-[10px] t-text-muted">
                      {f.size_mb.toFixed(1)}MB ·{" "}
                      <span style={{ color: f.status === "indexed" ? "#10b981" : f.status === "failed" ? "#ef4444" : "#f59e0b" }}>
                        {f.status}
                      </span>
                    </p>
                  </div>
                </div>
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
              </div>
            ))
          )}
        </div>
      ),
    },
  ];

  // Collapsed sidebar
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
      <div className="flex items-center justify-between px-4 py-4 border-b" style={{ borderColor: "var(--border-color)" }}>
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-brand-500 to-brand-700 flex items-center justify-center" style={{ background: "linear-gradient(135deg, #6366f1, #4338ca)" }}>
            <span className="text-white text-sm font-bold">A</span>
          </div>
          <div>
            <h1 className="text-sm font-bold t-text leading-none">Acadia Log IQ</h1>
            <p className="text-[10px] t-text-muted mt-0.5">Hybrid Search · AI Analysis</p>
          </div>
        </div>
        <Button type="text" icon={<MenuFoldOutlined style={{ color: "var(--text-muted)" }} />} onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })} size="small" />
      </div>

      {/* New Chat + Theme Toggle */}
      <div className="px-3 pt-3 flex items-center gap-2">
        <Button type="primary" icon={<PlusOutlined />} onClick={() => dispatch({ type: "NEW_CHAT" })} block className="rounded-lg h-9 font-medium text-sm flex-1">
          New Chat
        </Button>
        <Tooltip title={isDark ? "Switch to Light" : "Switch to Dark"}>
          <Button
            type="text"
            icon={<BulbOutlined style={{ color: isDark ? "#fbbf24" : "#6366f1" }} />}
            onClick={toggleTheme}
            className="flex-shrink-0"
          />
        </Tooltip>
      </div>

      {/* Tabs */}
      <div className="flex-1 overflow-hidden px-3 pt-2">
        <Tabs defaultActiveKey="chat" items={tabItems} size="small" className="sidebar-tabs" />
      </div>

      {/* User Profile (Clerk) */}
      <UserProfile />

      {/* Footer */}
      <div className="px-3 py-3 border-t" style={{ borderColor: "var(--border-color)" }}>
        <Popconfirm title="Clear all chat history?" onConfirm={handleClearAll} okText="Clear" cancelText="Cancel" okButtonProps={{ danger: true }}>
          <Button type="text" icon={<ClearOutlined />} block size="small" className="t-text-muted text-xs">
            Clear All History
          </Button>
        </Popconfirm>
      </div>
    </div>
  );
}