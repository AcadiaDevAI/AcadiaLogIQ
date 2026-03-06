import React from "react";
import { Button } from "antd";
import { MenuOutlined } from "@ant-design/icons";
import { SignedIn, UserButton } from "@clerk/clerk-react";
import { useChat } from "../hooks/ChatContext";

const CLERK_KEY = process.env.REACT_APP_CLERK_PUBLISHABLE_KEY;

export default function MobileHeader() {
  const { state, dispatch } = useChat();

  return (
    <div
      className="flex items-center justify-between px-4 py-3 t-bg-secondary border-b md:hidden"
      style={{ borderColor: "var(--border-color)" }}
    >
      <Button
        type="text"
        icon={<MenuOutlined style={{ color: "var(--text-muted)" }} />}
        onClick={() => dispatch({ type: "TOGGLE_SIDEBAR" })}
      />
      <div className="flex items-center gap-2">
        <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ background: "linear-gradient(135deg, #6366f1, #4338ca)" }}>
          <span className="text-white text-[10px] font-bold">A</span>
        </div>
        <span className="text-sm font-semibold t-text">Acadia Log IQ</span>
      </div>
      {CLERK_KEY ? (
        <SignedIn>
          <UserButton appearance={{ elements: { avatarBox: "w-7 h-7" } }} afterSignOutUrl="/" />
        </SignedIn>
      ) : (
        <div className="w-8" />
      )}
    </div>
  );
}
