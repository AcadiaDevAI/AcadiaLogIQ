import React, { useState, useRef, useEffect } from "react";
import { Button, Tooltip } from "antd";
import { SendOutlined, LoadingOutlined } from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";

export default function ChatInput({ onSend }) {
  const { state } = useChat();
  const [input, setInput] = useState("");
  const textareaRef = useRef(null);

  const disabled = state.isLoading || state.isUploading;

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 160) + "px";
    }
  }, [input]);

  const handleSubmit = () => {
    const q = input.trim();
    if (!q || disabled) return;
    onSend(q);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <div className="border-t t-bg-primary px-4 py-3 md:px-8 lg:px-16 xl:px-24" style={{ borderColor: "var(--border-color)" }}>
      <div
        className="relative flex items-end gap-2 border rounded-xl px-4 py-2 transition-colors max-w-4xl mx-auto"
        style={{ backgroundColor: "var(--bg-tertiary)", borderColor: "var(--border-color)" }}
      >
        <textarea
          ref={textareaRef}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            state.isUploading ? "Upload in progress..." : state.isLoading ? "Thinking..." : "Ask about your logs..."
          }
          disabled={disabled}
          rows={1}
          className="flex-1 text-sm resize-none outline-none py-1.5 max-h-40 font-sans disabled:opacity-50"
          style={{ background: "transparent", color: "var(--text-primary)", caretColor: "var(--brand-accent)" }}
        />
        <Tooltip title={disabled ? "Please wait..." : !input.trim() ? "Type a question" : "Send (Enter)"}>
          <Button
            type="primary"
            shape="circle"
            size="small"
            icon={state.isLoading ? <LoadingOutlined spin /> : <SendOutlined className="text-xs" />}
            onClick={handleSubmit}
            disabled={disabled || !input.trim()}
            className="mb-0.5 flex-shrink-0"
          />
        </Tooltip>
      </div>
      <p className="text-center text-[10px] t-text-faint mt-2 max-w-4xl mx-auto">
        Powered by Hybrid Search (Vector + BM25) + LLM Re-ranking · Mistral 7B
      </p>
    </div>
  );
}
