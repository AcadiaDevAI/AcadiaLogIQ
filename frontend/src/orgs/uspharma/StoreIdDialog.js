// StoreIdDialog — US Pharma "Discuss Store Specific with LogIQ" launcher.
//
// A small modal that asks for a Store ID. On submit it pre-creates an EMPTY
// chat session server-side, hard-scoped to that store (POST
// /chat/sessions/store-scoped → chat_sessions.scope_store_id, migration 065),
// then opens that session as an empty chat via NEW_CHAT. Every /ask in the
// chat is then restricted to that store's indexed content.
//
// Accepts ANY Store ID (no validation) — if nothing is indexed for it,
// retrieval returns the normal "no content for that store" message. The chat
// opens BLANK (no prefilled question); the engineer types their own.
//
// US-Pharma-only: mounted solely by the US Pharma landing / intake surfaces.

import React, { useState } from "react";
import { Modal, Input, message } from "antd";
import { useChat } from "../../hooks/ChatContext";
import { createStoreScopedChat } from "../../services/api";

export default function StoreIdDialog({ open, onClose }) {
  const { dispatch } = useChat();
  const [storeId, setStoreId] = useState("");
  const [busy, setBusy] = useState(false);

  const close = () => {
    setStoreId("");
    onClose && onClose();
  };

  const submit = async () => {
    const sid = (storeId || "").trim();
    if (!sid) {
      message.warning("Please enter a Store ID.");
      return;
    }
    setBusy(true);
    try {
      const res = await createStoreScopedChat(sid);
      const sessionId = res?.data?.session_id;
      // Open the pre-created store-scoped session as an empty chat. The
      // kbSearchFromLanding flag surfaces ChatArea's "Back to screen" button;
      // scopeStoreId drives its banner label.
      dispatch({
        type: "NEW_CHAT",
        payload: {
          kbSearchFromLanding: true,
          sessionId,
          scopeStoreId: sid,
        },
      });
      close();
    } catch (err) {
      const detail =
        err?.response?.data?.detail || err?.message || "Could not open the store chat.";
      message.error(detail);
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      title="Discuss Store Specific with LogIQ"
      okText="Start chat"
      cancelText="Cancel"
      confirmLoading={busy}
      onOk={submit}
      onCancel={close}
      destroyOnClose
      maskClosable={!busy}
    >
      <p style={{ marginBottom: 10, color: "var(--text-muted, #6b7280)" }}>
        Enter the Store ID to open a chat scoped to that store.
      </p>
      <Input
        autoFocus
        placeholder="e.g. 3000"
        value={storeId}
        onChange={(e) => setStoreId(e.target.value)}
        onPressEnter={submit}
        disabled={busy}
        allowClear
      />
    </Modal>
  );
}
