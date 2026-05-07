// Sprint 10.2 Stage 4 — Search KB / SOP handoff (corrected).
//
// Replaces Sprint 10's dispatch-only handoff (which left the chat
// empty because nothing was persisted to the backend). The new flow:
//   1. Click → POST /tier1/journey/{sid}/search-kb-handoff
//   2. Backend creates a real chat session + saves the user turn +
//      either invokes /ask (when SOP/KB corpus exists) or inserts an
//      upload-prompt assistant message (when corpus is empty).
//   3. Frontend dispatches SET_MODE → "troubleshooting" (M1.3 state-
//      based routing pattern), then ADD_USER_MESSAGE so the prefilled
//      question is visible immediately, then optionally a toast when
//      has_corpus=false.
//   4. The new chat appears in the left sidebar via the existing
//      chat-list polling.

import React, { useState } from "react";
import { Alert, Button, Card, Spin, Typography, message } from "antd";
import { LoadingOutlined, MessageOutlined } from "@ant-design/icons";

import { useChat } from "../../../hooks/ChatContext";
import { askQuestion, getSession } from "../../../services/api";
import DislikeButton from "./DislikeButton";
import HelpfulButton from "./HelpfulButton";
import NextStageButton from "./NextStageButton";
import { STAGE_LABELS } from "../tier1Constants";
import { postJourneyEvent, searchKbHandoff } from "./journeyApi";

const { Title, Paragraph, Text } = Typography;


export default function Stage4SearchKBHandoff({
  data,
  sessionId,
  onMarkedHelpful,
  onStartNewTicket,
  onReveal,
  helpfulMarked,
}) {
  const { dispatch } = useChat();
  const [busy, setBusy] = useState(false);

  if (!data) return null;

  const handleOpenChat = async () => {
    if (busy) return;
    setBusy(true);
    try {
      // Sprint 10.6 §3 — backend response no longer carries
      // has_corpus; the handoff invokes /ask the same way regular chat
      // does (no doc-kind gate, no filter), so the boolean is gone.
      const { chat_session_id } = await searchKbHandoff(sessionId);

      // Switch the main pane to chat. AppLayout reads
      // state.selectedMode; when non-null it falls through from
      // LandingPage/LandingRouter to ChatArea (M1.3 state-routing).
      dispatch({
        type: "SET_MODE",
        payload: { selectedMode: "troubleshooting", subMode: null },
      });

      // Sprint 10.5 §2.1 — hydrate the new chat session into
      // ChatContext via SET_SESSION so state.sessionMetadata picks up
      // journey_session_id (powers the inline JourneyMessageActions
      // buttons). Sprint 10.6 §4: with the route now stamping the
      // authenticated owner_id correctly, this GET no longer 404s.
      try {
        const sessRes = await getSession(chat_session_id);
        dispatch({ type: "SET_SESSION", payload: sessRes.data });
      } catch (sessErr) {
        // eslint-disable-next-line no-console
        console.warn("[journey.stage4] SET_SESSION hydration failed", sessErr);
        dispatch({
          type: "ADD_USER_MESSAGE",
          payload: data.prefilled_message,
        });
      }

      // Sprint 10.6 §3 — auto-fire /ask exactly like regular chat
      // would. The handoff already persisted the user turn; we now
      // generate the assistant turn through the SAME pipeline regular
      // chat uses (no allowed_doc_kinds, no filter). Whatever
      // retrieval finds — the engineer's PDF, SOPs, KBs — surfaces
      // as a grounded answer. If retrieval finds nothing, /ask's
      // natural low-confidence reply handles it gracefully.
      dispatch({ type: "SET_LOADING", payload: true });
      try {
        const askRes = await askQuestion(data.prefilled_message, chat_session_id);
        const askData = askRes.data;
        dispatch({
          type: "ADD_ASSISTANT_MESSAGE",
          payload: {
            answer: askData.answer,
            sources: askData.sources || [],
            confidence: askData.confidence,
            processing_time_ms: askData.processing_time_ms,
            sessionId: askData.session_id,
            context_stats: askData.context_stats || null,
            needs_clarification: askData.needs_clarification,
            clarification_id: askData.clarification_id,
            clarification_options: askData.clarification_options,
            clarification_context: askData.clarification_context,
          },
        });
        // Sprint 11 — engagement signal. Posted ONLY when /ask
        // returned successfully (i.e., the engineer actually saw an
        // assistant answer in the chat). The escalation traversal
        // log uses this to distinguish "opened KB chat (1 exchange)"
        // from a click-through that never triggered any retrieval.
        // Fire-and-forget; a telemetry failure shouldn't disrupt UX.
        postJourneyEvent(
          sessionId, "stage_4", "kb_chat_engaged",
          { chat_session_id },
        ).catch((tErr) => {
          // eslint-disable-next-line no-console
          console.warn("[journey.stage4] kb_chat_engaged telemetry failed", tErr);
        });
      } catch (askErr) {
        // eslint-disable-next-line no-console
        console.warn("[journey.stage4] /ask follow-up failed", askErr);
      } finally {
        dispatch({ type: "SET_LOADING", payload: false });
      }
    } catch (err) {
      // eslint-disable-next-line no-console
      console.error("[journey.stage4] handoff failed", err);
      message.error("Could not open Search KB. Please try again.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card style={{ marginBottom: 16, borderLeft: "4px solid #7c3aed" }}>
      <Title level={5} style={{ marginTop: 0 }}>
        {STAGE_LABELS.stage_4}
      </Title>
      <Paragraph type="secondary">
        Click below to ask follow-up questions in the chat — your alert is pre-loaded.
      </Paragraph>

      <Alert
        type="default"
        message="Pre-filled chat message"
        description={
          <pre style={{ whiteSpace: "pre-wrap", margin: 0, fontSize: 13 }}>
            {data.prefilled_message}
          </pre>
        }
        style={{ marginBottom: 12 }}
      />

      <Paragraph style={{ marginBottom: 12 }}>
        <Text type="secondary" style={{ fontSize: 12 }}>
          Scoped to: {(data.allowed_doc_kinds || []).join(", ")}
        </Text>
      </Paragraph>

      <Button
        type="primary"
        icon={busy ? <LoadingOutlined /> : <MessageOutlined />}
        onClick={handleOpenChat}
        disabled={busy}
        style={{ marginBottom: 16 }}
      >
        {busy ? "Opening chat…" : "Open Chat with this alert"}
      </Button>
      {busy ? (
        <span style={{ marginLeft: 8, color: "#94A3B8" }}>
          <Spin size="small" /> creating session + checking knowledge base…
        </span>
      ) : null}

      <div
        style={{
          marginTop: 8,
          paddingTop: 12,
          borderTop: "1px solid var(--border-color, #f0f0f0)",
          display: "flex",
          flexWrap: "wrap",
          gap: 12,
          justifyContent: "space-between",
          alignItems: "center",
        }}
      >
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          <HelpfulButton
            sessionId={sessionId}
            stage="stage_4"
            onMarkedHelpful={onMarkedHelpful}
            onStartNewTicket={onStartNewTicket}
            disabled={helpfulMarked}
          />
          <DislikeButton
            sessionId={sessionId}
            stage="stage_4"
          />
        </div>
        <NextStageButton
          sessionId={sessionId}
          fromStage="stage_4"
          toStage="stage_5"
          label="Escalate to Tier-2"
          onReveal={onReveal}
        />
      </div>
    </Card>
  );
}
