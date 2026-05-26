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
import EscalateButton from "./EscalateButton";
import HelpfulButton from "./HelpfulButton";
import CardWatermark from "./CardWatermark";
import { STAGE_LABELS } from "../tier1Constants";
import { postJourneyEvent, searchKbHandoff } from "./journeyApi";

const { Title, Paragraph, Text } = Typography;


// ─────────────────────────────────────────────────────────────
// Block 03 — Knowledge Base & SOP Reference (heading only).
// Mirrors the Block 02 (Guided Workflow) eyebrow + display +
// gradient italic accent treatment so all four cohort/journey
// panels read as a consistent visual family.
// Scope: `.b03-*` so nothing bleeds into any other panel.
// ─────────────────────────────────────────────────────────────
const B03_BLUE = {
  50:  "#EFF6FF",
  200: "#BFDBFE",
  400: "#60A5FA",
  500: "#3B82F6",
  600: "#2563EB",
  700: "#1D4ED8",
};
const B03_CSS = `
.b03-eyebrow {
  display: inline-flex; align-items: center; gap: 8px;
  padding: 4px 12px; border-radius: 9999px;
  background: ${B03_BLUE[50]};
  border: 1px solid ${B03_BLUE[200]};
  color: ${B03_BLUE[700]};
  font-family: var(--font-mono, 'Geist Mono', 'JetBrains Mono', Consolas, monospace);
  font-size: 10.5px; font-weight: 500; letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.b03-eyebrow__dot {
  width: 6px; height: 6px; border-radius: 50%;
  background: ${B03_BLUE[500]};
  box-shadow: 0 0 8px ${B03_BLUE[400]};
}
.b03-display {
  font-family: var(--font-display, 'Instrument Serif', Georgia, serif);
  font-size: clamp(24px, 2.8vw, 32px);
  font-weight: 400; line-height: 1.12; letter-spacing: -0.018em;
  margin: 0 0 6px 0; color: #0F172A;
}
.b03-accent {
  font-style: italic; font-weight: 400;
  background: linear-gradient(135deg, ${B03_BLUE[400]} 0%, ${B03_BLUE[600]} 60%, ${B03_BLUE[700]} 100%);
  -webkit-background-clip: text; background-clip: text;
  -webkit-text-fill-color: transparent; color: transparent;
}
`;
let _b03StylesInjected = false;
function _ensureB03Styles() {
  if (typeof document === "undefined" || _b03StylesInjected) return;
  const style = document.createElement("style");
  style.setAttribute("data-acadia-block03", "1");
  style.textContent = B03_CSS;
  document.head.appendChild(style);
  _b03StylesInjected = true;
}


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

  // Block 03 — inject scoped CSS once on first mount. Idempotent.
  React.useEffect(() => { _ensureB03Styles(); }, []);

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

      // Premium revamp — expand the sidebar when the engineer lands
      // in the KB chat. The journey collapses it for focus; chat
      // wants history + scope visible. Idempotent if already open.
      dispatch({ type: "SET_SIDEBAR", payload: true });

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
    <Card
      style={{
        marginBottom: 16,
        borderLeft: "4px solid #7c3aed",
        // Premium revamp — per-card Acadia watermark.
        position: "relative",
        overflow: "hidden",
      }}
    >
      <CardWatermark />
      {/* ─── Block 03 — premium heading replacement (heading only) ───
          OLD heading preserved below for reference. Card wrapper +
          <CardWatermark/> + footer (Helpful/Dislike/Escalate) all
          UNCHANGED — only the title block is re-typeset to match the
          Best Historical Match / Guided Workflow header treatment.

          ── OLD ──
          <Title level={5} style={{ marginTop: 0, position: "relative", zIndex: 1 }}>
            {STAGE_LABELS.stage_4}
          </Title>
          ── /OLD ── */}
      <div style={{ position: "relative", zIndex: 1, marginTop: 0, marginBottom: 8 }}>
        <div className="b03-eyebrow">
          <span aria-hidden className="b03-eyebrow__dot" />
          Knowledge Base
        </div>
        <h2 className="b03-display">
          Knowledge base &amp; SOP <em className="b03-accent">reference</em>
        </h2>
      </div>
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
        {/* Sprint 13.28 — Stage 4's "advance" CTA IS the escalation, so
            we wire it through EscalateButton (same component every other
            stage uses) instead of a relabeled NextStageButton. This
            restores the trigger-classification modal that NextStageButton
            doesn't know about. EscalateButton internally posts the same
            `next_stage_clicked` telemetry with `{to: "stage_5"}` and
            invokes onReveal("stage_5"), so the post-modal flow is
            byte-identical to the previous behaviour — only the modal
            popup is added on top. */}
        <EscalateButton
          sessionId={sessionId}
          fromStage="stage_4"
          onReveal={onReveal}
          label="Escalate to Tier-2"
        />
      </div>
    </Card>
  );
}
