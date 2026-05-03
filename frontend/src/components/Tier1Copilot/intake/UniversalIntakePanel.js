import React, { useCallback, useState } from "react";
import { Alert, Button, Card, Input, message } from "antd";
import { ThunderboltOutlined } from "@ant-design/icons";
import { INTAKE_MAX_RAW_CHARS } from "../tier1Constants";
import SuggestionCarousel from "./SuggestionCarousel";
import { extractIntake, sendExtractionFeedback } from "./intakeApi";

const SOURCE_LABEL = {
  email: "email",
  phone: "phone transcript",
  portal: "portal ticket",
  chat: "chat message",
  note: "note",
};

/**
 * Sprint 9 — UniversalIntakePanel
 *
 * Engineer pastes raw text → /intake/extract → SuggestionCarousel.
 * Picking a card calls onCardPicked(card) so the parent IntakeForm
 * can prefill its existing 9 fields, then the engineer hits the
 * standard "Analyze alert" CTA.
 */
export default function UniversalIntakePanel({
  source,
  sessionId,
  onCardPicked,
  // Sprint 11 — optional UI overrides for the Reactive single-mode use.
  // When omitted, the panel falls back to the Sprint 9 channel-derived
  // copy ("Paste content from <channel>"). When the parent passes
  // explicit strings, they win — keeps the panel reusable for any
  // generic paste flow without leaking channel-specific copy.
  header: headerOverride,
  helperText: helperTextOverride,
  placeholder: placeholderOverride,
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [candidates, setCandidates] = useState([]);
  const [extractionId, setExtractionId] = useState(null);

  const handleExtract = useCallback(async () => {
    const raw = (text || "").trim();
    if (!raw) return;
    setBusy(true);
    setError(null);
    try {
      const data = await extractIntake({
        source,
        raw_text: raw,
        session_id: sessionId || null,
      });
      if (data && Array.isArray(data.candidates)) {
        setCandidates(data.candidates);
        setExtractionId(data.extraction_id || null);
        if (data.candidates.length === 0) {
          setError(
            data.error
              || "Extraction returned no candidates. Fill the form manually below.",
          );
        }
      } else {
        setError("Extraction failed. Fill the form manually below.");
      }
    } catch (err) {
      const detail = err?.response?.data?.detail?.error
        || err?.response?.data?.detail
        || err?.message
        || "Extraction failed.";
      setError(typeof detail === "string" ? detail : JSON.stringify(detail));
    } finally {
      setBusy(false);
    }
  }, [text, source, sessionId]);

  const handleUse = useCallback(
    async (card) => {
      if (!card) return;
      // Map the validated candidate back into the Sprint 6 form shape.
      const prefill = {
        severity: card.severity || null,
        asset_name: card.asset_name || "",
        alert_type: card.alert_type || "",
        customer: card.customer || "",
        location: card.location || "",
      };
      if (onCardPicked) onCardPicked(prefill);
      message.success("Form pre-filled. Edit fields if needed and click Analyze.");

      if (extractionId) {
        const idx = candidates.findIndex((c) => c === card);
        try {
          await sendExtractionFeedback(extractionId, {
            picked_index: idx >= 0 ? idx : null,
            was_rejected: false,
          });
        } catch {
          /* telemetry failure is non-fatal */
        }
      }
    },
    [extractionId, candidates, onCardPicked],
  );

  const charCount = text.length;
  const overLimit = charCount > INTAKE_MAX_RAW_CHARS;

  return (
    <Card
      bodyStyle={{ padding: 18 }}
      style={{
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
        borderRadius: 12,
        marginBottom: 16,
      }}
    >
      <div className="t-text text-sm font-semibold mb-1">
        {headerOverride || `Paste content from ${SOURCE_LABEL[source] || source}`}
      </div>
      <div className="t-text-muted text-xs mb-2">
        {helperTextOverride
          || ("The system will extract up to 4 structured interpretations and"
              + " validate them against the ingested ticket corpus. Pick one to"
              + " pre-fill the form below.")}
      </div>

      <Input.TextArea
        rows={6}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={
          placeholderOverride
          || `Paste the ${SOURCE_LABEL[source] || source} content here…`
        }
        maxLength={INTAKE_MAX_RAW_CHARS}
        showCount
        disabled={busy}
      />
      {overLimit && (
        <Alert
          type="warning"
          message={`Pasted text exceeds ${INTAKE_MAX_RAW_CHARS} chars; trim before extracting.`}
          showIcon
          style={{ marginTop: 8 }}
        />
      )}

      <div className="flex justify-between items-center mt-3">
        <span className="t-text-muted text-xs">
          {charCount}/{INTAKE_MAX_RAW_CHARS} chars
        </span>
        <Button
          type="primary"
          icon={<ThunderboltOutlined />}
          onClick={handleExtract}
          disabled={busy || !text.trim() || overLimit}
          loading={busy}
          // Sprint 11 — match the Proactive "Analyze alert" button's
          // gradient look (navy → light blue) so the two CTAs feel
          // visually identical across the split landing page. The
          // gradient endpoints are the same Acadia brand tokens.
          style={{
            background:
              "linear-gradient(135deg, var(--acadia-primary) 0%, var(--acadia-primary-light) 100%)",
            borderColor: "transparent",
          }}
        >
          {busy ? "Extracting…" : "Extract & Suggest"}
        </Button>
      </div>

      {error && (
        <Alert
          type="info"
          message={error}
          showIcon
          style={{ marginTop: 12 }}
        />
      )}

      {candidates.length > 0 && (
        <div style={{ marginTop: 16 }}>
          <SuggestionCarousel
            candidates={candidates}
            onUse={handleUse}
          />
          <div className="t-text-muted text-xs mt-2">
            Edit the form fields below if the AI got it wrong.
          </div>
        </div>
      )}
    </Card>
  );
}
