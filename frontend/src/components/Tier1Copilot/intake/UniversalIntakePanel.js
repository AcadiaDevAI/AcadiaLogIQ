import React, { useCallback, useState } from "react";
import { Alert, Button, Card, Input, message } from "antd";
import { ThunderboltOutlined } from "@ant-design/icons";
import { INTAKE_MAX_RAW_CHARS } from "../tier1Constants";
// SuggestionCarousel intentionally NOT imported here — the cards UI is
// commented out in the render below per the new tab UX (top candidate
// auto-applies to the Proactive form). Re-add the import if you revive
// the cards block.
// import SuggestionCarousel from "./SuggestionCarousel";
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
        } else {
          // Auto-pick the top candidate (the one the diversifier ranked
          // first) instead of waiting for the engineer to click one of
          // the suggestion cards. The cards UI is commented out below;
          // the top match's extracted fields flow straight into the
          // Proactive form via the shared onCardPicked callback, and
          // the parent (SourceAwareIntake) switches the tab back to
          // Proactive so the engineer sees the pre-filled form.
          handleUse(data.candidates[0], data.extraction_id || null);
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
    // handleUse intentionally NOT in the dep list — its own deps
    // (extractionId, candidates, onCardPicked) don't affect what
    // `handleExtract` does; we just need the latest reference at
    // call time, which closing over it captures fine. Adding handleUse
    // would re-create handleExtract on every candidates change.
  }, [text, source, sessionId]);

  const handleUse = useCallback(
    async (card, freshExtractionId) => {
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
      message.success("Form pre-filled from your text. Edit if needed and click Analyze.");

      // `freshExtractionId` is the id returned by the just-completed
      // extract call — passed in by `handleExtract` so we don't have to
      // wait for the state set to flush. Falls back to the state value
      // for any future caller that picks a card without the latest id.
      const effectiveId = freshExtractionId || extractionId;
      if (effectiveId) {
        const idx = candidates.findIndex((c) => c === card);
        try {
          await sendExtractionFeedback(effectiveId, {
            picked_index: idx >= 0 ? idx : 0,
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
      // Body padding matches the Proactive Card (14px vertical / 28px
      // horizontal) so the two tabs are visually identical in chrome.
      bodyStyle={{ padding: "14px 28px" }}
      style={{
        backgroundColor: "var(--bg-secondary)",
        borderColor: "var(--border-color)",
        // Spherical (heavily rounded) border to match the Proactive
        // form's outer Card so the two tabs feel like the same vessel
        // with different inner content. Soft ambient shadow lifts the
        // panel off the page background the same way the Proactive
        // Card does — keeps the two views visually identical in
        // weight even though only one is mounted at a time.
        borderRadius: 28,
        boxShadow: "0 6px 24px -8px rgba(15, 23, 42, 0.10), 0 2px 6px -2px rgba(15, 23, 42, 0.06)",
        marginBottom: 16,
      }}
    >
      {/* Sprint 12 — header centered + sized to match the Proactive
          column's "What's happening?" block (20/13px) so the two
          columns of the split intake landing read as visually
          symmetric. UniversalIntakePanel has only one usage today
          (Reactive column inside SourceAwareIntake), so this style
          bump is local to that surface. */}
      {/* Header trimmed (20/13px → 18/12px, smaller gap) so the
          Reactive panel matches the Proactive box's shorter height. */}
      <div style={{ marginBottom: 10, textAlign: "center" }}>
        <h2
          className="t-text"
          style={{
            fontSize: 18,
            fontWeight: 600,
            letterSpacing: "-0.01em",
            margin: 0,
            marginBottom: 2,
          }}
        >
          {headerOverride || `Paste content from ${SOURCE_LABEL[source] || source}`}
        </h2>
        <p
          className="t-text-muted"
          style={{ fontSize: 12, margin: 0, lineHeight: 1.45 }}
        >
          {helperTextOverride
            || ("The system extracts the most likely interpretation from your"
                + " text and applies it to the Proactive form automatically.")}
        </p>
      </div>

      {/* rows lowered 6 → 4 so the Reactive panel is shorter — with
          the wider parent (maxWidth 1400) each row now holds more
          characters, so 4 rows still offers a comfortable paste area. */}
      <Input.TextArea
        rows={4}
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder={
          placeholderOverride
          || `Paste the ${SOURCE_LABEL[source] || source} content here…`
        }
        maxLength={INTAKE_MAX_RAW_CHARS}
        // `showCount` deliberately OFF — the manual counter below is
        // styled to match the premium theme + has the explicit one-line
        // gap before the Extract & Suggest CTA. Keeping AntD's built-in
        // showCount would render the same number twice (duplicate UX).
        disabled={busy}
        // Pill-rounded corners to match the Proactive box's spherical
        // border language. Stops short of full pill on a multi-line
        // textarea (would look odd) — 18px reads as clearly rounded
        // without distorting the corner glyphs.
        style={{ borderRadius: 18 }}
      />
      {overLimit && (
        <Alert
          type="warning"
          message={`Pasted text exceeds ${INTAKE_MAX_RAW_CHARS} chars; trim before extracting.`}
          showIcon
          style={{ marginTop: 8 }}
        />
      )}

      {/* Counter + CTA stack — gaps tightened so the panel fits in
          the new shorter height. The vertical rhythm (textarea →
          counter → CTA) is preserved, just compressed. */}
      <div style={{ marginTop: 8 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "flex-end",
            fontSize: 12,
            color: "var(--text-muted)",
          }}
          className="t-text-muted"
        >
          {charCount}/{INTAKE_MAX_RAW_CHARS} chars
        </div>
        <div
          style={{
            marginTop: 8,
            display: "flex",
            justifyContent: "flex-end",
          }}
        >
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
      </div>

      {error && (
        <Alert
          type="info"
          message={error}
          showIcon
          style={{ marginTop: 12 }}
        />
      )}

      {/* Suggestion cards intentionally commented out per the new tab
          UX. The engineer pastes text → clicks Extract & Suggest → the
          top candidate is auto-applied to the Proactive form and the
          parent flips the active tab back to Proactive (see
          `handleExtract` above and SourceAwareIntake's onCardPicked).

          Preserved here for reference / easy revert:

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
      */}
    </Card>
  );
}
