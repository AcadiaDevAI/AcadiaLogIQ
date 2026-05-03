import React, { useState } from "react";
import { Input, Button, Card, message } from "antd";
import { SearchOutlined } from "@ant-design/icons";
import { useChat } from "../hooks/ChatContext";
import { fingerprintLookup, fingerprintSkip } from "../services/api";

/**
 * FingerprintInputScreen — Sprint 4 landing step.
 *
 * The user types a fingerprint code (e.g., BGP-5-ADJCHANGE) and either
 * gets routed to an Expert Copilot answer (match) or to the "no match"
 * fallback that falls through to the normal mode picker. Skip also
 * falls through to the mode picker, just with an audit flag recording
 * entered_via='skip'.
 *
 * Client-side shape validation was intentionally removed as part of the
 * Sprint 4 hotfix — the backend passes the raw trimmed string straight
 * through to the JSONB exact-match query, so any non-empty input is
 * submittable and the user's keystrokes are preserved verbatim.
 */
export default function FingerprintInputScreen({ onMatch, onNoMatch, onSkip }) {
  const { state, dispatch } = useChat();
  const [fingerprint, setFingerprint] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const trimmed = fingerprint.trim();
  const canSubmit = !!trimmed && !submitting;

  const handleSubmit = async () => {
    if (!trimmed) return;
    setSubmitting(true);
    try {
      const res = await fingerprintLookup(state.sessionId || null, trimmed);
      const { match, answer, session_id: newSid } = res.data || {};
      if (newSid) {
        dispatch({ type: "SET_SESSION_ID", payload: newSid });
      }
      if (match) {
        onMatch({ fingerprint: trimmed, answer, sessionId: newSid });
      } else {
        onNoMatch({ fingerprint: trimmed, sessionId: newSid });
      }
    } catch (err) {
      if (err?.response?.status === 404) {
        // Flag mismatch — silently fall through to classic landing.
        onSkip();
      } else {
        message.error("Fingerprint lookup failed. Please try again.");
      }
    }
    setSubmitting(false);
  };

  const handleSkip = async () => {
    setSubmitting(true);
    try {
      await fingerprintSkip(state.sessionId || null);
    } catch {
      // Skip is best-effort audit; fall through on any failure.
    }
    setSubmitting(false);
    onSkip();
  };

  return (
    <div className="flex-1 flex items-center justify-center px-4 py-8 t-bg-primary">
      <div className="w-full max-w-2xl">
        <div className="text-center mb-6">
          <img
            src="/logo.png"
            alt="LogIQ"
            className="h-12 mx-auto mb-3 object-contain"
          />
          {/* <h1 className="text-xl font-bold t-text">Expert Troubleshooting Copilot</h1> */}
          <p className="t-text-muted text-sm mt-1">
            Start with a fingerprint code if you have one
          </p>
        </div>

        <Card
          className="mb-4"
          bodyStyle={{ padding: 20 }}
          style={{
            backgroundColor: "var(--bg-secondary)",
            borderColor: "var(--border-color)",
          }}
        >
          <p className="text-sm font-medium t-text mb-3">
            Enter the fingerprint for the signature you&apos;re seeing
            (e.g., <code>BGP-5-ADJCHANGE</code>):
          </p>
          <Input
            size="large"
            placeholder="e.g., BGP-5-ADJCHANGE"
            value={fingerprint}
            onChange={(e) => setFingerprint(e.target.value)}
            onPressEnter={handleSubmit}
            autoFocus
            disabled={submitting}
            prefix={<SearchOutlined />}
            style={{ fontFamily: "monospace" }}
          />
          <p className="t-text-faint text-[11px] mt-3">
            Fingerprints are exact-match. We look up the best historical
            resolution and walk you through it in three phases.
          </p>
        </Card>

        <div className="flex justify-center gap-3">
          <Button
            size="large"
            onClick={handleSkip}
            disabled={submitting}
          >
            Skip
          </Button>
          <Button
            type="primary"
            size="large"
            onClick={handleSubmit}
            disabled={!canSubmit}
            loading={submitting}
            style={{
              backgroundColor: "var(--acadia-primary)",
              borderColor: "var(--acadia-primary)",
              minWidth: 160,
            }}
          >
            Submit
          </Button>
        </div>
      </div>
    </div>
  );
}
