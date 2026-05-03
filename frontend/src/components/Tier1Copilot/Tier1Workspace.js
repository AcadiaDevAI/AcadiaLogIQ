import React, { useCallback, useMemo, useState } from "react";
import { Button, Space, message } from "antd";

import Tier1AnswerCard from "./Tier1AnswerCard";
import Tier1FollowupChips from "./Tier1FollowupChips";
import DeeperDiagnosticsCard from "./DeeperDiagnosticsCard";
import EscalationPackageCard from "./EscalationPackageCard";
import ExplainRecommendationCard from "./ExplainRecommendationCard";
import SessionTimer from "./SessionTimer";
import SkeletonCard from "./SkeletonCard";
import StuckDetectionModal from "./StuckDetectionModal";
import { TIER1_JOURNEY_ON, TIER1_UX_FIXES_ON } from "./tier1Constants";
import Tier1ThemeProvider from "../../theme/ThemeProvider";

// Sprint 10 — Resolution Journey replaces the answer-card path when
// the build-time flag is on AND the backend returned a session_id.
import ResolutionJourney from "./journey/ResolutionJourney";

import useTier1Session from "../../hooks/useTier1Session";
import {
  fetchDeeperDiagnostics,
  fetchEscalationPackage,
  fetchExplain,
  fetchMatchByIndex, // Sprint 8.1 demo — passed to AnswerCard for bulk download
  logAction,
} from "./tier1Api";

/**
 * Sprint 7 — Tier1Workspace
 *
 * Stateful parent that orchestrates Sprint 7's progressive UX:
 *   - AnswerCard with arrow pagination + trust-calibrated banner
 *   - Progressive reveal of action chips behind 👎
 *   - Conditionally-rendered DeeperDiagnostics / Escalation /
 *     Explain cards
 *   - Session timer + stuck-detection modal
 *
 * The Sprint 6 LandingRouter path (single AnswerCard + chips) is
 * preserved when this workspace is not mounted.
 */
function Tier1WorkspaceInner({ result, onNewAlert }) {
  const tier1 = useTier1Session(result);
  const [whatTried, setWhatTried] = useState([]);
  const [activeCard, setActiveCard] = useState(null);
  // "deeper" | "escalation" | "explain" | null

  const [deeper, setDeeper] = useState(null);
  const [pkg, setPkg] = useState(null);
  const [explain, setExplain] = useState(null);
  const [loadingKey, setLoadingKey] = useState(null);

  // Sprint 8 — the answer card now needs to swap content on arrow
  // click. We mirror the initial /analyze response into local state and
  // overwrite it with the rank-N /match/{index} payload on pagination.
  const [currentResult, setCurrentResult] = useState(result);

  const totalMatches =
    (currentResult && Array.isArray(currentResult.top_5_match_ids)
      ? currentResult.top_5_match_ids.length
      : 0) || 1;

  // Arrow pagination handlers. Sprint 7 only swapped server-side
  // current_match_index; Sprint 8 (Track A) additionally fetches the
  // rank-N match's full Tier1AnalyzeResponse and swaps the card body.
  const handleNavigate = useCallback(
    async (newIndex) => {
      const clamped = Math.max(0, Math.min(totalMatches - 1, newIndex));
      if (TIER1_UX_FIXES_ON) {
        const next = await tier1.fetchMatch(clamped);
        if (next) {
          // Preserve session_id + top_5 from the initial response so the
          // Workspace's action chips continue to operate against the
          // same session.
          setCurrentResult({
            ...next,
            session_id: next.session_id || result.session_id,
            top_5_match_ids:
              next.top_5_match_ids && next.top_5_match_ids.length
                ? next.top_5_match_ids
                : result.top_5_match_ids,
            started_at: next.started_at || result.started_at,
          });
        }
      } else {
        await tier1.swapMatch(clamped);
      }
    },
    [tier1.fetchMatch, tier1.swapMatch, totalMatches, result],
  );

  const handlePrev = useCallback(
    () => handleNavigate(tier1.matchIndex - 1),
    [handleNavigate, tier1.matchIndex],
  );
  const handleNext = useCallback(
    () => handleNavigate(tier1.matchIndex + 1),
    [handleNavigate, tier1.matchIndex],
  );

  // Log a step outcome into the server-side what_tried (also mirrored
  // locally so the Escalation Package has context even if the API call
  // briefly fails).
  const pushWhatTried = useCallback(
    async (entry) => {
      if (!entry) return;
      setWhatTried((prev) => [...prev, entry]);
      if (tier1.sessionId) {
        try {
          await logAction(tier1.sessionId, entry);
        } catch {
          /* swallow — local copy already recorded */
        }
      }
    },
    [tier1.sessionId],
  );

  const openDeeper = useCallback(async () => {
    if (!tier1.sessionId) return true;
    setLoadingKey("deeper");
    try {
      const resp = await fetchDeeperDiagnostics(
        tier1.sessionId,
        result && result.matched_incident,
      );
      setDeeper(resp);
      setActiveCard("deeper");
    } catch (err) {
      message.error("Could not load deeper diagnostics.");
    } finally {
      setLoadingKey(null);
    }
    return true;
  }, [tier1.sessionId, result && result.matched_incident]);

  const openEscalation = useCallback(
    async (extraEntry) => {
      if (!tier1.sessionId) return true;
      if (extraEntry) await pushWhatTried(extraEntry);
      setLoadingKey("escalation");
      try {
        const resp = await fetchEscalationPackage(
          tier1.sessionId,
          result && result.matched_incident,
          extraEntry ? [...whatTried, extraEntry] : whatTried,
        );
        setPkg(resp && resp.package);
        setActiveCard("escalation");
      } catch (err) {
        message.error("Could not assemble escalation package.");
      } finally {
        setLoadingKey(null);
      }
      return true;
    },
    [
      tier1.sessionId,
      result && result.matched_incident,
      whatTried,
      pushWhatTried,
    ],
  );

  const openExplain = useCallback(async () => {
    if (!tier1.sessionId) return true;
    setLoadingKey("explain");
    try {
      const resp = await fetchExplain(
        tier1.sessionId,
        result && result.matched_incident,
      );
      setExplain(resp);
      setActiveCard("explain");
    } catch (err) {
      message.error("Could not load recommendation explanation.");
    } finally {
      setLoadingKey(null);
    }
    return true;
  }, [tier1.sessionId, result && result.matched_incident]);

  const handleAction = useCallback(
    (key) => {
      if (key === "deeper_diagnostics") {
        openDeeper();
        return true;
      }
      if (key === "escalation_note") {
        openEscalation();
        return true;
      }
      if (key === "explain_recommendation") {
        openExplain();
        return true;
      }
      if (key === "next_best_solution") {
        // Arrow-pagination already handles cycling; encourage the user.
        message.info("Use the ◀ / ▶ arrows above to see the next match.");
        return true;
      }
      if (key === "search_kb_sop") {
        message.info(
          "KB/SOP search opens the Acadia chat with this alert as the starter query.",
        );
        return true;
      }
      return false;
    },
    [openDeeper, openEscalation, openExplain],
  );

  const dismissStuck = useCallback(() => {
    tier1.setStuckNudge(false);
  }, [tier1]);

  const escalateFromStuck = useCallback(async () => {
    tier1.setStuckNudge(false);
    await openEscalation();
  }, [tier1, openEscalation]);

  const subCardProps = useMemo(
    () => ({
      onClose: () => setActiveCard(null),
    }),
    [],
  );

  // Sprint 10 — when the journey flag is on AND the backend returned
  // a session_id, render the Resolution Journey in place of the legacy
  // answer-card + chips path. The legacy branch below stays untouched
  // so flag-off behaviour is byte-identical.
  if (TIER1_JOURNEY_ON && result && result.session_id) {
    return (
      <div className="flex-1 overflow-y-auto px-4 py-6 t-bg-primary">
        <div className="w-full max-w-6xl mx-auto">
          <div className="flex justify-between items-center mb-3">
            <Space>
              <SessionTimer elapsedSeconds={tier1.elapsed} />
            </Space>
            <Space>
              <Button onClick={onNewAlert}>Start a new alert</Button>
            </Space>
          </div>
          <ResolutionJourney
            sessionId={result.session_id}
            onNewAlert={onNewAlert}
          />
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 t-bg-primary">
      <div className="w-full max-w-6xl mx-auto">
        <div className="flex justify-between items-center mb-3">
          <Space>
            <SessionTimer elapsedSeconds={tier1.elapsed} />
            {tier1.thumbsDownCount > 0 && (
              <span className="t-text-muted text-xs">
                {tier1.thumbsDownCount} feedback note
                {tier1.thumbsDownCount === 1 ? "" : "s"}
              </span>
            )}
          </Space>
          <Space>
            <Button onClick={onNewAlert}>Start a new alert</Button>
          </Space>
        </div>

        <Tier1AnswerCard
          result={currentResult}
          progressive
          matchIndex={tier1.matchIndex}
          totalMatches={totalMatches}
          onPrev={handlePrev}
          onNext={handleNext}
          paginating={tier1.paginating}
          /* Sprint 8.1 demo — bulk download of all 5 matches. */
          sessionId={tier1.sessionId}
          fetchMatchByIndex={fetchMatchByIndex}
          alertPayload={null /* LandingRouter doesn't thread the intake payload yet; header shows em-dashes */}
        />

        <Tier1FollowupChips
          responseId={currentResult && currentResult.response_id}
          sessionId={tier1.sessionId || "tier1-local"}
          onNewAlert={onNewAlert}
          onAction={handleAction}
        />

        {loadingKey === "deeper" && activeCard !== "deeper" && (
          <div className="mt-4">
            <SkeletonCard variant="diagnostics" />
          </div>
        )}
        {activeCard === "deeper" && (
          <div className="mt-4">
            <DeeperDiagnosticsCard
              diagnostics={deeper}
              onContinueStep={pushWhatTried}
              onSkipStep={pushWhatTried}
              onSkipToEscalation={(entry) => openEscalation(entry)}
              onAnswerQuestion={pushWhatTried}
              /* Sprint 8.1 demo — download .txt export uses these. */
              whatTried={whatTried}
              matchedIncident={currentResult && currentResult.matched_incident}
              {...subCardProps}
            />
          </div>
        )}

        {loadingKey === "escalation" && activeCard !== "escalation" && (
          <div className="mt-4">
            <SkeletonCard variant="escalation" />
          </div>
        )}
        {activeCard === "escalation" && (
          <div className="mt-4">
            <EscalationPackageCard pkg={pkg} {...subCardProps} />
          </div>
        )}

        {loadingKey === "explain" && activeCard !== "explain" && (
          <div className="mt-4">
            <SkeletonCard variant="escalation" />
          </div>
        )}
        {activeCard === "explain" && (
          <div className="mt-4">
            <ExplainRecommendationCard explain={explain} {...subCardProps} />
          </div>
        )}

        <StuckDetectionModal
          open={tier1.stuckNudge}
          elapsedSeconds={tier1.elapsed}
          onEscalate={escalateFromStuck}
          onDismiss={dismissStuck}
        />
      </div>
    </div>
  );
}


/**
 * Default export — wraps the Workspace in Tier1ThemeProvider so every
 * descendant component can read theme tokens via useTier1Theme(). The
 * provider is a no-op (renders children + classic tokens) when
 * LOGIQ_TIER1_MODERN_THEME is off.
 */
export default function Tier1Workspace(props) {
  return (
    <Tier1ThemeProvider>
      <Tier1WorkspaceInner {...props} />
    </Tier1ThemeProvider>
  );
}
