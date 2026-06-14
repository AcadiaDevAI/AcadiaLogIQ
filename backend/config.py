"""
Configuration — Phases 2-6 Complete + Accuracy Fixes + Performance Tuning.
Includes: contextual ingestion, hybrid retrieval, model routing,
multi-agent troubleshooting, answer validation guardrails,
accuracy fixes, and performance optimizations.
"""

import sys
from pathlib import Path
from typing import ClassVar, Dict, List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Global application configuration.
    """

    # ----------------------------------------------------------------
    # Storage
    # ----------------------------------------------------------------

    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = [
        "log", "txt", "md", "json", "pdf", "docx",
    ]

    # ----------------------------------------------------------------
    # Upload storage backend selector (Phase 1 — S3 direct upload).
    # ----------------------------------------------------------------
    # STORAGE_TYPE="local"  → legacy: files saved under UPLOAD_DIR
    # STORAGE_TYPE="s3"     → new path: browser uploads directly to
    #                         S3_UPLOAD_BUCKET via presigned PUT URLs
    #                         issued by /upload/presign; ingestion is
    #                         confirmed by /upload/finalize.
    #
    # The legacy /upload (multipart) route is unaffected by this knob;
    # it always uses LocalStorageProvider. Switching this to "s3" only
    # enables the new endpoints — clients choose which path to use.
    STORAGE_TYPE: str = "local"
    S3_UPLOAD_BUCKET: str = ""
    S3_KEY_PREFIX: str = "tenants"
    S3_PRESIGN_PUT_EXPIRY_SECONDS: int = 300   # one-shot, 5 minutes
    S3_PRESIGN_GET_EXPIRY_SECONDS: int = 900   # reserved for Phase 2 view-original

    # ----------------------------------------------------------------
    # AWS / Bedrock — model IDs
    # ----------------------------------------------------------------

    AWS_REGION: str = "us-east-1"

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"
    BEDROCK_HAIKU_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    BEDROCK_SONNET_MODEL: str = "us.anthropic.claude-sonnet-4-6"

    # ----------------------------------------------------------------
    # Database
    # ----------------------------------------------------------------

    DATABASE_URL: str

    # ----------------------------------------------------------------
    # Phase-1 chunking compatibility
    # ----------------------------------------------------------------

    MAX_CHARS: int = 8000
    OVERLAP: int = 300
    BATCH_SIZE: int = 10

    # ----------------------------------------------------------------
    # Phase-2 contextual chunking (accuracy fix: generous limits)
    # ----------------------------------------------------------------

    CHUNK_MAX_CHARS: int = 6000
    CHUNK_MIN_CHARS: int = 200
    # Overlap is applied ONLY on size-based mid-section splits (build_chunks).
    # Structured docs whose chunks are bounded by detected headings are
    # unaffected. 600 ≈ 10% of CHUNK_MAX_CHARS — preserves cross-boundary
    # context without producing duplicate retrieval hits. Set to 0 to disable.
    CHUNK_OVERLAP_CHARS: int = 600
    CHUNK_BATCH_SIZE: int = 10          # ← was 6, now 10 chunks per Haiku call (fewer API calls)

    LLM_CHUNK_FALLBACK_PREVIEW_CHARS: int = 8000
    ENABLE_LLM_CHUNK_FALLBACK: bool = True

    # ----------------------------------------------------------------
    # Ingestion safety caps — protect against huge / malformed uploads.
    # MAX_DOC_PAGES applies to PDFs (page count from fitz).
    # MAX_DOC_CHARS is a pre-chunk projection guard (sum of block chars).
    # MAX_DOC_CHUNKS is enforced after build_chunks; oversized docs raise
    # so we never silently truncate, and never embed a 10k-chunk runbook.
    # ----------------------------------------------------------------
    MAX_DOC_PAGES: int = 1000
    MAX_DOC_CHARS: int = 8_000_000      # ~1.5M tokens worth of source text
    MAX_DOC_CHUNKS: int = 5000

    # ----------------------------------------------------------------
    # Extraction guard — per-page quality check for PDFs.
    # When a page returns suspiciously little text AND contains images,
    # we flag it as a likely scanned page that needs OCR. In SHADOW mode
    # we only LOG the decision (no Textract calls, no cost). Flip
    # OCR_SHADOW_MODE off in a later iteration once we've tuned the
    # threshold against real corpus data.
    # ----------------------------------------------------------------
    EXTRACTION_GUARD_ENABLED: bool = True
    EXTRACTION_GUARD_MIN_CHARS_PER_PAGE: int = 50
    OCR_SHADOW_MODE: bool = True        # log-only; do not call Textract

    # Garbled-text guard — catches PDFs with broken CID fonts where fitz
    # returns characters but they're unreadable (private-use codepoints,
    # replacement chars). Runs only on pages that pass the length check;
    # threshold is the fraction of "good" chars required to NOT flag.
    ENABLE_GARBLED_TEXT_GUARD: bool = True
    GARBLED_TEXT_PRINTABLE_THRESHOLD: float = 0.60

    # ----------------------------------------------------------------
    # Live Textract OCR — only fires when OCR_SHADOW_MODE = False.
    # Per-doc cap caps the worst case where a 1000-page scanned upload
    # would otherwise route every page to Textract. The render scale
    # controls the DPI of the PNG sent to Textract (higher = better
    # OCR accuracy at higher Textract memory usage; 2.0 is the sweet
    # spot for most scanned docs).
    # ----------------------------------------------------------------
    MAX_OCR_PAGES_PER_DOC: int = 100
    TEXTRACT_RENDER_DPI_SCALE: float = 2.0

    # ----------------------------------------------------------------
    # PDF table extraction (fitz.find_tables). When enabled, each PDF
    # page is scanned for table structures AFTER text extraction; any
    # tables found are appended as block_type="table" blocks (matching
    # the existing DOCX table path). Strict mode reduces false positives
    # from code listings and aligned text. Disable to revert to
    # text-only PDF parsing.
    # ----------------------------------------------------------------
    ENABLE_PDF_TABLE_EXTRACTION: bool = True
    PDF_TABLE_STRATEGY: str = "lines_strict"   # fitz find_tables strategy

    # ----------------------------------------------------------------
    # Empty-result fallback chain — runs ONLY after the existing
    # variant-retry path has already failed to return any chunks.
    #
    # Stage 1: Haiku query rewriter generates N alternative phrasings
    #          (semantically different, not just glossary swaps) and
    #          retries the orchestrator on each until one returns hits.
    # Stage 2: BM25-only sweep over the same query — last-ditch lexical
    #          attempt for cases where vector + hybrid still returned 0.
    # Stage 3: Graceful decline. The orchestrator returns a sentinel
    #          RetrievalResult with stats["search_mode"] =
    #          "empty_after_fallback". The chat endpoint MUST honor
    #          this sentinel and return a canned "no info" answer —
    #          NEVER call the LLM with an empty context (that's the
    #          hallucination path we're closing).
    # ----------------------------------------------------------------
    ENABLE_EMPTY_RESULT_FALLBACK: bool = True
    EMPTY_FALLBACK_LLM_REWRITE_COUNT: int = 2     # Haiku rewrites to try
    EMPTY_FALLBACK_BM25_TOP_K: int = 20           # BM25-only sweep top-K
    EMPTY_FALLBACK_DECLINE_MESSAGE: str = (
        "I couldn't find relevant information in the indexed documents "
        "for your question. Try rephrasing, or upload a document that "
        "covers this topic."
    )

    MAX_METADATA_INPUT_CHARS: int = 1800
    MAX_CONTEXT_SUMMARY_CHARS: int = 120
    MAX_OPERATIONAL_LABELS: int = 8
    MAX_METADATA_RETRIES: int = 2

    HAIKU_TEMPERATURE: float = 0.0
    HAIKU_MAX_TOKENS: int = 4096

    # ----------------------------------------------------------------
    # Adaptive ingestion batching — sizes Haiku batches by estimated
    # output tokens so verbose PDFs do not truncate JSON mid-response.
    # ----------------------------------------------------------------

    ADAPTIVE_INGESTION_BATCHING_ENABLED: bool = True
    ADAPTIVE_TARGET_OUTPUT_TOKENS: int = 4000
    ADAPTIVE_BATCH_SIZE_MIN: int = 3
    ADAPTIVE_BATCH_SIZE_MAX: int = 25
    ADAPTIVE_MAX_TOKENS_BUFFER: float = 1.3
    ADAPTIVE_MAX_TOKENS_MIN: int = 2000
    ADAPTIVE_MAX_TOKENS_MAX: int = 8000

    # ----------------------------------------------------------------
    # CSV column-aware parsing
    # ----------------------------------------------------------------
    CSV_COLUMN_AWARE_PARSING_ENABLED: bool = True
    CSV_MAX_ROWS_PER_UPLOAD: int = 50000

    # ----------------------------------------------------------------
    # Dynamic per-organization schema inference
    # ----------------------------------------------------------------
    DYNAMIC_SCHEMA_INFERENCE_ENABLED: bool = True
    SCHEMA_CONFIDENCE_THRESHOLD: float = 0.5

    # ----------------------------------------------------------------
    # Retrieval: prefer latest version of a document family
    # ----------------------------------------------------------------
    PREFER_LATEST_VERSION_IN_RETRIEVAL: bool = True

    # ----------------------------------------------------------------
    # Concurrency — PERFORMANCE TUNED
    # ----------------------------------------------------------------

    METADATA_CONCURRENCY: int = 6       # ← was 4, now 6 parallel Haiku calls
    EMBED_CONCURRENCY: int = 12         # ← was 8, now 12 parallel embed calls

    # ----------------------------------------------------------------
    # Phase-3: Hybrid Retrieval Orchestration
    # ----------------------------------------------------------------

    VECTOR_WEIGHT: float = 0.45
    BM25_WEIGHT: float = 0.30
    KEYWORD_WEIGHT: float = 0.25

    VECTOR_CANDIDATES: int = 25
    BM25_CANDIDATES: int = 20
    KEYWORD_CANDIDATES: int = 15

    FTS_MIN_RANK: float = 0.01
    EXACT_TERM_BOOST: float = 2.0
    RRF_K: int = 60

    RERANKER_BACKEND: str = "llm"
    RERANK_TOP_K: int = 10
    RERANK_CANDIDATES: int = 20
    RERANK_SCORE_WEIGHT: float = 0.70
    RERANK_FUSION_WEIGHT: float = 0.30

    KEYWORD_QUERY_THRESHOLD: float = 0.60
    SEMANTIC_QUERY_THRESHOLD: float = 0.60
    ENABLE_QUERY_CLASSIFICATION: bool = True

    ENABLE_METADATA_FILTER: bool = True
    METADATA_FILTER_CANDIDATES: int = 10

    MIN_GROUNDING_SCORE: float = 0.18
    MIN_KEYWORD_OVERLAP: int = 1

    # ----------------------------------------------------------------
    # Phase-4: Model Routing
    # ----------------------------------------------------------------

    ENABLE_MODEL_ROUTING: bool = True
    ROUTING_DEFAULT_MODEL: str = "haiku"

    COMPLEXITY_SIMPLE_THRESHOLD: float = 0.30
    COMPLEXITY_COMPLEX_THRESHOLD: float = 0.70

    SONNET_MAX_TOKENS: int = 4096
    SONNET_TEMPERATURE: float = 0.1
    SONNET_MONTHLY_BUDGET_USD: float = 50.0

    HAIKU_ANSWER_MAX_TOKENS: int = 2048
    HAIKU_ANSWER_TEMPERATURE: float = 0.1

    SESSION_CONTEXT_MAX_MESSAGES: int = 4
    SESSION_CONTEXT_MAX_CHARS: int = 2000
    INCLUDE_METADATA_IN_PROMPT: bool = True
    INCLUDE_CONFIDENCE_IN_PROMPT: bool = True

    CX_WEIGHT_MULTI_STEP: float = 0.30
    CX_WEIGHT_REASONING: float = 0.25
    CX_WEIGHT_CONTEXT_SIZE: float = 0.15
    CX_WEIGHT_LOW_CONFIDENCE: float = 0.20
    CX_WEIGHT_MULTI_DOC: float = 0.10

    # ----------------------------------------------------------------
    # Phase-5: Multi-Agent Troubleshooting
    # ----------------------------------------------------------------

    ENABLE_AGENT_MODE: bool = True
    AGENT_COMPLEXITY_THRESHOLD: float = 0.65
    AGENT_MIN_SOURCES: int = 1
    AGENT_MAX_STEPS: int = 4

    AGENT_PLANNER_MODEL: str = "sonnet"
    AGENT_ANALYSIS_MODEL: str = "haiku"
    AGENT_COMPOSER_MODEL: str = "haiku"

    AGENT_PLANNER_MAX_TOKENS: int = 1024
    AGENT_ANALYSIS_MAX_TOKENS: int = 1500
    AGENT_COMPOSER_MAX_TOKENS: int = 2048

    AGENT_MAX_TOTAL_TOKENS: int = 8000
    AGENT_TIMEOUT_SECONDS: int = 45

    # ----------------------------------------------------------------
    # Goal 4: Dynamic agent token budget
    # ----------------------------------------------------------------
    # Static AGENT_MAX_TOTAL_TOKENS above is a legacy fallback. When
    # ENABLE_DYNAMIC_AGENT_BUDGET is True, the orchestrator computes
    # max_total = CLARIFIER + PLANNER + steps*PER_STEP + COMPOSER after
    # the Planner returns its step list, clamped by HARD_CAP so a runaway
    # plan can't explode cost. Analyst stops early if remaining budget
    # would dip below the Composer reserve.

    AGENT_BUDGET_PER_STEP_TOKENS: int = 3500
    AGENT_BUDGET_PLANNER_TOKENS: int = 1500
    AGENT_BUDGET_COMPOSER_TOKENS: int = 3000
    AGENT_BUDGET_CLARIFIER_TOKENS: int = 500
    AGENT_BUDGET_HARD_CAP_TOKENS: int = 20000
    ENABLE_DYNAMIC_AGENT_BUDGET: bool = True

    ENABLE_CLARIFIER: bool = True
    AGENT_CLARIFIER_MODEL: str = "haiku"
    AGENT_CLARIFIER_MAX_TOKENS: int = 400
    CLARIFIER_MIN_QUERY_LEN: int = 3
    CLARIFIER_MAX_QUESTIONS: int = 2

    # ----------------------------------------------------------------
    # Tier 2 aggregation intent classifier (Brief 2)
    # ----------------------------------------------------------------
    # When regex detect_aggregation_intent misses, fall through to a
    # Haiku classifier that reads one-shot intent off the query text.
    # Runtime-toggleable: flip AGGREGATION_CLASSIFIER_ENABLED=False to
    # fall back to regex-only behavior without a deploy.
    AGGREGATION_CLASSIFIER_ENABLED: bool = True
    AGGREGATION_CLASSIFIER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    AGGREGATION_CLASSIFIER_MAX_TOKENS: int = 300
    AGGREGATION_CLASSIFIER_TEMPERATURE: float = 0.0  # deterministic
    AGGREGATION_CLASSIFIER_MIN_CONFIDENCE: float = 0.6
    AGGREGATION_CLASSIFIER_TIMEOUT_SECONDS: float = 4.0

    # ----------------------------------------------------------------
    # Same-session query rewriter (Brief 3)
    # ----------------------------------------------------------------
    # Resolves pronouns, ordinals, ellipsis, and filter swaps against
    # the last few turns BEFORE aggregation detect / retrieval run, so
    # "what caused it?" after "tell me about INC-10015" becomes
    # "What caused INC-10015?" automatically. Toggle the enabled flag
    # off to revert to pre-Brief-3 pipeline behavior without a deploy.
    QUERY_REWRITER_ENABLED: bool = True
    QUERY_REWRITER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    QUERY_REWRITER_MAX_TOKENS: int = 200
    QUERY_REWRITER_TEMPERATURE: float = 0.0
    QUERY_REWRITER_TIMEOUT_SECONDS: float = 3.5
    QUERY_REWRITER_MAX_HISTORY_TURNS: int = 8
    QUERY_REWRITER_MIN_QUERY_LEN_FOR_REWRITE: int = 2

    # ----------------------------------------------------------------
    # Brief 4 — LLM optimization sprint
    # ----------------------------------------------------------------
    # Opt 1 — Query-rewriter micro-cache
    QUERY_REWRITER_CACHE_ENABLED: bool = True
    QUERY_REWRITER_CACHE_TTL_SECONDS: int = 60
    QUERY_REWRITER_CACHE_MAX_ENTRIES: int = 1000

    # Opt 2 — Parallel rewriter + aggregation classifier
    PARALLEL_REWRITE_AND_CLASSIFY_ENABLED: bool = True

    # Opt 3 — Merged triage classifier (complexity + intent + mode in one call)
    MERGED_TRIAGE_ENABLED: bool = True
    MERGED_TRIAGE_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    MERGED_TRIAGE_MAX_TOKENS: int = 150
    MERGED_TRIAGE_TIMEOUT_SECONDS: float = 3.0

    # Opt 4 — Skip reranker on tiny result sets
    SKIP_RERANK_ON_TINY_RESULTS_ENABLED: bool = True
    RERANK_MIN_CHUNKS: int = 3

    # Opt 5 — Context compression for generation
    CONTEXT_COMPRESSION_ENABLED: bool = True
    CONTEXT_COMPRESSION_MIN_CHUNK_CHARS: int = 5000

    # ----------------------------------------------------------------
    # Brief 5 — Cost optimization sprint
    # ----------------------------------------------------------------
    # Part 1 — Semantic Answer Cache
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_SIMILARITY_THRESHOLD: float = 0.94
    SEMANTIC_CACHE_MIN_CONFIDENCE_TO_CACHE: float = 0.75
    SEMANTIC_CACHE_TTL_DAYS: int = 7
    SEMANTIC_CACHE_MAX_ENTRIES: int = 50000
    SEMANTIC_CACHE_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    SEMANTIC_CACHE_LOG_HITS: bool = True

    # Part 2 — Response Token Caps
    RESPONSE_TOKEN_CAPS_ENABLED: bool = True
    RESPONSE_TOKENS_CLASSIFICATION: int = 30
    RESPONSE_TOKENS_SHORT_FACT: int = 120
    # Sprint 2.7 Bug D — was 350; raised to 600 so multi-section
    # explanations (hardware + application + network) fit in one Haiku
    # call instead of triggering the walkthrough-reclassification retry.
    RESPONSE_TOKENS_EXPLANATION: int = 600
    RESPONSE_TOKENS_WALKTHROUGH: int = 600
    RESPONSE_TOKENS_ANALYTICAL: int = 1200
    RESPONSE_TOKENS_DEFAULT: int = 350

    # Part 3 — Input Guardrails + Output Sanitizer
    INPUT_GUARDRAILS_ENABLED: bool = True
    INPUT_GUARDRAIL_REGEX_BLOCK: bool = True
    INPUT_GUARDRAIL_CLASSIFIER_ENABLED: bool = True
    INPUT_GUARDRAIL_CLASSIFIER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    INPUT_GUARDRAIL_CLASSIFIER_MAX_TOKENS: int = 80
    INPUT_GUARDRAIL_CLASSIFIER_MIN_CONFIDENCE: float = 0.75
    INPUT_GUARDRAIL_CLASSIFIER_TIMEOUT_SECONDS: float = 2.0
    INPUT_GUARDRAIL_SCRUB_PII: bool = True
    OUTPUT_SANITIZER_ENABLED: bool = True
    OUTPUT_SANITIZER_SCRUB_PII: bool = True

    # ----------------------------------------------------------------
    # Two-bug fix brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Bug 1: classifier-path group_by_customer now defaults to
    # ranking_direction="highest" (most-first). Flip to False only if
    # a regression surfaces on "fewest/lowest" phrasings.
    GROUP_BY_DEFAULT_DESC_ENABLED: bool = True
    # Bug 2: carry original-query identifiers as a scope-lock across
    # all Analyst sub-steps so per-step retrieval stays grounded in
    # the same tickets the user asked about. Flip to False to revert
    # to step-text-only retrieval.
    ANALYST_IDENTIFIER_LOCK_ENABLED: bool = True
    # Bug 3: aggregation classifier is stateless — filters it returns must
    # be textually present in the query. Strips hallucinated customer /
    # priority / sla / component filters that bleed from prior turns.
    # Flip to False to allow raw classifier output without the guard.
    AGG_CLASSIFIER_STATELESS_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Rewriter + regex fix brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Issue 1: defensive guard that rejects rewriter output which injects
    # filters (priority, SLA, customer names) into already-self-contained
    # queries. Flip to False to disable the safety check without a deploy.
    REWRITER_SAFETY_GUARD_ENABLED: bool = True
    # Issue 2: regex aggregation classifier strips P1..P4 priority tokens
    # before running the customer-name regex and rejects non-customer
    # tokens (P1, SLA, SBC, etc.) so "How many P1 tickets?" no longer
    # captures "P1" as a customer name. Flip to False to revert.
    REGEX_PRIORITY_AWARE_PARSING_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Four-bug fix brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Bug 1: refuse queries where scrubbed PII was the lookup key (e.g.
    # "Find tickets for SSN 123-45-6789"). Incidental PII still scrubs.
    PII_CENTRAL_REFUSAL_ENABLED: bool = True
    # Bug 2: apply the customer/priority/sla filters to the ENTIRE ranking
    # result set, not just the top row, so "lowest Nebula-Corp quality
    # score" returns Nebula-Corp-only tickets across top-1 + others-near.
    RANKING_CUSTOMER_SCOPE_FIX_ENABLED: bool = True
    # Bug 3: first-class rework_detected filter threaded classifier →
    # AggIntent → SQL. Requires ingestion to populate metadata_json's
    # rework_detected key; the clause is safe when the field is absent.
    REWORK_FILTER_ENABLED: bool = True
    # Bug 4A: reject the aggregation fast-path when the query mentions
    # content-only keywords (brand names, change-request, "involved",
    # etc.) that need chunk-text retrieval instead of metadata SQL.
    AGG_CONTENT_FILTER_REJECT_ENABLED: bool = True
    # Bug 4B: enable avg/sum/min/max numeric aggregation operations over
    # whitelisted metadata_json numeric fields.
    NUMERIC_AGGREGATION_OPS_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Final cleanup brief — feature flags for safe rollback
    # ----------------------------------------------------------------
    # Bug 1: populate metadata_json->>'rework_detected' during gold-ticket
    # ingestion so the rework SQL filter can actually match rows. Flipping
    # to False doesn't un-ingest existing rows, but it stops the field
    # from being written on subsequent uploads.
    INGEST_REWORK_METADATA_ENABLED: bool = True
    # Bug 2: thread priority/sla_met/rework_detected/component filters
    # through _run_group_by_customer, mirroring the ranking fix. Without
    # this, "Among P1 tickets, which customers had the most SLA misses?"
    # returns the unfiltered P1 distribution.
    GROUP_BY_FULL_FILTER_ENABLED: bool = True
    # Bug 3: multi-strategy JSON parsing for Mistral reranker output so
    # "Extra data" edge cases recover the ranking instead of silently
    # dropping to fusion order. Fallback behavior unchanged on total
    # parse failure — we just reduce WHEN that fallback fires.
    RERANKER_ROBUST_PARSE_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Interactive Clarifier (Brief 6) — feature flags
    # ----------------------------------------------------------------
    # Presents 3-4 clickable clarification options when retrieval surfaces
    # multiple distinct candidates with no clear winner. Runs AFTER retrieval,
    # BEFORE agent pipeline / full RAG generation. Falls through safely on
    # any error so existing behavior is preserved when the flag is off.
    INTERACTIVE_CLARIFIER_ENABLED: bool = True
    INTERACTIVE_CLARIFIER_AMBIGUITY_THRESHOLD: float = 0.6
    INTERACTIVE_CLARIFIER_MAX_PER_SESSION: int = 3
    INTERACTIVE_CLARIFIER_MAX_TOKENS: int = 600
    INTERACTIVE_CLARIFIER_TIMEOUT_SECONDS: float = 4.0
    INTERACTIVE_CLARIFIER_MODEL: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

    # ----------------------------------------------------------------
    # Goal 2: Schema-agnostic identifier extraction
    # ----------------------------------------------------------------
    # Config-only extensibility — add new schemas here WITHOUT code changes.
    #
    # IDENTIFIER_PATTERNS maps id_type → regex. The id_type string MUST
    # match what the ingestion schema writes into chunks.metadata_json
    # ['id_type'] (see STRUCTURED_SCHEMAS in contextual_ingestion_service).
    # Each regex must have exactly ONE capture group yielding the raw
    # identifier body (digits or the whole token). Patterns are evaluated
    # in declaration order; earlier matches own their span so later
    # patterns can't re-tag the same token (prevents 'INC-10005' from
    # being misclassified as an issue_key).
    IDENTIFIER_PATTERNS: Dict[str, str] = {
        "ticket_number": r"\bINC[-\s]?(\d{3,})\b",
        "issue_key":     r"\b([A-Za-z][A-Za-z0-9]+-\d+)\b",
    }
    # IDENTIFIER_CANONICAL_FORMAT maps id_type → str.format template. The
    # capture group from the matching regex is passed as positional arg 0.
    # The canonical string is UPPER()-compared against
    # chunks.metadata_json->>'primary_id', so formats must match whatever
    # the ingestion schema wrote at ingest time.
    IDENTIFIER_CANONICAL_FORMAT: Dict[str, str] = {
        "ticket_number": "INC-{0}",
        "issue_key":     "{0}",
    }

    # ----------------------------------------------------------------
    # Phase-6: Validation Guardrails & Confidence Scoring
    # ----------------------------------------------------------------

    ENABLE_ANSWER_VALIDATION: bool = True

    CONF_WEIGHT_RETRIEVAL: float = 0.30
    CONF_WEIGHT_COVERAGE: float = 0.25
    CONF_WEIGHT_GROUNDING: float = 0.25
    CONF_WEIGHT_CONSISTENCY: float = 0.20

    VALIDATION_MIN_CONFIDENCE: float = 0.35
    # Tightened from 0.25 → 0.40 in response to the manual-eval feedback:
    # several Q's (Q3 timer hallucination, Q10 CLI expansion) passed the
    # looser 0.25 gate despite containing fabricated specifics that the
    # grounding checker only weakly penalised. The prompt-side
    # "DO NOT FABRICATE SPECIFICS" directive does the prevention work;
    # this threshold raise catches what still gets through.
    VALIDATION_MIN_GROUNDING: float = 0.40
    VALIDATION_MIN_COVERAGE: float = 0.20

    # ----------------------------------------------------------------
    # Response Relevancy gate — Haiku scores "does this answer address
    # this question?" post-generation. Independent of grounding
    # (grounded but off-topic still fails). Fails OPEN on Haiku error
    # so the judge being unreachable doesn't block legitimate answers.
    # ----------------------------------------------------------------
    # ----------------------------------------------------------------
    # KB-search system prompt — activates ONLY when the request's
    # allowed_doc_kinds is a non-empty subset of {kb, sop}. Adds the
    # RAG Knowledge Architect addendum to the existing Claude system
    # prompt. Set to false to disable without touching code.
    # ----------------------------------------------------------------
    ENABLE_KB_SEARCH_PROMPT: bool = True

    ENABLE_RELEVANCY_CHECK: bool = True
    MIN_RELEVANCY_SCORE: float = 0.50          # 0.0-1.0; below = off-topic
    RELEVANCY_MAX_TOKENS: int = 128            # Haiku output budget for the score
    RELEVANCY_FALLBACK_MESSAGE: str = (
        "- The generated answer did not appear to directly address your "
        "question.\n"
        "- Please try rephrasing — make sure the question is specific and "
        "covered by the uploaded content."
    )

    # ----------------------------------------------------------------
    # LLM hard timeout + fast-model fallback
    #
    # Previous boto3 config used read_timeout=120 and max_attempts=10.
    # Worst-case user wait was 10 × 120s = 20 minutes for a hung call.
    # These knobs replace that with a tight per-attempt timeout plus a
    # capped retry count, and add a fast-model fallback (Haiku 4.5)
    # when the primary model (Mistral) times out or throttles.
    #
    # Wall-clock worst case after this iteration:
    #   primary attempts (3 × 45s = 135s) + Haiku fallback (~30s) ≈ 165s
    # Typical case: ~10–20s (no retry needed).
    #
    # Fails graceful: if both primary AND Haiku error, returns
    # LLM_FALLBACK_DECLINE_MESSAGE rather than hanging the user.
    # ----------------------------------------------------------------
    LLM_TIMEOUT_FALLBACK_ENABLED: bool = True
    LLM_READ_TIMEOUT_S: int = 45               # per-attempt socket read timeout
    LLM_CONNECT_TIMEOUT_S: int = 10            # TCP connect timeout
    LLM_MAX_ATTEMPTS: int = 3                  # boto3 max_attempts (down from 10)
    LLM_HAIKU_FALLBACK_MAX_TOKENS: int = 2048  # output budget on fallback path
    LLM_FALLBACK_DECLINE_MESSAGE: str = (
        "- The system is taking longer than expected to respond. Please "
        "try again in a moment, or rephrase your question."
    )

    VALIDATION_HALLUCINATION_PHRASES: List[str] = [
        "as an AI",
        "I don't have access",
        "based on my training",
        "in general",
        "typically",
        "it is commonly known",
        "from my knowledge",
        "as of my last update",
    ]

    VALIDATION_WARN_SUPERSEDED: bool = True
    VALIDATION_SUPERSEDED_PENALTY: float = 0.20

    VALIDATION_MAX_RETRIES: int = 1
    VALIDATION_RETRY_EXPAND_K: int = 3

    ENABLE_EVAL_LOGGING: bool = True
    EVAL_LOG_FILE: Optional[str] = None

    # ----------------------------------------------------------------
    # Feature flags
    # ----------------------------------------------------------------

    ENABLE_METADATA_EXTRACTION: bool = True
    ENABLE_VERSION_DETECTION: bool = True
    ENABLE_DUPLICATE_CHECK: bool = True
    ENABLE_CHUNK_SUMMARY: bool = True
    ENABLE_TABLE_PARSING: bool = True
    ENABLE_CODE_BLOCK_DETECTION: bool = True
    INCLUDE_OLD_VERSIONS: bool = False

    # ----------------------------------------------------------------
    # API
    # ----------------------------------------------------------------

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"
    # Observability output format. ``json`` emits newline-delimited
    # JSON for CloudWatch Logs Insights ingestion; ``text`` emits a
    # compact human-readable line that's friendlier to ``docker logs``
    # tailing on a laptop. Both formats carry the same fields
    # (request_id, user_id, route, status, duration_ms) so swapping
    # is purely a deployment knob — application code never branches.
    LOG_FORMAT: str = "json"

    # ----------------------------------------------------------------
    # Sentry — error monitoring + performance tracing
    # ----------------------------------------------------------------
    # ``SENTRY_DSN`` is the only required value; when empty,
    # ``init_sentry()`` is a no-op so the app boots normally without
    # Sentry. The other three are advisory:
    #   * SENTRY_ENV       — labels events (``dev``, ``staging``, ``prod``)
    #   * SENTRY_RELEASE   — usually ``BUILD_TIMESTAMP`` from the deploy
    #                         pipeline so each release is tagged.
    #   * SENTRY_TRACES_SAMPLE_RATE — 0.0–1.0 fraction of requests that
    #                         get a performance trace; 0.1 = 10%.
    SENTRY_DSN: Optional[str] = None
    SENTRY_ENV: str = "dev"
    SENTRY_RELEASE: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = 0.1

    # ----------------------------------------------------------------
    # DB connection pool — sized per worker process.
    # ----------------------------------------------------------------
    # The total DB connections consumed by the deployment is
    # ``(pool_size + max_overflow) × workers × containers + worker_pool``.
    # This MUST stay below the RDS instance's ``max_connections``
    # (minus a small reservation for admin sessions). Currently:
    #   RDS max_connections = 80  (db.t3.small / db.t4g.micro class)
    #   Reservation         = 10  (psql sessions + future SQS worker)
    #   Available           = 70
    #
    # Defaults below assume a single-worker, single-container laptop /
    # dev configuration. Production overrides via env vars (see
    # docker-compose.ec2.yml / Terraform task definitions):
    #   DB_POOL_SIZE=3, DB_MAX_OVERFLOW=5  → 8 conns/worker
    #   At 4 workers × 2 containers = 64 conns + 6 admin = fits in 80.
    #
    # When the RDS class is upgraded (Phase 3 prereq → db.t3.medium with
    # max_connections ~150) these can return to the looser laptop
    # values without touching code.
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    # Recycle stale connections after this many seconds. 1800 = 30 min,
    # comfortably below the RDS server-side idle disconnect (~8 h)
    # while still letting connections live long enough to amortise
    # TLS handshake cost.
    DB_POOL_RECYCLE: int = 1800
    # Ping a connection before checking it out of the pool — costs a
    # microsecond per checkout, pays for itself the first time the
    # network blip happens.
    DB_POOL_PRE_PING: bool = True

    # ----------------------------------------------------------------
    # Glossary store — per-process in-memory cache refreshed from
    # ``document_metadata`` + ``chunks`` every N seconds. Postgres is
    # the source of truth; this knob controls how stale the cache can
    # be. Bounded drift across replicas is acceptable for acronym
    # expansion (no one expects a new acronym to propagate in <60 s).
    # Set to 0 to disable the refresher (tests, single-shot tooling).
    # ----------------------------------------------------------------
    GLOSSARY_REFRESH_SECONDS: int = 300

    # ----------------------------------------------------------------
    # Retrieval — BM25 → FTS migration (Phase 0.1).
    # ----------------------------------------------------------------
    # The in-process BM25 index is per-replica state that doesn't
    # survive horizontal scaling cleanly. Postgres FTS (migration 045
    # added a STORED tsvector column + GIN index on chunks) replaces
    # it. During the 2-week shadow window we run BOTH channels and
    # log the diff to ``retrieval_eval``; once the eval set confirms
    # FTS is within tolerance, flip RETRIEVAL_BM25_ENABLED to false.
    #
    # 30 days after the flip the BM25 module is deleted from the
    # codebase entirely — no fallback flag, no dead code.
    RETRIEVAL_BM25_ENABLED: bool = True
    # Shadow-mode diff logging — separate flag so we can stop the
    # writes after the verdict is in, without removing the FTS path.
    RETRIEVAL_SHADOW_LOG_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Ingestion routing — Phase 5 worker migration safety net.
    # ----------------------------------------------------------------
    # ``true``  (production posture, set explicitly in Fargate task
    #            definitions): /upload + /upload/finalize enqueue an
    #            ``ingest_document`` job; the worker container's
    #            ingest service claims and processes it. No in-process
    #            work on the API tier.
    #
    # ``false`` (default — local dev, single-uvicorn deployments):
    #            Routes use the legacy
    #            ``background_tasks.add_task(index_file_job, ...)``
    #            path. Heavy work runs in the API process. This is
    #            the only safe default for local dev because running
    #            a worker process is an EXPLICIT opt-in — without it,
    #            queued jobs would sit pending forever and the
    #            frontend would poll indefinitely.
    INGESTION_VIA_WORKER: bool = False

    # ----------------------------------------------------------------
    # Report routing — same pattern as ingestion. RCA + Gap Analysis
    # generations are heavy LLM calls (30 s – 4 min). When
    # ``REPORTS_VIA_WORKER=true`` the API enqueues a job and returns
    # 202 + job_id; a worker container claims and runs the LLM.
    # When ``false`` (the local-dev default), the LLM runs in-process
    # via ``asyncio.to_thread`` — identical to the pre-Phase-1
    # behaviour the local stack expects.
    # ----------------------------------------------------------------
    REPORTS_VIA_WORKER: bool = False

    REQUEST_TIMEOUT: int = 30
    API_KEY: Optional[str] = None
    UI_API_KEY: Optional[str] = None
    API_BASE: str = "http://localhost:8000"

    # ----------------------------------------------------------------
    # Clerk authentication
    # ----------------------------------------------------------------

    CLERK_SECRET_KEY: Optional[str] = None
    CLERK_PUBLISHABLE_KEY: Optional[str] = None
    CLERK_ENABLED: str = "false"

    # ----------------------------------------------------------------
    # SES email
    # ----------------------------------------------------------------

    SES_SENDER_EMAIL: str = "noreply@acadiaconsultants.com"
    SES_FEEDBACK_RECIPIENT: str = "dev@acadiaconsultants.com"
    SES_REGION: Optional[str] = None
    SES_ENABLED: str = "true"

    # ----------------------------------------------------------------
    # Pattern Analytics (Layer 3 selective stats engine)
    # ----------------------------------------------------------------
    PATTERN_ANALYTICS_ENABLED: bool = True
    PATTERN_ANALYTICS_MIN_SIMILAR_TICKETS: int = 3
    PATTERN_ANALYTICS_CONFIDENCE_THRESHOLD: float = 0.7
    PATTERN_ANALYTICS_SIMILARITY_THRESHOLD: float = 0.65
    PATTERN_ANALYTICS_RECENCY_WINDOW_DAYS: int = 30
    PATTERN_ANALYTICS_CACHE_TTL_HOURS: int = 24
    PATTERN_ANALYTICS_TOP_N_ACTIONS: int = 3
    PATTERN_ANALYTICS_FORCE_ENABLE_IN_TROUBLESHOOTING_MODE: bool = True

    # ----------------------------------------------------------------
    # Rich Response Formatting Brief (Approach A — pure markdown)
    # ----------------------------------------------------------------
    # Master flag for rich formatting prompt guidance. When enabled, backend
    # adds a markdown formatting addendum to the Claude system prompt, telling
    # the LLM when to use tables, code blocks, bold, inline code, blockquotes.
    # When False, the prompt is byte-identical to the pre-brief conversational
    # engineer voice (zero behavioral change).
    RICH_FORMATTING_PROMPT_ENABLED: bool = True

    # When True, the grounding rules emitted by context_builder include the
    # "Formatting guidance" block that tells the model when to reach for
    # markdown tables / fenced code / bold. When False, only the original
    # answering guidance is emitted. Independent of the system-prompt flag
    # so the two prompt layers can be A/B tested.
    RICH_FORMATTING_GROUNDING_RULES_ENABLED: bool = True

    # ----------------------------------------------------------------
    # Pattern Analytics Polish Brief — 3 surgical fixes, each flag-gated.
    # ----------------------------------------------------------------
    # Fix 1 — queries.load_similar_tickets_for_topic splits topic into
    # individual terms and OR-matches tsvector + ILIKE across summary /
    # resolution_text / component / chunk content. Flip False to fall
    # back to the legacy ILIKE-OR-on-chunks path preserved under
    # _load_similar_tickets_legacy.
    PATTERN_ANALYTICS_SQL_OR_MATCHING_ENABLED: bool = True

    # Fix 2 — when pattern_context is active, orchestrator caps planner
    # steps, bumps the dynamic budget, and reserves composer tokens so
    # the Composer never gets starved. All other (non-pattern) queries
    # use the legacy dynamic budget exactly as before.
    PATTERN_ANALYTICS_BUDGET_TUNING_ENABLED: bool = True
    PATTERN_ANALYTICS_MAX_PLANNER_STEPS: int = 3
    PATTERN_ANALYTICS_AGENT_BUDGET: int = 25000
    PATTERN_ANALYTICS_COMPOSER_RESERVE: int = 4000

    # Fix 3 — Luhn-validate candidate credit-card digit sequences before
    # redacting. Prevents ticket IDs / long numeric identifiers from
    # being falsely masked as PII. Only affects the credit_card PII type
    # — SSN, email, AWS keys, private keys are untouched.
    PII_CREDIT_CARD_LUHN_VALIDATION_ENABLED: bool = True

    # ────────────────────────────────────────────────
    # Rich Formatting Polish — Fix 5: Agent Composer Markdown
    # ────────────────────────────────────────────────
    #
    # When True, the multi-agent Composer's system prompt receives a
    # markdown formatting addendum instructing it to use tables for
    # compare queries, bold for identifiers, fenced code blocks for
    # commands, etc. When False, Composer behaves exactly as before
    # (prose-heavy synthesis, no markdown instructions) — the prompt
    # is byte-for-byte identical to the pre-fix composer prompt.
    #
    # Rollback: set to False, restart backend. Zero impact on other flags.
    AGENT_COMPOSER_MARKDOWN_ENABLED: bool = True

    # ────────────────────────────────────────────────
    # Cross-Cutting Analytical Router
    # ────────────────────────────────────────────────
    #
    # When True, analytical queries ("common root causes", "recurring
    # patterns", "most frequently recommended improvements") are detected
    # before the SQL aggregation classifier and routed to the agent
    # pipeline for proper content synthesis. When False, all queries flow
    # through existing agg_classifier exactly as before.
    CROSS_CUTTING_ANALYTICAL_ROUTING_ENABLED: bool = True

    # Minimum confidence to route as analytical (0.0-1.0).
    # Higher = fewer false positives, some analytical queries miss.
    # Lower = more analytical routing, some SQL queries misrouted.
    CROSS_CUTTING_DETECTOR_CONFIDENCE_THRESHOLD: float = 0.75

    # Minimum distinct signal categories required (prevents single-word
    # triggers like "list all tickets" with only one scope signal).
    CROSS_CUTTING_MIN_SIGNAL_CATEGORIES: int = 2

    # Fast mode ticket limit (top reranked tickets the analyst reads).
    CROSS_CUTTING_FAST_MODE_TICKET_LIMIT: int = 20

    # Deep mode ticket limit (all matching, capped for cost).
    CROSS_CUTTING_DEEP_MODE_TICKET_LIMIT: int = 50

    # Analytical result cache TTL in seconds (1 hour default).
    CROSS_CUTTING_CACHE_TTL_SECONDS: int = 3600

    # ────────────────────────────────────────────────
    # Agent Pipeline Performance Tuning
    # Additive flags — all existing budget/flag behavior preserved when
    # these four are flipped False.
    # ────────────────────────────────────────────────

    # Budget ceiling for analytical + compare queries (cross-cutting, deep
    # analysis). Separate from PATTERN_ANALYTICS_AGENT_BUDGET (pattern-specific
    # queries). Higher ceiling prevents Composer truncation on complex work.
    AGENT_ANALYTICAL_BUDGET: int = 25000

    # Concurrency limit for parallel analyst step execution.
    # 3 = reasonable balance of speed vs Bedrock rate limits.
    ANALYST_PARALLEL_CONCURRENCY: int = 3

    # Run independent analyst steps in parallel (wave-based).
    # Dependent steps (synthesize/combine/cross-reference) stay sequential.
    ANALYST_PARALLEL_EXECUTION_ENABLED: bool = True

    # Cache retrieval results keyed by identifier list within one pipeline run.
    AGENT_RETRIEVAL_CACHE_ENABLED: bool = True

    # Early-skip clarifier when identifiers/high-confidence retrieval/simple
    # patterns make clarification unnecessary.
    CLARIFIER_EARLY_SKIP_ENABLED: bool = True

    # ────────────────────────────────────────────────
    # Dynamic Agent Budget
    # Replaces static AGENT_ANALYTICAL_BUDGET for analytical queries with a
    # formula: base + (tickets × per_ticket) + (steps × per_step), × deep
    # multiplier when applicable, clamped to [floor, ceiling].
    # AGENT_ANALYTICAL_BUDGET (25000) is preserved as the fallback used
    # when AGENT_DYNAMIC_BUDGET_ENABLED is False.
    # ────────────────────────────────────────────────

    # Master flag — when True, analytical queries use formula-based dynamic
    # budget. When False, falls back to static AGENT_ANALYTICAL_BUDGET for
    # byte-identical pre-brief behavior.
    AGENT_DYNAMIC_BUDGET_ENABLED: bool = True

    # Base synthesis budget (Composer minimum — prevents truncation floor).
    AGENT_DYNAMIC_BUDGET_BASE: int = 8000

    # Per-ticket overhead (synthesis tokens needed per ticket analyzed).
    AGENT_DYNAMIC_BUDGET_PER_TICKET: int = 1500

    # Per-step overhead (planner step contribution to final synthesis).
    AGENT_DYNAMIC_BUDGET_PER_STEP: int = 2000

    # Floor — minimum viable budget regardless of inputs.
    AGENT_DYNAMIC_BUDGET_FLOOR: int = 10000

    # Ceiling — hard cap to prevent runaway cost on huge queries.
    AGENT_DYNAMIC_BUDGET_CEILING: int = 35000

    # Multiplier for deep analysis mode (explicit user request for depth).
    AGENT_DYNAMIC_BUDGET_DEEP_MULTIPLIER: float = 1.5

    # ─────────────────────────────────────────────────────────────
    # Guided Workflow (PRD: User Journey - LogIQ Landing Page Guidance)
    # Sprint 1 — session state foundation + landing page shell.
    # ─────────────────────────────────────────────────────────────

    # When True, a session with a locked mode rejects /ask queries that
    # look clearly cross-mode. Sprint 1 keeps this OFF (soft mode); the
    # real enforcement arrives in Sprint 2 once context-break detection
    # is wired. Flag exists now so Sprint 1 is forward-compatible.
    MODE_LOCK_STRICT: bool = False

    # Sprint 2 placeholder — regex-based context-break phrase detection.
    # Declared now so Sprint 1 deploys don't need a config reload later.
    CONTEXT_BREAK_DETECTION_ENABLED: bool = False

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.5 — Retrieval & Pattern Hotfix tunables. The three
    # HOTFIX_* values below are defaults; operators may override
    # them in .env if needed.
    # ─────────────────────────────────────────────────────────────
    HOTFIX_VOCAB_MIN_OCCURRENCE: int = 1
    HOTFIX_SEMANTIC_CACHE_THRESHOLD: float = 0.985
    HOTFIX_COMPOSER_RESERVE_TOKENS: int = 5000

    # ─────────────────────────────────────────────────────────────
    # Production Retrieval Fix v2 — 5 deeper architectural bug fixes.
    # ─────────────────────────────────────────────────────────────

    # Bug #2: JSON-tree-aware vocabulary learning — keys become
    # field_names, values matching identifier pattern become
    # identifiers, short repeated uppercase values become enums.
    VOCABULARY_JSON_STRUCTURAL_PARSING: bool = True
    VOCABULARY_ENUM_MIN_OCCURRENCE: int = 3

    # Bug #1: identifier extractor consults learned_vocabulary and
    # skips tokens classified as 'field_name' so
    # Resolution_Quality_Score isn't treated as a ticket ID.
    IDENTIFIER_VOCAB_TYPE_CHECK: bool = True

    # Bug #5: multi-column exact lookup across incident_number /
    # vector_id / external_incident_id + suffix stripping so
    # INC-ALPHA-001_SEMANTIC_UNIT maps to INC-ALPHA-001.
    IDENTIFIER_MULTI_COLUMN_LOOKUP: bool = True
    IDENTIFIER_SUFFIX_STRIP_ENABLED: bool = True

    # Bug #4: reserve Composer tokens in a separate pool so the
    # Analyst can't drain the budget and starve the synthesis step.
    AGENT_BUDGET_SEPARATE_COMPOSER_POOL: bool = True

    # Bug #3: admin-only ingestion verification + reindex endpoints.
    INGESTION_VERIFICATION_ENABLED: bool = True

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.7 — Accuracy tunables. HOTFIX_EXPLANATION_TOKENS_CAP
    # lets operators retune the explanation length cap.
    # ─────────────────────────────────────────────────────────────
    HOTFIX_EXPLANATION_TOKENS_CAP: int = 600

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.8 — Compound filter tunable. RERANK_TOP_K_AGGREGATION
    # controls the aggregation-intent reranker ceiling.
    # ─────────────────────────────────────────────────────────────
    RERANK_TOP_K_AGGREGATION: int = 40

    # ─────────────────────────────────────────────────────────────
    # Sprint 2.8.1 — Rewriter ellipsis tightening
    # LOGIQ_REWRITER_STRICT_ELLIPSIS gates the TRUE-ellipsis
    # structural requirement (referential pronoun / sentence
    # conjunction / short fragment without named entity).
    # REWRITER_ELLIPSIS_MIN_CONF raises the confidence floor for
    # accepting an ellipsis_expanded rewrite from 0.85 → 0.92 —
    # SAFE to apply even when the master flag is off.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_REWRITER_STRICT_ELLIPSIS: bool = False
    REWRITER_ELLIPSIS_MIN_CONF: float = 0.92

    # ─────────────────────────────────────────────────────────────
    # Sprint 3B — Low-similarity confidence band tunable.
    # Top-chunk score below which the answer is tagged
    # confidence_band="low" so the frontend can render the
    # "⚠️ Low similarity match" banner.
    # ─────────────────────────────────────────────────────────────
    LOW_SIMILARITY_THRESHOLD: float = 0.35

    # ─────────────────────────────────────────────────────────────
    # Sprint 4 — Fingerprint-First Expert Copilot tunables.
    # FINGERPRINT_REGEX accepts any non-empty string — the JSONB `?`
    # operator in retrieve_by_fingerprint returns no rows on misses.
    # ─────────────────────────────────────────────────────────────
    FINGERPRINT_REGEX: ClassVar[str] = r".+"
    FINGERPRINT_MIN_QUALITY_SCORE: int = 3

    # ─────────────────────────────────────────────────────────────
    # Sprint 11 — Journey-aware retrieval context for /ask.
    # When ON, /ask looks up the chat session's journey_session_id,
    # loads the originating ticket's intake (severity / asset_name /
    # alert_type / customer / ...), and prepends a compact context
    # prefix to the BM25 / keyword query so retrieval narrows toward
    # the right corner of the corpus when the chat originated from a
    # Stage 0 / Stage 3 "Ask in chat" link.
    # The LLM prompt + the visible chat message are NOT touched.
    # Flag-off path: /ask behaves bit-for-bit as before.
    # ─────────────────────────────────────────────────────────────
    LOGIQ_JOURNEY_CHAT_RETRIEVAL_CONTEXT: bool = True

    # ─────────────────────────────────────────────────────────────
    # Sprint 5 — Template-First Expert Copilot + Answer Cache.
    # Cache is invalidated automatically because Sprint 2.9 ingestion
    # DELETEs + re-INSERTs the chunk row on re-upload.
    # ─────────────────────────────────────────────────────────────
    EXPERT_COPILOT_CACHE_TTL_DAYS: int = 30

    # ─────────────────────────────────────────────────────────────
    # Sprint 6 — Tier-1 Alert Copilot tunables.
    # ─────────────────────────────────────────────────────────────
    TIER1_CACHE_TTL_DAYS: int = 7
    TIER1_TOP_K: int = 5
    TIER1_HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    TIER1_MIN_CONFIDENCE_THRESHOLD: float = 0.60

    # ─────────────────────────────────────────────────────────────
    # Sprint 7 — Tier-1 Progressive Workflow tunables.
    # ─────────────────────────────────────────────────────────────
    TIER1_STUCK_THRESHOLD_SECONDS: int = 480   # 8 minutes
    TIER1_TOP_N_MATCHES: int = 5

    # ─────────────────────────────────────────────────────────────
    # Sprint 10 — Tier-1 Resolution Journey tunables.
    # ─────────────────────────────────────────────────────────────
    TIER1_JOURNEY_TOP_N: int = 5
    TIER1_JOURNEY_LOOKBACK_MONTHS: int = 18
    TIER1_JOURNEY_LLM_POLISH: bool = False
    TIER1_JOURNEY_BUNDLE_CACHE_TTL_SECONDS: int = 600
    TIER1_JOURNEY_STAGE0_DOMINANT_THRESHOLD: float = 0.4

    # ─────────────────────────────────────────────────────────────
    # Sprint 9 — Universal Intake (Email/Phone/Portal/Chat/Note)
    # tunables. Catalog builder is lazy: first /intake/extract call
    # materialises the in-memory IntakeCatalogs from
    # chunks.metadata_json (no startup cost).
    # ─────────────────────────────────────────────────────────────
    INTAKE_MAX_RAW_CHARS: int = 10000
    INTAKE_MAX_CANDIDATES: int = 5            # asked of the LLM
    INTAKE_MAX_CARDS: int = 4                 # shown to the engineer
    INTAKE_CATALOG_MAX_TERMS_PER_TYPE: int = 10000
    # Sprint 9 baseline (kept for backward compat — Sprint 9.2 adds
    # per-field thresholds below that take precedence in the validator).
    INTAKE_FUZZY_MATCH_THRESHOLD: float = 0.70
    INTAKE_FUZZY_MATCH_THRESHOLD_CUSTOMER: float = 0.85

    # ─────────────────────────────────────────────────────────────
    # Sprint 9.2 — per-field fuzzy-match thresholds + token bump.
    # The Sprint 9 single-threshold (0.70) was too permissive on short
    # asset names (e.g. NY4-CORE-RTR-01 vs v-bay-core-rtr scored 0.69
    # — a near-miss the validator was rubber-stamping). 9.2 raises the
    # bars per-field so the long-tail asset names that the reproduction
    # tests proved exist in the corpus actually win their lookups.
    # The per-field constants below are the operative thresholds; the
    # legacy INTAKE_FUZZY_MATCH_THRESHOLD constant remains for any
    # external callers but the new validator code never reads it.
    # ─────────────────────────────────────────────────────────────
    INTAKE_FUZZY_THRESHOLD_ASSET: float = 0.85
    INTAKE_FUZZY_THRESHOLD_ALERT: float = 0.80
    INTAKE_FUZZY_THRESHOLD_CUSTOMER: float = 0.85
    # Bumped from Sprint 9's 1500 — evidence-substring fields add tokens.
    INTAKE_EXTRACTION_MAX_TOKENS: int = 2000
    # Weights MUST sum to 1.00. Declared ClassVar so pydantic treats it as
    # a constant, not a settable field (mutable dicts aren't a valid
    # Settings field type and spec §12 explicitly wants a single source of
    # truth callers reuse for the /tier1/explain breakdown).
    TIER1_RANKING_WEIGHTS: ClassVar[Dict[str, float]] = {
        "alert_type_match":      0.25,
        "asset_match":           0.15,
        "fingerprint_match":     0.15,
        "technology_match":      0.05,
        "vector_similarity":     0.05,
        "resolution_quality":    0.10,
        "recency":               0.05,
        "success_frequency":     0.05,
        "same_customer_boost":   0.075,
        "same_asset_family":     0.075,
    }

    # ─────────────────────────────────────────────────────────────
    # Sprint 3-PREP-A — doc_kind multi-corpus tagging.
    # VALID_DOC_KINDS is a ClassVar so pydantic treats it as a constant,
    # not a settable field — the set is immutable and referenced via
    # settings.VALID_DOC_KINDS from ingestion for input validation.
    # ─────────────────────────────────────────────────────────────
    VALID_DOC_KINDS: ClassVar[frozenset] = frozenset({
        "ticket",            # JSON gold-ticket history (Troubleshooting)
        "sop",               # Standard operating procedures, runbooks
        "kb",                # Knowledge base articles
        "contact_customer",  # Customer contact directory (Escalation)
        "contact_vendor",    # Vendor contact directory (Vendor-OEM)
        "vendor_case",       # Historical vendor case records
    })

    # ─────────────────────────────────────────────────────────────
    # Sprint 3-PREP-B — Bulk ingestion (folder / S3 → corpus) tunable.
    # ─────────────────────────────────────────────────────────────
    BULK_INGEST_MAX_FILES_PER_RUN: int = 10000

    model_config = SettingsConfigDict(
        # Production runtime config now comes from AWS Secrets Manager
        # — see backend/core/secrets.py. The bootstrap call in
        # backend/api.py populates os.environ before this Settings
        # instance is built, so pydantic reads the same values without
        # needing to touch the .env file directly.
        #
        # The old behavior (read keys from backend/.env) is kept as a
        # documented fallback path: bootstrap_environment() invokes
        # python-dotenv on backend/.env whenever AWS is unreachable
        # or AWS_SECRETS_DISABLED=true is set. So .env still works on
        # a laptop without AWS access — we just don't let pydantic
        # double-load it (which would override an operator's exported
        # env vars on the command line).
        #
        # env_file=str(BASE_DIR / ".env"),   # ← legacy; loaded via secrets.bootstrap fallback now
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    @field_validator("CLERK_ENABLED", "SES_ENABLED", mode="before")
    @classmethod
    def normalize_string_flags(cls, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        return value


settings = Settings()

AWS_REGION = settings.AWS_REGION
BEDROCK_EMBED_MODEL = settings.BEDROCK_EMBED_MODEL
BEDROCK_LLM_MODEL = settings.BEDROCK_LLM_MODEL

settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
