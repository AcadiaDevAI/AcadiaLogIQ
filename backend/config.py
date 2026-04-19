"""
Configuration — Phases 2-6 Complete + Accuracy Fixes + Performance Tuning.
Includes: contextual ingestion, hybrid retrieval, model routing,
multi-agent troubleshooting, answer validation guardrails,
accuracy fixes, and performance optimizations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    CHUNK_OVERLAP_CHARS: int = 0
    CHUNK_BATCH_SIZE: int = 10          # ← was 6, now 10 chunks per Haiku call (fewer API calls)

    LLM_CHUNK_FALLBACK_PREVIEW_CHARS: int = 8000
    ENABLE_LLM_CHUNK_FALLBACK: bool = True

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
    RESPONSE_TOKENS_EXPLANATION: int = 350
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
    VALIDATION_MIN_GROUNDING: float = 0.25
    VALIDATION_MIN_COVERAGE: float = 0.20

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

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
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
