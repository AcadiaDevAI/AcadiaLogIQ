"""
Phase 5 Configuration — Multi-Agent Troubleshooting Layer.
Adds agent escalation thresholds, per-agent model/token budgets,
and cost controls. Includes all Phase 2 + 3 + 4 settings.
"""

import sys
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")


BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Global application configuration.

    Phase 2: contextual ingestion, duplicate/version detection, Haiku metadata
    Phase 3: hybrid retrieval orchestration, query classification, fusion, reranking
    Phase 4: complexity classification, cost-optimized model routing, context builder
    Phase 5: selective multi-agent troubleshooting for complex queries only
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
    # Phase-2 contextual chunking
    # ----------------------------------------------------------------

    CHUNK_MAX_CHARS: int = 2200
    CHUNK_MIN_CHARS: int = 450
    CHUNK_OVERLAP_CHARS: int = 150
    CHUNK_BATCH_SIZE: int = 6

    MAX_METADATA_INPUT_CHARS: int = 1800
    MAX_CONTEXT_SUMMARY_CHARS: int = 120
    MAX_OPERATIONAL_LABELS: int = 8
    MAX_METADATA_RETRIES: int = 2

    HAIKU_TEMPERATURE: float = 0.0
    HAIKU_MAX_TOKENS: int = 4096

    # ----------------------------------------------------------------
    # Concurrency (Phase 2)
    # ----------------------------------------------------------------

    METADATA_CONCURRENCY: int = 4
    EMBED_CONCURRENCY: int = 8

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
    RERANK_TOP_K: int = 6
    RERANK_CANDIDATES: int = 15
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
    # Phase-4: Model Routing & Context-Aware Generation
    # ----------------------------------------------------------------

    ENABLE_MODEL_ROUTING: bool = True
    ROUTING_DEFAULT_MODEL: str = "mistral"

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

    # --- Master switch ---
    ENABLE_AGENT_MODE: bool = True            # False = Phase 4 behavior, no agents

    # --- Escalation gate ---
    # Agent mode activates ONLY when complexity tier is 'complex' AND
    # the query matches one of the agent-eligible patterns.
    # This prevents agents from firing on simple or moderate queries.
    AGENT_COMPLEXITY_THRESHOLD: float = 0.65  # complexity score must exceed this
    AGENT_MIN_SOURCES: int = 1                # minimum source docs to justify agents
    AGENT_MAX_STEPS: int = 4                  # max planning steps (cost ceiling)

    # --- Per-agent model assignment ---
    AGENT_PLANNER_MODEL: str = "sonnet"       # Planner needs deep reasoning
    AGENT_ANALYSIS_MODEL: str = "haiku"       # Analysis is mid-tier
    AGENT_COMPOSER_MODEL: str = "haiku"       # Composer assembles final answer

    # --- Per-agent token budgets ---
    AGENT_PLANNER_MAX_TOKENS: int = 1024      # plan is compact
    AGENT_ANALYSIS_MAX_TOKENS: int = 1500     # per-step analysis
    AGENT_COMPOSER_MAX_TOKENS: int = 2048     # final answer

    # --- Cost controls ---
    AGENT_MAX_TOTAL_TOKENS: int = 8000        # hard ceiling across all agents
    AGENT_TIMEOUT_SECONDS: int = 45           # total wall-clock limit for agent flow

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
