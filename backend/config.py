"""
Phase 3 Configuration — Hybrid Retrieval Orchestration.
Adds search weights, reranker config, query classifier thresholds,
and metadata filter settings on top of Phase 2 ingestion config.
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

    Phase-2: contextual ingestion, duplicate/version detection, Haiku metadata
    Phase-3: hybrid retrieval orchestration, query classification,
             weighted fusion, modular reranking, metadata filters
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
    # AWS / Bedrock
    # ----------------------------------------------------------------

    AWS_REGION: str = "us-east-1"

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"
    BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"
    BEDROCK_HAIKU_MODEL: str = "us.anthropic.claude-3-5-haiku-20241022-v1:0"

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

    # --- Search channel weights (used in RRF fusion) ---
    VECTOR_WEIGHT: float = 0.45       # semantic similarity channel
    BM25_WEIGHT: float = 0.30         # BM25 term-frequency channel
    KEYWORD_WEIGHT: float = 0.25      # PostgreSQL full-text / exact match channel

    # --- Candidate pool sizes per channel ---
    VECTOR_CANDIDATES: int = 25
    BM25_CANDIDATES: int = 20
    KEYWORD_CANDIDATES: int = 15

    # --- Full-text / keyword search ---
    FTS_MIN_RANK: float = 0.01        # minimum pg ts_rank to keep a result
    EXACT_TERM_BOOST: float = 2.0     # extra RRF weight for exact-match terms

    # --- Reciprocal Rank Fusion ---
    RRF_K: int = 60                   # RRF smoothing constant

    # --- Reranker ---
    RERANKER_BACKEND: str = "llm"     # "llm" | "cross_encoder" | "none"
    RERANK_TOP_K: int = 6             # final chunks sent to LLM context
    RERANK_CANDIDATES: int = 15       # how many fused results enter reranker
    RERANK_SCORE_WEIGHT: float = 0.70 # reranker score contribution in final blend
    RERANK_FUSION_WEIGHT: float = 0.30  # original fusion score contribution

    # --- Query classifier ---
    KEYWORD_QUERY_THRESHOLD: float = 0.60   # classifier score above this → keyword-heavy
    SEMANTIC_QUERY_THRESHOLD: float = 0.60  # classifier score above this → semantic-heavy
    ENABLE_QUERY_CLASSIFICATION: bool = True

    # --- Metadata filter ---
    ENABLE_METADATA_FILTER: bool = True
    METADATA_FILTER_CANDIDATES: int = 10

    # --- Grounding gate ---
    MIN_GROUNDING_SCORE: float = 0.18
    MIN_KEYWORD_OVERLAP: int = 1

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
