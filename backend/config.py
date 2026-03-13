import sys
from pathlib import Path
from typing import List, Optional

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -------------------------------------------------------------------
# Python version safety check
# -------------------------------------------------------------------

if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")


BASE_DIR = Path(__file__).resolve().parent


# -------------------------------------------------------------------
# Settings
# -------------------------------------------------------------------

class Settings(BaseSettings):
    """
    Global application configuration.

    Phase-2 additions:
    - contextual ingestion controls
    - duplicate/version detection flags
    - Bedrock Claude Haiku metadata extraction
    """

    # ----------------------------------------------------------------
    # Vector store / storage
    # ----------------------------------------------------------------

    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    COLLECTION_NAME: str = "logs_titan_v2_1024"

    UPLOAD_DIR: Path = BASE_DIR / "uploads"

    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = [
        "log",
        "txt",
        "md",
        "json",
        "pdf",
        "docx",
    ]

    # ----------------------------------------------------------------
    # AWS / Bedrock
    # ----------------------------------------------------------------

    AWS_REGION: str = "us-east-1"

    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    # Titan embeddings
    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"

    # generation model
    BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"

    # metadata extraction model
    BEDROCK_HAIKU_MODEL: str = "anthropic.claude-3-5-haiku-20241022-v1:0"

    # ----------------------------------------------------------------
    # Database
    # ----------------------------------------------------------------

    DATABASE_URL: str

    # ----------------------------------------------------------------
    # Phase-1 chunking
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

    MAX_METADATA_INPUT_CHARS: int = 12000
    MAX_CONTEXT_SUMMARY_CHARS: int = 280

    MAX_OPERATIONAL_LABELS: int = 12
    MAX_METADATA_RETRIES: int = 3

    HAIKU_TEMPERATURE: float = 0.0
    HAIKU_MAX_TOKENS: int = 900

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

    # ----------------------------------------------------------------
    # Pydantic config
    # ----------------------------------------------------------------

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ----------------------------------------------------------------
    # Validators
    # ----------------------------------------------------------------

    @field_validator("CLERK_ENABLED", "SES_ENABLED", mode="before")
    @classmethod
    def normalize_string_flags(cls, value):
        if isinstance(value, bool):
            return "true" if value else "false"
        return value


# -------------------------------------------------------------------
# Settings instance
# -------------------------------------------------------------------

settings = Settings()


# -------------------------------------------------------------------
# Global exports (used across the project)
# -------------------------------------------------------------------

AWS_REGION = settings.AWS_REGION
BEDROCK_EMBED_MODEL = settings.BEDROCK_EMBED_MODEL
BEDROCK_LLM_MODEL = settings.BEDROCK_LLM_MODEL
CHROMA_PERSIST_DIR = settings.CHROMA_PERSIST_DIR
COLLECTION_NAME = settings.COLLECTION_NAME


# -------------------------------------------------------------------
# Ensure directories exist
# -------------------------------------------------------------------

settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

Path(settings.CHROMA_PERSIST_DIR).mkdir(
    exist_ok=True,
    parents=True,
)