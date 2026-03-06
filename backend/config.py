import sys
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


if sys.version_info < (3, 11):
    raise RuntimeError("This application requires Python 3.11 or higher")

BASE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    CHROMA_PERSIST_DIR: str = str(BASE_DIR / "data" / "chroma")
    COLLECTION_NAME: str = "logs_titan_v2_1024"

    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_FILE_TYPES: List[str] = ["log", "txt", "md", "json", "pdf", "docx"]

    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_SESSION_TOKEN: Optional[str] = None

    # Embeddings: Titan Embed V2 (8192 token input limit)
    BEDROCK_EMBED_MODEL: str = "amazon.titan-embed-text-v2:0"

    # LLM: Mistral 7B Instruct (32K context window)
    BEDROCK_LLM_MODEL: str = "mistral.mistral-7b-instruct-v0:2"

    # Titan Embed V2 input limit ~8192 tokens.
    # Chunks are max 6000 chars, so 8000 gives headroom.
    MAX_CHARS: int = 8000
    OVERLAP: int = 300
    BATCH_SIZE: int = 10

    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "INFO"

    API_KEY: Optional[str] = None
    UI_API_KEY: Optional[str] = None
    API_BASE: str = "http://localhost:8000"
    REQUEST_TIMEOUT: int = 30

    # ── Clerk Authentication ──────────────────────────────────
    # Get these from https://dashboard.clerk.com → API Keys
    CLERK_SECRET_KEY: Optional[str] = None
    CLERK_PUBLISHABLE_KEY: Optional[str] = None
    # Set to "true" to enforce Clerk JWT auth on all endpoints
    CLERK_ENABLED: str = "false"

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()

# Backward-compatible exports
AWS_REGION = settings.AWS_REGION
BEDROCK_EMBED_MODEL = settings.BEDROCK_EMBED_MODEL
BEDROCK_LLM_MODEL = settings.BEDROCK_LLM_MODEL
CHROMA_PERSIST_DIR = settings.CHROMA_PERSIST_DIR
COLLECTION_NAME = settings.COLLECTION_NAME

settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
Path(settings.CHROMA_PERSIST_DIR).mkdir(exist_ok=True, parents=True)
