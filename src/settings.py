import os
from dataclasses import dataclass

from dotenv import load_dotenv


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


@dataclass(frozen=True)
class Settings:
    database_url: str
    collection_name: str
    pdf_path: str
    openai_api_key: str
    google_api_key: str
    openai_embedding_model: str
    google_embedding_model: str
    openai_chat_model: str
    google_chat_model: str
    provider_priority: str
    enable_local_fallback: bool
    local_embedding_dim: int
    llm_timeout_seconds: int
    input_max_chars: int
    max_context_distance: float


def _get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if value == "":
        return default

    return value


def get_settings() -> Settings:
    return Settings(
        database_url=_get_env("DATABASE_URL"),
        collection_name=_get_env("PG_VECTOR_COLLECTION_NAME", "pdf_chunks"),
        pdf_path=_get_env("PDF_PATH", os.path.join(PROJECT_ROOT, "document.pdf")),
        openai_api_key=_get_env("OPENAI_API_KEY"),
        google_api_key=_get_env("GOOGLE_API_KEY"),
        openai_embedding_model=_get_env("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        google_embedding_model=_get_env("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"),
        openai_chat_model=_get_env("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        google_chat_model=_get_env("GOOGLE_CHAT_MODEL", "gemini-2.0-flash"),
        provider_priority=_get_env("LLM_PROVIDER_PRIORITY", "google,openai"),
        enable_local_fallback=_get_env("ENABLE_LOCAL_FALLBACK", "true").lower() == "true",
        local_embedding_dim=int(_get_env("LOCAL_EMBEDDING_DIM", "256")),
        llm_timeout_seconds=int(_get_env("LLM_TIMEOUT_SECONDS", "30")),
        input_max_chars=int(_get_env("INPUT_MAX_CHARS", "1000")),
        max_context_distance=float(_get_env("MAX_CONTEXT_DISTANCE", "0.85")),
    )


def require_database_url() -> str:
    settings = get_settings()
    if not settings.database_url:
        raise ValueError("Defina DATABASE_URL no arquivo .env")
    return settings.database_url


def require_provider() -> str:
    settings = get_settings()
    if settings.openai_api_key:
        return "openai"
    if settings.google_api_key:
        return "google"
    raise ValueError("Defina OPENAI_API_KEY ou GOOGLE_API_KEY no arquivo .env")
