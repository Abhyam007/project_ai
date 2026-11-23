from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Google Gemini (env var: LLM_API_KEY)
    google_api_key: str = Field(alias="LLM_API_KEY")

    # Supabase Storage
    supabase_url: str = Field(default="", alias="SUPABASE_URL")
    supabase_service_key: str

    # Direct Postgres connection (for pgvector queries)
    database_url: str

    # Upload limits
    max_upload_bytes: int = 25 * 1024 * 1024  # 25 MB

    # RAG settings
    top_k_passages: int = 8
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Embedding
    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = 768

    # LLM
    llm_model: str = "gemini-3.1-flash-lite"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
