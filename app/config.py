from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration, loaded from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    openai_api_key: str = ""

    llm_model: str = "claude-sonnet-5"
    embedding_model: str = "text-embedding-3-small"

    database_url: str = "postgresql://postgres:devpassword@localhost:5432/rag_db"

    default_company_id: str = "demo-company"

    app_env: str = "development"


settings = Settings()
