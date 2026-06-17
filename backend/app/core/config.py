from pydantic_settings import BaseSettings, SettingsConfigDict


API_VERSION = "0.1.0"


class Settings(BaseSettings):
    APP_NAME: str = "SmartBuyAgent"
    APP_ENV: str = "development"
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DATABASE_URL: str = "sqlite:///./data/smartbuy.db"
    CHROMA_DIR: str = "./data/chroma"
    LOG_LEVEL: str = "INFO"
    REDIS_URL: str | None = None
    SESSION_CACHE_TTL_SECONDS: int = 1800
    RETRIEVAL_PRODUCT_CACHE_TTL_SECONDS: int = 600
    RETRIEVAL_KNOWLEDGE_CACHE_TTL_SECONDS: int = 1800
    SSE_TRACE_TTL_SECONDS: int = 1800
    RATE_LIMIT_WINDOW_SECONDS: int = 10
    RATE_LIMIT_MAX_REQUESTS: int = 5
    FEEDBACK_CACHE_TTL_SECONDS: int = 604800
    EMBEDDING_PROVIDER: str = "mock"
    EMBEDDING_DIM: int = 32
    EMBEDDING_API_BASE: str | None = None
    EMBEDDING_API_KEY: str | None = None
    EMBEDDING_MODEL: str | None = None
    EMBEDDING_TIMEOUT_SECONDS: float = 30.0
    LLM_PROVIDER: str = "mock"
    LLM_API_BASE: str | None = None
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "mock-chat"
    LLM_TIMEOUT_SECONDS: float = 30.0
    LLM_MAX_TOKENS: int = 800
    LLM_TEMPERATURE: float = 0.2

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
