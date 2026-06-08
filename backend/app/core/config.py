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

    model_config = SettingsConfigDict(
        env_file=(".env", "../.env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
