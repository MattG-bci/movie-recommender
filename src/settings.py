from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from urllib.parse import quote


class DBSettings(BaseSettings):
    HOST: str
    USER: str
    PASS: str
    NAME: str
    PORT: int

    model_config = SettingsConfigDict(
        env_prefix="DB_", env_file=Path(__file__).parents[1] / ".env", extra="ignore"
    )

    def get_postgres_dsn(self, prefix: str = "postgres") -> str:
        dsn = f"{prefix}://{self.USER}:{quote(self.PASS)}@{self.HOST}:{self.PORT}/{self.NAME}"
        return dsn


class WebScraperSettings(BaseSettings):
    USERNAME_PAGE: str
    RATINGS_PAGE: str
    MOVIES_PAGE: str

    HEADERS: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "identity",
        "Connection": "keep-alive",
    }

    model_config = SettingsConfigDict(
        env_prefix="SCRAPER_",
        env_file=Path(__file__).parents[1] / ".env",
        extra="ignore",
    )


class LLMSettings(BaseSettings):
    API_KEY: str
    MODEL: str = "claude-sonnet-4-6"

    model_config = SettingsConfigDict(
        env_prefix="ANTHROPIC_",
        env_file=Path(__file__).parents[1] / ".env",
        extra="ignore",
    )
