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

    model_config = SettingsConfigDict(
        env_prefix="SCRAPER_",
        env_file=Path(__file__).parents[1] / ".env",
        extra="ignore",
    )
