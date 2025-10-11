from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class DBSettings(BaseSettings):
    HOST: str
    USER: str
    PASS: str
    NAME: str
    PORT: int

    model_config = SettingsConfigDict(
        env_prefix="DB_", env_file=Path(__file__).parents[1] / ".env", extra="ignore"
    )


class WebScraperSettings(BaseSettings):
    USERNAME_PAGE: str
    RATINGS_PAGE: str
    MOVIES_PAGE: str

    model_config = SettingsConfigDict(
        env_prefix="SCRAPER_",
        env_file=Path(__file__).parents[1] / ".env",
        extra="ignore",
    )
