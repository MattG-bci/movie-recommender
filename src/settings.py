import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


class DBSettings(BaseSettings):
    HOST: str
    USER: str
    PASS: str
    NAME: str
    PORT: int

    model_config = SettingsConfigDict(env_prefix="DB_")


class WebScraperSettings(BaseSettings):
    USERNAME_PAGE: str
    RATINGS_PAGE: str

    model_config = SettingsConfigDict(env_prefix="SCRAPER_")
