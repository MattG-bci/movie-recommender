from fastapi import APIRouter, HTTPException

from etl.generation.generate import create_uuid_for_user
from etl.sql_queries import DatabaseConnector, fetch_user_by_username, insert_usernames
from schemas.users import User, UserWithUUID
import httpx
from settings import WebScraperSettings
import logging


logging.basicConfig(level=logging.INFO)

users = APIRouter()


@users.get("/{username}")
async def get_user(username: str) -> User:
    async with DatabaseConnector() as conn:
        user = await fetch_user_by_username(conn, username)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user


@users.post("/{username}")
async def ingest_username(username: str) -> None:
    settings = WebScraperSettings()
    username = username.lower().strip()
    path = str(settings.BASE_URL + username)
    try:
        with httpx.Client(
            timeout=30.0, follow_redirects=True, headers=settings.HEADERS
        ) as client:
            response = client.get(path)
            response.raise_for_status()
    except httpx.HTTPStatusError:
        raise HTTPException(status_code=404, detail="User not found")

    user_to_insert = UserWithUUID(username=username, id=create_uuid_for_user(username))
    async with DatabaseConnector() as conn:
        await insert_usernames(conn, [user_to_insert])
