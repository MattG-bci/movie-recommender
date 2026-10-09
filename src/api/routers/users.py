from fastapi import APIRouter, HTTPException

from etl.sql_queries import DatabaseConnector, fetch_user_by_username
from schemas.users import User

users = APIRouter()


@users.get("/{username}")
async def get_user(username: str) -> User:
    async with DatabaseConnector() as conn:
        user = await fetch_user_by_username(conn, username)

    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    return user
